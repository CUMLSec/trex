# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from command import configs
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('similarity')
class SimilarityCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name, classification_pair_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.classification_pair_head_name = classification_pair_head_name
        self.fields = configs.fields

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='similarity',
                            help='name of the classification head to use')
        parser.add_argument('--classification-pair-head-name',
                            default='similarity_pair',
                            help='name of the classification head to use to classify on function pair')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
                hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads
                and self.classification_pair_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=similarity'

        logits0 = model(
            sample['net_input0']['src_tokens'],
            classification_head_name=self.classification_head_name
        )[0]['features']
        logits1 = model(
            sample['net_input1']['src_tokens'],
            classification_head_name=self.classification_head_name
        )[0]['features']
        targets = model.get_targets(sample, [logits0]).view(-1)

        sample_size = targets.numel()

        targets_cosine_thresholded = targets > configs.cosine_embedding_loss_margin

        loss_cosine = F.cosine_embedding_loss(logits0, logits1, targets, margin=configs.cosine_embedding_loss_margin,
                                              reduction='sum')

        func_embedding0 = model(
            sample['net_input0']['src_tokens'],
            features_only=True,
            classification_head_name=None
        )[0]['features']
        func_embedding1 = model(
            sample['net_input1']['src_tokens'],
            features_only=True,
            classification_head_name=None
        )[0]['features']

        # pool the embedding
        func1_embedding_mean = torch.mean(func_embedding0, dim=1)
        func2_embedding_mean = torch.mean(func_embedding1, dim=1)

        # concatenate mean(func), u, v, |u-v|, u*v
        concat_in = torch.cat((func1_embedding_mean,
                               func2_embedding_mean,
                               torch.abs(func1_embedding_mean - func2_embedding_mean),
                               func1_embedding_mean * func2_embedding_mean),
                              dim=-1)

        logits = model.classification_heads[self.classification_pair_head_name](concat_in)

        # target, 0 - not similar, 1 - similar
        targets_pair = (sample["target"] > 0).long()

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss_pair = F.nll_loss(lprobs, targets_pair, reduction='sum')

        loss = loss_cosine + loss_pair

        logging_output = {
            'loss': loss.data,
            'loss_cosine': loss_cosine.data,
            'loss_pair': loss_pair.data,
            'ntokens': sample['ntokens0'] + sample['ntokens1'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds_cosine = torch.cosine_similarity(logits0, logits1, dim=1)
        preds_cosine_thresholded = torch.cosine_similarity(logits0, logits1,
                                                           dim=1) > configs.cosine_embedding_loss_margin
        preds_pair = logits.argmax(dim=1)
        preds_pair_scaled = F.softmax(logits, dim=-1)[:, 1] * 2 - 1

        if random.random() < 0.001:
            print(preds_cosine, preds_pair_scaled, targets)

        logging_output['ncorrect_total_cosine'] = (preds_cosine_thresholded == targets_cosine_thresholded).sum().item()
        logging_output['ncorrect_cosine'] = ((preds_cosine_thresholded == targets_cosine_thresholded) * (
                targets_cosine_thresholded == 1)).sum().item()
        logging_output['ncorrect_actual_cosine'] = (targets_cosine_thresholded == 1).sum().item()
        logging_output['ncorrect_pred_cosine'] = (preds_cosine_thresholded == 1).sum().item()
        logging_output['preds_cosine'] = preds_cosine.detach().cpu().numpy().tolist()

        logging_output['targets'] = targets.detach().cpu().numpy().tolist()

        logging_output['ncorrect_total_pair'] = (preds_pair == targets_pair).sum().item()
        logging_output['ncorrect_pair'] = ((preds_pair == targets_pair) * (targets_pair == 1)).sum().item()
        logging_output['ncorrect_actual_pair'] = (targets_pair == 1).sum().item()
        logging_output['ncorrect_pred_pair'] = (preds_pair == 1).sum().item()
        logging_output['preds_pair_scaled'] = preds_pair_scaled.detach().cpu().numpy().tolist()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_cosine_sum = sum(log.get('loss_cosine', 0) for log in logging_outputs)
        loss_pair_sum = sum(log.get('loss_pair', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('loss_cosine', loss_cosine_sum / sample_size, ntokens, round=3)
        metrics.log_scalar('loss_pair', loss_pair_sum / sample_size, ntokens, round=3)

        if len(logging_outputs) > 0 and len(
                set(itertools.chain.from_iterable([log.get('targets', 0) for log in logging_outputs]))) == 2:
            preds_cosine = list(itertools.chain.from_iterable([log.get('preds_cosine', 0) for log in logging_outputs]))
            preds_pair_scaled = list(
                itertools.chain.from_iterable([log.get('preds_pair_scaled', 0) for log in logging_outputs]))
            targets = list(itertools.chain.from_iterable([log.get('targets', 0) for log in logging_outputs]))

            auc_cosine = roc_auc_score(targets, preds_cosine)
            auc_pair = roc_auc_score(targets, preds_pair_scaled)

            metrics.log_scalar('AUC_cosine', auc_cosine, nsentences, round=4)
            metrics.log_scalar('AUC_pair', auc_pair, nsentences, round=4)

            # F1 might be messy as it depends on chosen threshold
            ncorrect_cosine = sum(log.get('ncorrect_cosine', 0) for log in logging_outputs)
            ncorrect_pred_cosine = sum(log.get('ncorrect_pred_cosine', 0) for log in logging_outputs)
            ncorrect_actual_cosine = sum(log.get('ncorrect_actual_cosine', 0) for log in logging_outputs)
            ncorrect_total_cosine = sum(log.get('ncorrect_total_cosine', 0) for log in logging_outputs)

            precision_cosine = 100 * ncorrect_cosine / (ncorrect_pred_cosine + 1e-5)
            recall_cosine = 100 * ncorrect_cosine / (ncorrect_actual_cosine + 1e-5)

            metrics.log_scalar('accuracy_cosine', 100.0 * ncorrect_total_cosine / nsentences, nsentences, round=1)
            metrics.log_scalar('precision_cosine', precision_cosine, nsentences, round=1)
            metrics.log_scalar('recall_cosine', recall_cosine, nsentences, round=1)
            metrics.log_scalar('F1_cosine',
                               2 * (precision_cosine * recall_cosine) / (precision_cosine + recall_cosine + 1e-8),
                               nsentences, round=1)

            ncorrect_pair = sum(log.get('ncorrect_pair', 0) for log in logging_outputs)
            ncorrect_pred_pair = sum(log.get('ncorrect_pred_pair', 0) for log in logging_outputs)
            ncorrect_actual_pair = sum(log.get('ncorrect_actual_pair', 0) for log in logging_outputs)
            ncorrect_total_pair = sum(log.get('ncorrect_total_pair', 0) for log in logging_outputs)

            precision_pair = 100 * ncorrect_pair / (ncorrect_pred_pair + 1e-5)
            recall_pair = 100 * ncorrect_pair / (ncorrect_actual_pair + 1e-5)

            metrics.log_scalar('accuracy_pair', 100.0 * ncorrect_total_pair / nsentences, nsentences, round=1)
            metrics.log_scalar('precision_pair', precision_pair, nsentences, round=1)
            metrics.log_scalar('recall_pair', recall_pair, nsentences, round=1)
            metrics.log_scalar('F1_pair', 2 * (precision_pair * recall_pair) / (precision_pair + recall_pair + 1e-8),
                               nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
