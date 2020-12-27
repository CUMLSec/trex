# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('clr')
class CLRCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='clr_head',
                            help='name of the classification head to use')
        parser.add_argument('--freeze-encoder', action='store_true', default=False,
                            help='Freeze encoder weights and disable encoder dropout during training')
        parser.add_argument('--freeze-embedding', action='store_true', default=False,
                            help='Freeze embedding weights for zero shot transfer')
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
                and self.args.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=clr'

        logits0, _ = model(
            **sample['net_input0'],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            freeze_encoder=self.args.freeze_encoder,
            freeze_embedding=self.args.freeze_embedding
        )
        logits1, _ = model(
            **sample['net_input1'],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            freeze_encoder=self.args.freeze_encoder,
            freeze_embedding=self.args.freeze_embedding
        )
        targets = model.get_targets(sample, [logits0]).view(-1)
        sample_size = targets.numel()

        # loss = F.cosine_embedding_loss(logits0, logits1, targets, margin=0, reduction='sum')
        # targets = targets.float()
        loss = F.mse_loss(
            torch.cosine_similarity(logits0, logits1, dim=1),
            targets,
            reduction='sum',
        )

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens0'] + sample['ntokens1'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = torch.cosine_similarity(logits0, logits1, dim=1)
        logging_output['ncorrect'] = utils.item((((preds > 0) == (targets > 0)) * (targets > 0)).sum())
        logging_output['ncorrect_total'] = utils.item(((preds > 0) == (targets > 0)).sum())
        logging_output['ncorrect_pred'] = utils.item((preds > 0).sum())
        logging_output['ncorrect_actual'] = utils.item((targets > 0).sum())

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0] \
                and 'ncorrect_pred' in logging_outputs[0] and 'ncorrect_actual' in logging_outputs[0] \
                and 'ncorrect_total' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            ncorrect_pred = sum(log.get('ncorrect_pred', 0) for log in logging_outputs)
            ncorrect_actual = sum(log.get('ncorrect_actual', 0) for log in logging_outputs)
            ncorrect_total = sum(log.get('ncorrect_total', 0) for log in logging_outputs)

            precision = 100 * ncorrect / (ncorrect_pred + 1e-5)
            recall = 100 * ncorrect / (ncorrect_actual + 1e-5)

            metrics.log_scalar('accuracy', 100.0 * ncorrect_total / nsentences, nsentences, round=1)
            metrics.log_scalar('precision', precision, nsentences, round=1)
            metrics.log_scalar('recall', recall, nsentences, round=1)
            metrics.log_scalar('F1', 2 * (precision * recall) / (precision + recall + 1e-5), nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
