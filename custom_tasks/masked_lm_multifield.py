# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    PadMultipleDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils

from command import configs

logger = logging.getLogger(__name__)


@register_task('masked_lm_multifield')
class MaskedLMMultifieldTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')

        # Output specification
        parser.add_argument('--output-lang', default='static', type=str,
                            help='the language the model needs to predict')
        parser.add_argument('--trace-weight', default=0.1, type=float,
                            help='weigh the loss for dynamic traces')

    def __init__(self, args, dictionary_dict):
        super().__init__(args)
        self.dictionary_dict = dictionary_dict
        self.seed = args.seed
        # self.multiple = self.args.bucket_size * 2

        # add mask token
        self.mask_idx_dict = {}
        for field, dictionary in dictionary_dict.items():
            self.mask_idx_dict[field] = dictionary.add_symbol('<mask>')

        # All field of each token
        self.fields = configs.fields

    @classmethod
    def setup_task(cls, args, **kwargs):
        # paths = utils.split_paths(args.data)
        paths = os.listdir(args.data)
        # assert len(paths) > 0
        assert len(paths) == len(configs.fields)
        for path, field in zip(sorted(paths), sorted(configs.fields)):
            assert path == field

        dictionary_dict = {}
        for field in configs.fields:
            dictionary_dict[field] = Dictionary.load(os.path.join(args.data, field, 'dict.txt'))
            logger.info(f'{field} dictionary: {dictionary_dict[field]} types')
        return cls(args, dictionary_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        src_tokens = {}
        target = {}
        for field in self.fields:
            split_path = os.path.join(self.args.data, field, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary[field].pad(),
                eos=self.source_dictionary[field].eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info('field {} loaded {} blocks from: {}'.format(field, len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary[field].bos())

            # create masked input and targets
            mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary[field]) \
                if self.args.mask_whole_words else None

            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary[field],
                pad_idx=self.source_dictionary[field].pad(),
                mask_idx=self.mask_idx_dict[field],
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            src_tokens[field] = PadDataset(
                src_dataset,
                pad_idx=self.source_dictionary[field].pad(),
                left_pad=False,
            )

            # For Reformer
            # src_tokens[field] = PadMultipleDataset(
            #     src_dataset,
            #     pad_idx=self.source_dictionary[field].pad(),
            #     left_pad=False,
            #     multiple=self.multiple
            # )

            target[field] = PadDataset(
                tgt_dataset,
                pad_idx=self.source_dictionary[field].pad(),
                left_pad=False,
            )

            # For Reformer
            # target[field] = PadMultipleDataset(
            #     tgt_dataset,
            #     pad_idx=self.source_dictionary[field].pad(),
            #     left_pad=False,
            #     multiple=self.multiple
            # )

        net_input = dict()
        net_input['src_tokens'] = src_tokens
        net_input['src_lengths'] = NumelDataset(src_dataset, reduce=False)

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        # Net input has multiple fields
        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'net_input': net_input,
                    'target': target,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        for field in self.fields:
            src_dataset = PadDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    self.args.tokens_per_sample - 1,  # one less for <s>
                    pad=self.source_dictionary[field].pad(),
                    eos=self.source_dictionary[field].eos(),
                    break_mode='eos',
                ),
                pad_idx=self.source_dictionary[field].pad(),
                left_pad=False,
            )
            src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())

        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict
