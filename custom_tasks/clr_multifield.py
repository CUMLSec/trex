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
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
)
from fairseq.tasks import FairseqTask, register_task
from command import configs

logger = logging.getLogger(__name__)


@register_task('clr_multifield')
class CLRMultifieldTask(FairseqTask):
    """
    Contrastive Learning Representation task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='truncate sequence to max-positions')

    def __init__(self, args, data_dictionary_dict, label_dictionary):
        super().__init__(args)
        self.dictionary_dict = data_dictionary_dict
        self._label_dictionary = label_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load data dictionary
        data_dictionary_dict = {}
        for field in configs.fields:
            data_dictionary_dict[field] = cls.load_dictionary(
                args,
                os.path.join(args.data, 'input0', f'{field}', 'dict.txt'),
                source=True,
            )
            logger.info('[input {}] dictionary: {} types'.format(field, len(data_dictionary_dict[field])))

        label_dict = data_dictionary_dict  # dummy set as we don't have discrete label
        return CLRMultifieldTask(args, data_dictionary_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, field, split):
            return os.path.join(self.args.data, type, field, split)

        def make_dataset(type, field, dictionary):
            split_path = get_path(type, field, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = {}
        input1 = {}
        for field in configs.fields:
            input0[field] = make_dataset('input0', field, self.source_dictionary[field])
            assert input0[field] is not None, 'could not find dataset: {}'.format(get_path('input0', field, split))
            input1[field] = make_dataset('input1', field, self.source_dictionary[field])
            assert input1[field] is not None, 'could not find dataset: {}'.format(get_path('input1', field, split))
            assert len(input0[field]) == len(input1[field]), 'input pair different length'

            if self.args.init_token is not None:
                input0[field] = PrependTokenDataset(input0[field], self.args.init_token)
                input1[field] = PrependTokenDataset(input1[field], self.args.init_token)

            if self.args.truncate_sequence:
                input0[field] = TruncateDataset(input0[field], self.args.max_positions)
                input1[field] = TruncateDataset(input1[field], self.args.max_positions)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(input0[field]))

        dataset = {
            'id': IdDataset(),
            'net_input0': {
                'src_tokens': {field: RightPadDataset(input0[field], pad_idx=self.source_dictionary[field].pad()) for
                               field in configs.fields},
                'src_lengths': NumelDataset(input0[field], reduce=False),
            },
            'net_input1': {
                'src_tokens': {field: RightPadDataset(input1[field], pad_idx=self.source_dictionary[field].pad()) for
                               field in configs.fields},
                'src_lengths': NumelDataset(input1[field], reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens0': NumelDataset(input0[field], reduce=True),
            'ntokens1': NumelDataset(input1[field], reduce=True),
        }

        label_path = "{0}.label".format(get_path('label', '', split))
        if os.path.exists(label_path):
            dataset.update(
                target=RawLabelDataset([
                    float(x.strip()) for x in open(label_path).readlines()
                ])
            )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum(input0[field].sizes, input1[field].sizes)],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_clr_head(getattr(args, 'classification_head_name', 'clr_head'))

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def label_dictionary(self):
        return self._label_dictionary
