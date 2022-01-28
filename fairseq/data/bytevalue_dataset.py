# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch

from fairseq.data import data_utils
from . import BaseWrapperDataset


# newly added
class BytevalueDataset(BaseWrapperDataset):

    def __init__(self, dataset, vocab):
        super().__init__(dataset)
        self.vocab = vocab
        # real bytes index
        hexval = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
        self.real_bytes_idx = set(self.vocab.index(f'{i}{j}') for i, j in product(hexval, repeat=2))

    def __getitem__(self, index):
        item = self.dataset[index]
        output = torch.ones_like(item, dtype=torch.float16)
        for i, idx in enumerate(item):
            if idx.item() not in self.real_bytes_idx:
                output[i] = float(1)
            else:
                output[i] = int(self.vocab[idx], 16) / 256
        return output

    def collater(self, samples):
        return data_utils.collate_tokens(samples, float(1))
