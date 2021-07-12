# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders

from command import configs


class TrexHubInterface(nn.Module):
    """A simple PyTorch Hub interface adapted from RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(cfg.bpe)

        print('init hub')
        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def __process_token_dict(self, tokens: dict):
        token_fields = tokens.keys()
        assert len(token_fields) == len(configs.fields)

        for field in configs.fields:
            if tokens[field].dim() == 1:
                tokens[field] = tokens[field].unsqueeze(0)
            if tokens[field].size(-1) > self.model.max_positions():
                raise ValueError(
                    "tokens exceeds maximum length: {} > {}".format(
                        tokens[field].size(-1), self.model.max_positions()
                    )
                )
            tokens[field] = tokens[field].to(device=self.device).long()

        return tokens

    def encode(self, sentence: dict) -> torch.LongTensor:
        """
        encode a code piece to its embedding index.
        This is executed individually with extract_features or predict
        """
        sentence_fields = sentence.keys()
        assert len(sentence_fields) == len(configs.fields)

        token_dict = dict()

        for field in configs.fields:
            token_dict[field] = self.task.source_dictionary[field].encode_line(
                sentence[field], append_eos=False, add_if_not_exist=False
            )

        return token_dict

    def decode(self, token_dict: dict):
        token_fields = token_dict.keys()
        assert len(token_fields) == len(configs.fields)

        sentence_dict = dict()

        for field in configs.fields:
            tokens = token_dict[field]
            assert tokens.dim() == 1
            tokens = tokens.numpy()
            if tokens[0] == self.task.source_dictionary[field].bos():
                tokens = tokens[1:]  # remove <s>
            eos_mask = tokens == self.task.source_dictionary[field].eos()
            doc_mask = eos_mask[1:] & eos_mask[:-1]
            sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
            sentences = [
                self.task.source_dictionary[field].string(s) for s in sentences
            ]
            if len(sentences) == 1:
                sentence_dict[field] = sentences[0]
            else:
                sentence_dict[field] = sentences

        return sentence_dict

    def extract_features(self, tokens: dict) -> torch.Tensor:
        tokens = self.__process_token_dict(tokens)

        features = self.model(
            tokens,
            features_only=True
        )[0]

        return features  # just the last layer's features

    def predict(self, head: str, tokens: dict):
        features = self.extract_features(tokens)
        logits = self.model.classification_heads[head](features)
        return logits
