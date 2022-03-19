import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from command import configs
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TrexEncoder(FairseqEncoder):
    """
    Adapted from Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (dict of torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens[configs.static_field].embedding_dim

        # assume the padding will be the same for a sequences,
        self.padding_idx = embed_tokens[configs.static_field].padding_idx

        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        if args.seq_combine == 'concat':
            self.seq_agg = SeqCombineConcat(embed_dim)
        else:
            self.seq_agg = None

        if args.input_combine == 'cnn':
            self.byte_combine = ByteCombineCNN(1, embed_dim)
        else:  # input_combine == 'sum'
            self.byte_combine = ByteCombineSUM()

        # tailored for torchscript
        self.drop_field = args.drop_field

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.byte_fields: Dict[str, torch.Tensor] = {byte_field: torch.tensor([]) for byte_field in configs.byte_fields}

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(self, src_tokens: Dict[str, torch.Tensor]):
        if self.seq_agg is not None:
            token_embeddings = []
            # embed tokens (static)
            token_embeddings.append(self.embed_tokens[configs.static_field](src_tokens[configs.static_field]))
            # embed auxiliary annotations (inst_pos, op_pos, arch)
            for field in configs.aux_fields:
                token_embeddings.append(self.embed_tokens[field](src_tokens[field]))
            # embed bytes
            byte_embedding_stack = []
            for field in self.byte_fields:
                byte_embedding_stack.append(src_tokens[field].unsqueeze(-1).type_as(token_embeddings[0]))
            byte_embedding = self.byte_combine(torch.stack(byte_embedding_stack, dim=2))

            token_embeddings.append(byte_embedding)

            token_embeddings_agg = self.seq_agg(torch.cat(token_embeddings, dim=-1))
            x = embed = self.embed_scale * token_embeddings_agg

        else:
            # embed tokens (static)
            if self.drop_field != 'static':
                token_embedding = self.embed_tokens[configs.static_field](src_tokens[configs.static_field])
            else:
                token_embedding = torch.tensor([])
            # embed auxiliary annotations (inst_pos, op_pos, arch)
            if len(token_embedding.size()) == 0:
                token_embedding = self.embed_tokens['inst_pos_emb'](src_tokens['inst_pos_emb'])
            else:
                if self.drop_field != 'inst_pos_emb':
                    token_embedding += self.embed_tokens['inst_pos_emb'](src_tokens['inst_pos_emb'])

            if self.drop_field != 'op_pos_emb':
                token_embedding += self.embed_tokens['op_pos_emb'](src_tokens['op_pos_emb'])
            if self.drop_field != 'arch_emb':
                token_embedding += self.embed_tokens['arch_emb'](src_tokens['arch_emb'])

            if self.drop_field != 'bytes':
                byte_embedding_stack = []
                for field in self.byte_fields:
                    byte_embedding_stack.append(src_tokens[field].unsqueeze(-1).type_as(token_embedding[0]))
                byte_embedding = self.byte_combine(torch.stack(byte_embedding_stack, dim=2))
                token_embedding += byte_embedding

            x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens: Dict[str, torch.Tensor],
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (Dict[str, torch.Tensor]): dictionary of tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens,
                                       src_lengths,
                                       return_all_hiddens)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
            self,
            src_tokens: Dict[str, torch.Tensor],
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens[configs.static_field].eq(self.padding_idx)  # padding is from byte sequences
        has_pads = (src_tokens[configs.static_field].device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens[configs.static_field] \
            .ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class ByteCombineCNN(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn='relu',
                 filters=[(1, 4), (2, 8), (3, 12)] + [(i, 4 * i) for i in range(4, configs.byte_len)],
                 highway_layers=2):

        # Pytorch will search for the most efficient convolution implementation
        torch.backends.cudnn.benchmark = True

        # TODO: increase filters once the byte fields went to 8
        super().__init__()

        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(embed_dim, out_c, kernel_size=width)
            )

        last_dim = sum(f[1] for f in filters)

        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None

        self.projection = nn.Linear(last_dim, output_dim)

    def forward(self, features):
        # features size: Batch x Seq x byte_len x Emb_dim
        B = features.size(0)
        T = features.size(1)
        byte_len = features.size(2)
        emb_dim = features.size(3)

        # BTC -> BCT, BTC: batch, sequence, embedding size
        features = features.transpose(2, 3).view(-1, emb_dim, byte_len)

        conv_result = []

        for conv in self.convolutions:
            x = conv(features)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)

        x = torch.cat(conv_result, dim=-1)

        if self.highway is not None:
            x = self.highway(x)
        x = self.projection(x)
        x = x.view(B, T, -1)

        return x


class SeqCombineConcat(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.projection = nn.Linear(output_dim * 5, output_dim)

    def forward(self, x):
        # expect input of size [Batch x Seq x (5 x Emb_dim)]
        return self.projection(x)


class ByteCombineSUM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # expect input of size Batch x Seq x byte_len x Emb_dim
        return torch.sum(x, dim=-2)


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: int, num_layers: int = 1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)

            nn.init.constant_(layer.bias[: self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (torch.tensor([1], device=gate.device) - gate) * proj_x
        return x
