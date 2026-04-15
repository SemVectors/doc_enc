#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_in import EncoderInputType, SeqEncoderBatchedInput
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.base_pooler import BasePoolerConf, BasePooler


class TransformerPooler(BasePooler):
    def __init__(self, input_type: EncoderInputType, emb_dim, conf: BasePoolerConf):
        super().__init__(emb_dim, conf)
        self.input_type = input_type
        if conf.pooling_strategy not in (
            PoolingStrategy.MAX,
            PoolingStrategy.MEAN,
            PoolingStrategy.FIRST,
        ):
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")

    def _jagged_pooling_impl(self, hidden_states: torch.Tensor, lengths: torch.Tensor):
        if self.conf.pooling_strategy == PoolingStrategy.FIRST:
            sel = torch.cat(
                (
                    torch.tensor([0], dtype=torch.int32, device=lengths.device),
                    lengths.cumsum(-1)[:-1],
                )
            )
            pooled_out = hidden_states.values()[sel]
        elif self.conf.pooling_strategy == PoolingStrategy.MEAN:
            # hidden_states is already nested tensor
            pooled_out = hidden_states.sum(1) / lengths.unsqueeze(-1)
        elif self.conf.pooling_strategy == PoolingStrategy.MAX:
            pooled_out = torch.max(hidden_states, dim=1)[0]
        else:
            raise RuntimeError(f"Unsupported pooling strategy: {self.conf.pooling_strategy}")

        return pooled_out

    def _padded_pooling_impl(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states shape: batch_sz, seq_len, hidden_dim
        if self.conf.pooling_strategy == PoolingStrategy.FIRST:
            return hidden_states[:, 0]

        if mask is None:
            raise RuntimeError("Mask is None")

        if self.conf.pooling_strategy == PoolingStrategy.MEAN:
            masked = hidden_states.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
            sum_embeddings = torch.sum(masked, dim=1)
            return sum_embeddings / lengths.unsqueeze(-1)
        if self.conf.pooling_strategy == PoolingStrategy.MAX:
            masked = hidden_states.masked_fill(mask.logical_not().unsqueeze(-1), float('-inf'))
            return torch.max(masked, dim=1)[0]
        raise RuntimeError("Unsupported pooling strategy trans")

    def _pooling_impl(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.input_type == EncoderInputType.PADDED:
            return self._padded_pooling_impl(hidden_states, lengths, mask)
        elif self.input_type == EncoderInputType.JAGGED:
            return self._jagged_pooling_impl(hidden_states, lengths)
        else:
            raise RuntimeError(f"Unsupported input type: {self.input_type}")


def _create_activation(act_str: str):
    if not hasattr(nn.functional, act_str):
        raise RuntimeError(f"No such activation ({act_str}) in torch.nn.functional")
    return getattr(nn.functional, act_str)


class _BasicLayer(nn.Module):
    """Based on implementation from transformers library"""

    def __init__(self, conf: BaseEncoderConf, attention: nn.Module):
        super().__init__()
        self.attn = attention
        if conf.intermediate_size is None or conf.intermediate_activation is None:
            raise RuntimeError(
                "check that encoder's intermediate_size, intermediate_activation are not None"
            )
        self.linear1 = nn.Linear(conf.hidden_size, conf.intermediate_size)
        self.activation = _create_activation(conf.intermediate_activation)

        self.linear2 = nn.Linear(conf.intermediate_size, conf.hidden_size)
        self.norm = nn.LayerNorm(conf.hidden_size)
        self.dropout = nn.Dropout(conf.dropout)

    def _ff(self, x_attn):
        x_lin1 = self.linear1(x_attn)
        x_act = self.activation(x_lin1)
        x_lin2 = self.linear2(x_act)
        x_drop = self.dropout(x_lin2)
        return self.norm(x_drop + x_attn)

    def forward(self, x, **kwargs):
        x_attn = self.attn(x, **kwargs)
        return self._ff(x_attn)


class _FullLayer(nn.Module):
    """Based on implementation of TransformerEncoderLayer from pytorch"""

    def __init__(self, conf: BaseEncoderConf, attention: nn.Module):
        super().__init__()
        if conf.intermediate_size is None or conf.intermediate_activation is None:
            raise RuntimeError(
                "check that encoder's intermediate_size, intermediate_activation are not None"
            )
        self.attn = attention

        self.linear1 = torch.nn.Linear(conf.hidden_size, conf.intermediate_size)
        self.linear2 = torch.nn.Linear(conf.intermediate_size, conf.hidden_size)
        self.activation = _create_activation(conf.intermediate_activation)

        self.norm1 = torch.nn.LayerNorm(conf.hidden_size)
        self.norm2 = torch.nn.LayerNorm(conf.hidden_size)
        self.dropout1 = torch.nn.Dropout(conf.dropout)
        self.dropout2 = torch.nn.Dropout(conf.dropout)

    def _ff(self, x_attn, input_x):
        # dropout of x_attn is intentionally skipped
        x_norm1 = self.norm1(x_attn + input_x)
        x_lin1 = self.linear1(x_norm1)
        x_act = self.activation(x_lin1)
        x_drop1 = self.dropout1(x_act)
        x_lin2 = self.linear2(x_drop1)
        x_drop2 = self.dropout2(x_lin2)
        return self.norm2(x_norm1 + x_drop2)

    def forward(self, x, **kwargs):
        x_attn = self.attn(x, **kwargs)
        return self._ff(x_attn, x)


class BaseTransformerEncoder(BaseEncoder):
    def __init__(self, conf: BaseEncoderConf, attention_cls):
        super().__init__()
        self.conf = conf
        self._input_type = self._init_input_type()

        layer_cls = _BasicLayer
        if conf.full_intermediate:
            layer_cls = _FullLayer

        attention = attention_cls(conf)
        get_attn_for_layer = lambda _: attention
        if not conf.share_attn:
            get_attn_for_layer = lambda i: attention_cls(conf, i)

        self.layers = nn.ModuleList(
            [layer_cls(conf, get_attn_for_layer(i)) for i in range(conf.num_layers)]
        )

        self.pooler = TransformerPooler(self.input_type(), conf.hidden_size, conf.pooler)
        self.output_units = conf.hidden_size
        if self.conf.pooler.out_size is not None:
            self.output_units = self.conf.pooler.out_size

    def _init_input_type(self):
        if self.conf.input_type is None:
            return EncoderInputType.PADDED
        if self.conf.input_type not in (EncoderInputType.JAGGED, EncoderInputType.PADDED):
            raise RuntimeError(f"Unsupported input type: {self.conf.input_type}")
        return self.conf.input_type

    def input_type(self) -> EncoderInputType:
        return self._input_type

    def out_embs_dim(self):
        return self.output_units

    def forward(
        self,
        input_batch: SeqEncoderBatchedInput,
        **kwargs,
    ) -> BaseEncoderOut:
        if self.input_type() == EncoderInputType.JAGGED:
            input_embs = input_batch.get_nested()
            lengths = input_batch.get_jagged().lengths
            key_padding_mask = None
        elif self.input_type() == EncoderInputType.PADDED:
            inp = input_batch.get_padded()
            # embs shape: batch_sz, seq_len, hidden_dim
            input_embs = inp.data
            lengths = inp.lengths
            key_padding_mask = inp.padding_mask
            assert key_padding_mask is not None, "Padding mask should be already created!"

        else:
            raise RuntimeError(
                f"BaseTransformer:forward - Unsupported input type {self.input_type()}"
            )

        output = input_embs
        for layer_module in self.layers:
            output = layer_module(output, key_padding_mask=key_padding_mask)

        pooled_out = self.pooler(output, lengths, mask=key_padding_mask)
        return BaseEncoderOut(pooled_out, output, lengths)


class GlobalSelfAttention(nn.Module):
    def __init__(self, conf: BaseEncoderConf, layer_id: int = 0):
        super().__init__()
        if conf.num_heads is None:
            raise RuntimeError("Should set num_heads in encoding config")

        self.self_attn = MultiheadAttention(conf.hidden_size, conf.num_heads, dropout=conf.dropout)

    def forward(self, x, **kwargs):
        return self.self_attn(x, **kwargs)


class TransformerEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__(conf, GlobalSelfAttention)


# Stripped down version of nn.MultiheadAttention with support of jagged nested
# tensors. We have to keep parameter names the same as in nn.MultiheadAttention
# to keep compatibility with the previously trained models.
# See also https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html#multiheadattention
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # adjust dropout probability
        dropout_p = self.dropout
        if not self.training:
            dropout_p = 0.0

        proj = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        query, key, value = torch.chunk(proj, 3, dim=-1)

        # Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        if key_padding_mask is not None:
            bs, max_len = key_padding_mask.shape
            attn_mask = key_padding_mask.view(bs, 1, 1, max_len)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, attn_mask=attn_mask
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
