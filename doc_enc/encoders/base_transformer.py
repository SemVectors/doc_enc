#!/usr/bin/env python3

import logging
from typing import Optional

import torch
from torch import nn

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.base_pooler import BasePoolerConf, BasePooler


class TransformerPooler(BasePooler):
    def __init__(self, emb_dim, conf: BasePoolerConf):
        super().__init__(emb_dim, conf)
        if conf.pooling_strategy not in (
            PoolingStrategy.MAX,
            PoolingStrategy.MEAN,
            PoolingStrategy.FIRST,
        ):
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")

    def _pooling_impl(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states shape: seq_len, batch_sz, hidden_dim
        if self.conf.pooling_strategy == PoolingStrategy.FIRST:
            return hidden_states[0]

        if mask is None:
            raise RuntimeError("Mask is None")

        if self.conf.pooling_strategy == PoolingStrategy.MEAN:
            masked = hidden_states.masked_fill(mask.t().unsqueeze(-1), 0.0)
            sum_embeddings = torch.sum(masked, dim=0)
            return sum_embeddings / lengths.unsqueeze(-1)
        if self.conf.pooling_strategy == PoolingStrategy.MAX:
            masked = hidden_states.masked_fill(mask.t().unsqueeze(-1), float('-inf'))
            return torch.max(masked, dim=0)[0]
        raise RuntimeError("Unsupported pooling strategy trans")


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
        layer_cls = _BasicLayer
        if conf.full_intermediate:
            layer_cls = _FullLayer

        attention = attention_cls(conf)
        get_attn_for_layer = lambda i: attention
        if not conf.share_attn:
            get_attn_for_layer = lambda i: attention_cls(conf, i)

        self.layers = nn.ModuleList(
            [layer_cls(conf, get_attn_for_layer(i)) for i in range(conf.num_layers)]
        )

        self.pooler = TransformerPooler(conf.hidden_size, conf.pooler)
        self.output_units = conf.hidden_size
        if self.conf.pooler.out_size is not None:
            self.output_units = self.conf.pooler.out_size

    def out_embs_dim(self):
        return self.output_units

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = src_lengths.shape[0]
        mask = torch.full((bs, max_len), True, dtype=torch.bool, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = False

        return mask

    def forward(
        self,
        input_embs: torch.Tensor | None,
        lengths: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseEncoderOut:
        if input_embs is None:
            raise RuntimeError("Only embs as input are supported")

        # embs shape: batch_sz, seq_len, hidden_dim
        embs = input_embs.transpose(0, 1)
        if key_padding_mask is None:
            key_padding_mask = self._create_key_padding_mask(embs.size()[0], lengths, embs.device)

        output = embs
        for layer_module in self.layers:
            output = layer_module(output, key_padding_mask=key_padding_mask)

        sentemb = self.pooler(output, lengths, mask=key_padding_mask)
        return BaseEncoderOut(sentemb, output, lengths)


class GlobalSelfAttention(nn.Module):
    def __init__(self, conf: BaseEncoderConf, layer_id=0):
        super().__init__()
        if conf.num_heads is None:
            raise RuntimeError("Should set num_heads in encoding config")

        self.self_attn = nn.modules.activation.MultiheadAttention(
            conf.hidden_size, conf.num_heads, dropout=conf.dropout
        )

    def forward(self, x, key_padding_mask=None, **kwargs):
        return self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)[0]


class TransformerEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__(conf, GlobalSelfAttention)
