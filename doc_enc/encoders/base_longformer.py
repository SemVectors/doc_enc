#!/usr/bin/env python3

import logging
from typing import Optional

import torch
from torch import nn

from transformers.models.longformer.modeling_longformer import LongformerAttention

from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_transformer import BaseTransformerEncoder


class CompatConfig:
    def __init__(self, hidden_size, dropout, num_heads, attention_window) -> None:
        self.hidden_size = hidden_size
        self.layer_norm_eps = 1e-05
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.num_attention_heads = num_heads
        self.attention_window = attention_window


def create_compat_config(conf: BaseEncoderConf):
    if len(conf.attention_window) > 1 and conf.share_attn:
        raise RuntimeError(
            "Longformer: shared attention and different window sizes is not supported"
        )
    if len(conf.attention_window) == 1 and conf.num_layers > 1:
        attention_window = list(conf.attention_window) * conf.num_layers
    elif len(conf.attention_window) != conf.num_layers:
        raise RuntimeError("len of attention_window should be equal to num_layers")
    else:
        attention_window = list(conf.attention_window)

    return CompatConfig(
        hidden_size=conf.hidden_size,
        dropout=conf.dropout,
        num_heads=conf.num_heads,
        attention_window=attention_window,
    )


class LocalSelfAttention(nn.Module):
    def __init__(self, conf: BaseEncoderConf, layer_id=0):
        super().__init__()
        if conf.num_heads is None or not conf.attention_window:
            raise RuntimeError("Should set num_heads and attention_window in encoding config")

        compat_conf = create_compat_config(conf)
        self.self_attn = LongformerAttention(compat_conf, layer_id)
        self._max_window_size = max(compat_conf.attention_window)
        if self._max_window_size % 2 != 0:
            raise RuntimeError("attention window should be an even value")

    def _create_mask(self, key_padding_mask: torch.Tensor):
        attn_mask = key_padding_mask.logical_not().int()
        # enable global attention of the first tokens
        attn_mask[:, 0] = 2

        # TODO why?
        # extended_attention_mask = attn_mask[:, None, None, :]
        extended_attention_mask = attn_mask
        """
        The *attention_mask* is changed  from 0, 1, 2 to:
            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, **kwargs):
        seq_len = x.size(0)
        if seq_len % self._max_window_size != 0:
            raise RuntimeError("For local attention you need to enable padding of batches seq len")
        if key_padding_mask is None:
            raise RuntimeError("key_padding_mask is None")
        attn_mask = self._create_mask(key_padding_mask)

        is_index_masked = attn_mask < 0
        is_index_global_attn = attn_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        outputs = self.self_attn(
            x.transpose(0, 1),  # longformer attention expects batch_first
            attn_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=False,
        )
        attn_output = outputs[0]
        return attn_output.transpose(0, 1)


class LongformerEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__(conf, LocalSelfAttention)
