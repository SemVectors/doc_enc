#!/usr/bin/env python3

import logging
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from doc_enc.encoders.enc_config import BaseEncoderConf, LookAroundMode
from doc_enc.encoders.base_transformer import BaseTransformerEncoder

# based on https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
def look_around(x, backward=1, forward=1, pad_value=-1, dim=2):
    if backward + forward == 0:
        return x
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


def adjust_mask_dims(tensor, dim, k):
    tensor = tensor.unsqueeze(dim)
    expand_shape = [-1] * len(tensor.shape)
    expand_shape[dim] = k
    tensor = tensor.expand(*expand_shape)
    return tensor.reshape(-1, *tensor.shape[2:])


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class LocalAttention(nn.Module):
    def __init__(self, conf: BaseEncoderConf, layer_id=0):
        super().__init__()

        self.conf = conf
        if conf.num_heads is None or not conf.attention_window:
            raise RuntimeError("Should set num_heads and attention_window in encoding config")
        if len(conf.attention_window) == 1:
            self.window_size = conf.attention_window[0]
        else:
            self.window_size = conf.attention_window[layer_id]
        self.num_heads = conf.num_heads

        if conf.hidden_size % conf.num_heads != 0:
            raise RuntimeError(
                f"Wrong number of attention heads {conf.num_heads}: "
                "hidden_size should be divisible on num_heads"
            )

        self.look_backward = 0
        self.look_forward = 0
        if self.conf.window_look_around_mode in (
            LookAroundMode.BACK,
            LookAroundMode.BACK_AND_FORWARD,
        ):
            self.look_backward = 1
        if self.conf.window_look_around_mode in (
            LookAroundMode.FORWARD,
            LookAroundMode.BACK_AND_FORWARD,
        ):
            self.look_forward = 1
        logging.info(
            "LookAroundMode: %s, window_size: %s, extended_window_size: %s",
            self.conf.window_look_around_mode,
            self.window_size,
            self._extended_window_size(),
        )

        self.query = nn.Linear(conf.hidden_size, conf.hidden_size)
        self.key = nn.Linear(conf.hidden_size, conf.hidden_size)
        self.value = nn.Linear(conf.hidden_size, conf.hidden_size)
        self.dropout = nn.Dropout(conf.dropout)

        self.global_query = nn.Linear(conf.hidden_size, conf.hidden_size)

    def _split_input_into_heads(self, x: torch.Tensor):
        """x should have shape: (seq_len,bs,embs_dim)
        out has shape: (batch_size * attention_head_size, seq_length, attention_head_size)
        """
        l, bs, dim = x.shape
        x = x.contiguous().view(l, bs * self.num_heads, dim // self.num_heads)
        return x.transpose(0, 1)

    def _restore_input_shape(self, x: torch.Tensor):
        """input shape: bs_with_heads, seq_len, head_embs_dim;
        output shape: seq_len, bs, embs_dim
        """
        bs_with_heads, seq_len, head_embs_dim = x.shape
        return (
            x.transpose(0, 1)
            .contiguous()
            .view(seq_len, bs_with_heads // self.num_heads, head_embs_dim * self.num_heads)
        )

    def _extended_window_size(self):
        return self.window_size * (1 + self.look_forward + self.look_backward)

    def _look_around(self, x: torch.Tensor, **kwargs):
        return look_around(x, backward=self.look_backward, forward=self.look_forward, **kwargs)

    def _attn(self, q, k, v, input_mask):
        """
        input should be of shape (batch_size * attention_head_size, seq_length, attention_head_size)
        input_mask - positions with True are not allowed to attend. Shape is (bs x SeqLen)
        """

        bs_with_heads, seq_len, head_embs_dim = q.shape
        assert (
            seq_len % self.window_size
        ) == 0, f'sequence length {seq_len} must be divisible by window size {self.window_size} for local attention'

        windows_cnt = seq_len // self.window_size

        bucket_fn = lambda t: t.reshape(bs_with_heads, windows_cnt, self.window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        bk = self._look_around(bk)
        assert (bs_with_heads, windows_cnt, self._extended_window_size(), head_embs_dim) == bk.shape
        bv = self._look_around(bv)
        assert (bs_with_heads, windows_cnt, self._extended_window_size(), head_embs_dim) == bv.shape

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (head_embs_dim**-0.5)
        assert (
            bs_with_heads,
            windows_cnt,
            self.window_size,
            self._extended_window_size(),
        ) == dots.shape

        mask_value = max_neg_value(dots)

        input_mask = input_mask.reshape(-1, windows_cnt, self.window_size)
        key_mask = self._look_around(input_mask, pad_value=True)
        # input_mask: bs, windows, window_size, 1
        # key_mask: bs, windows, 1, extended_window_size
        mask = input_mask[:, :, :, None] + key_mask[:, :, None, :]
        mask = adjust_mask_dims(mask, 1, self.num_heads)
        # mask: bs_with_heads, windows, window_size, 3 * window_size
        dots.masked_fill_(mask, mask_value)
        del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        assert (bs_with_heads, windows_cnt, self.window_size, head_embs_dim) == out.shape

        out = out.reshape(-1, seq_len, head_embs_dim)

        return out

    def _global_attn(
        self,
        input_tensor: torch.Tensor,
        attn_out: torch.Tensor,
        key_padding_mask: torch.Tensor,
        key_with_heads: torch.Tensor,
        value_with_heads: torch.Tensor,
    ):
        """input and attn_out shapes: seq_len, bs, embs_dim"""
        seq_len, bs, embs_dim = input_tensor.shape
        query_tensor = input_tensor[0].unsqueeze(0)
        assert (1, bs, embs_dim) == query_tensor.shape

        query = self.global_query(query_tensor)

        query_with_heads = self._split_input_into_heads(query)
        bs_with_heads, seq_len, head_embs_dim = key_with_heads.shape

        dots = torch.einsum('bie,bje->bij', query_with_heads, key_with_heads) * (
            head_embs_dim**-0.5
        )
        assert (bs_with_heads, 1, seq_len) == dots.shape

        mask = adjust_mask_dims(key_padding_mask, 1, self.num_heads)
        # mask: bs_with_heads, seq_len
        dots.masked_fill_(mask.unsqueeze(1), max_neg_value(dots))
        del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bij,bje->bie', attn, value_with_heads)
        assert (bs_with_heads, 1, head_embs_dim) == out.shape
        out = self._restore_input_shape(out)
        assert (1, bs, embs_dim) == out.shape

        attn_out[0] = out
        return attn_out

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """hidden_states shape: seq_len,bs,embs_dim"""

        if key_padding_mask is None:
            raise RuntimeError("Input mask is None")

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query_with_heads, key_with_heads, value_with_heads = map(
            self._split_input_into_heads, (query, key, value)
        )

        out = self._attn(query_with_heads, key_with_heads, value_with_heads, key_padding_mask)

        out = self._restore_input_shape(out)
        out = self._global_attn(
            hidden_states,
            out,
            key_padding_mask,
            key_with_heads=key_with_heads,
            value_with_heads=value_with_heads,
        )

        return out


class LocalAttnEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__(conf, LocalAttention)
