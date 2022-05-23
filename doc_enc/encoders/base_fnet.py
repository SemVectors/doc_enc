#!/usr/bin/env python3

import torch
from torch import nn

from doc_enc.encoders.base_transformer import BaseTransformerEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf


class FnetEncoderLayer(nn.Module):
    def __init__(self, dim, dim_feedforward=2048, dropout=0, **kwargs):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, dim)

        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, src, src_key_padding_mask, **kwargs):
        # src shape: seq_len, batch_sz, hidden_dim
        residual = src
        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            src = torch.fft.fft2(src.float(), dim=(0, 2)).real
        src = src.masked_fill(src_key_padding_mask.t().unsqueeze(-1), 0.0)

        src = self.norm1(src + residual)
        residual = src
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BaseFNetEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__(conf, FnetEncoderLayer)
