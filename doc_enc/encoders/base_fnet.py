#!/usr/bin/env python3

import torch
from torch import nn

from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.encoders.enc_config import PoolingStrategy


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
        residual = src
        src = torch.fft.fft2(src, dim=(-1, 0)).real
        src = src.masked_fill(src_key_padding_mask.t().unsqueeze(-1), 0.0)

        src = self.norm1(src + residual)
        residual = src
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SentFNetEncoder(SentTransformerEncoder):
    def __init__(
        self,
        num_embeddings,
        padding_idx=0,
        num_heads=8,
        hidden_size=512,
        num_layers=1,
        dropout=0.1,
        dim_feedforward=512,
        pooling_strategy=PoolingStrategy.FIRST,
        **kwargs,
    ):
        super().__init__(
            num_embeddings,
            padding_idx,
            num_heads,
            hidden_size,
            num_layers,
            dropout,
            dim_feedforward,
            pooling_strategy,
            FnetEncoderLayer,
        )
