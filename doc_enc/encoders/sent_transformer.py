#!/usr/bin/env python3

import math

import torch
from torch import nn

from doc_enc.encoders.pos_enc import PositionalEncoding
from doc_enc.encoders.enc_config import PoolingStrategy


class SentTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx=0,
        num_heads=8,
        hidden_size=512,
        num_layers=1,
        dropout=0.1,
        filter_size=2048,
        pooling_strategy=PoolingStrategy.FIRST,
        layer_cls=nn.TransformerEncoderLayer,
        **kwargs,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings, hidden_size, padding_idx=padding_idx)
        nn.init.uniform_(self.embed_tokens.weight, -0.1, 0.1)
        nn.init.constant_(self.embed_tokens.weight[padding_idx], 0)
        self.token_type_embeddings = torch.nn.Embedding(2, hidden_size)
        torch.nn.init.uniform_(self.token_type_embeddings.weight, -0.1, 0.1)

        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = layer_cls(
            hidden_size, nhead=num_heads, dim_feedforward=filter_size, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.hidden_size = hidden_size

        if pooling_strategy not in (
            PoolingStrategy.MAX,
            PoolingStrategy.MEAN,
            PoolingStrategy.FIRST,
        ):
            raise RuntimeError(f"Unsupported pooling strategy: {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

    def embs_dim(self):
        return self.hidden_size

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = len(src_lengths)
        mask = torch.full((bs, max_len), True, dtype=torch.bool, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = False

        return mask

    def forward(self, tokens, lengths, token_types=None, **kwargs):
        src = self.embed_tokens(tokens) * math.sqrt(self.hidden_size)
        src = src.transpose(0, 1)
        if token_types is not None:
            token_type_embeddings = self.token_type_embeddings(token_types)
            src = src + token_type_embeddings.transpose(0, 1)

        src = self.pos_encoder(src)

        mask = self._create_key_padding_mask(src.size()[0], lengths, tokens.device)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)

        if self._pooling_strategy == PoolingStrategy.FIRST:
            sentemb = output[0]
        elif self._pooling_strategy == PoolingStrategy.MEAN:
            masked = output.masked_fill(mask.t().unsqueeze(-1), 0.0)
            sum_embeddings = torch.sum(masked, dim=0)
            sentemb = sum_embeddings / lengths.unsqueeze(-1)
        elif self._pooling_strategy == PoolingStrategy.MAX:
            masked = output.masked_fill(mask.t().unsqueeze(-1), float('-inf'))
            sentemb = torch.max(masked, dim=0)[0]
        else:
            raise RuntimeError("Unsupported pooling strategy trans")

        return {'pooled_out': sentemb, 'encoder_out': output, 'out_lengths': lengths}
