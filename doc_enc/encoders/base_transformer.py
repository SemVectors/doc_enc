#!/usr/bin/env python3


import torch
from torch import nn

from doc_enc.encoders.enc_config import PoolingStrategy
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_out import BaseEncoderOut


class BaseTransformerEncoder(nn.Module):
    def __init__(self, conf: BaseEncoderConf, layer_cls=nn.TransformerEncoderLayer):
        super().__init__()
        self.conf = conf
        if conf.num_heads is None or conf.filter_size is None:
            raise RuntimeError("set num_heads and filter_size")

        encoder_layer = layer_cls(
            conf.hidden_size,
            nhead=conf.num_heads,
            dim_feedforward=conf.filter_size,
            dropout=conf.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, conf.num_layers)

        # self.hidden_size = hidden_size

        if conf.pooling_strategy not in (
            PoolingStrategy.MAX,
            PoolingStrategy.MEAN,
            PoolingStrategy.FIRST,
        ):
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")
        # self.pooling_strategy = pooling_strategy

    def out_embs_dim(self):
        return self.conf.hidden_size

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = len(src_lengths)
        mask = torch.full((bs, max_len), True, dtype=torch.bool, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = False

        return mask

    def forward(self, embs, lengths, **kwargs):
        mask = self._create_key_padding_mask(embs.size()[0], lengths, embs.device)
        output = self.transformer_encoder(embs, src_key_padding_mask=mask)

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

        return BaseEncoderOut(sentemb, output, lengths)
