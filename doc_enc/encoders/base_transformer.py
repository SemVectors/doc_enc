#!/usr/bin/env python3

from typing import Optional

import torch
from torch import nn

from doc_enc.common_types import PoolingStrategy
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

        self.pooler = TransformerPooler(conf.hidden_size, conf.pooler)
        self.output_units = conf.hidden_size
        if self.conf.pooler.out_size is not None:
            self.output_units = self.conf.pooler.out_size

    def out_embs_dim(self):
        return self.output_units

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = len(src_lengths)
        mask = torch.full((bs, max_len), True, dtype=torch.bool, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = False

        return mask

    def forward(self, embs, lengths, **kwargs):
        # embs shape: batch_sz, seq_len, hidden_dim
        embs = embs.transpose(0, 1)
        mask = self._create_key_padding_mask(embs.size()[0], lengths, embs.device)
        output = self.transformer_encoder(embs, src_key_padding_mask=mask)

        sentemb = self.pooler(output, lengths, mask=mask)
        return BaseEncoderOut(sentemb, output, lengths)
