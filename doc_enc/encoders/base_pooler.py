#!/usr/bin/env python3

import dataclasses
from typing import Optional

import torch
from torch import nn

from doc_enc.common_types import PoolingStrategy


@dataclasses.dataclass
class BasePoolerConf:
    pooling_strategy: PoolingStrategy
    out_size: Optional[int] = None
    force_dense_layer: bool = False
    use_activation: bool = False


class BasePooler(nn.Module):
    def __init__(self, emb_dim, conf: BasePoolerConf):
        super().__init__()

        self.conf = conf

        self.dense = None
        if conf.out_size is not None and emb_dim != conf.out_size:
            self.dense = nn.Linear(emb_dim, conf.out_size)
        elif conf.force_dense_layer:
            self.dense = nn.Linear(emb_dim, emb_dim)

        self.activation = None
        if conf.use_activation:
            self.activation = nn.Tanh()

    def _post_proc(self, pooled_output):
        if self.dense is not None:
            pooled_output = self.dense(pooled_output)
        if self.activation is not None:
            pooled_output = self.activation(pooled_output)

        return pooled_output

    def _pooling_impl(
        self, hidden_states: torch.Tensor, lengths: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("base pooler forward")

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        pooled_out = self._pooling_impl(hidden_states, lengths, **kwargs)
        return self._post_proc(pooled_out)
