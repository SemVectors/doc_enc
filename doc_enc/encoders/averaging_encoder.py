#!/usr/bin/env python3

import torch

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_pooler import BasePoolerConf
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_out import BaseEncoderOut


class AveragingEncoder(BaseEncoder):
    def __init__(self, config: BaseEncoderConf) -> None:
        super().__init__()
        self.config = config

        # set missing params for config
        # It is required to be able to save this config later
        config.num_layers = 1
        config.dropout = 0
        config.pooler = BasePoolerConf(pooling_strategy=PoolingStrategy.UNDEFINED)

    def out_embs_dim(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        **kwargs,
    ):
        if input_embs is None or lengths is None:
            raise RuntimeError("Averaging encoder accepts only input_embs as input!")

        out_embs = []
        for num, l in enumerate(lengths):
            out_embs.append(torch.mean(input_embs[num, :l], 0))

        stacked = torch.vstack(out_embs)
        return BaseEncoderOut(stacked, input_embs, lengths)
