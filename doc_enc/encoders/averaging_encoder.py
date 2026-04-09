#!/usr/bin/env python3

import torch

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_pooler import BasePoolerConf
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_in import EncoderInputType, SeqEncoderBatchedInput
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

    def input_type(self) -> EncoderInputType:
        return EncoderInputType.PADDED

    def out_embs_dim(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        input_batch: SeqEncoderBatchedInput,
        **kwargs,
    ):
        if not input_batch.embedded:
            raise RuntimeError("Averaging encoder accepts only input_embs as input!")

        pd = input_batch.get_padded()

        assert pd.padding_mask is not None, "AveragingEncoder: Padding mask is None"
        mask = pd.padding_mask.logical_not().unsqueeze(-1)
        embs = pd.data.masked_fill(mask, 0).sum(dim=1) / pd.lengths.unsqueeze(-1)
        return BaseEncoderOut(embs, None, pd.lengths)
