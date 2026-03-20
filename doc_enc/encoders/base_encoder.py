#!/usr/bin/env python3

import torch

from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.enc_in import EncoderInputType, SeqEncoderBatchedInput


class BaseEncoder(torch.nn.Module):
    def input_type(self) -> EncoderInputType:
        raise NotImplementedError("input_type not implemented")

    def out_embs_dim(self) -> int:
        raise NotImplementedError("out_embs_dim not implemented")

    def forward(self, input_batch: SeqEncoderBatchedInput, **kwargs) -> BaseEncoderOut:
        raise NotImplementedError("forward not implemented")
