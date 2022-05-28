#!/usr/bin/env python3

import torch

from doc_enc.encoders.enc_out import BaseEncoderOut


class BaseEncoder(torch.nn.Module):
    def out_embs_dim(self) -> int:
        raise NotImplementedError("out_embs_dim not implemented")

    def forward(self, embs: torch.Tensor, lengths: torch.Tensor, **kwargs) -> BaseEncoderOut:
        raise NotImplementedError("forward not implemented")
