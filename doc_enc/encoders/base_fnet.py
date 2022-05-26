#!/usr/bin/env python3

import torch
from torch import nn

from doc_enc.encoders.base_transformer import BaseTransformerEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf


class FourierTransform(nn.Module):
    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is None:
            raise RuntimeError("key_padding_mask is None")

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            x = torch.fft.fft2(x.float(), dim=(0, 2)).real
        x = x.masked_fill(key_padding_mask.t().unsqueeze(-1), 0.0)
        return x


class FNetEncoder(BaseTransformerEncoder):
    def __init__(self, conf: BaseEncoderConf):
        fourier = FourierTransform()
        super().__init__(conf, fourier)
