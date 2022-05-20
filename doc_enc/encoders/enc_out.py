#!/usr/bin/env python3

from typing import NamedTuple

import torch


class BaseEncoderOut(NamedTuple):
    pooled_out: torch.Tensor
    encoder_out: torch.Tensor
    out_lengths: torch.Tensor


# torch gradient checkpointing replaces NamedTuples with the plain tuples
POOLED_OUT = 0
ENCODER_OUT = 1
OUT_LENGTHS = 2
