#!/usr/bin/env python3

from enum import Enum


class EmbKind(Enum):
    TOKEN = 1
    TOKEN_WITH_POSITIONAL_EMB = 2
    TOKEN_WITH_POSITIONAL_ENC = 3


class EncoderKind(Enum):
    UNDEFINED = 0
    LSTM = 1
    TRANSFORMER = 2
    FNET = 3
    LONGFORMER = 4
    GRU = 5
    LOCAL_ATTN_TRANSFORMER = 6


class PoolingStrategy(Enum):
    UNDEFINED = 0
    MAX = 1
    MEAN = 2
    FIRST = 3
