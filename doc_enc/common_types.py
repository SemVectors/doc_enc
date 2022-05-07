#!/usr/bin/env python3

from enum import Enum


class EmbKind(Enum):
    TOKEN = 1
    POSITIONAL = 2
    TOKEN_WITH_POSITIONAL = 3


class EncoderKind(Enum):
    UNDEFINED = 0
    LSTM = 1
    TRANSFORMER = 2
    FNET = 3


class PoolingStrategy(Enum):
    UNDEFINED = 0
    MAX = 1
    MEAN = 2
    FIRST = 3
