#!/usr/bin/env python3

from typing import Optional
from enum import Enum
import dataclasses


class SentEncoderKind(Enum):
    UNDEFINED = 0
    LSTM = 1
    TRANSFORMER = 2
    FNET = 3


class PoolingStrategy(Enum):
    UNDEFINED = 0
    MAX = 1
    MEAN = 2
    FIRST = 3


@dataclasses.dataclass
class BaseEncoderConf:
    encoder_kind: SentEncoderKind
    hidden_size: int
    num_layers: int
    dropout: float
    pooling_strategy: PoolingStrategy
    # lstm opts
    embedding_size: Optional[int] = None
    bidirectional: Optional[bool] = None
    # transformer opts
    num_heads: Optional[int] = None
    filter_size: Optional[int] = None


@dataclasses.dataclass
class SentEncoderConf(BaseEncoderConf):
    pass
