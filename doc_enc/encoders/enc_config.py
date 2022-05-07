#!/usr/bin/env python3

from typing import Optional
import dataclasses

from doc_enc.common_types import EncoderKind, PoolingStrategy
from doc_enc.embs.emb_config import BaseEmbConf


@dataclasses.dataclass
class BaseEncoderConf:
    encoder_kind: EncoderKind
    hidden_size: int
    num_layers: int
    dropout: float
    pooling_strategy: PoolingStrategy

    emb_conf: Optional[BaseEmbConf] = None

    # lstm opts
    input_size: Optional[int] = None
    bidirectional: Optional[bool] = None
    # transformer opts
    num_heads: Optional[int] = None
    filter_size: Optional[int] = None


@dataclasses.dataclass
class SentEncoderConf(BaseEncoderConf):
    pass


@dataclasses.dataclass
class FragmentEncoderConf(BaseEncoderConf):
    pass


@dataclasses.dataclass
class DocEncoderConf(BaseEncoderConf):
    pass
