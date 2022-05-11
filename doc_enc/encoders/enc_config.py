#!/usr/bin/env python3

from typing import Optional
import dataclasses

from omegaconf import MISSING

from doc_enc.common_types import EncoderKind, PoolingStrategy
from doc_enc.embs.emb_config import BaseEmbConf


@dataclasses.dataclass
class BaseEncoderConf:
    encoder_kind: EncoderKind
    hidden_size: int
    num_layers: int
    dropout: float
    pooling_strategy: PoolingStrategy

    # lstm opts
    input_size: Optional[int] = None
    bidirectional: Optional[bool] = None
    # transformer opts
    num_heads: Optional[int] = None
    filter_size: Optional[int] = None


@dataclasses.dataclass
class SentEncoderConf(BaseEncoderConf):
    emb_conf: BaseEmbConf = MISSING


@dataclasses.dataclass
class EmbSeqEncoderConf(BaseEncoderConf):
    add_beg_seq_token: bool = False
    # add_end_seq_token: bool = False
