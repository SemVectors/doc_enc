#!/usr/bin/env python3

from enum import Enum
from typing import Optional, List
import dataclasses

from omegaconf import MISSING

from doc_enc.common_types import EncoderKind
from doc_enc.embs.emb_config import BaseEmbConf
from doc_enc.encoders.base_pooler import BasePoolerConf


class LookAroundMode(Enum):
    NONE = 0
    BACK = 1
    FORWARD = 2
    BACK_AND_FORWARD = 3


@dataclasses.dataclass
class BaseEncoderConf:
    encoder_kind: EncoderKind
    hidden_size: int
    num_layers: int
    dropout: float
    pooler: BasePoolerConf

    # lstm opts
    input_size: Optional[int] = None
    bidirectional: Optional[bool] = None
    # transformer opts
    num_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    intermediate_activation: Optional[str] = None
    full_intermediate: bool = False
    share_attn: bool = True
    # longformer and local transformer
    attention_window: List[int] = dataclasses.field(default_factory=list)
    window_look_around_mode: LookAroundMode = LookAroundMode.BACK


@dataclasses.dataclass
class SentEncoderConf(BaseEncoderConf):
    emb_conf: BaseEmbConf = MISSING


@dataclasses.dataclass
class EmbSeqEncoderConf(BaseEncoderConf):
    add_beg_seq_token: bool = False
    input_dropout: float = 0.0
    add_pos_emb: bool = False
