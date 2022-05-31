#!/usr/bin/env python3

from typing import Optional
import dataclasses

from omegaconf import MISSING

from doc_enc.common_types import EncoderKind
from doc_enc.embs.emb_config import BaseEmbConf
from doc_enc.encoders.base_pooler import BasePoolerConf


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


@dataclasses.dataclass
class SentEncoderConf(BaseEncoderConf):
    emb_conf: BaseEmbConf = MISSING


@dataclasses.dataclass
class EmbSeqEncoderConf(BaseEncoderConf):
    add_beg_seq_token: bool = False
    input_dropout: float = 0.0
    add_pos_emb: bool = False
