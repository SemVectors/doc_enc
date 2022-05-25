#!/usr/bin/env python3

from typing import Optional
import dataclasses
from enum import Enum

from omegaconf import MISSING

from doc_enc.encoders.enc_config import (
    BaseEncoderConf,
    SentEncoderConf,
    EmbSeqEncoderConf,
)


class ModelKind(Enum):
    UNDEFINED = 0
    DUAL_ENC = 1


@dataclasses.dataclass
class BaseModelConf:
    kind: ModelKind = ModelKind.DUAL_ENC
    # dual enc model opts
    normalize: bool = True
    scale: float = 20.0
    margin: float = 0.0

    # encoders
    encoder: BaseEncoderConf = MISSING


@dataclasses.dataclass
class SentModelConf(BaseModelConf):
    encoder: SentEncoderConf = MISSING

    split_target_sents: bool = False
    split_size: int = 512


@dataclasses.dataclass
class DocModelConf:
    kind: ModelKind

    sent: SentModelConf
    doc: EmbSeqEncoderConf
    fragment: Optional[EmbSeqEncoderConf] = None

    split_sents: bool = True
    split_size: int = 512

    # dual enc model opts
    normalize: bool = True
    scale: float = 20.0
    margin: float = 0.0
