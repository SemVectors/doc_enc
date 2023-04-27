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

from doc_enc.training.index.index_train_conf import IndexTrainConf


class ModelKind(Enum):
    UNDEFINED = 0
    DUAL_ENC = 1


@dataclasses.dataclass
class BaseModelConf:
    load_params_from: str = ''
    kind: ModelKind = ModelKind.DUAL_ENC
    # dual enc model opts
    normalize: bool = True
    scale: float = 20.0
    margin: float = 0.0

    split_sents: bool = True
    max_chunk_size: int = 512
    max_tokens_in_chunk: int = 48_000

    # index training
    index: IndexTrainConf = IndexTrainConf()


@dataclasses.dataclass
class SentModelConf(BaseModelConf):
    encoder: SentEncoderConf = MISSING


@dataclasses.dataclass
class DocModelConf(BaseModelConf):
    sent: SentModelConf = MISSING
    sent_for_doc: Optional[BaseEncoderConf] = None
    fragment: Optional[EmbSeqEncoderConf] = None
    doc: EmbSeqEncoderConf = MISSING

    freeze_base_sents_layer: bool = False

    grad_tgt_sents: bool = True
    grad_src_senst: bool = True

    # index training
    index: IndexTrainConf = IndexTrainConf()
