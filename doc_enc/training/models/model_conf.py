#!/usr/bin/env python3

import dataclasses
from enum import Enum

from omegaconf import MISSING

from doc_enc.embs.emb_config import BaseEmbConf
from doc_enc.encoders.enc_config import (
    BaseEncoderConf,
    SeqEncoderConf,
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

    split_input: bool = True
    max_chunk_size: int = 512
    max_tokens_in_chunk: int = 48_000

    cross_device_sample: bool = False

    # index training
    index: IndexTrainConf = dataclasses.field(default_factory=IndexTrainConf)


@dataclasses.dataclass
class SentModelConf(BaseModelConf):
    encoder: SeqEncoderConf = MISSING


@dataclasses.dataclass
class DocModelConf(BaseModelConf):
    embed: BaseEmbConf | None = None
    sent: SentModelConf | None = None
    sent_for_doc: BaseEncoderConf | None = None
    fragment: SeqEncoderConf | None = None
    doc: SeqEncoderConf = MISSING

    freeze_base_sents_layer: bool = False

    grad_tgt_sents: bool = True
    grad_src_sents: bool = True

    # index training
    index: IndexTrainConf = dataclasses.field(default_factory=IndexTrainConf)
