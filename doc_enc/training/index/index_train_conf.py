#!/usr/bin/env python3

import dataclasses
from enum import Enum


class IndexLossType(Enum):
    BASIC = 1
    DISTIL = 2


@dataclasses.dataclass
class _BaseConf:
    loss_type: IndexLossType = IndexLossType.DISTIL
    margin: float = 0.0
    scale: float = 20.0
    weight: float = 1.0


@dataclasses.dataclass
class IvfConf(_BaseConf):
    scale_ivf_weight_to_pq: bool = False
    lr: float = 0.0001
    fixed: bool = False


@dataclasses.dataclass
class PQConf(_BaseConf):
    pass


@dataclasses.dataclass
class IndexTrainConf:
    enable: bool = False
    init_index_file: str = ''
    # create index if init_index_file is empty
    train_sample: float = 0.4

    ivf_centers_num: int = 10000
    subvector_num: int = 32
    subvector_bits: int = 8

    readd_vectors_while_training: bool = False

    dense_weight: float = 1.0

    ivf: IvfConf = dataclasses.field(default_factory=IvfConf)
    pq: PQConf = dataclasses.field(default_factory=PQConf)
