#!/usr/bin/env python3

from typing import NamedTuple, List, Tuple
from enum import Enum


class TaskType(Enum):
    UNDEFINED = 0
    SENT_RETR = 1
    DOC_RETR = 2


class SentRetrLossType(Enum):
    CE = 1
    BICE = 2


class Example(NamedTuple):
    src_id: int
    src: List[int]
    tgt: List[int]
    dups: List[int]
    hns: Tuple[List[List[int]], List[int]]


class SentsBatch(NamedTuple):
    bs: int
    src_id: List[int]
    src: List[List[int]]
    src_len: List[int]
    tgt_id: List[int]
    tgt: List[List[int]]
    tgt_len: List[int]
    hn_idxs: List[List[int]]


class DocsBatch(NamedTuple):
    pass
