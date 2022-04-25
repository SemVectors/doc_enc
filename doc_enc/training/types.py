#!/usr/bin/env python3

from typing import NamedTuple, List, Tuple, Dict
from enum import Enum


class TaskType(Enum):
    UNDEFINED = 0
    SENT_RETR = 1
    DOC_RETR = 2


class SentRetrLossType(Enum):
    CE = 1
    BICE = 2


class DocRetrLossType(Enum):
    CE = 1


class SentsBatch(NamedTuple):
    src_id: List[int]
    src: List[List[int]]
    src_len: List[int]
    tgt_id: List[int]
    tgt: List[List[int]]
    tgt_len: List[int]
    hn_idxs: List[List[int]]
    info: Dict[str, int]


class DocsBatch(NamedTuple):
    src_sents: List[List[int]]
    src_sent_len: List[int]
    src_fragment_len: List[int]
    src_doc_len_in_sents: List[int]
    src_doc_len_in_frags: List[int]
    src_ids: List[int]

    tgt_sents: List[List[int]]
    tgt_sent_len: List[int]
    tgt_fragment_len: List[int]
    tgt_doc_len_in_sents: List[int]
    tgt_doc_len_in_frags: List[int]
    tgt_ids: List[int]

    positive_idxs: List[List[int]]
    info: Dict[str, int]
