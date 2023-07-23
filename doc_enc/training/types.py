#!/usr/bin/env python3

from typing import NamedTuple
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
    src_id: list[int]
    src: list[list[int]]
    src_len: list[int]
    tgt_id: list[int]
    tgt: list[list[int]]
    tgt_len: list[int]
    hn_idxs: list[list[int]]
    info: dict[str, int]


class DocsBatch(NamedTuple):
    # segmented and tokenized text for multiple documents
    src_texts: list[list[int]]
    # segments lengths for each document
    src_doc_segments_length: list[list[int]]

    src_sent_len: list[int]
    # lengths of fragments in sentences
    # filled only when document is segmented on sentences and fragments
    src_fragment_len: list[int]

    # filled only when document is segmented on sentences
    src_doc_len_in_sents: list[int]
    # filled only when document is segmented on fragments
    src_doc_len_in_frags: list[int]
    src_ids: list[int]

    tgt_texts: list[list[int]]
    tgt_doc_segments_length: list[list[int]]

    tgt_sent_len: list[int]
    tgt_fragment_len: list[int]
    tgt_doc_len_in_sents: list[int]
    tgt_doc_len_in_frags: list[int]
    tgt_ids: list[int]

    positive_idxs: list[list[int]]
    info: dict[str, int]
