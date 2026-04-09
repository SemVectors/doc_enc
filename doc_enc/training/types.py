#!/usr/bin/env python3


from typing import NamedTuple
from enum import Enum

import torch

from doc_enc.encoders.enc_in import EncoderInData


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
    src_data: EncoderInData
    tgt_data: EncoderInData

    labels: torch.Tensor

    # for tests
    hn_idxs: list[list[int]]

    def batch_size(self):
        return len(self.src_data.text_ids)


class DocRetrPairs(NamedTuple):
    src_texts: list[list[int]]
    src_text_lengths: list[list[int]]
    src_ids: list[str | int]

    tgt_texts: list[list[int]]
    tgt_text_lengths: list[list[int]]
    tgt_ids: list[str | int]

    positive_idxs: list[list[int]]
    info: dict[str, int]


class DocsBatch(NamedTuple):
    src_data: EncoderInData
    tgt_data: EncoderInData

    labels: torch.Tensor

    def get_src_docs_cnt(self):
        return len(self.src_data.text_ids)

    def get_tgt_docs_cnt(self):
        return len(self.tgt_data.text_ids)

    def batch_size(self):
        return self.get_src_docs_cnt()

    def get_positive_idxs(self) -> list[list[int]]:
        pos_ids = []
        pos_ids_tmp = self.labels.nonzero().tolist()
        idx = 0
        cur_row = []
        for i, j in pos_ids_tmp:
            if i == idx:
                cur_row.append(j)
            else:
                idx += 1
                if idx == i:
                    pos_ids.append(cur_row)
                cur_row = [j]
        if cur_row:
            pos_ids.append(cur_row)
        return pos_ids

    def max_positives_per_doc(self):
        return int(torch.max(self.labels.sum(1)).item())
