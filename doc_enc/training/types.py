#!/usr/bin/env python3

from typing import NamedTuple, List, Tuple

# = collections.namedtuple('Example', ['src_id', 'src', 'tgt', 'dups', 'hns'])


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
