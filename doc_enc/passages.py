#!/usr/bin/env python3

from typing import List


def split_into_fragments_by_len(sents: List, fragment_size: int):
    l = len(sents)
    fragment_len_list = []

    for offs in range(0, l, fragment_size):
        cnt = min(l - offs, fragment_size)
        fragment_len_list.append(cnt)
    return fragment_len_list
