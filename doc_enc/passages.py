#!/usr/bin/env python3


def split_into_fragments_by_len(sents: list, fragment_size: int):
    l = len(sents)
    fragment_len_list = []

    for offs in range(0, l, fragment_size):
        cnt = min(l - offs, fragment_size)
        fragment_len_list.append(cnt)
    return fragment_len_list
