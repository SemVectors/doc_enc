#!/usr/bin/env python3

from typing import List, Optional
import dataclasses
import itertools

from doc_enc.utils import open_file
from doc_enc.tokenizer import TokenizerConf, AbcTokenizer, create_tokenizer

from doc_enc.passages import split_into_fragments_by_len


@dataclasses.dataclass
class TextProcessorConf:
    tokenizer: TokenizerConf
    max_sent_len: int = 512
    min_sent_len: int = 4

    fragment_size: int = 24


class TextProcessor:
    def __init__(self, conf: TextProcessorConf):
        self._conf = conf

        self._tokenizer = create_tokenizer(conf.tokenizer)

    def vocab(self):
        return self._tokenizer

    def prepare_text_from_file(self, path, split_into_fragments=True, return_strings=False):
        with open_file(path) as f:
            sents = []
            sents_str = []
            for l in f:
                if not l.strip():
                    continue

                tokens = self._tokenizer(l.rstrip())
                if len(tokens) >= self._conf.min_sent_len:
                    tokens = tokens[: self._conf.max_sent_len]
                    sents.append(tokens)
                    if return_strings:
                        sents_str.append(l.rstrip())

        fragment_len_list = []
        if split_into_fragments:
            fragment_len_list = split_into_fragments_by_len(sents, self._conf.fragment_size)

        if not return_strings:
            return sents, fragment_len_list
        return sents_str, fragment_len_list

    def state_dict(self):
        return {'tok': self._tokenizer.state_dict()}

    def load_state_dict(self, d):
        self._tokenizer.load_state_dict(d['tok'])


def pad_sent_sequences(sents: List[List[int]], lengths: List[int], vocab: AbcTokenizer):
    """sents - is sentences of all documents; lengths - is a length of each document.
    This function makes all docs the same length by padding with [<pad>].
    """
    max_len = max(lengths)
    pad_seq = [vocab.pad_idx()]
    padded_sents = []
    offs = 0
    for l in lengths:
        padded_sents.extend(sents[offs : offs + l])
        offs += l
        pad_cnt = max_len - l
        padded_sents.extend(itertools.repeat(pad_seq, pad_cnt))
    return padded_sents, max_len


def pad_fragment_sequences(
    sents: List[List[int]],
    lengths: List[int],
    fragment_len: int,
    fragment_len_list: List[int],
    vocab: AbcTokenizer,
):
    """sents - is sentences of all documents; lengths - is a length of each document in fragments.
    Fragments should be already padded with function `pad_sent_sequences`.
    All fragments should have the same length `fragment_len`.
    fragment_len_list - Lengths of fragments in sents.
    Adjusted version of this list is returned if any new fragment is added.
    This function makes all docs the same length by padding with empty fragments: ([<pad>], [<pad>], ...).
    """
    max_len = max(lengths)
    pad_seq = [vocab.pad_idx()]
    padded_fragments = []
    frag_lens_with_padding = []
    offs = 0
    frag_offs = 0
    for l in lengths:
        sents_cnt = l * fragment_len
        padded_fragments.extend(sents[offs : offs + sents_cnt])
        offs += sents_cnt
        frag_lens_with_padding.extend(fragment_len_list[frag_offs : frag_offs + l])
        frag_offs += l
        pad_cnt = max_len - l
        padded_fragments.extend(itertools.repeat(pad_seq, pad_cnt * fragment_len))
        frag_lens_with_padding.extend(itertools.repeat(1, pad_cnt))
    return padded_fragments, frag_lens_with_padding, max_len
