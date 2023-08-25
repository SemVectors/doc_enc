#!/usr/bin/env python3

from typing import Iterable, List
import dataclasses
import itertools
import re

from doc_enc.utils import open_file
from doc_enc.tokenizer import TokenizerConf, AbcTokenizer, create_tokenizer

from doc_enc.passages import split_into_fragments_by_len


@dataclasses.dataclass
class TextProcessorConf:
    tokenizer: TokenizerConf
    max_sent_len: int = 128
    min_sent_len: int = 4
    num_alpha_max_ratio: float = 1.0

    split_into_sents: bool = True
    split_into_fragments: bool = True
    fragment_size: int = 24


class TextProcessor:
    def __init__(self, conf: TextProcessorConf, inference_mode=False):
        self._conf = conf

        self._tokenizer = create_tokenizer(conf.tokenizer, inference_mode=inference_mode)
        self._alpha_nums_regex = re.compile(r'(\d+[-–.]?\d*)|(\w+)')

    def conf(self):
        return self._conf

    def vocab(self):
        return self._tokenizer

    def _filter_sent(self, tokens, sent_str):
        if len(tokens) < self._conf.min_sent_len:
            return True

        if self._conf.num_alpha_max_ratio:
            nums = 0
            alphas = 0
            occ_list = self._alpha_nums_regex.finditer(sent_str)
            for match in occ_list:
                if match.group(1):
                    nums += 1
                if match.group(2):
                    alphas += 1

            if alphas == 0 or nums / alphas > self._conf.num_alpha_max_ratio:
                return True

        return False

    def prepare_text(
        self, text_sents: Iterable[str], truncate_length_in_tokens: int = 0
    ) -> tuple[list[list[int]], list[int]]:
        segmented_text: list[list[int]] = []
        doc_segments_length: list[int] = []

        if self._conf.split_into_sents:
            # 1. document is a sequence of sentences (maybe grouped into fragments)
            cur_len_in_tokens = 0
            for sent in text_sents:
                if not sent.strip():
                    continue

                tokens = self._tokenizer(sent)
                if not self._filter_sent(tokens, sent):
                    if self._conf.max_sent_len:
                        tokens = tokens[: self._conf.max_sent_len]
                    segmented_text.append(tokens)
                    cur_len_in_tokens += len(tokens)
                    if truncate_length_in_tokens and cur_len_in_tokens > truncate_length_in_tokens:
                        break

            if self._conf.split_into_fragments:
                # document is a sequence of fragments
                doc_segments_length = split_into_fragments_by_len(
                    segmented_text, self._conf.fragment_size
                )
            else:
                # document is a sequence of sentences
                doc_segments_length.append(len(segmented_text))

        elif self._conf.split_into_fragments:
            # 2. document is a sequence of fragments
            if not isinstance(text_sents, list):
                text_sents = list(text_sents)
            fragment_lengths = split_into_fragments_by_len(text_sents, self._conf.fragment_size)
            offset = 0
            cur_len_in_tokens = 0
            for frag_len in fragment_lengths:
                tokens = self._tokenizer('\n'.join(text_sents[offset : offset + frag_len]))
                offset += frag_len
                segmented_text.append(tokens)
                cur_len_in_tokens += len(tokens)
                if truncate_length_in_tokens and cur_len_in_tokens > truncate_length_in_tokens:
                    break

            # save number of fragments for this doc
            doc_segments_length.append(len(segmented_text))
        else:
            # 3. document is a sequence of tokens
            text = '\n'.join(text_sents)
            tokens = self._tokenizer(text)
            if truncate_length_in_tokens and len(tokens) > truncate_length_in_tokens:
                tokens = tokens[:truncate_length_in_tokens]
            segmented_text.append(tokens)
            # document is one sequence
            doc_segments_length.append(1)

        return segmented_text, doc_segments_length

    def prepare_text_from_file(self, path):
        with open_file(path) as f:
            sent_gen = (l.rstrip() for l in f)
            return self.prepare_text(sent_gen)

    def prepare_sent(self, sent_str):
        tokens = self._tokenizer(sent_str)
        tokens = tokens[: self._conf.max_sent_len]
        return tokens

    def prepare_sents(self, sent_strs):
        sent_tokens = []
        for s in sent_strs:
            tokens = self.prepare_sent(s)
            if not tokens:
                tokens = [self.vocab().pad_idx()]
            sent_tokens.append(tokens)
        return sent_tokens

    def state_dict(self):
        return {'tok': self._tokenizer.state_dict()}

    def load_state_dict(self, d):
        if 'tok' in d:
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
