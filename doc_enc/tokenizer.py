#!/usr/bin/env python3

from typing import List
from enum import Enum

import sentencepiece as spm


class TokenizerType(Enum):
    PRETOKENIZED = 1
    SENTENCEPIECE = 2


class AbcTokenizer:
    def __call__(self, sent: str) -> List[int]:
        raise NotImplementedError("Not implemented")

    def pad_idx(self):
        raise NotImplementedError("Not implemented")


class Pretokenized(AbcTokenizer):
    def __init__(self, **kwargs) -> None:
        pass

    def pad_idx(self):
        return 0

    def __call__(self, sent: str) -> List[int]:
        return [int(s) for s in sent.split()]


class SentencepieceTokenizer(AbcTokenizer):
    def __init__(self, vocab_path) -> None:
        self._vocab = spm.SentencePieceProcessor()
        self._vocab.Load(vocab_path)

    def pad_idx(self):
        return self._vocab.pad_id()

    def _modify_sent_for_retr_task(self, sents):
        # TODO
        if self._add_special_symbols_in_retr:
            return [[self._bos] + sent + [self._eos] for sent in sents]
        return sents

    def __call__(self, sent: str) -> List[int]:
        raise NotImplementedError("Not implemented")


def create_tokenizer(tok_type: TokenizerType, **kwargs):
    if tok_type == TokenizerType.PRETOKENIZED:
        return Pretokenized(**kwargs)

    raise RuntimeError(f"{tok_type} is not supported")
