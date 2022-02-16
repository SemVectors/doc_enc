#!/usr/bin/env python3

import dataclasses
from typing import List, Optional
from enum import Enum

import sentencepiece as spm


class TokenizerType(Enum):
    PRETOKENIZED = 1
    SENTENCEPIECE = 2


@dataclasses.dataclass
class TokenizerConf:
    tokenizer_type: TokenizerType = TokenizerType.SENTENCEPIECE
    vocab_path: Optional[str] = None


class AbcTokenizer:
    def pad_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def vocab_size(self) -> int:
        raise NotImplementedError("Not implemented")

    def __call__(self, sent: str) -> List[int]:
        raise NotImplementedError("Not implemented")


class Pretokenized(AbcTokenizer):
    def __init__(self, conf: TokenizerConf) -> None:
        pass

    def pad_idx(self):
        return 0

    def vocab_size(self) -> int:
        return 0

    def __call__(self, sent: str) -> List[int]:
        return [int(s) for s in sent.split()]


class SentencepieceTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf) -> None:
        if conf.vocab_path is None:
            raise RuntimeError("Missing vocab_path option in SentencepieceTokenizer")

        self._vocab = spm.SentencePieceProcessor()
        self._vocab.Load(conf.vocab_path)

    def pad_idx(self):
        return self._vocab.pad_id()

    def vocab_size(self) -> int:
        return len(self._vocab)

    def _modify_sent_for_retr_task(self, sents):
        # TODO
        if self._add_special_symbols_in_retr:
            return [[self._bos] + sent + [self._eos] for sent in sents]
        return sents

    def __call__(self, sent: str) -> List[int]:
        return self._vocab.EncodeAsIds(sent)


def create_tokenizer(conf: TokenizerConf) -> AbcTokenizer:
    if conf.tokenizer_type == TokenizerType.PRETOKENIZED:
        return Pretokenized(conf)
    if conf.tokenizer_type == TokenizerType.SENTENCEPIECE:
        return SentencepieceTokenizer(conf)

    raise RuntimeError(f"{conf.tokenizer_type} is not supported")
