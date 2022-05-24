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
    add_bos: bool = False
    add_eos: bool = False


class AbcTokenizer:
    def get_idx(self, token):
        raise NotImplementedError("Not implemented")

    def pad_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def bos_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def eos_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def vocab_size(self) -> int:
        raise NotImplementedError("Not implemented")

    def __call__(self, sent: str) -> List[int]:
        raise NotImplementedError("Not implemented")

    def state_dict(self):
        raise NotImplementedError("Not implemented")

    def load_state_dict(self, d):
        raise NotImplementedError("Not implemented")


class Pretokenized(AbcTokenizer):
    def __init__(self, conf: TokenizerConf) -> None:
        pass

    def pad_idx(self):
        return 0

    def bos_idx(self) -> int:
        return -1

    def eos_idx(self) -> int:
        return -2

    def vocab_size(self) -> int:
        return 0

    def __call__(self, sent: str) -> List[int]:
        return [int(s) for s in sent.split()]


class SentencepieceTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf) -> None:
        self._conf = conf
        self._vocab = spm.SentencePieceProcessor()
        if conf.vocab_path is not None:
            self._vocab.Load(conf.vocab_path)

    def get_idx(self, token):
        self._vocab.PieceToId(token)

    def pad_idx(self):
        return self._vocab.pad_id()

    def bos_idx(self) -> int:
        return self._vocab.bos_id()

    def eos_idx(self) -> int:
        return self._vocab.eos_id()

    def vocab_size(self) -> int:
        return len(self._vocab)

    def _modify_sent(self, sent):
        prefix = []
        if self._conf.add_bos:
            prefix = [self.bos_idx()]
        suffix = []
        if self._conf.add_eos:
            suffix = [self.eos_idx()]
        if not prefix and not suffix:
            return sent

        return prefix + sent + suffix

    def __call__(self, sent: str) -> List[int]:
        sent = self._vocab.EncodeAsIds(sent)
        return self._modify_sent(sent)

    def state_dict(self):
        return {
            'spm': self._vocab.serialized_model_proto(),
        }

    def load_state_dict(self, d):
        self._vocab.Load(model_proto=d['spm'])


def create_tokenizer(conf: TokenizerConf) -> AbcTokenizer:
    if conf.tokenizer_type == TokenizerType.PRETOKENIZED:
        return Pretokenized(conf)
    if conf.tokenizer_type == TokenizerType.SENTENCEPIECE:
        return SentencepieceTokenizer(conf)

    raise RuntimeError(f"{conf.tokenizer_type} is not supported")
