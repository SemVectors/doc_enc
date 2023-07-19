#!/usr/bin/env python3

import dataclasses
from enum import Enum

from transformers import AutoTokenizer, AutoConfig
import sentencepiece as spm


class TokenizerType(Enum):
    PRETOKENIZED = 1
    SENTENCEPIECE = 2
    TRANSFORMERS_AUTO = 3


@dataclasses.dataclass
class TokenizerConf:
    tokenizer_type: TokenizerType = TokenizerType.SENTENCEPIECE
    vocab_path: str | None = None
    transformers_auto_name: str = ''
    transformers_cache_dir: str | None = None

    add_bos: bool = False
    add_eos: bool = False

    enable_sampling: bool = False
    alpha: float = 0.1
    nbest_size: int = -1


class AbcTokenizer:
    def get_idx(self, token: str) -> int:
        raise NotImplementedError("Not implemented")

    def pad_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def bos_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def eos_idx(self) -> int:
        raise NotImplementedError("Not implemented")

    def vocab_size(self) -> int:
        raise NotImplementedError("Not implemented")

    def __call__(self, sent: str) -> list[int]:
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

    def __call__(self, sent: str) -> list[int]:
        return [int(s) for s in sent.split()]


class SentencepieceTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        self._conf = conf
        self._inference_mode = inference_mode
        self._vocab = spm.SentencePieceProcessor()
        if conf.vocab_path is not None:
            self._vocab.Load(conf.vocab_path)

        spm.set_random_generator_seed(42 * 42 + 52)

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

    def _modify_sent(self, sent: list[int]):
        prefix = []
        if self._conf.add_bos:
            prefix = [self.bos_idx()]
        suffix = []
        if self._conf.add_eos:
            suffix = [self.eos_idx()]
        if not prefix and not suffix:
            return sent

        return prefix + sent + suffix

    def __call__(self, sent: str) -> list[int]:
        if not self._inference_mode and self._conf.enable_sampling:
            sent = self._vocab.SampleEncodeAsIds(
                sent, alpha=self._conf.alpha, nbest_size=self._conf.nbest_size
            )
        else:
            sent = self._vocab.EncodeAsIds(sent)
        return self._modify_sent(sent)

    def state_dict(self):
        return {
            'spm': self._vocab.serialized_model_proto(),
        }

    def load_state_dict(self, d):
        self._vocab.Load(model_proto=d['spm'])


class TransformersTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        self._conf = conf
        self._inference_mode = inference_mode
        self._tokenizer = AutoTokenizer.from_pretrained(
            conf.transformers_auto_name,
            cache_dir=conf.transformers_cache_dir,
        )

    def get_idx(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def pad_idx(self) -> int:
        return self._tokenizer.pad_token_id

    def bos_idx(self) -> int:
        return self._tokenizer.bos_token_id

    def eos_idx(self) -> int:
        return self._tokenizer.eos_token_id

    def vocab_size(self) -> int:
        return len(self._tokenizer)

    def __call__(self, sent: str) -> list[int]:
        return self._tokenizer(sent, truncation=True)["input_ids"]

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


def create_tokenizer(conf: TokenizerConf, inference_mode=False) -> AbcTokenizer:
    if conf.tokenizer_type == TokenizerType.PRETOKENIZED:
        return Pretokenized(conf)
    if conf.tokenizer_type == TokenizerType.SENTENCEPIECE:
        return SentencepieceTokenizer(conf, inference_mode=inference_mode)
    if conf.tokenizer_type == TokenizerType.TRANSFORMERS_AUTO:
        return TransformersTokenizer(conf, inference_mode=inference_mode)

    raise RuntimeError(f"{conf.tokenizer_type} is not supported")
