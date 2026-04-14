#!/usr/bin/env python3

import logging
import time
import json
from pathlib import Path
import dataclasses
from enum import Enum


import requests
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import sentencepiece as spm


class TokenizerType(Enum):
    PRETOKENIZED = 1
    SENTENCEPIECE = 2
    TRANSFORMERS_AUTO = 3
    SBERT_AUTO = 4


@dataclasses.dataclass
class TokenizerConf:
    tokenizer_type: TokenizerType = TokenizerType.SENTENCEPIECE
    vocab_path: str | None = None

    max_seq_length: int | None = None

    transformers_auto_name: str = ''
    transformers_cache_dir: str | None = None
    # Deprecated: use max_seq_length
    auto_tokenizer_max_seq_len: int | None = None

    add_bos: bool = False
    add_eos: bool = False

    enable_sampling: bool = False
    alpha: float = 0.1
    nbest_size: int = -1


class AbcTokenizer:
    def get_max_seq_length(self) -> int | None:
        raise NotImplementedError("Not implemented")

    def special_tokens_cnt(self) -> int:
        raise NotImplementedError("Not implemented")

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

    def __call__(
        self, sent: str, add_special_tokens: bool = True, max_length: int | None = None
    ) -> list[int]:
        raise NotImplementedError("Not implemented")

    def prepare_for_model(self, tokens: list[int], max_length: int | None = None):
        """Truncate and add special tokens."""
        raise NotImplementedError("Not implemented")

    def state_dict(self):
        raise NotImplementedError("Not implemented")

    def load_state_dict(self, d):
        raise NotImplementedError("Not implemented")


class Pretokenized(AbcTokenizer):
    def __init__(self, conf: TokenizerConf) -> None:
        self._conf = conf

    def get_max_seq_length(self) -> int | None:
        return None

    def special_tokens_cnt(self) -> int:
        return 0

    def pad_idx(self):
        return 0

    def bos_idx(self) -> int:
        return -1

    def eos_idx(self) -> int:
        return -2

    def vocab_size(self) -> int:
        return 0

    def __call__(
        self, sent: str, add_special_tokens: bool = True, max_length: int | None = None
    ) -> list[int]:
        tokens = [int(s) for s in sent.split()]
        return _modify_sent(self._conf, self, tokens, add_special_tokens, max_length)

    def prepare_for_model(self, tokens: list[int], max_length: int | None = None):
        return _modify_sent(
            self._conf, self, tokens, add_special_tokens=True, max_length=max_length
        )


def _modify_sent(
    conf: TokenizerConf,
    tokenizer: AbcTokenizer,
    sent: list[int],
    add_special_tokens: bool,
    max_length: int | None,
):
    if max_length is None:
        max_length = conf.max_seq_length
    elif conf.max_seq_length is not None:
        max_length = min(max_length, conf.max_seq_length)

    prefix = []
    if add_special_tokens and conf.add_bos:
        prefix = [tokenizer.bos_idx()]
    suffix = []
    if add_special_tokens and conf.add_eos:
        suffix = [tokenizer.eos_idx()]

    if not prefix and not suffix:
        if max_length is not None:
            return sent[:max_length]
        return sent

    if max_length is not None:
        extra_len = len(prefix) + len(suffix)
        sent = sent[: max_length - extra_len]
    return prefix + sent + suffix


def _get_max_length(conf: TokenizerConf, max_length: int | None):
    if max_length is None:
        max_length = conf.max_seq_length
    elif conf.max_seq_length is not None:
        max_length = min(max_length, conf.max_seq_length)
    return max_length


class SentencepieceTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        self._conf = conf
        self._inference_mode = inference_mode
        self._vocab = spm.SentencePieceProcessor()
        if conf.vocab_path is not None:
            self._vocab.Load(conf.vocab_path)

        spm.set_random_generator_seed(42 * 42 + 52)

    def get_max_seq_length(self) -> int | None:
        return self._conf.max_seq_length

    def special_tokens_cnt(self) -> int:
        cnt = 0
        if self._conf.add_bos:
            cnt += 1
        if self._conf.add_eos:
            cnt += 1
        return cnt

    def get_idx(self, token: str) -> int:
        return self._vocab.PieceToId(token)

    def pad_idx(self):
        return self._vocab.pad_id()

    def bos_idx(self) -> int:
        return self._vocab.bos_id()

    def eos_idx(self) -> int:
        return self._vocab.eos_id()

    def vocab_size(self) -> int:
        return len(self._vocab)

    def _modify_sent(self, sent: list[int], add_special_tokens: bool, max_length: int | None):
        max_length = _get_max_length(self._conf, max_length)
        return _modify_sent(self._conf, self, sent, add_special_tokens, max_length)

    def __call__(
        self, sent: str, add_special_tokens: bool = True, max_length: int | None = None
    ) -> list[int]:
        if not self._inference_mode and self._conf.enable_sampling:
            tokens: list[int] = self._vocab.SampleEncodeAsIds(
                sent, alpha=self._conf.alpha, nbest_size=self._conf.nbest_size
            )
        else:
            tokens: list[int] = self._vocab.EncodeAsIds(sent)

        return self._modify_sent(tokens, add_special_tokens, max_length=max_length)

    def prepare_for_model(self, tokens: list[int], max_length: int | None = None):
        """Truncate and add special tokens."""

        return self._modify_sent(tokens, add_special_tokens=True, max_length=max_length)

    def state_dict(self):
        return {
            'spm': self._vocab.serialized_model_proto(),
        }

    def load_state_dict(self, d):
        self._vocab.Load(model_proto=d['spm'])


class BaseTransformersTokenizer(AbcTokenizer):
    def __init__(self, conf: TokenizerConf, tokenizer, inference_mode=False) -> None:
        self._tok_conf = conf
        self._inference_mode = inference_mode
        self._tokenizer = tokenizer
        if conf.auto_tokenizer_max_seq_len is not None and conf.max_seq_length is None:
            conf.max_seq_length = conf.auto_tokenizer_max_seq_len
            logging.warning("auto_tokenizer_max_seq_len is deprecated use max_seq_length instead.")

    def get_max_seq_length(self):
        return self._tok_conf.max_seq_length or self._tokenizer.model_max_length

    def special_tokens_cnt(self) -> int:
        return self._tokenizer.num_special_tokens_to_add()

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

    def __call__(
        self, sent: str, add_special_tokens: bool = True, max_length: int | None = None
    ) -> list[int]:
        max_length = _get_max_length(self._tok_conf, max_length)

        return self._tokenizer(
            sent, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length
        )["input_ids"]

    def prepare_for_model(self, tokens: list[int], max_length: int | None = None):
        """Truncate and add special tokens."""
        return self._tokenizer.prepare_for_model(
            tokens, add_special_tokens=True, truncation=True, max_length=max_length
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        # Compat with previous versions
        if msl := d.get('max_seq_length'):
            self._tok_conf.max_seq_length = msl


class TransformersTokenizer(BaseTransformersTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        tries = 0
        while True:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    conf.transformers_auto_name,
                    cache_dir=conf.transformers_cache_dir,
                )
                if tries > 0:
                    logging.warning("Eventually connected!")
                break
            except requests.exceptions.SSLError:
                # annoying error that pops up very often.
                # this library always tries to connect to smth online, even if all models are cached...
                logging.warning("Huggingface cant connect to its resources, waiting a bit..")
                tries += 1
                time.sleep(1)
        super().__init__(conf, tokenizer, inference_mode=inference_mode)


class SbertTokenizer(BaseTransformersTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        self._conf = conf
        temp_sbert = SentenceTransformer(
            conf.transformers_auto_name,
            cache_folder=conf.transformers_cache_dir,
            device='cpu',
        )
        if not hasattr(temp_sbert[0], 'tokenizer'):
            raise RuntimeError("Unsupported sbert model. It has no tokenizer attribute")

        conf.max_seq_length = temp_sbert.get_max_seq_length()
        super().__init__(
            conf,
            temp_sbert[0].tokenizer,
            inference_mode=inference_mode,
        )

    def create_conf_for_transformers_tok(self):
        path = Path(self._tokenizer.name_or_path)
        # should be cached downloaded dir for the whole sentenceBert model
        if not path.exists():
            raise RuntimeError(f"{path} should contained sentenceBert model")
        auto_tok_conf = path / 'tokenizer_config.json'
        if not auto_tok_conf.exists():
            raise RuntimeError(f"{path} should contain tokenizer_config.json file")

        with open(auto_tok_conf, 'r', encoding='utf8') as inpf:
            tokenizer_conf = json.load(inpf)
            if not (auto_name := tokenizer_conf['name_or_path']):
                raise RuntimeError(f"tokenizer conf {auto_tok_conf} should contain name_or_path")
            if auto_name.startswith('old_models'):
                auto_name = auto_name.replace('old_models', 'sentence-transformers')
            if auto_name.endswith('/0_Transformer'):
                auto_name = auto_name.replace('/0_Transformer', '')

            self._conf.transformers_auto_name = auto_name
            self._conf.tokenizer_type = TokenizerType.TRANSFORMERS_AUTO

        return self._conf


def create_tokenizer(conf: TokenizerConf, inference_mode=False) -> AbcTokenizer:
    if conf.tokenizer_type == TokenizerType.PRETOKENIZED:
        return Pretokenized(conf)
    if conf.tokenizer_type == TokenizerType.SENTENCEPIECE:
        return SentencepieceTokenizer(conf, inference_mode=inference_mode)
    if conf.tokenizer_type == TokenizerType.TRANSFORMERS_AUTO:
        return TransformersTokenizer(conf, inference_mode=inference_mode)
    if conf.tokenizer_type == TokenizerType.SBERT_AUTO:
        return SbertTokenizer(conf, inference_mode=inference_mode)

    raise RuntimeError(f"{conf.tokenizer_type} is not supported")
