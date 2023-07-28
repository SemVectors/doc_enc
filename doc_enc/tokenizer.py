#!/usr/bin/env python3

import json
from pathlib import Path
import dataclasses
from enum import Enum


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
    transformers_auto_name: str = ''
    transformers_cache_dir: str | None = None

    add_bos: bool = False
    add_eos: bool = False

    enable_sampling: bool = False
    alpha: float = 0.1
    nbest_size: int = -1


class AbcTokenizer:
    def get_max_seq_length(self) -> int | None:
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

    def get_max_seq_length(self):
        return None

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


class BaseTransformersTokenizer(AbcTokenizer):
    def __init__(self, tokenizer, max_seq_length=None, inference_mode=False) -> None:
        self._inference_mode = inference_mode
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def get_max_seq_length(self):
        return self._max_seq_length

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
        return self._tokenizer(sent, truncation=True, max_length=self._max_seq_length)["input_ids"]

    def state_dict(self):
        d = {}
        if self._max_seq_length is not None:
            d['max_seq_length'] = self._max_seq_length
        return d

    def load_state_dict(self, di):
        if msl := di.get('max_seq_length'):
            self._max_seq_length = msl


class TransformersTokenizer(BaseTransformersTokenizer):
    def __init__(self, conf: TokenizerConf, inference_mode=False) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            conf.transformers_auto_name,
            cache_dir=conf.transformers_cache_dir,
        )
        super().__init__(tokenizer, inference_mode=inference_mode)


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

        super().__init__(
            temp_sbert[0].tokenizer,
            max_seq_length=temp_sbert.get_max_seq_length(),
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
