#!/usr/bin/env python3

import dataclasses

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
