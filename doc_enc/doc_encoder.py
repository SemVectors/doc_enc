#!/usr/bin/env python3

import logging
from typing import Optional
import dataclasses
from pathlib import Path

import torch

from doc_enc.text_processor import TextProcessor, TextProcessorConf

from doc_enc.encoders.enc_factory import (
    create_sent_encoder,
    create_emb_seq_encoder,
)
from doc_enc.encoders.sent_encoder import split_sents_and_embed


@dataclasses.dataclass
class DocEncoderConf:
    model_path: str
    use_gpu: Optional[int] = None

    max_sents: int = 1024
    max_tokens: int = 0


class DocEncoder:
    def __init__(self, conf: DocEncoderConf) -> None:
        self._conf = conf

        state_dict = torch.load(conf.model_path)
        tp_conf: TextProcessorConf = state_dict['tp_conf']
        tp_conf.tokenizer.vocab_path = None
        self._tp = TextProcessor(tp_conf)
        self._tp.load_state_dict(state_dict['tp'])

        if conf.use_gpu is not None and torch.cuda.is_available():
            logging.info("Computing on gpu:%s", conf.use_gpu)
            self._device = torch.device(f'cuda:{conf.use_gpu}')
        else:
            logging.info("Computing on cpu")
            self._device = torch.device('cpu')

        mc = state_dict['model_conf']
        self._sent_layer = create_sent_encoder(mc.sent.encoder, self._tp.vocab())
        self._sent_layer.load_state_dict(state_dict['sent_enc'])
        self._sent_layer = self._sent_layer.to(device=self._device).eval()
        logging.debug("sent layer\n%s", self._sent_layer)

        self._fragment_layer = None
        sent_embs_out_size = self._sent_layer.out_embs_dim()
        if 'frag_enc' in state_dict:
            self._fragment_layer = create_emb_seq_encoder(mc.fragment, sent_embs_out_size)
            self._fragment_layer.load_state_dict(state_dict['frag_enc'])
            self._fragment_layer = self._fragment_layer.to(device=self._device).eval()
            doc_input_size = self._fragment_layer.out_embs_dim()
            logging.debug("fragment layer\n:%s", self._fragment_layer)
        else:
            doc_input_size = sent_embs_out_size

        self._doc_layer = create_emb_seq_encoder(mc.doc, doc_input_size)
        self._doc_layer.load_state_dict(state_dict['doc_enc'])
        self._doc_layer = self._doc_layer.to(device=self._device).eval()

        logging.debug("doc layer\n:%s", self._doc_layer)

    def _encode_sents(self, sents):
        cnt = len(sents)
        sent_lengths = [len(t) for t in sents]
        sent_lengths = torch.as_tensor(sent_lengths, dtype=torch.int64, device=self._device)
        if cnt > self._conf.max_sents:
            return split_sents_and_embed(
                self._sent_layer,
                sents,
                sent_lengths,
                self._conf.max_sents,
                pad_idx=self._tp.vocab().pad_idx(),
            )

        max_len = len(max(sents, key=len))
        sent_tensor = torch.full((cnt, max_len), self._tp.vocab().pad_idx(), dtype=torch.int32)
        for i in range(cnt):
            sent_tensor[i, 0 : len(sents[i])] = torch.as_tensor(sents[i])

        sent_tensor = sent_tensor.to(device=self._device)

        sent_embs = self._sent_layer(sent_tensor, sent_lengths, enforce_sorted=False)['pooled_out']
        return sent_embs

    def _encode_docs_impl(self, docs, doc_fragments):
        """Each doc is a list of tokenized sentences."""

        all_sents = [s for d in docs for s in d]
        sent_embs = self._encode_sents(all_sents)

        if self._fragment_layer is not None:
            frag_len = []
            len_list = []
            for fragments in doc_fragments:
                frag_len.extend(fragments)
                len_list.append(len(fragments))

            embs = self._fragment_layer(sent_embs, frag_len, enforce_sorted=False)['pooled_out']
        else:
            embs = sent_embs
            len_list = [len(d) for d in docs]

        doc_embs = self._doc_layer(embs, len_list, enforce_sorted=False)['pooled_out']
        return doc_embs

    def _encode_docs(self, docs, doc_fragments):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                return self._encode_docs_impl(docs, doc_fragments)

    def encode_docs_from_dir(self, path: Path):
        filenames = []
        docs = []
        doc_fragments = []
        cur_token_cnt = 0
        cur_sent_cnt = 0

        embs = []
        for p in path.iterdir():
            sents, fragment_len_list = self._tp.prepare_text_from_file(p)
            token_cnt = sum(len(s) for s in sents)

            if (
                docs
                and self._conf.max_sents
                and cur_sent_cnt + len(sents) > self._conf.max_sents
                or self._conf.max_tokens
                and cur_token_cnt + token_cnt > self._conf.max_tokens
            ):
                doc_embs = self._encode_docs(docs, doc_fragments)
                embs.append(doc_embs.to(device='cpu', dtype=torch.float32))
                docs = []
                doc_fragments = []
                cur_sent_cnt = 0
                cur_token_cnt = 0

            docs.append(sents)
            doc_fragments.append(fragment_len_list)
            filenames.append(p.name)
            cur_sent_cnt += len(sents)
            cur_token_cnt += token_cnt
        if docs:
            doc_embs = self._encode_docs(docs, doc_fragments)
            embs.append(doc_embs.to(device='cpu', dtype=torch.float32))

        return filenames, torch.vstack(embs)
