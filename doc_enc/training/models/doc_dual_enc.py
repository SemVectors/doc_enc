#!/usr/bin/env python3
#!/usr/bin/env python3

import contextlib
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.models.base_doc_model import BaseDocModel
from doc_enc.training.types import DocsBatch
from doc_enc.encoders.sent_encoder import split_sents_and_embed, SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import EmbSeqEncoder


class DocDualEncoder(BaseDocModel):
    def __init__(
        self,
        conf: DocModelConf,
        sent_encoder: SentForDocEncoder,
        doc_encoder: EmbSeqEncoder,
        frag_encoder: Optional[EmbSeqEncoder] = None,
    ):
        super().__init__(conf, sent_encoder, doc_encoder, frag_encoder)
        self._src_sents_ctx_mgr = contextlib.nullcontext
        if not conf.grad_src_senst:
            self._src_sents_ctx_mgr = torch.no_grad
        self._tgt_sents_ctx_mgr = contextlib.nullcontext
        if not conf.grad_tgt_sents:
            self._tgt_sents_ctx_mgr = torch.no_grad

    def _embed_sents(self, sents, sent_len):
        return split_sents_and_embed(
            self.sent_encoder,
            sents,
            sent_len,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
        )

    def _embed_fragments(self, sent_embs, frag_len, padded_seq_len):
        if self.frag_encoder is None:
            raise RuntimeError("Logic error")
        frag_embs = self.frag_encoder(
            sent_embs, frag_len, padded_seq_len=padded_seq_len, enforce_sorted=False
        ).pooled_out
        assert len(frag_embs) == len(frag_len)
        return frag_embs

    def _embed_docs(self, embs, len_list, padded_seq_len):
        doc_embs = self.doc_encoder(
            embs, len_list, padded_seq_len=padded_seq_len, enforce_sorted=False
        ).pooled_out
        assert len(doc_embs) == len(len_list)
        return doc_embs

    def calc_sim_matrix(self, batch: DocsBatch):

        with self._src_sents_ctx_mgr():
            src_sent_embs = self._embed_sents(batch.src_sents, batch.src_sent_len)

        with self._tgt_sents_ctx_mgr():
            tgt_sent_embs = self._embed_sents(batch.tgt_sents, batch.tgt_sent_len)

        if self.frag_encoder is not None:

            src_embs = self._embed_fragments(
                src_sent_embs, batch.src_fragment_len, batch.info.get('src_fragment_len')
            )
            tgt_embs = self._embed_fragments(
                tgt_sent_embs, batch.tgt_fragment_len, batch.info.get('tgt_fragment_len')
            )
            src_len_list = batch.src_doc_len_in_frags
            tgt_len_list = batch.tgt_doc_len_in_frags
            src_padded_len = batch.info.get('src_doc_len_in_frags')
            tgt_padded_len = batch.info.get('tgt_doc_len_in_frags')
        else:
            src_embs = src_sent_embs
            tgt_embs = tgt_sent_embs
            src_len_list = batch.src_doc_len_in_sents
            tgt_len_list = batch.tgt_doc_len_in_sents
            src_padded_len = batch.info.get('src_doc_len_in_sents')
            tgt_padded_len = batch.info.get('tgt_doc_len_in_sents')
        src_doc_embs = self._embed_docs(src_embs, src_len_list, src_padded_len)

        tgt_doc_embs = self._embed_docs(tgt_embs, tgt_len_list, tgt_padded_len)

        if self.conf.normalize:
            src_doc_embs = F.normalize(src_doc_embs, p=2, dim=1)
            tgt_doc_embs = F.normalize(tgt_doc_embs, p=2, dim=1)

        m = torch.mm(src_doc_embs, tgt_doc_embs.t())  # bsz x target_bsz
        return m

    def _forward_doc_task(self, batch: DocsBatch, labels):
        m = self.calc_sim_matrix(batch)
        if self.conf.margin:
            m[labels.to(dtype=torch.bool)] -= self.conf.margin

        if self.conf.scale:
            return m * self.conf.scale
        return m

    def forward(self, batch, labels):
        return self._forward_doc_task(batch, labels)
