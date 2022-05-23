#!/usr/bin/env python3
#!/usr/bin/env python3

import logging

import torch
from torch import nn
import torch.nn.functional as F

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.types import DocsBatch
from doc_enc.encoders.sent_encoder import split_sents_and_embed


class DocDualEncoder(nn.Module):
    def __init__(self, conf: DocModelConf, sent_model, doc_encoder, frag_encoder=None):
        super().__init__()
        self.conf = conf
        self.sent_model = sent_model
        self.doc_encoder = doc_encoder
        self.frag_encoder = frag_encoder

    def _embed_sents(self, sents, sent_len):
        if not self.conf.split_sents or len(sents) <= self.conf.split_size:
            res = self.sent_model.encoder(sents, sent_len, enforce_sorted=False)
            return res.pooled_out
        return split_sents_and_embed(
            self.sent_model.encoder,
            sents,
            sent_len,
            split_size=self.conf.split_size,
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
        src_sent_embs = self._embed_sents(batch.src_sents, batch.src_sent_len)
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
