#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F

from doc_enc.training.models.base_sent_model import BaseSentModel
from doc_enc.encoders.sent_encoder import split_sents_and_embed


class SentDualEncoder(BaseSentModel):
    def _embed_target(self, batch):
        if not self.conf.split_target_sents:
            # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
            res = self.encoder.forward(batch.tgt, batch.tgt_len, enforce_sorted=False)
            return res.pooled_out
        return split_sents_and_embed(
            self.encoder,
            batch.tgt,
            batch.tgt_len,
            split_size=self.conf.split_size,
        )

    def calc_sim_matrix(self, batch):
        # bsz x hidden
        source_embeddings = self.encoder(batch.src, batch.src_len).pooled_out
        target_embeddings = self._embed_target(batch)
        if self.conf.normalize:
            source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        m = torch.mm(source_embeddings, target_embeddings.t())  # bsz x target_bsz
        return m

    def forward(self, batch):
        m = self.calc_sim_matrix(batch)

        if self.conf.margin:
            diag = m.diagonal()
            diag[:] = diag - self.conf.margin

        if self.conf.scale:
            return m * self.conf.scale
        return m
