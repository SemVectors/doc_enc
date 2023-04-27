#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F

from doc_enc.training.models.base_sent_model import BaseSentModel
from doc_enc.encoders.sent_encoder import split_sents_and_embed
from doc_enc.training.models.base_model import DualEncModelOutput


class SentDualEncoder(BaseSentModel):
    def _embed_sents(self, sents: torch.Tensor, sent_lengths: torch.Tensor, already_sorted=False):
        if not self.conf.split_sents:
            res = self.encoder.forward(sents, sent_lengths, enforce_sorted=already_sorted)
            return res.pooled_out
        return split_sents_and_embed(
            self.encoder,
            sents,
            sent_lengths,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
            already_sorted=already_sorted,
        )

    def calc_sim_matrix(self, batch) -> DualEncModelOutput:
        # bsz x hidden
        source_embeddings = self._embed_sents(batch.src, batch.src_len, already_sorted=True)
        # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
        target_embeddings = self._embed_sents(batch.tgt, batch.tgt_len, already_sorted=False)
        if self.conf.normalize:
            source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        m = torch.mm(source_embeddings, target_embeddings.t())  # bsz x target_bsz

        ivf_sm = None
        pq_sm = None
        if self.index is not None:
            ivf_sm, pq_sm = self.index(
                source_embeddings, target_embeddings, batch.tgt_id, normalize=self.conf.normalize
            )
        return DualEncModelOutput(m, ivf_score_matrix=ivf_sm, pq_score_matrix=pq_sm)

    def _finalize_sim_matrix(self, sm: torch.Tensor, margin, scale):
        if margin:
            diag = sm.diagonal()
            diag[:] = diag - margin

        if scale:
            return sm * scale
        return sm

    def forward(self, batch) -> DualEncModelOutput:
        output = self.calc_sim_matrix(batch)

        output.dense_score_matrix = self._finalize_sim_matrix(
            output.dense_score_matrix, self.conf.margin, self.conf.scale
        )
        if output.ivf_score_matrix is not None:
            output.ivf_score_matrix = self._finalize_sim_matrix(
                output.ivf_score_matrix, self.conf.index.ivf.margin, self.conf.index.ivf.scale
            )
        if output.pq_score_matrix is not None:
            output.pq_score_matrix = self._finalize_sim_matrix(
                output.pq_score_matrix, self.conf.index.pq.margin, self.conf.index.pq.scale
            )
        return output
