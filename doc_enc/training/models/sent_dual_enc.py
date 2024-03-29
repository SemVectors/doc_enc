#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F

from doc_enc.training.models.base_sent_model import BaseSentModel
from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.dist_util import dist_gather_target_embs


class SentDualEncoder(BaseSentModel):
    def calc_sim_matrix(self, batch, dont_cross_device_sample=False) -> DualEncModelOutput:
        # bsz x hidden
        source_embeddings = self._encode_sents_impl(
            batch.src,
            already_sorted=True,
            split_sents=self.conf.split_input,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
        )

        # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
        target_embeddings = self._encode_sents_impl(
            batch.tgt,
            already_sorted=False,
            split_sents=self.conf.split_input,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
        )
        if self.conf.normalize:
            source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        if not dont_cross_device_sample and self.conf.cross_device_sample:
            all_targets = dist_gather_target_embs(target_embeddings)
        else:
            all_targets = target_embeddings

        m = torch.mm(source_embeddings, all_targets.t())  # bsz x target_bsz

        ivf_sm = None
        pq_sm = None
        if self.index is not None:
            ivf_sm, pq_sm = self.index(
                source_embeddings,
                target_embeddings,
                batch.tgt_id,
                normalize=self.conf.normalize,
                cross_device_sample=self.conf.cross_device_sample,
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
