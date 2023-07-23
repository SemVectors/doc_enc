#!/usr/bin/env python3
#!/usr/bin/env python3

import contextlib
import logging
from typing import Any

import torch
import torch.nn.functional as F

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.models.base_doc_model import BaseDocModel
from doc_enc.training.types import DocsBatch
from doc_enc.encoders.sent_encoder import SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoder
from doc_enc.training.models.base_model import DualEncModelOutput
from doc_enc.training.dist_util import dist_gather_target_embs


class DocDualEncoder(BaseDocModel):
    def __init__(
        self,
        conf: DocModelConf,
        doc_layer: SeqEncoder,
        pad_idx: int,
        device: torch.device,
        sent_layer: SentForDocEncoder | None = None,
        frag_layer: SeqEncoder | None = None,
    ):
        super().__init__(
            conf,
            doc_layer=doc_layer,
            sent_layer=sent_layer,
            frag_layer=frag_layer,
            pad_idx=pad_idx,
            device=device,
        )
        self._src_sents_ctx_mgr = contextlib.nullcontext
        if not conf.grad_src_sents:
            self._src_sents_ctx_mgr = torch.no_grad
        self._tgt_sents_ctx_mgr = contextlib.nullcontext
        if not conf.grad_tgt_sents:
            self._tgt_sents_ctx_mgr = torch.no_grad
        self._cur_sents_ctx_mgr = contextlib.nullcontext

    def _reset_cur_sents_ctx_mrg(self, ctx_mgr: Any = contextlib.nullcontext):
        self._cur_sents_ctx_mgr = ctx_mgr

    def _encode_sents_impl(self, *args, **kwargs):
        with self._cur_sents_ctx_mgr():
            return super()._encode_sents_impl(*args, **kwargs)

    def _create_batch_info_dict(self, prefix, batch: DocsBatch):
        d = {}
        for n in ['doc_len_in_sents', 'doc_len_in_frags', 'fragment_len']:
            d[n] = batch.info.get(f'{prefix}_{n}')
        return d

    def calc_sim_matrix(
        self, batch: DocsBatch, dont_cross_device_sample=False
    ) -> DualEncModelOutput:
        self._reset_cur_sents_ctx_mrg(self._src_sents_ctx_mgr)
        src_doc_embs = self._encode_docs_impl(
            batch.src_texts,
            batch.src_doc_segments_length,
            split_sents=self.conf.split_sents,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
            batch_info=self._create_batch_info_dict('src', batch),
        )
        self._reset_cur_sents_ctx_mrg(self._tgt_sents_ctx_mgr)
        tgt_doc_embs = self._encode_docs_impl(
            batch.tgt_texts,
            batch.tgt_doc_segments_length,
            split_sents=self.conf.split_sents,
            max_chunk_size=self.conf.max_chunk_size,
            max_tokens_in_chunk=self.conf.max_tokens_in_chunk,
            batch_info=self._create_batch_info_dict('tgt', batch),
        )

        if self.conf.normalize:
            src_doc_embs = F.normalize(src_doc_embs, p=2, dim=1)
            tgt_doc_embs = F.normalize(tgt_doc_embs, p=2, dim=1)

        if not dont_cross_device_sample and self.conf.cross_device_sample:
            all_targets = dist_gather_target_embs(tgt_doc_embs)
        else:
            all_targets = tgt_doc_embs

        m = torch.mm(src_doc_embs, all_targets.t())  # bsz x target_bsz

        ivf_sm = None
        pq_sm = None
        if self.index is not None:
            ivf_sm, pq_sm = self.index(
                src_doc_embs,
                tgt_doc_embs,
                batch.tgt_ids,
                normalize=self.conf.normalize,
                cross_device_sample=self.conf.cross_device_sample,
            )
        return DualEncModelOutput(m, ivf_score_matrix=ivf_sm, pq_score_matrix=pq_sm)

    def _finalize_sim_matrix(self, sm: torch.Tensor, labels, margin, scale):
        if margin:
            if self.conf.cross_device_sample:
                l = torch.zeros_like(sm)
                l[:, : labels.shape[1]] = labels
                labels = l
            sm[labels.to(dtype=torch.bool)] -= margin

        if scale:
            return sm * scale

        return sm

    def _forward_doc_task(self, batch: DocsBatch, labels) -> DualEncModelOutput:
        output = self.calc_sim_matrix(batch)

        output.dense_score_matrix = self._finalize_sim_matrix(
            output.dense_score_matrix, labels, self.conf.margin, self.conf.scale
        )
        if output.ivf_score_matrix is not None:
            output.ivf_score_matrix = self._finalize_sim_matrix(
                output.ivf_score_matrix,
                labels,
                self.conf.index.ivf.margin,
                self.conf.index.ivf.scale,
            )
        if output.pq_score_matrix is not None:
            output.pq_score_matrix = self._finalize_sim_matrix(
                output.pq_score_matrix, labels, self.conf.index.pq.margin, self.conf.index.pq.scale
            )
        return output

    def forward(self, batch: DocsBatch, labels) -> DualEncModelOutput:
        return self._forward_doc_task(batch, labels)
