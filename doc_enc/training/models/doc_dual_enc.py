#!/usr/bin/env python3
#!/usr/bin/env python3

import logging

import torch
from torch import nn
import torch.nn.functional as F

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.types import TaskType, DocsBatch


class DocDualEncoder(nn.Module):
    def __init__(
        self, conf: DocModelConf, sent_model, doc_encoder, frag_encoder=None, pad_idx=None
    ):
        super().__init__()
        self.conf = conf
        self.sent_model = sent_model
        self.doc_encoder = doc_encoder
        self.frag_encoder = frag_encoder
        self.pad_idx = pad_idx

    def _embed_sents(self, sents, sent_len):
        if not self.conf.split_sents:
            embeddings = self.sent_model.encoder(sents, sent_len, enforce_sorted=False)[
                'pooled_out'
            ]
            return embeddings

        lengths, sorted_indices = torch.sort(sent_len, descending=True)
        sorted_indices = sorted_indices.to(sent_len.device)
        sorted_sents = [sents[i] for i in sorted_indices]

        embs = []
        for offs in range(0, len(sents), self.conf.split_size):
            cnt = min(len(sents) - offs, self.conf.split_size)
            max_len = len(sorted_sents[offs])
            sents_tensor = torch.full((cnt, max_len), self.pad_idx, dtype=torch.int32)
            for i in range(cnt):
                sents_tensor[i, 0 : len(sorted_sents[offs + i])] = torch.as_tensor(
                    sorted_sents[offs + i]
                )
            sents_tensor = sents_tensor.to(device=sent_len.device)

            emb = self.sent_model.encoder(
                sents_tensor, lengths[offs : offs + cnt], enforce_sorted=True
            )['pooled_out']
            embs.append(emb)

        embeddings = torch.vstack(embs)

        unsorted_indices = torch.empty_like(
            sorted_indices, memory_format=torch.legacy_contiguous_format
        )
        unsorted_indices.scatter_(
            0, sorted_indices, torch.arange(0, sorted_indices.numel(), device=sorted_indices.device)
        )

        embeddings = embeddings.index_select(0, unsorted_indices)

        assert len(sents) == len(embeddings), "assert wrong size of tgt after concat"
        return embeddings

    def _embed_fragments(self, sent_embs, frag_len):
        if self.frag_encoder is None:
            raise RuntimeError("Logic error")
        frag_embs = self.frag_encoder(sent_embs, frag_len, enforce_sorted=False)['pooled_out']
        assert len(frag_embs) == len(frag_len)
        return frag_embs

    def _embed_docs(self, embs, len_list):
        doc_embs = self.doc_encoder(embs, len_list, enforce_sorted=False)['pooled_out']
        assert len(doc_embs) == len(len_list)
        return doc_embs

    def calc_sim_matrix(self, batch: DocsBatch):
        src_sent_embs = self._embed_sents(batch.src_sents, batch.src_sent_len)
        tgt_sent_embs = self._embed_sents(batch.tgt_sents, batch.tgt_sent_len)

        if self.frag_encoder is not None:
            src_embs = self._embed_fragments(src_sent_embs, batch.src_fragment_len)
            tgt_embs = self._embed_fragments(tgt_sent_embs, batch.tgt_fragment_len)
            src_len_list = batch.src_doc_len_in_frags
            tgt_len_list = batch.tgt_doc_len_in_frags
        else:
            src_embs = src_sent_embs
            tgt_embs = tgt_sent_embs
            src_len_list = batch.src_doc_len_in_sents
            tgt_len_list = batch.tgt_doc_len_in_sents
        src_doc_embs = self._embed_docs(src_embs, src_len_list)

        tgt_doc_embs = self._embed_docs(tgt_embs, tgt_len_list)

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

    def forward(self, task, batch, labels):
        if task == TaskType.SENT_RETR:
            return self.sent_model(batch)
        if task == TaskType.DOC_RETR:
            return self._forward_doc_task(batch, labels)
        raise RuntimeError(f"Unknown task {task}")
