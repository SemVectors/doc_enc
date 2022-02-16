#!/usr/bin/env python3

import math
import logging

import torch
from torch import nn
import torch.nn.functional as F

from doc_enc.training.models.model_conf import SentModelConf


class SentDualEncoder(nn.Module):
    def __init__(self, conf: SentModelConf, encoder, split_target=False):
        super().__init__()
        self.conf = conf
        self.encoder = encoder
        self.split_target = split_target

    def _embed_target(self, batch):
        if not self.split_target:
            # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
            target_embeddings = self.encoder(batch.tgt, batch.tgt_len, enforce_sorted=False)[
                'pooled_out'
            ]
            return target_embeddings
        parts = int(math.ceil(len(batch.tgt_len) / batch.bs))
        embs = []
        for tokens, lens in zip(torch.chunk(batch.tgt, parts), torch.chunk(batch.tgt_len, parts)):
            emb = self.encoder(tokens, lens, enforce_sorted=False)['pooled_out']
            embs.append(emb)

        target_embeddings = torch.vstack(embs)
        assert len(batch.tgt_len) == len(target_embeddings), "assert wrong size of tgt after concat"
        return target_embeddings

    def forward(self, batch):
        # bsz x hidden
        source_embeddings = self.encoder(batch.src, batch.src_len)['pooled_out']
        target_embeddings = self._embed_target(batch)
        if self.conf.normalize:
            source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        m = torch.mm(source_embeddings, target_embeddings.t())  # bsz x target_bsz

        if self.conf.margin:
            diag = m.diagonal()
            diag[:] = diag - self.conf.margin

        if self.conf.scale:
            return m * self.conf.scale
        return m
