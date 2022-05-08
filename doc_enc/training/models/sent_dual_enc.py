#!/usr/bin/env python3

import logging

import torch
from torch import nn
import torch.nn.functional as F

from doc_enc.training.models.model_conf import SentModelConf
from doc_enc.encoders.sent_encoder import split_sents_and_embed


class SentDualEncoder(nn.Module):
    def __init__(self, conf: SentModelConf, encoder, pad_idx):
        super().__init__()
        self.conf = conf
        self.encoder = encoder
        self.pad_idx = pad_idx

    def _embed_target(self, batch):
        if not self.conf.split_target_sents:
            # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
            target_embeddings = self.encoder(batch.tgt, batch.tgt_len, enforce_sorted=False)[
                'pooled_out'
            ]
            return target_embeddings
        return split_sents_and_embed(
            self.encoder,
            batch.tgt,
            batch.tgt_len,
            split_size=self.conf.split_size,
            pad_idx=self.pad_idx,
        )

    def calc_sim_matrix(self, batch):
        # bsz x hidden
        source_embeddings = self.encoder(batch.src, batch.src_len)['pooled_out']
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
