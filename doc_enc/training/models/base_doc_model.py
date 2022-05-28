#!/usr/bin/env python3

from typing import Optional

from torch import nn

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.types import DocsBatch
from doc_enc.encoders.sent_encoder import SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import EmbSeqEncoder


class BaseDocModel(nn.Module):
    def __init__(
        self,
        conf: DocModelConf,
        sent_encoder: SentForDocEncoder,
        doc_encoder: EmbSeqEncoder,
        frag_encoder: Optional[EmbSeqEncoder] = None,
    ):
        super().__init__()
        self.conf = conf
        self.sent_encoder = sent_encoder
        self.doc_encoder = doc_encoder
        self.frag_encoder = frag_encoder

    def calc_sim_matrix(self, batch: DocsBatch):
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch: DocsBatch, labels):
        raise NotImplementedError("forward is not implemented")
