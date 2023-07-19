#!/usr/bin/env python3

from torch import nn

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.types import DocsBatch
from doc_enc.encoders.sent_encoder import SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoder

from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.index.ivf_pq_model import TrainableIvfPQ


class BaseDocModel(nn.Module):
    def __init__(
        self,
        conf: DocModelConf,
        doc_encoder: SeqEncoder,
        sent_encoder: SentForDocEncoder | None = None,
        frag_encoder: SeqEncoder | None = None,
    ):
        super().__init__()
        self.conf = conf
        self.doc_encoder = doc_encoder
        self.sent_encoder = sent_encoder
        self.frag_encoder = frag_encoder

        self.index: TrainableIvfPQ | None = None
        if conf.index.enable:
            self.index = TrainableIvfPQ(conf.index)

    def calc_sim_matrix(
        self, batch: DocsBatch, dont_cross_device_sample=False
    ) -> DualEncModelOutput:
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch: DocsBatch, labels) -> DualEncModelOutput:
        raise NotImplementedError("forward is not implemented")
