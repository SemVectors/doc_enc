#!/usr/bin/env python3

from torch import nn
from doc_enc.training.models.model_conf import SentModelConf
from doc_enc.encoders.sent_encoder import SentEncoder

from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.index.ivf_pq_model import TrainableIvfPQ


class BaseSentModel(nn.Module):
    def __init__(self, conf: SentModelConf, encoder: SentEncoder):
        super().__init__()
        self.conf = conf
        self.encoder = encoder
        self.index: TrainableIvfPQ | None = None
        if conf.index.enable:
            self.index = TrainableIvfPQ(conf.index)

    def calc_sim_matrix(self, batch) -> DualEncModelOutput:
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch) -> DualEncModelOutput:
        raise NotImplementedError("forward is not implemented")
