#!/usr/bin/env python3

from torch import nn
from doc_enc.training.models.model_conf import SentModelConf
from doc_enc.encoders.sent_encoder import SentEncoder


class BaseSentModel(nn.Module):
    def __init__(self, conf: SentModelConf, encoder: SentEncoder):
        super().__init__()
        self.conf = conf
        self.encoder = encoder

    def calc_sim_matrix(self, batch):
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch):
        raise NotImplementedError("forward is not implemented")
