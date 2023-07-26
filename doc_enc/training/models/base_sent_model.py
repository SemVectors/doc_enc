#!/usr/bin/env python3

import torch
from doc_enc.doc_encoder import BaseSentEncodeModule
from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.encoders.emb_seq_encoder import SeqEncoder
from doc_enc.training.models.model_conf import SentModelConf

from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.index.ivf_pq_model import TrainableIvfPQ


class BaseSentModel(BaseSentEncodeModule):
    def __init__(
        self,
        conf: SentModelConf,
        encoder: SeqEncoder,
        pad_idx: int,
        device: torch.device,
        embed: TokenEmbedding | None = None,
    ):
        super().__init__(pad_idx=pad_idx, device=device, embed=embed, sent_layer=encoder)
        self.conf = conf
        self.index: TrainableIvfPQ | None = None
        if conf.index.enable:
            self.index = TrainableIvfPQ(conf.index)

    def calc_sim_matrix(self, batch, dont_cross_device_sample=False) -> DualEncModelOutput:
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch) -> DualEncModelOutput:
        raise NotImplementedError("forward is not implemented")
