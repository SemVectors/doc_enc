#!/usr/bin/env python3

import torch

from doc_enc.doc_encoder import BaseEncodeModule
from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.encoders.sent_encoder import SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoder
from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.types import DocsBatch

from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.index.ivf_pq_model import TrainableIvfPQ


class BaseDocModel(BaseEncodeModule):
    def __init__(
        self,
        conf: DocModelConf,
        doc_layer: SeqEncoder,
        pad_idx: int,
        device: torch.device,
        embed: TokenEmbedding | None = None,
        sent_layer: SentForDocEncoder | None = None,
        frag_layer: SeqEncoder | None = None,
    ):
        super().__init__(
            doc_layer=doc_layer,
            embed=embed,
            sent_layer=sent_layer,
            frag_layer=frag_layer,
            pad_idx=pad_idx,
            device=device,
        )
        self.conf = conf

        self.index: TrainableIvfPQ | None = None
        if conf.index.enable:
            self.index = TrainableIvfPQ(conf.index)

    def calc_sim_matrix(
        self, batch: DocsBatch, dont_cross_device_sample=False
    ) -> DualEncModelOutput:
        raise NotImplementedError("calc_sim_matrix is not implemented")

    def forward(self, batch: DocsBatch, labels) -> DualEncModelOutput:
        raise NotImplementedError("forward is not implemented")
