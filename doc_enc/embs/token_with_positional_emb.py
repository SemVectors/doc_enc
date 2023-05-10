#!/usr/bin/env python3

import logging

from doc_enc.embs.pos_emb import PositionalEmbedding
from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.embs.emb_config import BaseEmbConf


class TokenWithPositionalEmbedding(TokenEmbedding):
    def __init__(self, conf: BaseEmbConf, num_embeddings, padding_idx):
        super().__init__(conf, num_embeddings, padding_idx)

        self.pos_emb = PositionalEmbedding(conf.emb_dim)

    def forward(self, tokens, lengths):
        x = super()._embed(tokens)
        x = self.pos_emb(x, lengths)
        return super()._post_proc(x)
