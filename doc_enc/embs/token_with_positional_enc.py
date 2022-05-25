#!/usr/bin/env python3

import logging

from doc_enc.embs.pos_enc import PositionalEncoding
from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.embs.emb_config import BaseEmbConf


class TokenWithPositionalEncoding(TokenEmbedding):
    def __init__(self, conf: BaseEmbConf, num_embeddings, padding_idx):
        super().__init__(conf, num_embeddings, padding_idx)

        self.pos_encoder = PositionalEncoding(conf.emb_dim)

    def forward(self, tokens, lengths=None, token_types=None):
        x = super()._embed(tokens, token_types=token_types)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        return super()._post_proc(x.transpose(0, 1))
