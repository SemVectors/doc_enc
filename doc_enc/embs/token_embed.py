#!/usr/bin/env python3

import logging
import math

from torch import nn
from doc_enc.embs.emb_config import BaseEmbConf


class TokenEmbedding(nn.Module):
    def __init__(self, conf: BaseEmbConf, num_embeddings, padding_idx):
        super().__init__()

        self.conf = conf

        if num_embeddings % 8 != 0:
            num_embeddings = ((num_embeddings // 8) + 1) * 8
        logging.info("dictionary size %s", num_embeddings)

        self.embed_tokens = nn.Embedding(num_embeddings, conf.emb_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.embed_tokens.weight, -0.1, 0.1)
        nn.init.constant_(self.embed_tokens.weight[padding_idx], 0)

        self.layer_norm = None
        if conf.normalize_emb:
            self.layer_norm = nn.LayerNorm(conf.emb_dim)

        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)

    def _embed(self, tokens, token_types=None):
        embs = self.embed_tokens(tokens)
        if self.conf.scale_by_dim:
            embs = embs * math.sqrt(self.conf.emb_dim)
        return embs

    def _post_proc(self, embs):
        if self.layer_norm is not None:
            embs = self.layer_norm(embs)
        if self.dropout is not None:
            embs = self.dropout(embs)
        return embs

    def forward(self, tokens, lengths=None, token_types=None):
        embs = self._embed(tokens, token_types=token_types)
        return self._post_proc(embs)
