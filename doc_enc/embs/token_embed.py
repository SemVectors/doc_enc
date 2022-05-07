#!/usr/bin/env python3

import logging


from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, embs_dim, num_embeddings, padding_idx):
        super().__init__()

        if num_embeddings % 8 != 0:
            num_embeddings = ((num_embeddings // 8) + 1) * 8
        logging.info("dictionary size %s", num_embeddings)

        self.embed_tokens = nn.Embedding(num_embeddings, embs_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.embed_tokens.weight, -0.1, 0.1)
        nn.init.constant_(self.embed_tokens.weight[padding_idx], 0)

    def forward(self, tokens, token_types=None):
        return self.embed_tokens(tokens)
