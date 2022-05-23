#!/usr/bin/env python3
#!/usr/bin/env python3

import logging
import math

from doc_enc.embs.pos_enc import PositionalEncoding
from doc_enc.embs.token_embed import TokenEmbedding


class TokenWithPositionalEmbedding(TokenEmbedding):
    def __init__(self, embs_dim, num_embeddings, padding_idx):
        super().__init__(embs_dim, num_embeddings, padding_idx)

        self.pos_encoder = PositionalEncoding(embs_dim)

    def forward(self, tokens, token_types=None):
        x = self.embed_tokens(tokens) * math.sqrt(self.embed_tokens.embedding_dim)
        x = x.transpose(0, 1)
        return self.pos_encoder(x)
