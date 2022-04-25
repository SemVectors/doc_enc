#!/usr/bin/env python3

import logging
from torch import nn

from doc_enc.encoders.enc_config import PoolingStrategy
from doc_enc.encoders.base_lstm import BaseLSTMEncoder


class SentLSTMEncoder(BaseLSTMEncoder):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        input_size=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        dropout=0.1,
        pooling_strategy=PoolingStrategy.MAX,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )

        if num_embeddings % 8 != 0:
            num_embeddings = ((num_embeddings // 8) + 1) * 8
        logging.info("dictionary size %s", num_embeddings)

        self.embed_tokens = nn.Embedding(num_embeddings, input_size, padding_idx=padding_idx)
        nn.init.uniform_(self.embed_tokens.weight, -0.1, 0.1)
        nn.init.constant_(self.embed_tokens.weight[padding_idx], 0)

    def forward(self, tokens, lengths, enforce_sorted=True, token_types=None):
        # embed tokens
        x = self.embed_tokens(tokens)
        return super().forward(x, lengths, enforce_sorted=enforce_sorted)
