#!/usr/bin/env python3

import logging
import torch
from torch import nn

from doc_enc.encoders.enc_config import PoolingStrategy


class SentLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embedding_size=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        dropout=0.1,
        pooling_strategy=PoolingStrategy.MAX,
        **kwargs,
    ):
        super().__init__()

        self.padding_idx = padding_idx

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        if pooling_strategy not in (PoolingStrategy.MAX, PoolingStrategy.MEAN):
            raise RuntimeError(f"Unsupported pooling strategy: {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

        if num_embeddings % 8 != 0:
            num_embeddings = ((num_embeddings // 8) + 1) * 8
        logging.info("dictionary size %s", num_embeddings)
        self.embed_tokens = nn.Embedding(
            num_embeddings, embedding_size, padding_idx=self.padding_idx
        )
        nn.init.uniform_(self.embed_tokens.weight, -0.1, 0.1)
        nn.init.constant_(self.embed_tokens.weight[self.padding_idx], 0)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                param.data.uniform_(-0.1, 0.1)

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def embs_dim(self):
        return self.output_units

    def forward(self, tokens, lengths, enforce_sorted=True, token_types=None):
        bsz, seqlen = tokens.size()

        # embed tokens
        x = self.embed_tokens(tokens)

        # BS x SeqLen x Dim -> SeqLen x BS x DIM
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), enforce_sorted=enforce_sorted
        )

        packed_outs, _ = self.lstm(packed_x)

        # unpack outputs and apply dropout
        pad_value = 0.0
        if self.pooling_strategy == PoolingStrategy.MAX:
            pad_value = float('-inf')
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=pad_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.pooling_strategy == PoolingStrategy.MAX:
            # Build the sentence embedding by max-pooling over the encoder outputs
            sentemb = torch.max(x, dim=0)[0]
        elif self.pooling_strategy == PoolingStrategy.MEAN:
            sum_embeddings = torch.sum(x, dim=0)
            sentemb = sum_embeddings / lengths.unsqueeze(-1).to(sum_embeddings.device)
        else:
            raise RuntimeError("Logic error ps_lstm")

        return {'pooled_out': sentemb, 'encoder_out': x, 'out_lengths': out_lengths}
