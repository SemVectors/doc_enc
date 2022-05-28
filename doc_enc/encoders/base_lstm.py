#!/usr/bin/env python3
#!/usr/bin/env python3

import logging
import torch
from torch import nn

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.base_pooler import BasePoolerConf, BasePooler


class LSTMPooler(BasePooler):
    def __init__(self, emb_dim, conf: BasePoolerConf):
        super().__init__(emb_dim, conf)
        if conf.pooling_strategy not in (PoolingStrategy.MAX, PoolingStrategy.MEAN):
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")

    def _pooling_impl(
        self, hidden_states: torch.Tensor, lengths: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if self.conf.pooling_strategy == PoolingStrategy.MAX:
            # Build the sentence embedding by max-pooling over the encoder outputs
            sentemb = torch.max(hidden_states, dim=0)[0]

        elif self.conf.pooling_strategy == PoolingStrategy.MEAN:
            sum_embeddings = torch.sum(hidden_states, dim=0)
            sentemb = sum_embeddings / lengths.unsqueeze(-1).to(sum_embeddings.device)
        else:
            raise RuntimeError("Logic error ps_lstm")

        return sentemb


class BaseLSTMEncoder(BaseEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__()

        self.conf = conf

        self.lstm = nn.LSTM(
            input_size=conf.input_size,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            bidirectional=conf.bidirectional,
            dropout=conf.dropout,
        )
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                param.data.uniform_(-0.1, 0.1)

        lstm_output_units = conf.hidden_size
        if conf.bidirectional:
            lstm_output_units *= 2

        self.pooler = LSTMPooler(lstm_output_units, conf.pooler)

        self.lstm_output_units = lstm_output_units
        self.output_units = lstm_output_units
        if self.conf.pooler.out_size is not None:
            self.output_units = self.conf.pooler.out_size

    def out_embs_dim(self):
        return self.output_units

    def forward(
        self, embs: torch.Tensor, lengths: torch.Tensor, enforce_sorted=True, **kwargs
    ) -> BaseEncoderOut:

        bsz, seqlen = embs.size()[:2]
        # BS x SeqLen x Dim -> SeqLen x BS x DIM
        x = embs.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), enforce_sorted=enforce_sorted
        )

        packed_outs, _ = self.lstm(packed_x)

        # unpack outputs and apply dropout
        pad_value = 0.0
        if self.conf.pooler.pooling_strategy == PoolingStrategy.MAX:
            pad_value = float('-inf')
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=pad_value)
        assert list(x.size()) == [seqlen, bsz, self.lstm_output_units]

        sentemb = self.pooler(x, out_lengths)

        return BaseEncoderOut(sentemb, x, out_lengths)


class LSTMEncoder(BaseLSTMEncoder):
    pass
