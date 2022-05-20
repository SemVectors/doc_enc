#!/usr/bin/env python3
#!/usr/bin/env python3

import logging
import torch
from torch import nn

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_out import BaseEncoderOut


class BaseLSTMEncoder(nn.Module):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__()

        self.conf = conf
        if conf.pooling_strategy not in (PoolingStrategy.MAX, PoolingStrategy.MEAN):
            raise RuntimeError(f"Unsupported pooling strategy: {conf.pooling_strategy}")

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

        self.output_units = conf.hidden_size
        if conf.bidirectional:
            self.output_units *= 2

    def out_embs_dim(self):
        return self.output_units

    def forward(self, embs, lengths, enforce_sorted=True, token_types=None) -> BaseEncoderOut:

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
        if self.conf.pooling_strategy == PoolingStrategy.MAX:
            pad_value = float('-inf')
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=pad_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.conf.pooling_strategy == PoolingStrategy.MAX:
            # Build the sentence embedding by max-pooling over the encoder outputs
            sentemb = torch.max(x, dim=0)[0]

        elif self.conf.pooling_strategy == PoolingStrategy.MEAN:
            sum_embeddings = torch.sum(x, dim=0)
            sentemb = sum_embeddings / lengths.unsqueeze(-1).to(sum_embeddings.device)
        else:
            raise RuntimeError("Logic error ps_lstm")

        return BaseEncoderOut(sentemb, x, out_lengths)


class LSTMEncoder(BaseLSTMEncoder):
    pass
