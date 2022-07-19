#!/usr/bin/env python3
#!/usr/bin/env python3

import logging
import torch
from torch import nn

from doc_enc.common_types import EncoderKind, PoolingStrategy
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.base_pooler import BasePoolerConf, BasePooler


class RNNPooler(BasePooler):
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
            raise RuntimeError("Logic error rnn_pooler")

        return sentemb


class BaseRNNEncoder(BaseEncoder):
    def __init__(self, conf: BaseEncoderConf):
        super().__init__()

        self.conf = conf
        proj_size = 0
        if 'proj_size' in conf and conf.proj_size is not None:
            proj_size = conf.proj_size

        if conf.encoder_kind == EncoderKind.LSTM:
            rnn_cls = nn.LSTM
            conf.get
            kwargs = {'proj_size': proj_size}
        elif conf.encoder_kind == EncoderKind.GRU:
            rnn_cls = nn.GRU
            kwargs = {}
        else:
            raise RuntimeError(f"Unsuppored rnn kind: {conf.encoder_kind}")

        self.rnn = rnn_cls(
            input_size=conf.input_size,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            bidirectional=conf.bidirectional,
            dropout=conf.dropout,
            **kwargs,
        )
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                param.data.uniform_(-0.1, 0.1)

        rnn_output_units = conf.hidden_size if not proj_size else proj_size
        if conf.bidirectional:
            rnn_output_units *= 2

        self.pooler = RNNPooler(rnn_output_units, conf.pooler)

        self.rnn_output_units = rnn_output_units
        self.output_units = rnn_output_units
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

        packed_outs, _ = self.rnn(packed_x)

        # unpack outputs and apply dropout
        pad_value = 0.0
        if self.conf.pooler.pooling_strategy == PoolingStrategy.MAX:
            pad_value = float('-inf')
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=pad_value)
        expected_shape = [seqlen, bsz, self.rnn_output_units]
        assert list(x.size()) == expected_shape, f"Got {x.size()}, but expected: {expected_shape}"

        sentemb = self.pooler(x, out_lengths)

        return BaseEncoderOut(sentemb, x, out_lengths)


class LSTMEncoder(BaseRNNEncoder):
    pass


class GRUEncoder(BaseRNNEncoder):
    pass
