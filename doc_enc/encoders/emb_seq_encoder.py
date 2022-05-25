#!/usr/bin/env python3

import logging
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn

from doc_enc.encoders.enc_config import EmbSeqEncoderConf


class EmbSeqEncoder(nn.Module):
    def __init__(self, conf: EmbSeqEncoderConf, encoder, prev_output_size):
        super().__init__()
        self.conf = conf
        self.encoder = encoder

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size

        self.inp_dropout = None
        if conf.input_dropout > 0.0:
            self.inp_dropout = nn.Dropout(conf.input_dropout)

        self.emb_to_hidden_mapping = None
        if prev_output_size != input_size:
            self.emb_to_hidden_mapping = nn.Linear(prev_output_size, input_size)

        self._beg_seq_param = None
        if conf.add_beg_seq_token:
            self._beg_seq_param = nn.parameter.Parameter(torch.zeros(input_size))
            nn.init.uniform_(self._beg_seq_param, -0.1, 0.1)
        # TODO end seq param
        # self._end_seq_param = None

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def _prepare_input(self, embs):
        if self.inp_dropout is not None:
            embs = self.inp_dropout(embs)

        if self.emb_to_hidden_mapping is not None:
            embs = self.emb_to_hidden_mapping(embs)
            if self.inp_dropout is not None:
                embs = self.inp_dropout(embs)
        return embs

    def forward(self, embs, lengths, padded_seq_len: Optional[int] = None, **kwargs):
        embs = self._prepare_input(embs)

        extra_len = int(self._beg_seq_param is not None)
        emb_sz = embs.size(1)
        if padded_seq_len is None:
            # pad sequence of embs
            max_len = max(lengths) + extra_len
            padded_seq = torch.zeros(
                (len(lengths) * max_len, emb_sz),
                device=embs.device,
                dtype=embs.dtype,
            )
            idx = []
            offs = 0 + extra_len
            for l in lengths:
                idx.extend([i] for i in range(offs, offs + l))
                offs += max_len
            idx = torch.tensor(idx, dtype=torch.int64, device=embs.device).expand(-1, emb_sz)
            padded_seq.scatter_(0, idx, embs)
            if self._beg_seq_param is not None:
                padded_seq[0 : padded_seq.size(0) : max_len] = self._beg_seq_param
        else:
            if self.conf.add_beg_seq_token:
                raise RuntimeError(
                    "Unsupported option add_beg_seq_token when batch  is preliminary padded"
                )
            padded_seq = embs
            max_len = padded_seq_len

        len_tensor = torch.as_tensor(lengths, dtype=torch.int64, device=embs.device)
        if extra_len:
            len_tensor += extra_len
        seqs_tensor = padded_seq.reshape(len(lengths), max_len, emb_sz)
        enc_result = self.encoder(seqs_tensor, len_tensor, **kwargs)

        return enc_result
