#!/usr/bin/env python3

import logging
import math

import torch
import torch.utils.checkpoint
from torch import nn

from doc_enc.embs.pos_emb import PositionalEmbedding
from doc_enc.encoders.enc_config import SeqEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder


class SeqEncoder(nn.Module):
    def __init__(
        self,
        conf: SeqEncoderConf,
        encoder: BaseEncoder,
        prev_output_size: int = 0,
        pad_to_multiple_of: int = 0,
    ):
        super().__init__()
        self.conf = conf
        self.encoder = encoder
        self.pad_to_multiple_of = pad_to_multiple_of

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size

        self.inp_dropout = None
        if conf.input_dropout > 0.0:
            self.inp_dropout = nn.Dropout(conf.input_dropout)

        self.emb_to_hidden_mapping = None
        if prev_output_size and prev_output_size != input_size:
            self.emb_to_hidden_mapping = nn.Linear(prev_output_size, input_size)

        self.beg_seq_param = None
        if conf.add_beg_seq_token:
            self.beg_seq_param = nn.parameter.Parameter(torch.zeros(input_size))
            nn.init.uniform_(self.beg_seq_param, -0.1, 0.1)

        self.pos_emb = None
        if conf.add_pos_emb:
            self.pos_emb = PositionalEmbedding(input_size)

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

    def _pad_embs_seq(self, embs: torch.Tensor, lengths: torch.IntTensor, extra_len: int):
        emb_sz = embs.size(1)
        # pad sequence of embs
        max_len: int = int(lengths.max().item()) + extra_len
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        padded_seq = torch.zeros(
            (len(lengths) * max_len, emb_sz),
            device=embs.device,
            dtype=embs.dtype,
        )
        idx = []
        offs = 0 + extra_len
        for l in lengths:
            idx.extend([i] for i in range(offs, offs + int(l.item())))
            offs += max_len
        idx = torch.tensor(idx, dtype=torch.int64, device=embs.device).expand(-1, emb_sz)
        padded_seq.scatter_(0, idx, embs)
        if self.beg_seq_param is not None:
            padded_seq[0 : padded_seq.size(0) : max_len] = self.beg_seq_param
        return padded_seq, max_len

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_seq_lengths: torch.IntTensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        padded_seq_len: int | None = None,
        **kwargs,
    ):
        # 1. input is token ids
        if input_token_ids is not None:
            enc_result = self.encoder(
                input_token_ids=input_token_ids, lengths=input_seq_lengths, **kwargs
            )
            return enc_result

        if input_seq_lengths is None or input_embs is None:
            raise RuntimeError("Pass either input_embs and input_seq_lengths or input_token_ids")

        # 2. input is embeddings
        embs = self._prepare_input(input_embs)

        extra_len = int(self.beg_seq_param is not None)
        if padded_seq_len is None:
            padded_seq, max_len = self._pad_embs_seq(embs, input_seq_lengths, extra_len)
            emb_sz = embs.size(1)
            padded_seq = padded_seq.reshape(len(input_seq_lengths), max_len, emb_sz)
        else:
            if self.conf.add_beg_seq_token or self.pad_to_multiple_of:
                raise RuntimeError(
                    "Unsupported option add_beg_seq_token or pad_to_multiple_of"
                    " when batch  is preliminary padded"
                )
            padded_seq = embs
            max_len = padded_seq_len

        # len_tensor = torch.as_tensor(input_seq_lengths, dtype=torch.int64, device=embs.device)
        if extra_len:
            input_seq_lengths += extra_len

        if self.pos_emb is not None:
            padded_seq = padded_seq * math.sqrt(self.conf.hidden_size)
            padded_seq = self.pos_emb(padded_seq, input_seq_lengths)

        enc_result = self.encoder(input_embs=padded_seq, lengths=input_seq_lengths, **kwargs)

        return enc_result
