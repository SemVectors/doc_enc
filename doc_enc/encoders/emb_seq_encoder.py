#!/usr/bin/env python3
import logging

import torch
from torch import nn

from doc_enc.encoders.enc_config import EmbSeqEncoderConf


class EmbSeqEncoder(nn.Module):
    def __init__(self, conf: EmbSeqEncoderConf, encoder, prev_output_size):
        super().__init__()
        self.conf = conf
        self.encoder = encoder

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size

        self.emb_to_hidden_mapping = None
        if prev_output_size != input_size:
            self.emb_to_hidden_mapping = nn.Linear(prev_output_size, input_size)

        # self._beg_seq_param = None
        # if conf.add_beg_seq_token:
        #     self._beg_seq_param = nn.parameter.Parameter(torch.zeros(input_size))
        #     nn.init.uniform_(self._beg_seq_param, -0.1, 0.1)
        # self._end_seq_param = None
        # if conf.add_end_seq_token:
        #     self._end_seq_param = nn.parameter.Parameter(torch.zeros(input_size))
        #     nn.init.uniform_(self._end_seq_param, -0.1, 0.1)

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def forward(self, sent_embs, lengths, padded_seq_len=0, **kwargs):
        if not padded_seq_len:
            raise RuntimeError("Pass padded_seq_len!")

        if self.emb_to_hidden_mapping is not None:
            sent_embs = self.emb_to_hidden_mapping(sent_embs)

        # extra_len = int(self.conf.add_beg_seq_token) + int(self.conf.add_end_seq_token)

        # cnt = len(lengths)
        # max_len = max(lengths) + extra_len

        # seqs_tensor = torch.full((cnt, max_len, sent_embs.size(1)), 0.0, dtype=sent_embs.dtype)
        # offs = 0
        # for i in range(cnt):
        #     l = lengths[i]
        #     k = 0
        #     if self._beg_seq_param is not None:
        #         seqs_tensor[i, 0] = self._beg_seq_param
        #         k = 1
        #     seqs_tensor[i, k : l + k] = sent_embs[offs : offs + l]
        #     offs += l
        #     if self._end_seq_param is not None:
        #         seqs_tensor[i, l + k] = self._end_seq_param
        # seqs_tensor = seqs_tensor.to(device=sent_embs.device)

        len_tensor = torch.as_tensor(lengths, dtype=torch.int64, device=sent_embs.device)
        # if extra_len:
        #     len_tensor += extra_len
        seqs_tensor = sent_embs.reshape(len(lengths), padded_seq_len, sent_embs.size(1))
        return self.encoder(seqs_tensor, len_tensor, **kwargs)
