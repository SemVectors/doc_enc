#!/usr/bin/env python3

import logging
from typing import Optional

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

    def forward(self, sent_embs, lengths, padded_seq_len: Optional[int] = None, **kwargs):
        if self.emb_to_hidden_mapping is not None:
            sent_embs = self.emb_to_hidden_mapping(sent_embs)

        emb_sz = sent_embs.size(1)
        if padded_seq_len is None:
            # pad sequence of sents
            max_len = max(lengths)
            padded_seq = torch.zeros(
                (len(lengths) * max_len, emb_sz),
                device=sent_embs.device,
                dtype=sent_embs.dtype,
            )
            idx = []
            offs = 0
            for l in lengths:
                idx.extend([i] for i in range(offs, offs + l))
                offs += max_len
            idx = torch.tensor(idx, dtype=torch.int64, device=sent_embs.device).expand(-1, emb_sz)
            padded_seq.scatter_(0, idx, sent_embs)
        else:
            padded_seq = sent_embs
            max_len = padded_seq_len

        len_tensor = torch.as_tensor(lengths, dtype=torch.int64, device=sent_embs.device)
        seqs_tensor = padded_seq.reshape(len(lengths), max_len, emb_sz)
        return self.encoder(seqs_tensor, len_tensor, **kwargs)
