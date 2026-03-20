#!/usr/bin/env python3

from typing import Any, Mapping

import torch
from torch import nn

from doc_enc.embs.pos_emb import PositionalEmbedding
from doc_enc.encoders.enc_config import SeqEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_in import SeqEncoderBatchedInput


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
            raise RuntimeError(
                f"SeqEncoder: prev_output_size={prev_output_size} != input_size={input_size}"
            )
            # TODO
            # self.emb_to_hidden_mapping = nn.Linear(prev_output_size, input_size)

        self.beg_seq_param = None
        if conf.add_beg_seq_token:
            self.beg_seq_param = nn.parameter.Parameter(torch.zeros(input_size))
            nn.init.uniform_(self.beg_seq_param, -0.1, 0.1)

        self.pos_emb = None
        if conf.add_pos_emb:
            raise RuntimeError("SeqEncoder: positional embeddings not supported")
            # TODO
            # self.pos_emb = PositionalEmbedding(input_size)

    def input_type(self):
        return self.encoder.input_type()

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def get_padding_side(self):
        padding_side = 'right'
        if self.conf.left_padding:
            padding_side = 'left'
        return padding_side

    def _prepare_input(self, embs):
        if self.inp_dropout is not None:
            embs = self.inp_dropout(embs)

        if self.emb_to_hidden_mapping is not None:
            embs = self.emb_to_hidden_mapping(embs)
            if self.inp_dropout is not None:
                embs = self.inp_dropout(embs)
        return embs

    def forward(
        self,
        input_batch: SeqEncoderBatchedInput,
        # padded_seq_len: int | None = None,
        **kwargs,
    ):

        # 1. input is token ids
        if not input_batch.embedded:
            enc_result = self.encoder(input_batch, **kwargs)
            return enc_result

        # if input_seq_lengths is None or input_embs is None:
        #     raise RuntimeError("Pass either input_embs and input_seq_lengths or input_token_ids")

        # 2. input is embeddings
        # TODO
        # embs = self._prepare_input(input_embs)

        # TODO
        # extra_len = int(self.beg_seq_param is not None)
        # if padded_seq_len is None:
        #     padded_seq, max_len = self._pad_embs_seq(embs, input_seq_lengths, extra_len)
        #     emb_sz = embs.size(1)
        #     padded_seq = padded_seq.reshape(len(input_seq_lengths), max_len, emb_sz)
        # else:
        #     if self.conf.add_beg_seq_token or self.pad_to_multiple_of:
        #         raise RuntimeError(
        #             "Unsupported option add_beg_seq_token or pad_to_multiple_of"
        #             " when batch  is preliminary padded"
        #         )
        #     padded_seq = embs
        #     max_len = padded_seq_len

        # # len_tensor = torch.as_tensor(input_seq_lengths, dtype=torch.int64, device=embs.device)
        # if extra_len:
        #     input_seq_lengths += extra_len

        # if self.pos_emb is not None:
        #     padded_seq = padded_seq * math.sqrt(self.conf.hidden_size)
        #     padded_seq = self.pos_emb(padded_seq, input_seq_lengths)

        if self.beg_seq_param is not None:
            input_batch.prepend_tensor_(self.beg_seq_param)
        enc_result = self.encoder(input_batch, **kwargs)

        return enc_result

    def state_dict(self, *args, **kwargs):
        st = {'encoder': self.encoder.state_dict(*args, **kwargs)}
        if self.beg_seq_param is not None:
            st['beg_seq_param'] = self.beg_seq_param.cpu()
        if self.pos_emb is not None:
            st['pos_emb'] = self.pos_emb.state_dict(*args, **kwargs)
        return st

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        if 'encoder' not in state_dict:
            # compat with previous versions
            return super().load_state_dict(state_dict, **kwargs)

        if enc_state := state_dict['encoder']:
            self.encoder.load_state_dict(enc_state, **kwargs)

        if 'beg_seq_param' in state_dict:
            self.beg_seq_param = state_dict['beg_seq_param']

        if 'pos_emb' in state_dict:
            self.pos_emb.load_state_dict(state_dict['pos_emb'], **kwargs)
