#!/usr/bin/env python3

import logging
from typing import Optional
import contextlib

import torch
from torch import nn

from doc_enc.encoders import enc_out
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_config import SeqEncoderConf
from doc_enc.encoders.seq_encoder import SeqEncoder


class SentForDocEncoder(SeqEncoder):
    def __init__(
        self,
        conf: SeqEncoderConf,
        encoder: BaseEncoder,
        emb_dim: int = 0,
        pad_to_multiple_of=0,
        doc_mode_encoder: Optional[BaseEncoder] = None,
        freeze_base_sents_layer=True,
    ):
        super().__init__(
            conf,
            encoder,
            prev_output_size=emb_dim,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.doc_mode_encoder = doc_mode_encoder
        self.doc_mode_dropout = None
        if self.doc_mode_encoder is not None:
            self.doc_mode_dropout = nn.Dropout(self.conf.dropout)

        self._maybe_no_grad = contextlib.nullcontext
        if freeze_base_sents_layer:
            self._maybe_no_grad = torch.no_grad

    @classmethod
    def from_base(
        cls,
        base_inst: SeqEncoder,
        doc_mode_encoder: Optional[BaseEncoder] = None,
        freeze_base_sents_layer=True,
    ):
        inst = cls(
            base_inst.conf,
            base_inst.encoder,
            emb_dim=0,
            pad_to_multiple_of=base_inst.pad_to_multiple_of,
            doc_mode_encoder=doc_mode_encoder,
            freeze_base_sents_layer=freeze_base_sents_layer,
        )
        inst.emb_to_hidden_mapping = base_inst.emb_to_hidden_mapping
        inst.beg_seq_param = base_inst.beg_seq_param
        inst.pos_emb = base_inst.pos_emb
        return inst

    def cast_to_base(self) -> SeqEncoder:
        inst = SeqEncoder(
            self.conf,
            self.encoder,
            prev_output_size=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        inst.emb_to_hidden_mapping = self.emb_to_hidden_mapping
        inst.beg_seq_param = self.beg_seq_param
        inst.pos_emb = self.pos_emb
        return inst

    def base_cls_forward(self, *args, **kwargs):
        return SeqEncoder.__call__(self, *args, **kwargs)

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_seq_lengths: torch.IntTensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> enc_out.BaseEncoderOut:
        with self._maybe_no_grad():
            sent_enc_result = super().forward(
                input_embs=input_embs,
                input_seq_lengths=input_seq_lengths,
                input_token_ids=input_token_ids,
                **kwargs,
            )

        if self.doc_mode_encoder is None or self.doc_mode_dropout is None:
            return sent_enc_result

        hidden_states = sent_enc_result.encoder_out.transpose(0, 1)
        hidden_states = self.doc_mode_dropout(hidden_states)
        enc_result = self.doc_mode_encoder(
            input_embs=hidden_states, lengths=sent_enc_result.out_lengths, **kwargs
        )
        return enc_result
