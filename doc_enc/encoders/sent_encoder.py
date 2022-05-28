#!/usr/bin/env python3

import logging
from typing import Optional
import contextlib

import torch
from torch import nn

from doc_enc.encoders import enc_out
from doc_enc.encoders.enc_config import SentEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.embs.token_embed import TokenEmbedding


class SentEncoder(nn.Module):
    def __init__(
        self,
        conf: SentEncoderConf,
        embed: TokenEmbedding,
        encoder: BaseEncoder,
        emb_to_hidden_mapping: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.conf = conf

        self.embed = embed
        self.encoder = encoder

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size
        self.emb_to_hidden_mapping = emb_to_hidden_mapping
        if emb_to_hidden_mapping is None and conf.emb_conf.emb_dim != input_size:
            self.emb_to_hidden_mapping = nn.Linear(conf.emb_conf.emb_dim, input_size)

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        *,
        enforce_sorted=False,
        token_types=None,
    ) -> enc_out.BaseEncoderOut:
        # embed tokens
        x = self.embed(tokens.int(), lengths=lengths, token_types=token_types)

        if self.emb_to_hidden_mapping is not None:
            x = self.emb_to_hidden_mapping(x)

        enc_result = self.encoder.forward(x, lengths, enforce_sorted=enforce_sorted)

        return enc_result


class SentForDocEncoder(SentEncoder):
    def __init__(
        self,
        conf: SentEncoderConf,
        embed: TokenEmbedding,
        encoder: BaseEncoder,
        emb_to_hidden_mapping: Optional[nn.Linear] = None,
        doc_mode_encoder: Optional[BaseEncoder] = None,
        freeze_base_sents_layer=True,
    ):
        super().__init__(conf, embed, encoder, emb_to_hidden_mapping)
        self.doc_mode_encoder = doc_mode_encoder
        self.dropout = None
        if self.doc_mode_encoder is not None:
            self.dropout = nn.Dropout(self.conf.dropout)

        self._maybe_no_grad = contextlib.nullcontext
        if freeze_base_sents_layer:
            self._maybe_no_grad = torch.no_grad

    @classmethod
    def from_base(
        cls,
        base_inst: SentEncoder,
        doc_mode_encoder: Optional[BaseEncoder] = None,
        freeze_base_sents_layer=True,
    ):
        return cls(
            base_inst.conf,
            base_inst.embed,
            base_inst.encoder,
            base_inst.emb_to_hidden_mapping,
            doc_mode_encoder=doc_mode_encoder,
            freeze_base_sents_layer=freeze_base_sents_layer,
        )

    def cast_to_base(self):
        return SentEncoder(self.conf, self.embed, self.encoder, self.emb_to_hidden_mapping)

    def base_cls_forward(self, *args, **kwargs):
        return SentEncoder.forward(self, *args, **kwargs)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        *,
        enforce_sorted=False,
        token_types=None,
    ) -> enc_out.BaseEncoderOut:

        with self._maybe_no_grad():
            sent_enc_result = super().forward(
                tokens, lengths, enforce_sorted=enforce_sorted, token_types=token_types
            )

        if self.doc_mode_encoder is None or self.dropout is None:
            return sent_enc_result

        hidden_states = sent_enc_result.encoder_out.transpose(0, 1)
        hidden_states = self.dropout(hidden_states)
        enc_result = self.doc_mode_encoder.forward(
            hidden_states,
            sent_enc_result.out_lengths,
            enforce_sorted=enforce_sorted,
        )
        return enc_result


def split_sents_and_embed(
    encoder: SentEncoder,
    sents: torch.Tensor,
    sent_lengths: torch.Tensor,
    split_size: int,
) -> torch.Tensor:
    sorted_lengths, sorted_indices = torch.sort(sent_lengths, descending=True)
    sorted_indices = sorted_indices.to(sent_lengths.device)
    sorted_sents = sents[sorted_indices]

    embs = []
    for offs in range(0, len(sents), split_size):
        cnt = min(len(sents) - offs, split_size)
        max_len = sorted_lengths[offs].item()
        chunk = sorted_sents[offs : offs + cnt, :max_len]

        res = encoder(chunk, sorted_lengths[offs : offs + cnt], enforce_sorted=True)
        embs.append(res.pooled_out)

    embeddings = torch.vstack(embs)

    unsorted_indices = torch.empty_like(
        sorted_indices, memory_format=torch.legacy_contiguous_format
    )
    unsorted_indices.scatter_(
        0, sorted_indices, torch.arange(0, sorted_indices.numel(), device=sorted_indices.device)
    )

    embeddings = embeddings.index_select(0, unsorted_indices)

    assert len(sents) == len(embeddings), "assert wrong size of tgt after concat"
    return embeddings
