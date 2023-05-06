#!/usr/bin/env python3

import logging
from typing import Optional
import contextlib

import torch
from torch import nn

from doc_enc.encoders import enc_out
from doc_enc.encoders.enc_config import SentEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.embs.token_embed import TokenEmbedding


class SentEncoder(nn.Module):
    def __init__(
        self,
        conf: SentEncoderConf,
        embed: TokenEmbedding,
        encoder: BaseEncoder,
        emb_to_hidden_mapping: Optional[nn.Linear] = None,
        pad_to_multiple_of=0,
    ):
        super().__init__()
        self.conf = conf

        self.embed = embed
        self.encoder = encoder
        self.pad_to_multiple_of = pad_to_multiple_of

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size
        self.emb_to_hidden_mapping = emb_to_hidden_mapping
        if emb_to_hidden_mapping is None and conf.emb_conf.emb_dim != input_size:
            self.emb_to_hidden_mapping = nn.Linear(conf.emb_conf.emb_dim, input_size)

    def emb_named_params(self):
        yield from self.embed.named_parameters()
        if self.emb_to_hidden_mapping is not None:
            yield from self.emb_to_hidden_mapping.named_parameters()

    def enc_named_params(self):
        yield from self.encoder.named_parameters()

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
        if self.pad_to_multiple_of and tokens.size(1) % self.pad_to_multiple_of != 0:
            raise RuntimeError(
                f"Sents should be padded in batch generator to {self.pad_to_multiple_of}"
            )
        # embed tokens
        x = self.embed(tokens.int(), lengths=lengths, token_types=token_types)

        if self.emb_to_hidden_mapping is not None:
            x = self.emb_to_hidden_mapping(x)

        enc_result = self.encoder(x, lengths, enforce_sorted=enforce_sorted)

        return enc_result


class SentForDocEncoder(SentEncoder):
    def __init__(
        self,
        conf: SentEncoderConf,
        embed: TokenEmbedding,
        encoder: BaseEncoder,
        emb_to_hidden_mapping: Optional[nn.Linear] = None,
        pad_to_multiple_of=0,
        doc_mode_encoder: Optional[BaseEncoder] = None,
        freeze_base_sents_layer=True,
    ):
        super().__init__(
            conf,
            embed,
            encoder,
            emb_to_hidden_mapping=emb_to_hidden_mapping,
            pad_to_multiple_of=pad_to_multiple_of,
        )
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
            base_inst.pad_to_multiple_of,
            doc_mode_encoder=doc_mode_encoder,
            freeze_base_sents_layer=freeze_base_sents_layer,
        )

    def cast_to_base(self):
        return SentEncoder(
            self.conf,
            self.embed,
            self.encoder,
            emb_to_hidden_mapping=self.emb_to_hidden_mapping,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def base_cls_forward(self, *args, **kwargs):
        return SentEncoder.__call__(self, *args, **kwargs)

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
        enc_result = self.doc_mode_encoder(
            hidden_states,
            sent_enc_result.out_lengths,
            enforce_sorted=enforce_sorted,
        )
        return enc_result


def _encoder_res_finalize(res: BaseEncoderOut, collect_on_cpu=False):
    if collect_on_cpu:
        return res.pooled_out.cpu()
    return res.pooled_out


def split_sents_and_embed(
    encoder: SentEncoder,
    sents: torch.Tensor,
    sent_lengths: torch.Tensor,
    max_chunk_size: int,
    max_tokens_in_chunk: int,
    collect_on_cpu: bool = False,
    already_sorted: bool = False,
) -> torch.Tensor:
    if not sent_lengths.numel():
        return torch.FloatTensor()

    max_len = torch.max(sent_lengths).item()
    sents_cnt = sents.size(0)
    if sents_cnt <= max_chunk_size and max_len * sents_cnt < max_tokens_in_chunk:
        res = encoder(sents, sent_lengths, enforce_sorted=False)
        return _encoder_res_finalize(res, collect_on_cpu)

    if not already_sorted:
        sorted_lengths, sorted_indices = torch.sort(sent_lengths, descending=True)
        sorted_indices = sorted_indices.to(sent_lengths.device)
        sorted_sents = sents[sorted_indices]
    else:
        sorted_sents = sents
        sorted_lengths = sent_lengths
        sorted_indices = None

    embs = []
    beg_offs = 0

    if sorted_lengths[0].item() > max_tokens_in_chunk:
        raise RuntimeError(
            f"max_tokens_in_chunk ({max_tokens_in_chunk}) is too low"
            " or max sentence size is too big"
        )

    pad_to_multiple_of = encoder.pad_to_multiple_of
    if pad_to_multiple_of and sents.size(1) % pad_to_multiple_of != 0:
        raise RuntimeError(f"Sents should be padded in batch generator to {pad_to_multiple_of}")

    while beg_offs < sents_cnt:
        max_len = sorted_lengths[beg_offs].item()
        if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
            max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

        cnt_by_tokens = max_tokens_in_chunk // max_len
        cnt = min(max_chunk_size, cnt_by_tokens, sents_cnt - beg_offs)
        chunk = sorted_sents[beg_offs : beg_offs + cnt, :max_len]

        res = encoder(chunk, sorted_lengths[beg_offs : beg_offs + cnt], enforce_sorted=True)
        embs.append(_encoder_res_finalize(res, collect_on_cpu))

        beg_offs += cnt

    embeddings = torch.vstack(embs)

    if sorted_indices is not None:
        unsorted_indices = torch.empty_like(
            sorted_indices, memory_format=torch.legacy_contiguous_format, device=embeddings.device
        )
        unsorted_indices.scatter_(
            0,
            sorted_indices.to(device=embeddings.device),
            torch.arange(0, sorted_indices.numel(), device=embeddings.device),
        )

        embeddings = embeddings.index_select(0, unsorted_indices)

    assert len(sents) == len(embeddings), "assert wrong size of tgt after concat"
    return embeddings
