#!/usr/bin/env python3

import logging

import torch
from torch import nn

from doc_enc.encoders import enc_out
from doc_enc.encoders.enc_config import SentEncoderConf


class SentEncoder(nn.Module):
    def __init__(self, conf: SentEncoderConf, embed, encoder):
        super().__init__()
        self.conf = conf

        self.embed = embed
        self.encoder = encoder

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size
        self.emb_to_hidden_mapping = None
        if conf.emb_conf.emb_dim != input_size:
            self.emb_to_hidden_mapping = nn.Linear(conf.emb_conf.emb_dim, input_size)

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def forward(
        self, tokens, lengths, enforce_sorted=True, token_types=None
    ) -> enc_out.BaseEncoderOut:
        # embed tokens
        x = self.embed(tokens.int(), lengths=lengths, token_types=token_types)

        if self.emb_to_hidden_mapping is not None:
            x = self.emb_to_hidden_mapping(x)

        enc_result = self.encoder.forward(x, lengths, enforce_sorted=enforce_sorted)
        return enc_result


def split_sents_and_embed(encoder: SentEncoder, sents, sent_lengths, split_size):
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
