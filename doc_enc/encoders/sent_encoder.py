#!/usr/bin/env python3

import logging
import torch
from torch import nn


class SentEncoder(nn.Module):
    def __init__(self, embed, encoder):
        super().__init__()

        self.embed = embed
        self.encoder = encoder

    def out_embs_dim(self):
        return self.encoder.out_embs_dim()

    def forward(self, tokens, lengths, enforce_sorted=True, token_types=None):
        # embed tokens
        x = self.embed(tokens, token_types)
        return self.encoder.forward(x, lengths, enforce_sorted=enforce_sorted)


def split_sents_and_embed(encoder: SentEncoder, sents, sent_lengths, split_size, pad_idx):
    lengths, sorted_indices = torch.sort(sent_lengths, descending=True)
    sorted_indices = sorted_indices.to(sent_lengths.device)
    sorted_sents = [sents[i] for i in sorted_indices]

    embs = []
    for offs in range(0, len(sents), split_size):
        cnt = min(len(sents) - offs, split_size)
        max_len = len(sorted_sents[offs])
        sents_tensor = torch.full((cnt, max_len), pad_idx, dtype=torch.int32)
        for i in range(cnt):
            sents_tensor[i, 0 : len(sorted_sents[offs + i])] = torch.as_tensor(
                sorted_sents[offs + i]
            )
        sents_tensor = sents_tensor.to(device=sent_lengths.device)

        emb = encoder(sents_tensor, lengths[offs : offs + cnt], enforce_sorted=True)['pooled_out']
        embs.append(emb)

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
