#!/usr/bin/env python3

import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pad_idx = 0

        self.position_embeddings = torch.nn.Embedding(
            max_len + 1, d_model, padding_idx=self.pad_idx
        )

    def forward(self, embs, seq_lengths):
        # embs shape: bsz, seq, emb_dim
        max_len = seq_lengths.max().item()
        cnt = len(seq_lengths)
        position_ids = torch.arange(start=1, end=max_len + 1).expand((cnt, -1)).clone()
        for i, l in enumerate(seq_lengths):
            position_ids[i, l:] = 0

        position_ids = position_ids.to(device=embs.device)
        position_embeddings = self.position_embeddings(position_ids)
        return embs + position_embeddings
