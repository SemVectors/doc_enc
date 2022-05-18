#!/usr/bin/env python3

import logging
import torch
from torch import nn

from doc_enc.encoders.enc_config import SentEncoderConf


class SentEncoder(nn.Module):
    def __init__(self, conf: SentEncoderConf, embed, encoder):
        super().__init__()
        self.conf = conf

        self.embed = embed
        self.encoder = encoder
        self.output_size = (
            conf.output_size if conf.output_size is not None else self.encoder.out_embs_dim()
        )

        self.hidden_to_output_mapping = None
        self.hidden_dropout = None
        if self.output_size != self.encoder.out_embs_dim():
            self.hidden_dropout = nn.Dropout(conf.dropout)
            self.hidden_to_output_mapping = nn.Linear(self.encoder.out_embs_dim(), self.output_size)

    def out_embs_dim(self):
        return self.output_size

    def _post_proc_enc_results(self, enc_result_dict):
        if self.hidden_to_output_mapping and self.hidden_dropout:
            embs = enc_result_dict.get('pooled_out')
            if embs is None:
                raise RuntimeError("pooled_out field was not found")
            embs = self.hidden_dropout(embs)
            enc_result_dict['pooled_out'] = self.hidden_to_output_mapping(embs)
        return enc_result_dict

    def forward(self, tokens, lengths, enforce_sorted=True, token_types=None):
        # embed tokens
        x = self.embed(tokens, token_types)
        enc_result_dict = self.encoder.forward(x, lengths, enforce_sorted=enforce_sorted)
        return self._post_proc_enc_results(enc_result_dict)


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
