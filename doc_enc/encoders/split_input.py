#!/usr/bin/env python3

import logging
from typing import Callable

import torch

from doc_enc.encoders.enc_out import BaseEncoderOut


def _encoder_res_finalize(res: BaseEncoderOut, collect_on_cpu=False):
    if collect_on_cpu:
        return res.pooled_out.cpu()
    return res.pooled_out


def split_input_and_embed(
    encoder: Callable,
    input: torch.Tensor,
    input_lengths: torch.Tensor,
    max_chunk_size: int,
    max_tokens_in_chunk: int,
    collect_on_cpu: bool = False,
    already_sorted: bool = False,
    pad_to_multiple_of: int = 0,
) -> torch.Tensor:
    if not input_lengths.numel():
        return torch.FloatTensor()

    max_len = torch.max(input_lengths).item()
    sents_cnt = input.size(0)
    if sents_cnt <= max_chunk_size and max_len * sents_cnt < max_tokens_in_chunk:
        res = encoder(input, input_lengths, enforce_sorted=already_sorted)
        return _encoder_res_finalize(res, collect_on_cpu)

    if not already_sorted:
        sorted_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
        sorted_indices = sorted_indices.to(input_lengths.device)
        sorted_sents = input[sorted_indices]
    else:
        sorted_sents = input
        sorted_lengths = input_lengths
        sorted_indices = None

    embs = []
    beg_offs = 0

    if sorted_lengths[0].item() > max_tokens_in_chunk:
        raise RuntimeError(
            f"max_tokens_in_chunk ({max_tokens_in_chunk}) is too low"
            " or max sentence size is too big"
        )

    if pad_to_multiple_of and input.size(1) % pad_to_multiple_of != 0:
        raise RuntimeError(f"Sents should be padded in batch generator to {pad_to_multiple_of}")

    sorted_lengths_list = sorted_lengths.tolist()
    while beg_offs < sents_cnt:
        max_len = sorted_lengths_list[beg_offs]
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

    assert len(input) == len(embeddings), "assert wrong size of tgt after concat"
    return embeddings
