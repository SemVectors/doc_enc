#!/usr/bin/env python3

import logging
from typing import Callable

import torch

from doc_enc.encoders.enc_in import EncoderInputType, SeqEncoderBatchedInput
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.pad_utils import PadOpts


def _encoder_res_finalize(res: BaseEncoderOut, collect_on_cpu=False):
    if collect_on_cpu:
        return res.pooled_out.cpu()
    return res.pooled_out


def split_padded_input_and_encode(
    encoder: Callable,
    orig_input_data: SeqEncoderBatchedInput,
    max_chunk_size: int,
    max_tokens_in_chunk: int,
    collect_on_cpu: bool = False,
    already_sorted: bool = False,
    pad_opts: PadOpts = PadOpts(),
) -> torch.Tensor:
    if orig_input_data.batch_size == 0:
        return torch.FloatTensor()

    max_len: int = orig_input_data.max_len
    batch_size = orig_input_data.batch_size
    if batch_size <= max_chunk_size and max_len * batch_size < max_tokens_in_chunk:
        res = encoder(orig_input_data, enforce_sorted=already_sorted)
        return _encoder_res_finalize(res, collect_on_cpu)

    # orig_data = copy.copy(orig_input_data)
    orig_padded = orig_input_data.get_padded()
    # TODO should be part of input data
    if not already_sorted:
        sorted_lengths, sorted_indices = torch.sort(orig_padded.lengths, descending=True)
        sorted_indices = sorted_indices.to(orig_padded.lengths.device)
        sorted_seqs = orig_padded.data[sorted_indices]
    else:
        sorted_seqs = orig_padded.data
        sorted_lengths = orig_padded.lengths
        sorted_indices = None

    embs = []
    beg_offs = 0

    if sorted_lengths[0].item() > max_tokens_in_chunk:
        raise RuntimeError(
            f"max seq len in chunk ({sorted_lengths[0].item()}) is larger"
            f"than max_tokens_in_chunk ({max_tokens_in_chunk})."
        )

    mult_of = pad_opts.pad_to_multiple_of
    if mult_of and orig_padded.data.size(1) % mult_of != 0:
        raise RuntimeError(f"Sents should be padded in batch generator to {mult_of}")

    sorted_lengths_list = sorted_lengths.tolist()
    while beg_offs < batch_size:
        chunk_max_len: int = sorted_lengths_list[beg_offs]
        if mult_of and chunk_max_len % mult_of != 0:
            chunk_max_len = ((chunk_max_len // mult_of) + 1) * mult_of

        cnt_by_tokens = max_tokens_in_chunk // chunk_max_len
        cnt = min(max_chunk_size, cnt_by_tokens, batch_size - beg_offs)
        if pad_opts.padding_side == 'right':
            chunk = sorted_seqs[beg_offs : beg_offs + cnt, :chunk_max_len]
        elif pad_opts.padding_side == 'left':
            chunk = sorted_seqs[beg_offs : beg_offs + cnt, max_len - chunk_max_len :]
        else:
            raise RuntimeError("Unknown value of padding_side: " + pad_opts.padding_side)

        input_data = SeqEncoderBatchedInput(EncoderInputType.PADDED)
        # logging.error("Split chunk: bs %s, max len %s", cnt, chunk_max_len)
        input_data.batch_size = cnt
        input_data.max_len = chunk_max_len
        input_data.batch = orig_padded._replace(
            data=chunk, lengths=sorted_lengths[beg_offs : beg_offs + cnt]
        )

        res = encoder(input_data, enforce_sorted=True)
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

    assert batch_size == len(embeddings), "assert wrong size of tgt after concat"
    return embeddings
