#!/usr/bin/env python3

import torch


def create_key_padding_mask(max_len, src_lengths, device, padding_side: str = 'right'):
    bs = src_lengths.shape[0]
    mask = torch.full((bs, max_len), 0, dtype=torch.float, device=device)
    if padding_side == 'right':
        for i, length in enumerate(src_lengths):
            mask[i, 0:length] = 1
    elif padding_side == 'left':
        for i, length in enumerate(src_lengths):
            mask[i, max_len - length :] = 1
    else:
        RuntimeError("Unknown padding_side value: " + padding_side)

    return mask


class PadOpts:
    def __init__(self, pad_to_multiple_of: int = 0, padding_side: str = 'right'):
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding_side = padding_side


def create_padded_tensor(
    tokens: list[list[int]], max_len: int, pad_idx: int, device: str | torch.device, opts: PadOpts
):
    bs: int = len(tokens)

    if opts.pad_to_multiple_of and max_len % opts.pad_to_multiple_of != 0:
        max_len = ((max_len // opts.pad_to_multiple_of) + 1) * opts.pad_to_multiple_of

    batch = torch.full([bs, max_len], pad_idx, dtype=torch.int32)
    if opts.padding_side == 'right':
        for i in range(bs):
            batch[i, 0 : len(tokens[i])] = torch.as_tensor(tokens[i])
    elif opts.padding_side == 'left':
        for i in range(bs):
            ll = len(tokens[i])
            batch[i, max_len - ll :] = torch.as_tensor(tokens[i])
    else:
        RuntimeError("Unknown padding_side value: " + opts.padding_side)

    batch = batch.to(device=device)
    lengths = torch.as_tensor([len(t) for t in tokens], dtype=torch.int32, device=device)
    return batch, lengths


def pad_embs_seq(
    embs: torch.Tensor,
    lengths: torch.Tensor,
    prepend_with_zero: bool = False,
    pad_to_multiple_of: int = 0,
):
    # embs: N, dim
    emb_dim = embs.size(1)
    # pad sequence of embs
    max_len: int = int(lengths.max().item()) + int(prepend_with_zero)
    if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    padded_seq = torch.zeros(
        (len(lengths) * max_len, emb_dim),
        device=embs.device,
        dtype=embs.dtype,
    )
    idx = []
    offs = 0 + int(prepend_with_zero)
    for lt in lengths:
        idx.extend([i] for i in range(offs, offs + int(lt.item())))
        offs += max_len
    idx = torch.tensor(idx, dtype=torch.int64, device=embs.device).expand(-1, emb_dim)
    padded_seq.scatter_(0, idx, embs)
    return padded_seq, max_len
