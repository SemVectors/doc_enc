#!/usr/bin/env python3

import logging
import torch


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
    """Pad sequence of embs based on lengths tensor."""
    # embs: [N, dim]
    nembs, emb_dim = embs.shape
    bs = lengths.shape[0]
    max_len: int = int(lengths.max().item()) + int(prepend_with_zero)
    if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    padded_seq = torch.zeros(
        (bs * max_len, emb_dim),
        device=embs.device,
        dtype=embs.dtype,
    )
    idx = []
    offs = 0 + int(prepend_with_zero)
    lengths_as_list = lengths.tolist()
    for lt in lengths_as_list:
        idx.extend([i] for i in range(offs, offs + lt))
        offs += max_len
    # idx: [N, 1]
    idx = torch.tensor(idx, dtype=torch.int64, device=embs.device)
    padded_seq.scatter_(0, idx.expand(-1, emb_dim), embs)
    padded_seq = padded_seq.reshape(-1, max_len, emb_dim)

    # Create padding mask
    mask = torch.full((bs * max_len,), False, dtype=torch.bool, device=embs.device)
    mask.scatter_(0, idx.squeeze(1), torch.tensor(True, device=idx.device).expand(nembs))
    mask = mask.reshape(bs, max_len)

    if prepend_with_zero:
        mask[:, 0] = True

    return padded_seq, mask
