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


def create_padded_tensor(
    tokens: list[list[int]],
    max_len: int,
    pad_idx: int,
    device: str | torch.device,
    pad_to_multiple_of: int = 0,
    padding_side: str = 'right',
):
    bs: int = len(tokens)

    if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    batch = torch.full([bs, max_len], pad_idx, dtype=torch.int32)
    if padding_side == 'right':
        for i in range(bs):
            batch[i, 0 : len(tokens[i])] = torch.as_tensor(tokens[i])
    elif padding_side == 'left':
        for i in range(bs):
            ll = len(tokens[i])
            batch[i, max_len - ll :] = torch.as_tensor(tokens[i])
    else:
        RuntimeError("Unknown padding_side value: " + padding_side)

    batch = batch.to(device=device)
    lengths = torch.as_tensor([len(t) for t in tokens], dtype=torch.int64, device=device)
    return batch, lengths
