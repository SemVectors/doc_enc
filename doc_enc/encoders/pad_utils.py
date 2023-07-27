#!/usr/bin/env python3

import torch


def create_padded_tensor(
    tokens: list[list[int]],
    max_len: int,
    pad_idx: int,
    device: str | torch.device,
    pad_to_multiple_of: int = 0,
):
    bs: int = len(tokens)

    if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    batch = torch.full([bs, max_len], pad_idx, dtype=torch.int32)
    for i in range(bs):
        batch[i, 0 : len(tokens[i])] = torch.as_tensor(tokens[i])

    batch = batch.to(device=device)
    lengths = torch.as_tensor([len(t) for t in tokens], dtype=torch.int64, device=device)
    return batch, lengths
