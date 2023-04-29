#!/usr/bin/env python3

import torch
import torch.distributed as dist


def dist_gather_tensor(vecs, world_size=0):
    if not world_size:
        world_size = dist.get_world_size()
    all_tensors = [torch.empty_like(vecs) for _ in range(world_size)]
    dist.all_gather(all_tensors, vecs)
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


# https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/3
def dist_gather_varsize_tensor(tensor: torch.Tensor, world_size=0):
    if not world_size:
        world_size = dist.get_world_size()

    cur_size = torch.tensor([tensor.size(0)], dtype=torch.int64, device=tensor.device)
    size_tens = [torch.empty_like(cur_size) for _ in range(world_size)]
    dist.all_gather(size_tens, cur_size)

    max_size = max(int(s.item()) for s in size_tens)

    if max_size == tensor.shape[0]:
        padded = tensor
    else:
        padded = torch.empty(max_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        padded[: tensor.shape[0]] = tensor

    all_tensors = [torch.empty_like(padded) for _ in size_tens]

    dist.all_gather(all_tensors, padded)

    for i, (padded_tensor, sz) in enumerate(zip(all_tensors, size_tens)):
        all_tensors[i] = padded_tensor[:sz]

    return all_tensors


def dist_gather_target_embs(tensor: torch.Tensor):
    all_targets = dist_gather_varsize_tensor(tensor)
    i = dist.get_rank()
    if i != 0:
        all_targets[0], all_targets[i] = tensor, all_targets[0]
    else:
        all_targets[0] = tensor
    return torch.cat(all_targets)
