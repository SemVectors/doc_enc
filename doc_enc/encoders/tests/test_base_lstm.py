#!/usr/bin/env python3

import torch

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_lstm import RNNPooler
from doc_enc.encoders.base_pooler import BasePoolerConf


def test_max_pooling_1():
    tens = [
        torch.tensor([[1, 2], [3, 1], [5, 0]], dtype=torch.float),
        torch.tensor([[0, -1], [3, 8]], dtype=torch.float),
        torch.tensor([[9, 10], [-1, 8], [7, 7], [0, 12]], dtype=torch.float),
    ]

    ps = torch.nn.utils.rnn.pack_sequence(tens, enforce_sorted=False)
    print('batch size', ps.batch_sizes)

    pooler = RNNPooler(2, BasePoolerConf(PoolingStrategy.MAX))
    pooled_out, lengths = pooler(ps)
    assert pooled_out.shape == (3, 2)
    assert lengths.tolist() == [3, 2, 4]
    assert pooled_out[0].tolist() == [5, 2]
    assert pooled_out[1].tolist() == [3, 8]
    assert pooled_out[2].tolist() == [9, 12]


def test_max_pooling_2():
    tens = [
        torch.tensor([[-4, 2], [3, 1], [5, 0]], dtype=torch.float),
        torch.tensor([[-6, -1]], dtype=torch.float),
        torch.tensor([[-1, 10], [-1, 8], [7, 7], [0, 12]], dtype=torch.float),
        torch.tensor([[-2, 1], [1, 2], [4, 1], [0, 2]], dtype=torch.float),
        torch.tensor([[-3, 1], [4, 10], [3, 2]], dtype=torch.float),
        torch.tensor([[-5, 10], [4, 1]], dtype=torch.float),
    ]

    ps = torch.nn.utils.rnn.pack_sequence(tens, enforce_sorted=False)
    print('batch sizes', ps.batch_sizes)

    pooler = RNNPooler(2, BasePoolerConf(PoolingStrategy.MAX))
    pooler.pooler.cap_m = 2
    pooled_out, lengths = pooler(ps)
    assert pooled_out.shape == (6, 2)
    assert lengths.tolist() == [3, 1, 4, 4, 3, 2]
    assert pooled_out[0].tolist() == [5, 2]
    assert pooled_out[1].tolist() == [-6, -1]
    assert pooled_out[2].tolist() == [7, 12]
    assert pooled_out[3].tolist() == [4, 2]
    assert pooled_out[4].tolist() == [4, 10]
    assert pooled_out[5].tolist() == [4, 10]


def test_mean_pooling_1():
    tens = [
        torch.tensor([[1, 2], [3, 1], [5, 0]], dtype=torch.float),
        torch.tensor([[0, -1]], dtype=torch.float),
        torch.tensor([[9, 10], [-1, 8], [7, 7], [0, 12]], dtype=torch.float),
    ]

    ps = torch.nn.utils.rnn.pack_sequence(tens, enforce_sorted=False)
    print('batch size', ps.batch_sizes)

    pooler = RNNPooler(2, BasePoolerConf(PoolingStrategy.MEAN))
    pooled_out, lengths = pooler(ps)
    assert pooled_out.shape == (3, 2)
    assert lengths.tolist() == [3, 1, 4]
    assert pooled_out[0].tolist() == [3, 1]
    assert pooled_out[1].tolist() == [0, -1]
    assert pooled_out[2].tolist() == [3.75, 9.25]


def test_mean_pooling_2():
    tens = [
        torch.tensor([[1, 2], [5, 1], [3, 0]], dtype=torch.float),
        torch.tensor([[9, 10], [-1, 8], [7, 7], [0, 12]], dtype=torch.float),
        torch.tensor([[1, -1], [2, -2], [3, -3], [6, -6]], dtype=torch.float),
        torch.tensor([[1, -1], [2, -2], [0, 0]], dtype=torch.float),
    ]

    ps = torch.nn.utils.rnn.pack_sequence(tens, enforce_sorted=False)
    print('batch sizes', ps.batch_sizes)

    pooler = RNNPooler(2, BasePoolerConf(PoolingStrategy.MEAN))
    pooler.pooler.cap_m = 2

    pooled_out, lengths = pooler(ps)
    assert pooled_out.shape == (4, 2)
    assert lengths.tolist() == [3, 4, 4, 3]
    assert pooled_out[0].tolist() == [3, 1]
    assert pooled_out[1].tolist() == [3.75, 9.25]
    assert pooled_out[2].tolist() == [3, -3]
    assert pooled_out[3].tolist() == [1, -1]
