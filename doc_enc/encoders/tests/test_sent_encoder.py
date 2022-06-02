#!/usr/bin/env python3

from typing import Any

import torch

from doc_enc.encoders.sent_encoder import split_sents_and_embed, SentEncoder


class DummyResponse:
    def __init__(self, t) -> None:
        self.pooled_out = t


class DummyEncoder:
    def __init__(self) -> None:
        self.enforce_sorted = None
        self.pad_to_multiple_of = 0

    def __call__(self, sent_tensor, *args: Any, enforce_sorted=None, **kwds: Any) -> Any:
        self.enforce_sorted = enforce_sorted
        cnt, max_len = sent_tensor.size()
        l = [[t[0], cnt, max_len] for t in sent_tensor]
        embs = torch.tensor(l)
        return DummyResponse(embs)


def test_split_sents():
    data = [
        [1, 2, 3, 0, 0, 0],
        [4, 5, 1, 6, 2, 0],
        [7, 8, 9, 1, 2, 3],
        [10, 11, 12, 5, 0, 0],
        [13, 15, 0, 0, 0, 0],
    ]
    test_tensor = torch.tensor(data)
    length_tensor = torch.tensor([sum(1 for t in s if t) for s in data])
    encoder = DummyEncoder()
    r = split_sents_and_embed(
        encoder, test_tensor, length_tensor, max_chunk_size=5, max_tokens_in_chunk=15
    )

    assert r[0].tolist() == [1, 3, 4]
    assert r[1].tolist() == [4, 2, 6]
    assert r[2].tolist() == [7, 2, 6]
    assert r[3].tolist() == [10, 3, 4]
    assert r[4].tolist() == [13, 3, 4]
    assert encoder.enforce_sorted is True

    r = split_sents_and_embed(
        encoder, test_tensor, length_tensor, max_chunk_size=2, max_tokens_in_chunk=85
    )

    assert r[0].tolist() == [1, 2, 4]
    assert r[1].tolist() == [4, 2, 6]
    assert r[2].tolist() == [7, 2, 6]
    assert r[3].tolist() == [10, 2, 4]
    assert r[4].tolist() == [13, 1, 2]

    assert encoder.enforce_sorted is True


def test_empty_tensor():
    data = []
    test_tensor = torch.tensor(data)
    length_tensor = torch.tensor([])
    encoder = DummyEncoder()
    r = split_sents_and_embed(
        encoder, test_tensor, length_tensor, max_chunk_size=5, max_tokens_in_chunk=15
    )
    assert r.numel() == 0


def test_fast_path():
    data = [
        [1, 2, 3, 0, 0],
        [4, 5, 1, 6, 2],
    ]
    test_tensor = torch.tensor(data)
    length_tensor = torch.tensor([sum(1 for t in s if t) for s in data])
    encoder = DummyEncoder()
    r = split_sents_and_embed(
        encoder, test_tensor, length_tensor, max_chunk_size=5, max_tokens_in_chunk=15
    )
    assert r[0].tolist() == [1, 2, 5]
    assert r[1].tolist() == [4, 2, 5]

    assert encoder.enforce_sorted is False


def test_already_sorted():
    data = [
        [4, 5, 1, 6, 2],
        [7, 8, 1, 6, 2],
        [1, 2, 3, 0, 0],
    ]
    test_tensor = torch.tensor(data)
    length_tensor = torch.tensor([sum(1 for t in s if t) for s in data])
    encoder = DummyEncoder()
    r = split_sents_and_embed(
        encoder,
        test_tensor,
        length_tensor,
        max_chunk_size=2,
        max_tokens_in_chunk=15,
        already_sorted=True,
    )
    assert r[0].tolist() == [4, 2, 5]
    assert r[1].tolist() == [7, 2, 5]
    assert r[2].tolist() == [1, 1, 3]

    assert encoder.enforce_sorted is True


def test_with_padding():
    data = [
        [4, 5, 1, 6, 2, 3, 4, 0],
        [7, 8, 1, 6, 2, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0, 0],
    ]
    test_tensor = torch.tensor(data)
    length_tensor = torch.tensor([sum(1 for t in s if t) for s in data])
    encoder = DummyEncoder()
    encoder.pad_to_multiple_of = 4
    r = split_sents_and_embed(
        encoder,
        test_tensor,
        length_tensor,
        max_chunk_size=1,
        max_tokens_in_chunk=15,
    )
    assert r[0].tolist() == [4, 1, 8]
    assert r[1].tolist() == [7, 1, 8]
    assert r[2].tolist() == [1, 1, 4]

    assert encoder.enforce_sorted is True
