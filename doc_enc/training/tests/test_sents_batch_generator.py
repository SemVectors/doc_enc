#!/usr/bin/env python3

import logging
import tempfile

import pytest

from doc_enc.training.types import SentsBatch

from doc_enc.training.sents_batch_generator import (
    SentsBatchGeneratorConf,
    SentsBatchGenerator,
    SentsBatchIterator,
    SentsBatchIteratorConf,
)

from doc_enc.tokenizer import TokenizerType, TokenizerConf


@pytest.fixture
def FakeTrainingData():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f'{tmpdirname}/train.id.src', 'w', encoding='utf8') as f:
            f.write(
                """1\t1 2 3 4
2\t5 6 7
3\t2 6 8
4\t3 8 9
5\t1 5 10 11
6\t11 9
7\t12 13 14"""
            )

        with open(f'{tmpdirname}/train.id.tgt', 'w', encoding='utf8') as f:
            f.write(
                """1\t1 2 3 4
2\t5 6 7
3\t2 6 8
4\t3 8 9
5\t1 5 10 11
6\t8 7
7\t13 15 20"""
            )

        with open(f'{tmpdirname}/train.dups', 'w', encoding='utf8') as f:
            f.write(
                """1\t6 9 99
2\t
3\t5
4\t
5\t3 7
6\t1
7\t5"""
            )

        with open(f'{tmpdirname}/train.hn', 'w', encoding='utf8') as f:
            f.write(
                """1\t10\t1 2
1\t11\t2 3
2\t20\t1 7
2\t21\t9 8
2\t22\t2 7 8 20 21 22
3\t
4\t30
5\t40\t8 9 11
5\t41\t5 7 10
5\t42\t9 10
6\t50\t11 2
7\t"""
            )

        yield tmpdirname


def _create_gen_opts(input_dir):
    conf = SentsBatchGeneratorConf(
        input_dir=input_dir,
        adjust_batch_size=False,
    )
    tok_conf = TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED)
    return conf, tok_conf


def test_gen_basic(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.bs == 4
    assert batch.src_id == [5, 1, 2, 4]
    assert batch.tgt_id == [5, 1, 2, 4, 40, 41, 42, 10, 11, 20, 21, 22]
    assert len(batch.src) == 4
    assert batch.src[0] == [1, 5, 10, 11]
    assert batch.hn_idxs == [[4, 5, 6], [7, 8], [9, 10, 11], []]


def test_gen_basic_line_offset(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(conf, tok_conf, split='train', line_offset=3)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.bs == 3
    assert batch.src_id == [5, 4, 6]
    assert batch.tgt_id == [5, 4, 6, 40, 41, 42, 50]
    assert len(batch.src) == 3
    assert batch.src[0] == [1, 5, 10, 11]
    assert batch.hn_idxs == [[3, 4, 5], [], [6]]


def test_gen_basic_line_cnt(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_cnt=3)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.bs == 3
    assert batch.src_id == [1, 2, 3]
    assert batch.tgt_id == [1, 2, 3, 10, 11, 20, 21, 22]
    assert len(batch.src) == 3
    assert batch.src[0] == [1, 2, 3, 4]
    assert batch.hn_idxs == [[3, 4], [5, 6, 7], []]


def test_gen_basic_line_cnt_and_offset(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=2, line_cnt=3)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.bs == 2
    assert batch.src_id == [5, 4]
    assert batch.tgt_id == [5, 4, 40, 41, 42]
    assert len(batch.src) == 2
    assert batch.src[0] == [1, 5, 10, 11]
    assert batch.hn_idxs == [[2, 3, 4], []]


def test_iterator_single_generator(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchIteratorConf(batch_generator_conf=gen_conf, async_generators=1)
    biter = SentsBatchIterator(iter_conf, tok_conf, 'train')

    biter.init_epoch(1)
    res = list(biter.batches())
    assert len(res) == 1
    _, batch, labels = res[0]
    assert batch.bs == 4
    assert batch.src_id == [5, 1, 2, 4]
    assert batch.tgt_id == [5, 1, 2, 4, 40, 41, 42, 10, 11, 20, 21, 22]

    assert batch.src.shape == (4, 4)
    assert batch.tgt.shape == (12, 6)
    assert labels.shape == (4,)


def test_iterator_1st_rank(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchIteratorConf(batch_generator_conf=gen_conf, async_generators=1)
    biter = SentsBatchIterator(iter_conf, tok_conf, 'train', rank=1, world_size=2)

    biter.init_epoch(1)
    res = list(biter.batches())
    assert len(res) == 1
    _, batch, _ = res[0]
    assert batch.bs == 2
    assert batch.src_id == [5, 6]
    assert batch.tgt_id == [5, 6, 40, 41, 42, 50]

    assert batch.src.shape == (2, 4)
    assert batch.tgt.shape == (6, 4)


def test_iterator_two_generators(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchIteratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = SentsBatchIterator(iter_conf, tok_conf, 'train')

    biter.init_epoch(1)
    res = list(biter.batches())
    assert len(res) == 2
    _, batch1, _ = res[0]
    assert batch1.bs == 4
    assert batch1.src_id == [1, 2, 4, 3]
    assert batch1.tgt_id == [1, 2, 4, 3, 10, 11, 20, 21, 22]

    assert batch1.src.shape == (4, 4)
    assert batch1.tgt.shape == (9, 6)

    _, batch2, _ = res[1]
    assert batch2.bs == 2
    assert batch2.src_id == [5, 6]
    assert batch2.tgt_id == [5, 6, 40, 41, 42, 50]

    assert batch2.src.shape == (2, 4)
    assert batch2.tgt.shape == (6, 4)
