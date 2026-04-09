#!/usr/bin/env python3

import logging
import tempfile

import pytest

from doc_enc.encoders.enc_in import EncoderInputType
from doc_enc.training.types import SentsBatch

from doc_enc.training.sents_batch_generator import (
    SentsBatchGeneratorConf,
    SentsBatchGenerator,
    SentsBatchAsyncGenerator,
    SentsBatchAsyncGeneratorConf,
)

from doc_enc.tokenizer import TokenizerType, TokenizerConf


@pytest.fixture
def FakeTrainingData():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(f'{tmpdirname}/train.src', 'w', encoding='utf8') as f:
            f.write(
                """1\t1 2 3 4
2\t5 6 7
3\t2 6 8
4\t3 8 9
5\t1 5 10 11
6\t11 9
7\t12 13 14"""
            )

        with open(f'{tmpdirname}/train.tgt', 'w', encoding='utf8') as f:
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
        min_hn_cnt=10,
    )
    tok_conf = TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED)
    return conf, tok_conf


def test_gen_basic(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(
        EncoderInputType.JAGGED, conf, tok_conf=tok_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.batch_size() == 4
    assert batch.src_data.text_ids == [5, 1, 2, 4]
    assert batch.tgt_data.text_ids == [5, 1, 2, 4, 40, 41, 42, 10, 11, 20, 21, 22]

    src_inp = batch.src_data.seq_encoder_input
    assert src_inp.batch_size == 4
    assert src_inp.get_jagged().data[:4].tolist() == [1, 5, 10, 11]

    assert batch.hn_idxs == [[4, 5, 6], [7, 8], [9, 10, 11], []]


def test_gen_basic_line_offset(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(EncoderInputType.JAGGED, conf, tok_conf, split='train', line_offset=3)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.batch_size() == 3
    assert batch.src_data.text_ids == [5, 4, 6]
    assert batch.tgt_data.text_ids == [5, 4, 6, 40, 41, 42, 50]

    src_inp = batch.src_data.seq_encoder_input
    assert src_inp.batch_size == 3
    assert src_inp.get_jagged().data[:4].tolist() == [1, 5, 10, 11]

    assert batch.hn_idxs == [[3, 4, 5], [], [6]]


def test_gen_basic_line_cnt(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(
        EncoderInputType.JAGGED, conf, tok_conf=tok_conf, split='train', line_cnt=3
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.batch_size() == 3
    assert batch.src_data.text_ids == [1, 2, 3]
    assert batch.tgt_data.text_ids == [1, 2, 3, 10, 11, 20, 21, 22]

    src_inp = batch.src_data.seq_encoder_input
    assert src_inp.batch_size == 3
    assert src_inp.get_jagged().data[:4].tolist() == [1, 2, 3, 4]

    assert batch.hn_idxs == [[3, 4], [5, 6, 7], []]


def test_gen_basic_line_cnt_and_offset(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = SentsBatchGenerator(
        EncoderInputType.JAGGED, conf, tok_conf=tok_conf, split='train', line_offset=2, line_cnt=3
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: SentsBatch = batches[0]
    assert batch.batch_size() == 2

    assert batch.src_data.text_ids == [5, 4]
    assert batch.tgt_data.text_ids == [5, 4, 40, 41, 42]

    src_inp = batch.src_data.seq_encoder_input
    assert src_inp.batch_size == 2
    assert src_inp.get_jagged().data[:4].tolist() == [1, 5, 10, 11]

    assert batch.hn_idxs == [[2, 3, 4], []]


# * Async generator


def test_asyng_single_generator(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=1)
    biter = SentsBatchAsyncGenerator(EncoderInputType.JAGGED, iter_conf, tok_conf, {}, 'train')

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        assert batch.batch_size() == 4
        assert batch.src_data.text_ids == [5, 1, 2, 4]
        assert batch.tgt_data.text_ids == [5, 1, 2, 4, 40, 41, 42, 10, 11, 20, 21, 22]

        src_inp = batch.src_data.seq_encoder_input
        assert src_inp.batch_size == 4
        assert src_inp.max_len == 4

        tgt_inp = batch.tgt_data.seq_encoder_input

        assert tgt_inp.batch_size == 12
        assert tgt_inp.max_len == 6
        assert batch.labels.shape == (4,)
    assert nbatches == 1


def test_iterator_1st_rank(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=1)
    biter = SentsBatchAsyncGenerator(
        EncoderInputType.JAGGED, iter_conf, tok_conf, {}, 'train', rank=1, world_size=2
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        assert batch.batch_size() == 2
        assert batch.src_data.text_ids == [5, 6]
        assert batch.tgt_data.text_ids == [5, 6, 40, 41, 42, 50]

        src_inp = batch.src_data.seq_encoder_input
        assert src_inp.batch_size == 2
        assert src_inp.max_len == 4

        tgt_inp = batch.tgt_data.seq_encoder_input

        assert tgt_inp.batch_size == 6
        assert tgt_inp.max_len == 4

    assert nbatches == 1


def test_iterator_two_generators(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData)
    iter_conf = SentsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = SentsBatchAsyncGenerator(EncoderInputType.PACKED, iter_conf, tok_conf, {}, 'train')

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1

        if batch.src_data.text_ids == [1, 2, 4, 3]:
            assert batch.batch_size() == 4
            assert batch.tgt_data.text_ids == [1, 2, 4, 3, 10, 11, 20, 21, 22]

            src_inp = batch.src_data.seq_encoder_input
            assert src_inp.batch_size == 4
            assert src_inp.max_len == 4
            src_ps = src_inp.get_packed_seq()
            assert src_ps.sorted_indices is None

            tgt_inp = batch.tgt_data.seq_encoder_input
            assert tgt_inp.batch_size == 9
            assert tgt_inp.max_len == 6

            tgt_ps = tgt_inp.get_packed_seq()
            assert tgt_ps.sorted_indices is not None

        elif batch.src_data.text_ids == [5, 6]:
            assert batch.batch_size() == 2
            assert batch.tgt_data.text_ids == [5, 6, 40, 41, 42, 50]

            src_inp = batch.src_data.seq_encoder_input
            assert src_inp.batch_size == 2
            assert src_inp.max_len == 4

            tgt_inp = batch.tgt_data.seq_encoder_input
            assert tgt_inp.batch_size == 6
            assert tgt_inp.max_len == 4
        else:
            raise RuntimeError("Unexpected batch!")

    assert nbatches == 2
