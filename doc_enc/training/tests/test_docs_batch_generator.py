#!/usr/bin/env python3

import logging
from pathlib import Path
import tempfile
import random

import pytest

from doc_enc.encoders.enc_in import EncoderInputType
from doc_enc.training.types import DocsBatch

from doc_enc.training.docs_batch_generator import (
    DocsBatchGeneratorConf,
    DocsBatchGenerator,
    DocsBatchAsyncGenerator,
    DocsBatchAsyncGeneratorConf,
)

from doc_enc.tokenizer import TokenizerType, TokenizerConf
from doc_enc.text_processor import TextProcessorConf


@pytest.fixture
def FakeTrainingData():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for n in ("ds1", "ds2"):
            ds = tmpdirname / n
            ds_docs = ds / "texts"
            ds_docs.mkdir(parents=True)

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("1 2\n3 4\n5 6")
        with open(ds1_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("15 2 3 4 5\n")
            f.write("1 2 3 4 5\n" * 14)
        with open(ds1_docs_dir / '16.txt', 'w', encoding='utf8') as f:
            f.write("16 2 3 4 6 7\n")
            f.write("1 2 3 4 6\n" * 15)
        with open(ds1_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("30 2 3 4 7 8\n")
            f.write("1 2 3 4 7\n" * 29)
        with open(ds1_docs_dir / '40.txt', 'w', encoding='utf8') as f:
            f.write("40 2 3 4 8 9\n")
            f.write("1 2 3 4 8\n" * 39)
        with open(ds1_docs_dir / '120.txt', 'w', encoding='utf8') as f:
            f.write("120 2 3 4 8\n")
            f.write("1 2 3 4 8\n" * 119)

        ds2_docs_dir = tmpdirname / "ds2" / "texts"
        with open(ds2_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3\n" * 15)
        with open(ds2_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("30 8 7 9 10 11\n")
            f.write("9 8 7\n" * 29)
        with open(ds2_docs_dir / '50.txt', 'w', encoding='utf8') as f:
            f.write("50 11 12 13 14 15\n")
            f.write("10 11 12\n" * 49)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            # f.write("ds,src,tgt,label,slen,tlen,shash,thash\n")
            f.write("ds1,3,40,1,3,40,3hash,40hash\n")
            f.write("ds1,3,16,0,3,16,3hash,16hash\n")
            f.write("ds1,3,30,0,3,30,3hash,30hash\n")
            f.write("ds1,15,30,1,15,30,15hash,30hash\n")
            f.write("ds2,15,50,1,15,50,15hash,50hash\n")
            f.write("ds2,15,30,0,15,30,15hash,30хэш\n")
            f.write("ds1,16,40,0,16,40,16hash,40hash\n")
            f.write("ds1,120,40,0,120,40,120hash,40hash\n")

        yield tmpdirname


@pytest.fixture
def FakeTrainingDataWithDups():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for n in ("ds1",):
            ds = tmpdirname / n
            ds_docs = ds / "texts"
            ds_docs.mkdir(parents=True)

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("1 2\n3 4\n5 6")
        with open(ds1_docs_dir / '2.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 5\n" * 2)
        with open(ds1_docs_dir / '16.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6\n" * 16)
        with open(ds1_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 7\n" * 30)
        with open(ds1_docs_dir / '40.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 8\n" * 40)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            # f.write("ds,src,tgt,label,slen,tlen,shash,thash\n")
            f.write("ds1,3,40,1,3,40,3hash,40hash\n")
            f.write("ds1,3,16,1,3,16,3hash,16hash\n")
            f.write("ds1,3,30,1,3,30,3hash,30hash\n")
            f.write("ds1,2,30,1,2,30,2hash,30hash\n")
            f.write("ds1,2,16,0,2,16,2hash,16hash\n")
            f.write("ds1,2,40,0,2,40,2hash,40hash\n")

        yield tmpdirname


@pytest.fixture
def FakeTrainingFiltering():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for n in ("ds1",):
            ds = tmpdirname / n
            ds_docs = ds / "texts"
            ds_docs.mkdir(parents=True)

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4\n3 4 5 6\n5 6 7 8")
        with open(ds1_docs_dir / '4.txt', 'w', encoding='utf8') as f:
            f.write("1 2 \n3 4 5 6\n5 6 \n7 8 10")
        with open(ds1_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 5\n" * 15)
        with open(ds1_docs_dir / '16.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6\n" * 16)
        with open(ds1_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6 7\n" * 30)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            # f.write("ds,src,tgt,label,slen,tlen,shash,thash\n")
            f.write("ds1,15,3,1,15,3,15hash,3hash\n")
            f.write("ds1,15,16,0,15,16,15hash,16hash\n")
            f.write("ds1,15,30,0,15,30,15hash,30hash\n")
            f.write("ds1,4,15,1,4,15,4hash,15hash\n")
            f.write("ds1,4,30,1,4,30,4hash,30hash\n")
            f.write("ds1,4,16,0,4,16,4hash,16hash\n")
            f.write("ds1,3,15,1,3,15,3hash,15hash\n")
            f.write("ds1,3,30,1,3,30,3hash,30hash\n")
            f.write("ds1,3,16,0,3,16,3hash,16hash\n")
        yield tmpdirname


@pytest.fixture
def FakeTrainingPadding():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for n in ("ds1",):
            ds = tmpdirname / n
            ds_docs = ds / "texts"
            ds_docs.mkdir(parents=True)

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4\n3 4 5 6\n5 6 7 8")
        with open(ds1_docs_dir / '5.txt', 'w', encoding='utf8') as f:
            f.write("1 2 \n3 4 5 6\n5 6 \n7 8 10\n11 12 13 14")
        with open(ds1_docs_dir / '8.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 5\n" * 8)
        with open(ds1_docs_dir / '10.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6\n" * 10)
        with open(ds1_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6 7\n" * 30)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            f.write("ds1,3,10,1,3,10,3hash,10hash\n")
            f.write("ds1,3,8,0,3,8,3hash,8hash\n")
            f.write("ds1,5,10,1,5,10,5hash,10hash\n")
            f.write("ds1,8,30,1,8,30,8hash,30hash\n")
        yield tmpdirname


def _create_gen_opts(
    input_dir,
    positives_per_doc=[2, 2],
    min_sents_per_doc=1,
    min_sent_len=1,
    **kwargs,
):
    conf = DocsBatchGeneratorConf(
        input_dir=input_dir,
        positives_per_doc=positives_per_doc,
        negatives_per_doc=[2, 2],
        min_sents_per_doc=min_sents_per_doc,
        min_tgt_docs_per_src_doc=1,
        **kwargs,
    )
    tp_conf = TextProcessorConf(
        TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED),
        fragment_size=16,
        min_sent_len=min_sent_len,
        num_alpha_max_ratio=0.0,
    )
    return conf, tp_conf


def test_gen_basic(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(FakeTrainingData)
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data
    assert sd.texts_repr.nsents() == 18
    assert td.texts_repr.nsents() == 166

    assert sd.texts_repr.fragment_lengths_in_sents() == [15, 3]
    assert td.texts_repr.fragment_lengths_in_sents() == [
        # doc30
        16,
        14,
        # doc50
        16,
        16,
        16,
        2,
        # doc30
        16,
        14,
        # doc 40
        16,
        16,
        8,
        # doc16
        16,
    ]

    assert sd.texts_repr.text_lengths_in_fragments() == [1, 1]
    assert sd.texts_repr.text_lengths_in_sents() == [15, 3]

    assert td.texts_repr.text_lengths_in_fragments() == [2, 4, 2, 3, 1]
    assert td.texts_repr.text_lengths_in_sents() == [30, 50, 30, 40, 16]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids[0] == [0, 1]
    assert pos_ids[1] == [3]

    assert batch.get_src_docs_cnt() == 2
    assert batch.get_tgt_docs_cnt() == 5


def test_two_batches(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData,
        batch_total_sents_cnt=200,
        allow_docs_without_positives=True,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]
    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.texts_repr.nsents() == 136
    assert td.texts_repr.nsents() == 40

    assert sd.texts_repr.fragment_lengths_in_sents() == [16] * 7 + [8, 16]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8]

    assert batch1.get_src_docs_cnt() == 2
    assert batch1.get_tgt_docs_cnt() == 1

    batch2: DocsBatch = batches[1]

    sd = batch2.src_data
    td = batch2.tgt_data
    assert sd.texts_repr.nsents() == 18
    assert td.texts_repr.nsents() == 166

    assert sd.texts_repr.fragment_lengths_in_sents() == [15, 3]

    assert batch2.get_src_docs_cnt() == 2
    assert batch2.get_tgt_docs_cnt() == 5


def test_three_batches(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData,
        batch_total_sents_cnt=178,
        allow_docs_without_positives=True,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(batches)
    assert len(batches) == 3
    batch1: DocsBatch = batches[0]

    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.texts_repr.nsents() == 136
    assert td.texts_repr.nsents() == 40

    assert sd.texts_repr.fragment_lengths_in_sents() == [16] * 7 + [8, 16]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8]

    assert batch1.get_src_docs_cnt() == 2
    assert batch1.get_tgt_docs_cnt() == 1

    batch2: DocsBatch = batches[1]
    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.texts_repr.nsents() == 15
    assert td.texts_repr.nsents() == 110

    assert sd.texts_repr.fragment_lengths_in_sents() == [15]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 14, 16, 16, 16, 2, 16, 14]

    assert batch2.get_src_docs_cnt() == 1
    assert batch2.get_tgt_docs_cnt() == 3

    batch3: DocsBatch = batches[2]
    sd = batch3.src_data
    td = batch3.tgt_data

    assert sd.texts_repr.nsents() == 3
    assert td.texts_repr.nsents() == 86

    assert sd.texts_repr.fragment_lengths_in_sents() == [3]

    assert batch3.get_src_docs_cnt() == 1
    assert batch3.get_tgt_docs_cnt() == 3


def test_cant_fit_batch(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData,
        batch_total_sents_cnt=90,
        allow_docs_without_positives=True,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]
    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.texts_repr.nsents() == 16
    assert td.texts_repr.nsents() == 40

    assert batch1.get_src_docs_cnt() == 1
    assert batch1.get_tgt_docs_cnt() == 1

    batch2: DocsBatch = batches[1]
    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.texts_repr.nsents() == 3
    assert td.texts_repr.nsents() == 86

    assert sd.texts_repr.fragment_lengths_in_sents() == [3]

    assert batch2.get_src_docs_cnt() == 1
    assert batch2.get_tgt_docs_cnt() == 3


def test_gen_with_dups(FakeTrainingDataWithDups):
    random.seed(4)
    conf, tp_conf = _create_gen_opts(FakeTrainingDataWithDups, positives_per_doc=[1, 1])

    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [3, 2]
    assert td.text_ids == [30, 16, 40]
    assert sd.texts_repr.nsents() == 5
    assert td.texts_repr.nsents() == 86

    assert sd.texts_repr.fragment_lengths_in_sents() == [3, 2]

    assert sd.texts_repr.text_lengths_in_sents() == [3, 2]

    assert td.texts_repr.text_lengths_in_sents() == [30, 16, 40]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids == [[0, 1, 2], [0]]

    assert batch.get_src_docs_cnt() == 2
    assert batch.get_tgt_docs_cnt() == 3

    assert batch.max_positives_per_doc() == 3


def test_gen_with_dups2(FakeTrainingDataWithDups):
    # the same test with different seed
    random.seed(2)

    conf, tp_conf = _create_gen_opts(FakeTrainingDataWithDups, positives_per_doc=[1, 1])
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [3, 2]
    assert td.text_ids == [16, 30, 40]
    assert sd.texts_repr.nsents() == 5
    assert td.texts_repr.nsents() == 86

    assert sd.texts_repr.fragment_lengths_in_sents() == [3, 2]

    assert sd.texts_repr.text_lengths_in_sents() == [3, 2]

    assert td.texts_repr.text_lengths_in_sents() == [16, 30, 40]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids == [[0, 1, 2], [1]]

    assert batch.get_src_docs_cnt() == 2
    assert batch.get_tgt_docs_cnt() == 3

    assert batch.max_positives_per_doc() == 3


def test_gen_with_filters(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=4, max_sents_per_doc=20
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]

    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [4]
    assert td.text_ids == [15, 16]
    assert sd.texts_repr.nsents() == 4
    assert td.texts_repr.nsents() == 31

    assert sd.texts_repr.fragment_lengths_in_sents() == [4]

    assert sd.texts_repr.text_lengths_in_sents() == [4]
    assert td.texts_repr.text_lengths_in_sents() == [15, 16]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids == [[0]]

    assert batch.get_src_docs_cnt() == 1
    assert batch.get_tgt_docs_cnt() == 2

    assert batch.max_positives_per_doc() == 1


def test_gen_with_all_filtered(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=5, max_sents_per_doc=10
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 0


def test_gen_with_filtering_sents_by_len(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=3, max_sents_per_doc=16, min_sent_len=4
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [15, 3]
    assert td.text_ids == [3, 15]
    assert sd.texts_repr.nsents() == 18
    assert td.texts_repr.nsents() == 18

    assert sd.texts_repr.fragment_lengths_in_sents() == [15, 3]

    assert sd.texts_repr.text_lengths_in_sents() == [15, 3]
    assert td.texts_repr.text_lengths_in_sents() == [3, 15]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids == [[0], [1]]

    assert batch.get_src_docs_cnt() == 2
    assert batch.get_tgt_docs_cnt() == 2

    assert batch.max_positives_per_doc() == 1


# * Async generator


def test_async_gen_packed_1(FakeTrainingData):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData, allow_docs_without_positives=True)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.PACKED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [1, 0, 0]
            assert batch.labels[1].tolist() == [0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 40 + 16)
            assert tgt_enc_in.max_len == 6
            ps = tgt_enc_in.get_packed_seq()
            first3_tgt = sorted(ps.data[:3].tolist())
            assert first3_tgt == [16, 30, 40]
            assert int(ps.batch_sizes[-1]) == 3

        elif batch.src_data.text_ids == [120, 16, 15]:
            assert batch.tgt_data.text_ids == [40, 50, 30]
            assert batch.get_src_docs_cnt() == 3
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [0, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 0]
            assert batch.labels[2].tolist() == [0, 1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16 + 15)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 50 + 30)
            assert tgt_enc_in.max_len == 6
            ps = tgt_enc_in.get_packed_seq()
            first3_tgt = sorted(ps.data[:3].tolist())
            assert first3_tgt == [30, 40, 50]
            assert int(ps.batch_sizes[-1]) == 3

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2


def test_async_gen_padded_1(FakeTrainingData):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData, allow_docs_without_positives=True)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.PADDED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [1, 0, 0]
            assert batch.labels[1].tolist() == [0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 40 + 16)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_padded()
            assert pd.data.shape == (30 + 40 + 16, 6)
            assert pd.data[0, 0].item() == 30
            assert pd.data[30, 0].item() == 40

        elif batch.src_data.text_ids == [120, 16, 15]:
            assert batch.tgt_data.text_ids == [40, 50, 30]
            assert batch.get_src_docs_cnt() == 3
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [0, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 0]
            assert batch.labels[2].tolist() == [0, 1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16 + 15)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 50 + 30)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_padded()
            assert pd.data.shape == (40 + 50 + 30, 6)
            assert pd.data[0, 0].item() == 40
            assert pd.data[40, 0].item() == 50

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2


def test_async_gen_jagged_1(FakeTrainingData):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData, allow_docs_without_positives=True)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.JAGGED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [1, 0, 0]
            assert batch.labels[1].tolist() == [0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 40 + 16)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_jagged()
            assert pd.data[0].item() == 30
            assert pd.data[6 + 29 * 5].item() == 40

        elif batch.src_data.text_ids == [120, 16, 15]:
            assert batch.tgt_data.text_ids == [40, 50, 30]
            assert batch.get_src_docs_cnt() == 3
            assert batch.get_tgt_docs_cnt() == 3

            assert batch.labels[0].tolist() == [0, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 0]
            assert batch.labels[2].tolist() == [0, 1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16 + 15)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 50 + 30)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_jagged()
            assert pd.data[0].item() == 40
            assert pd.data[6 + 39 * 5].item() == 50

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2
