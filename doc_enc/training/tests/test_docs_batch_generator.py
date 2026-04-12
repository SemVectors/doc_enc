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
def FakeTrainingData2():
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
        with open(ds1_docs_dir / '10.txt', 'w', encoding='utf8') as f:
            f.write("10 2 3 4 8 9\n")
            f.write("1 2 3 4 8\n" * 9)

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
            f.write("ds1,16,40,1,16,40,16hash,40hash\n")
            f.write("ds1,120,40,1,120,40,120hash,40hash\n")
            f.write("ds1,120,10,0,120,10,120hash,10hash\n")

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
def FakeTrainingTextAsTokensSeq():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        for n in ("ds1",):
            ds = tmpdirname / n
            ds_docs = ds / "texts"
            ds_docs.mkdir(parents=True)

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '1.txt', 'w', encoding='utf8') as f:
            f.write("1 2 " * 100)
        with open(ds1_docs_dir / '10.txt', 'w', encoding='utf8') as f:
            f.write("4 5 " * 50)
        with open(ds1_docs_dir / '20.txt', 'w', encoding='utf8') as f:
            f.write("6 7 " * 200)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            f.write("ds1,1,10,1,3,10,1hash,10hash\n")
            f.write("ds1,1,20,0,3,20,1hash,20hash\n")
        yield tmpdirname


def _create_gen_opts(
    input_dir,
    positives_per_doc=[2, 2],
    min_sents_per_doc=1,
    min_sent_len=1,
    add_bos: bool = False,
    add_eos: bool = False,
    split_into_sents: bool = True,
    split_into_fragments: bool = True,
    max_seq_length: int | None = None,
    **kwargs,
):
    conf = DocsBatchGeneratorConf(
        input_dir=input_dir,
        positives_per_doc=positives_per_doc,
        negatives_per_doc=[2, 2],
        min_sents_per_doc=min_sents_per_doc,
        **kwargs,
    )
    tp_conf = TextProcessorConf(
        TokenizerConf(
            tokenizer_type=TokenizerType.PRETOKENIZED,
            add_bos=add_bos,
            add_eos=add_eos,
            max_seq_length=max_seq_length,
        ),
        fragment_size=16,
        min_sent_len=min_sent_len,
        num_alpha_max_ratio=0.0,
        split_into_sents=split_into_sents,
        split_into_fragments=split_into_fragments,
    )
    return conf, tp_conf


def test_gen_basic(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(FakeTrainingData)
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data
    assert sd.text_ids == [15, 3]

    assert td.text_ids == [
        # src15
        30,
        30,
        50,
        # src3
        40,
        16,
    ]
    assert sd.texts_repr.nsents() == 18
    assert td.texts_repr.nsents() == 166

    assert sd.texts_repr.fragment_lengths_in_sents() == [15, 3]
    assert td.texts_repr.fragment_lengths_in_sents() == [
        # doc30
        16,
        14,
        # doc30
        16,
        14,
        # doc50
        16,
        16,
        16,
        2,
        # doc 40
        16,
        16,
        8,
        # doc16
        16,
    ]

    assert sd.texts_repr.text_lengths_in_fragments() == [1, 1]
    assert sd.texts_repr.text_lengths_in_sents() == [15, 3]

    assert td.texts_repr.text_lengths_in_fragments() == [2, 2, 4, 3, 1]
    assert td.texts_repr.text_lengths_in_sents() == [30, 30, 50, 40, 16]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids[0] == [0, 2]
    assert pos_ids[1] == [3]

    assert batch.get_src_docs_cnt() == 2
    assert batch.get_tgt_docs_cnt() == 5


def test_two_batches(FakeTrainingData2):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData2,
        batch_total_sents_cnt=200,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert gen._stat.cant_fit == 1
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]
    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.text_ids == [120, 16]
    assert td.text_ids == [40, 10]
    assert sd.texts_repr.nsents() == 136
    assert td.texts_repr.nsents() == 50

    assert sd.texts_repr.fragment_lengths_in_sents() == [16] * 7 + [8, 16]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8, 10]

    assert batch1.get_src_docs_cnt() == 2
    assert batch1.get_tgt_docs_cnt() == 2
    assert batch1.get_positive_idxs() == [[0], [0]]

    batch2: DocsBatch = batches[1]

    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.text_ids == [15, 3]
    assert td.text_ids == [30, 30, 50, 40, 16]

    assert sd.texts_repr.nsents() == 18
    assert td.texts_repr.nsents() == 166

    assert sd.texts_repr.fragment_lengths_in_sents() == [15, 3]

    assert batch2.get_src_docs_cnt() == 2
    assert batch2.get_tgt_docs_cnt() == 5


def test_two_batches_2(FakeTrainingData2):
    # text with id (50, 50hash) is not included in the second batch since we
    # decreased the batch_total_sents_cnt. There is no room for it.

    conf, tp_conf = _create_gen_opts(
        FakeTrainingData2,
        batch_total_sents_cnt=178,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert gen._stat.cant_fit == 1
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]

    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.text_ids == [120]
    assert td.text_ids == [40, 10]

    assert sd.texts_repr.nsents() == 120
    assert td.texts_repr.nsents() == 50

    assert sd.texts_repr.fragment_lengths_in_sents() == [16] * 7 + [8]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8, 10]

    assert batch1.get_src_docs_cnt() == 1
    assert batch1.get_tgt_docs_cnt() == 2

    batch2: DocsBatch = batches[1]
    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.text_ids == [16, 15, 3]
    assert td.text_ids == [40, 30, 30, 16]

    assert sd.texts_repr.nsents() == 34
    assert td.texts_repr.nsents() == 116

    assert sd.texts_repr.fragment_lengths_in_sents() == [16, 15, 3]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8, 16, 14, 16, 14, 16]

    assert batch2.get_src_docs_cnt() == 3
    assert batch2.get_tgt_docs_cnt() == 4

    assert batch2.get_positive_idxs() == [[0], [1], [0]]


def test_truncation_of_batch(FakeTrainingData2):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData2,
        batch_total_sents_cnt=90,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())

    print(gen._stat)
    assert gen._stat.seqs_truncated == 1
    assert gen._stat.total_seqs_trunc_ratio > 0.48
    # src 16 has only positive example and no negatives.
    assert gen._stat.defective_batch == 1
    assert len(batches) == 3
    batch1: DocsBatch = batches[0]
    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.text_ids == [120]
    assert td.text_ids == [40, 10]

    # Truncation is done proportionally to their original lengths.
    # orig_total = 120 + 40 + 10
    # max = 90
    # trunc_src_len = 90 * (120/170)
    assert sd.texts_repr.nsents() == 63
    assert td.texts_repr.nsents() == 21 + 5

    assert sd.texts_repr.fragment_lengths_in_sents() == [16] * 3 + [15]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 5, 5]

    assert batch1.get_src_docs_cnt() == 1
    assert batch1.get_tgt_docs_cnt() == 2

    batch2: DocsBatch = batches[1]
    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.text_ids == [15]
    assert td.text_ids == [30, 30]

    assert sd.texts_repr.nsents() == 15
    assert td.texts_repr.nsents() == 60

    assert sd.texts_repr.fragment_lengths_in_sents() == [15]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 14, 16, 14]

    assert batch2.get_src_docs_cnt() == 1
    assert batch2.get_tgt_docs_cnt() == 2

    assert batch2.get_positive_idxs() == [[0]]

    batch3: DocsBatch = batches[2]
    sd = batch3.src_data
    td = batch3.tgt_data

    assert sd.text_ids == [3]
    assert td.text_ids == [40, 16, 30]

    assert sd.texts_repr.nsents() == 3
    assert td.texts_repr.nsents() == 86

    assert sd.texts_repr.fragment_lengths_in_sents() == [3]
    assert td.texts_repr.fragment_lengths_in_sents() == [16, 16, 8, 16, 16, 14]

    assert batch3.get_src_docs_cnt() == 1
    assert batch3.get_tgt_docs_cnt() == 3

    assert batch3.get_positive_idxs() == [[0]]


def test_truncation_of_batch_2(FakeTrainingData2):
    # Decrease batch_total_sents_cnt , truncate ratio drops below conf.acceptable_seqs_trunc_ratio
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData2,
        batch_total_sents_cnt=80,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())

    print(gen._stat)
    assert gen._stat.seqs_truncated == 0
    assert gen._stat.huge_src == 1
    assert len(batches) == 2

    batch1: DocsBatch = batches[0]
    sd = batch1.src_data
    td = batch1.tgt_data

    assert sd.text_ids == [15]
    assert td.text_ids == [30, 30]

    assert sd.texts_repr.nsents() == 15
    assert td.texts_repr.nsents() == 60

    batch2: DocsBatch = batches[1]
    sd = batch2.src_data
    td = batch2.tgt_data

    assert sd.text_ids == [3]
    assert td.text_ids == [40, 16]

    assert sd.texts_repr.nsents() == 3
    assert td.texts_repr.nsents() == 56


def test_add_special_tokens_1(FakeTrainingTextAsTokensSeq):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingTextAsTokensSeq,
        batch_total_tokens_cnt=1000,
        add_eos=True,
        add_bos=True,
        max_seq_length=1000,
        split_into_fragments=False,
        split_into_sents=False,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())

    print(gen._stat)
    assert len(batches) == 1

    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [1]
    assert td.text_ids == [10, 20]

    assert sd.seq_encoder_input.ntokens() == 202
    assert td.seq_encoder_input.ntokens() == 504

    sjgt = sd.seq_encoder_input.get_jagged()
    assert sjgt.data[0] == -1
    assert sjgt.data[1] == 1
    assert sjgt.data[-1] == -2

    tjgt = td.seq_encoder_input.get_jagged()
    assert tjgt.data[0] == -1
    assert tjgt.data[101] == -2
    assert tjgt.data[102] == -1
    assert tjgt.data[-1] == -2


def test_add_special_tokens_2(FakeTrainingTextAsTokensSeq):
    # Test truncation with special symbols.
    conf, tp_conf = _create_gen_opts(
        FakeTrainingTextAsTokensSeq,
        batch_total_tokens_cnt=500,
        add_eos=True,
        add_bos=True,
        max_seq_length=1000,
        split_into_fragments=False,
        split_into_sents=False,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert 0.28 < gen._stat.total_toks_trunc_ratio < 0.29
    assert gen._stat.toks_truncated == 1

    assert len(batches) == 1

    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [1]
    assert td.text_ids == [10, 20]

    assert sd.seq_encoder_input.ntokens() == 142
    assert td.seq_encoder_input.ntokens() == 71 + 285

    sjgt = sd.seq_encoder_input.get_jagged()
    assert sjgt.data[0] == -1
    assert sjgt.data[1] == 1
    assert sjgt.data[-1] == -2
    assert sjgt.data.shape[0] == 142

    tjgt = td.seq_encoder_input.get_jagged()
    assert tjgt.data[0] == -1
    assert tjgt.data[1] == 4
    assert tjgt.data[70] == -2
    assert tjgt.data[71] == -1
    assert tjgt.data[72] == 6
    assert tjgt.data[-1] == -2


def test_add_special_tokens_3(FakeTrainingTextAsTokensSeq):
    # Test truncation with special symbols on sent level.
    conf, tp_conf = _create_gen_opts(
        FakeTrainingTextAsTokensSeq,
        batch_total_tokens_cnt=500,
        add_eos=True,
        add_bos=True,
        max_seq_length=1000,
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert 0.28 < gen._stat.total_toks_trunc_ratio < 0.29
    assert gen._stat.toks_truncated == 1

    assert len(batches) == 1

    batch: DocsBatch = batches[0]
    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [1]
    assert td.text_ids == [10, 20]

    assert sd.seq_encoder_input.ntokens() == 142
    assert td.seq_encoder_input.ntokens() == 71 + 285

    sjgt = sd.seq_encoder_input.get_jagged()
    assert sjgt.data[0] == -1
    assert sjgt.data[1] == 1
    assert sjgt.data[-1] == -2
    assert sjgt.data.shape[0] == 142

    tjgt = td.seq_encoder_input.get_jagged()
    assert tjgt.data[0] == -1
    assert tjgt.data[1] == 4
    assert tjgt.data[70] == -2
    assert tjgt.data[71] == -1
    assert tjgt.data[72] == 6
    assert tjgt.data[-1] == -2


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
    conf, tp_conf = _create_gen_opts(FakeTrainingFiltering, min_sents_per_doc=4)
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    print(gen._stat)
    assert gen._stat.no_positives == 1
    assert gen._stat.bad_src == 1
    assert gen._stat.filtered_by_min_sents_per_doc == 2
    assert len(batches) == 1
    batch: DocsBatch = batches[0]

    sd = batch.src_data
    td = batch.tgt_data

    assert sd.text_ids == [4]
    assert td.text_ids == [15, 16, 30]
    assert sd.texts_repr.nsents() == 4
    assert td.texts_repr.nsents() == 61

    assert sd.texts_repr.fragment_lengths_in_sents() == [4]

    assert sd.texts_repr.text_lengths_in_sents() == [4]
    assert td.texts_repr.text_lengths_in_sents() == [15, 16, 30]

    pos_ids = batch.get_positive_idxs()
    assert pos_ids == [[0, 2]]

    assert batch.get_src_docs_cnt() == 1
    assert batch.get_tgt_docs_cnt() == 3

    assert batch.max_positives_per_doc() == 2


def test_gen_with_all_filtered(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(FakeTrainingFiltering, min_sents_per_doc=5)
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )
    batches = list(gen.batches())
    assert len(batches) == 0


def test_gen_with_filtering_sents_by_len(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=3, max_sents_per_doc=15, min_sent_len=4
    )
    gen = DocsBatchGenerator(
        EncoderInputType.JAGGED, conf, tp_conf=tp_conf, split='train', line_offset=0
    )

    batches = list(gen.batches())
    print(gen._stat)
    assert gen._stat.filtered_by_min_sents_per_doc == 1
    assert gen._stat.filtered_by_max_sents_per_doc == 4

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


def test_async_gen_packed_1(FakeTrainingData2):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData2)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.PACKED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        # total number of pairs = 9, will receive first 5 lines and the second
        # the rest. The last documen of src_id = 15 is in the first process so
        # it will not included in the first batch (hash = 30хэш).

        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 50, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 4

            assert batch.labels[0].tolist() == [1, 1, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 50 + 40 + 16)
            assert tgt_enc_in.max_len == 6
            ps = tgt_enc_in.get_packed_seq()
            first4_tgt = sorted(ps.data[:4].tolist())
            assert first4_tgt == [16, 30, 40, 50]
            assert int(ps.batch_sizes[-1]) == 4

        elif batch.src_data.text_ids == [120, 16]:
            assert batch.tgt_data.text_ids == [40, 10]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 2

            assert batch.labels[0].tolist() == [1, 0]
            assert batch.labels[1].tolist() == [1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 10)
            assert tgt_enc_in.max_len == 6
            ps = tgt_enc_in.get_packed_seq()
            first2_tgt = sorted(ps.data[:2].tolist())
            assert first2_tgt == [10, 40]
            assert int(ps.batch_sizes[-1]) == 2

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2


def test_async_gen_padded_1(FakeTrainingData2):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData2)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.PADDED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 50, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 4

            assert batch.labels[0].tolist() == [1, 1, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 50 + 40 + 16)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_padded()
            assert pd.data.shape == (30 + 50 + 40 + 16, 6)
            assert pd.data[0, 0].item() == 30
            assert pd.data[30, 0].item() == 50
            assert pd.data[80, 0].item() == 40

        elif batch.src_data.text_ids == [120, 16]:
            assert batch.tgt_data.text_ids == [40, 10]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 2

            assert batch.labels[0].tolist() == [1, 0]
            assert batch.labels[1].tolist() == [1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 10)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_padded()
            assert pd.data.shape == (40 + 10, 6)
            assert pd.data[0, 0].item() == 40
            assert pd.data[40, 0].item() == 10

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2


def test_async_gen_jagged_1(FakeTrainingData2):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData2)
    iter_conf = DocsBatchAsyncGeneratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchAsyncGenerator(
        EncoderInputType.JAGGED, iter_conf, tp_conf, logging_conf={}, split='train'
    )

    biter.init_epoch(1)
    nbatches = 0
    for batch in biter.batches():
        nbatches += 1
        if batch.src_data.text_ids == [15, 3]:
            assert batch.tgt_data.text_ids == [30, 50, 40, 16]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 4

            assert batch.labels[0].tolist() == [1, 1, 0, 0]
            assert batch.labels[1].tolist() == [0, 0, 1, 0]

            # check input for sent encoder
            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (15 + 3)
            assert src_enc_in.max_len == 5

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (30 + 50 + 40 + 16)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_jagged()
            assert pd.data[0].item() == 30
            assert pd.data[6 + 29 * 5].item() == 50

        elif batch.src_data.text_ids == [120, 16]:
            assert batch.tgt_data.text_ids == [40, 10]
            assert batch.get_src_docs_cnt() == 2
            assert batch.get_tgt_docs_cnt() == 2

            assert batch.labels[0].tolist() == [1, 0]
            assert batch.labels[1].tolist() == [1, 0]

            src_enc_in = batch.src_data.seq_encoder_input
            assert src_enc_in.batch_size == (120 + 16)
            assert src_enc_in.max_len == 6

            tgt_enc_in = batch.tgt_data.seq_encoder_input
            assert tgt_enc_in.batch_size == (40 + 10)
            assert tgt_enc_in.max_len == 6

            pd = tgt_enc_in.get_jagged()
            assert pd.data[0].item() == 40
            assert pd.data[6 + 39 * 5].item() == 10

        else:
            raise RuntimeError("Unknown batch!")

    assert nbatches == 2
