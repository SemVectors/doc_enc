#!/usr/bin/env python3

from pathlib import Path
import tempfile
import random

import pytest

from doc_enc.training.types import DocsBatch

from doc_enc.training.docs_batch_generator import (
    DocsBatchGeneratorConf,
    DocsBatchGenerator,
    DocsBatchIterator,
    DocsBatchIteratorConf,
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
            f.write("1 2 3 4 5\n" * 15)
        with open(ds1_docs_dir / '16.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 6\n" * 16)
        with open(ds1_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 7\n" * 30)
        with open(ds1_docs_dir / '40.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 8\n" * 40)

        ds2_docs_dir = tmpdirname / "ds2" / "texts"
        with open(ds2_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3\n" * 15)
        with open(ds2_docs_dir / '30.txt', 'w', encoding='utf8') as f:
            f.write("9 8 7\n" * 30)
        with open(ds2_docs_dir / '50.txt', 'w', encoding='utf8') as f:
            f.write("10 11 12\n" * 50)

        with open(tmpdirname / 'combined_train.csv', 'w', encoding='utf8') as f:
            # f.write("ds,src,tgt,label,slen,tlen,shash,thash\n")
            f.write("ds1,3,40,1,3,40,3hash,40hash\n")
            f.write("ds1,3,16,0,3,16,3hash,16hash\n")
            f.write("ds1,3,30,0,3,30,3hash,30hash\n")
            f.write("ds1,15,30,1,15,30,15hash,30hash\n")
            f.write("ds2,15,50,1,15,50,15hash,50hash\n")
            f.write("ds2,15,30,0,15,30,15hash,30хэш\n")
            f.write("ds1,16,40,0,16,40,16hash,40hash\n")

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
    pad_sentences=False,
    min_sent_len=1,
    **kwargs,
):

    conf = DocsBatchGeneratorConf(
        input_dir=input_dir,
        positives_per_doc=positives_per_doc,
        negatives_per_doc=[2, 2],
        min_sents_per_doc=min_sents_per_doc,
        pad_src_sentences=pad_sentences,
        pad_tgt_sentences=pad_sentences,
        **kwargs,
    )
    tp_conf = TextProcessorConf(
        TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED),
        fragment_size=16,
        min_sent_len=min_sent_len,
    )
    return conf, tp_conf


def test_gen_basic(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(FakeTrainingData)
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert len(batch.src_sents) == 18
    assert len(batch.tgt_sents) == 166

    assert batch.src_fragment_len == [15, 3]
    assert batch.tgt_fragment_len == [
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

    assert batch.src_doc_len_in_frags == [1, 1]
    assert batch.src_doc_len_in_sents == [15, 3]

    assert batch.tgt_doc_len_in_frags == [2, 4, 2, 3, 1]
    assert batch.tgt_doc_len_in_sents == [30, 50, 30, 40, 16]

    assert batch.positive_idxs[0] == [0, 1]
    assert batch.positive_idxs[1] == [3]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 5


def test_two_batches(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData, batch_src_sents_cnt=15, allow_docs_without_positives=True
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]
    assert len(batch1.src_sents) == 16
    assert len(batch1.tgt_sents) == 40

    assert batch1.src_fragment_len == [16]
    assert batch1.tgt_fragment_len == [16, 16, 8]

    assert batch1.info['src_docs_cnt'] == 1
    assert batch1.info['tgt_docs_cnt'] == 1

    batch2: DocsBatch = batches[1]

    assert len(batch2.src_sents) == 18
    assert len(batch2.tgt_sents) == 166

    assert batch2.src_fragment_len == [15, 3]

    assert batch2.info['src_docs_cnt'] == 2
    assert batch2.info['tgt_docs_cnt'] == 5


def test_three_batches(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData,
        batch_src_sents_cnt=15,
        max_sents_cnt_delta=2,
        allow_docs_without_positives=True,
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    print(batches)
    assert len(batches) == 3
    batch1: DocsBatch = batches[0]

    assert len(batch1.src_sents) == 16
    assert len(batch1.tgt_sents) == 40

    assert batch1.src_fragment_len == [16]
    assert batch1.tgt_fragment_len == [16, 16, 8]

    assert batch1.info['src_docs_cnt'] == 1
    assert batch1.info['tgt_docs_cnt'] == 1

    batch2: DocsBatch = batches[1]

    assert len(batch2.src_sents) == 15
    assert len(batch2.tgt_sents) == 110

    assert batch2.src_fragment_len == [15]
    assert batch2.tgt_fragment_len == [16, 14, 16, 16, 16, 2, 16, 14]

    assert batch2.info['src_docs_cnt'] == 1
    assert batch2.info['tgt_docs_cnt'] == 3

    batch3: DocsBatch = batches[2]

    assert len(batch3.src_sents) == 3
    assert len(batch3.tgt_sents) == 86

    assert batch3.src_fragment_len == [3]

    assert batch3.info['src_docs_cnt'] == 1
    assert batch3.info['tgt_docs_cnt'] == 3


def test_cant_fit_batch(FakeTrainingData):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingData,
        batch_src_sents_cnt=10,
        max_sents_cnt_delta=2,
        allow_docs_without_positives=True,
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch1: DocsBatch = batches[0]

    assert len(batch1.src_sents) == 3
    assert len(batch1.tgt_sents) == 86

    assert batch1.src_fragment_len == [3]

    assert batch1.info['src_docs_cnt'] == 1
    assert batch1.info['tgt_docs_cnt'] == 3


def test_iterator_two_generators(FakeTrainingData):
    gen_conf, tp_conf = _create_gen_opts(FakeTrainingData, allow_docs_without_positives=True)
    iter_conf = DocsBatchIteratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchIterator(iter_conf, tp_conf, logging_conf={}, split='train')

    biter.init_epoch(1)
    res = list(biter.batches())
    print(res[0])
    assert len(res) == 2
    batch1, labels1 = res[0]
    assert batch1.src_ids == [15, 3]
    assert batch1.info['src_docs_cnt'] == 2
    assert batch1.info['tgt_docs_cnt'] == 3
    assert labels1[0].tolist() == [1, 0, 0]
    assert labels1[1].tolist() == [0, 1, 0]

    batch2, labels2 = res[1]
    assert batch2.src_ids == [16, 15]
    assert batch2.info['src_docs_cnt'] == 2
    assert batch2.info['tgt_docs_cnt'] == 3
    assert labels2[0].tolist() == [0, 0, 0]
    assert labels2[1].tolist() == [0, 1, 0]


def test_gen_with_dups(FakeTrainingDataWithDups):
    random.seed(4)
    conf, tp_conf = _create_gen_opts(FakeTrainingDataWithDups, positives_per_doc=[1, 1])

    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [3, 2]
    assert batch.tgt_ids == [30, 16, 40]
    assert len(batch.src_sents) == 5
    assert len(batch.tgt_sents) == 86

    assert batch.src_fragment_len == [3, 2]

    assert batch.src_doc_len_in_sents == [3, 2]

    assert batch.tgt_doc_len_in_sents == [30, 16, 40]

    assert batch.positive_idxs[0] == [0, 1, 2]
    assert batch.positive_idxs[1] == [0]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 3
    assert batch.info['max_positives_per_doc'] == 3


def test_gen_with_dups2(FakeTrainingDataWithDups):
    # the same test with different seed
    random.seed(2)

    conf, tp_conf = _create_gen_opts(FakeTrainingDataWithDups, positives_per_doc=[1, 1])
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [3, 2]
    assert batch.tgt_ids == [16, 30, 40]
    assert len(batch.src_sents) == 5
    assert len(batch.tgt_sents) == 86

    assert batch.src_fragment_len == [3, 2]

    assert batch.src_doc_len_in_sents == [3, 2]

    assert batch.tgt_doc_len_in_sents == [16, 30, 40]

    assert batch.positive_idxs[0] == [0, 1, 2]
    assert batch.positive_idxs[1] == [1]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 3
    assert batch.info['max_positives_per_doc'] == 3


def test_gen_with_filters(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=4, max_sents_per_doc=20
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.positive_idxs)
    assert batch.src_ids == [4]
    assert batch.tgt_ids == [15, 16]
    assert len(batch.src_sents) == 4
    assert len(batch.tgt_sents) == 31

    assert batch.src_fragment_len == [4]

    assert batch.src_doc_len_in_sents == [4]
    assert batch.tgt_doc_len_in_sents == [15, 16]

    assert batch.positive_idxs[0] == [0]

    assert batch.info['src_docs_cnt'] == 1
    assert batch.info['tgt_docs_cnt'] == 2
    assert batch.info['max_positives_per_doc'] == 1


def test_gen_with_all_filtered(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=5, max_sents_per_doc=10
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 0


def test_gen_with_filtering_sents_by_len(FakeTrainingFiltering):
    conf, tp_conf = _create_gen_opts(
        FakeTrainingFiltering, min_sents_per_doc=3, max_sents_per_doc=16, min_sent_len=4
    )
    gen = DocsBatchGenerator(conf, tp_conf=tp_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.positive_idxs)
    assert batch.src_ids == [15, 3]
    assert batch.tgt_ids == [3, 15]
    assert len(batch.src_sents) == 18
    assert len(batch.tgt_sents) == 18

    assert batch.src_fragment_len == [15, 3]

    assert batch.src_doc_len_in_sents == [15, 3]
    assert batch.tgt_doc_len_in_sents == [3, 15]

    assert batch.positive_idxs == [[0], [1]]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 2
    assert batch.info['max_positives_per_doc'] == 1


def test_gen_with_padding_wo_fragments(FakeTrainingPadding):
    conf, tp_conf = _create_gen_opts(FakeTrainingPadding, pad_sentences=True)
    gen = DocsBatchGenerator(
        conf, tp_conf=tp_conf, split='train', line_offset=0, include_fragments_level=False
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [8, 5, 3]
    assert batch.tgt_ids == [30, 10, 8]
    assert len(batch.src_sents) == 24
    assert len(batch.tgt_sents) == 90

    assert batch.src_sents[4] == [1, 2, 3, 4, 5]
    assert batch.src_sents[8] == [1, 2]
    assert batch.src_sents[12] == [11, 12, 13, 14]
    assert batch.src_sents[13] == [0]
    assert batch.src_sents[15] == [0]
    assert batch.src_sents[16] == [1, 2, 3, 4]
    assert batch.src_sents[18] == [5, 6, 7, 8]
    assert batch.src_sents[19] == [0]
    assert batch.src_sents[23] == [0]

    assert batch.src_fragment_len is None
    assert batch.tgt_fragment_len is None

    assert batch.src_doc_len_in_sents == [8, 5, 3]
    assert batch.tgt_doc_len_in_sents == [30, 10, 8]

    assert batch.info['src_doc_len_in_sents'] == 8
    assert batch.info['tgt_doc_len_in_sents'] == 30

    assert batch.positive_idxs == [[0], [1], [1]]


def test_gen_with_padding_w_fragments(FakeTrainingPadding):
    conf, tp_conf = _create_gen_opts(FakeTrainingPadding, pad_sentences=True)
    gen = DocsBatchGenerator(
        conf, tp_conf=tp_conf, split='train', line_offset=0, include_fragments_level=True
    )
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [8, 5, 3]
    assert batch.tgt_ids == [30, 10, 8]
    assert len(batch.src_sents) == 24
    assert len(batch.tgt_sents) == 32 + 32 + 32
    #                            1st d  2nd and 3rd with extra fragment

    assert batch.src_sents[4] == [1, 2, 3, 4, 5]
    assert batch.src_sents[12] == [11, 12, 13, 14]
    assert batch.src_sents[15] == [0]
    assert batch.src_sents[16] == [1, 2, 3, 4]
    assert batch.src_sents[23] == [0]

    assert batch.tgt_sents[18] == [1, 2, 3, 4, 6, 7]
    assert batch.tgt_sents[29] == [1, 2, 3, 4, 6, 7]
    assert batch.tgt_sents[30] == [0]
    assert batch.tgt_sents[32] == [1, 2, 3, 4, 6]
    assert batch.tgt_sents[41] == [1, 2, 3, 4, 6]
    assert batch.tgt_sents[47] == [0]
    assert batch.tgt_sents[63] == [0]
    assert batch.tgt_sents[64] == [1, 2, 3, 4, 5]
    assert batch.tgt_sents[71] == [1, 2, 3, 4, 5]
    assert batch.tgt_sents[72] == [0]
    assert batch.tgt_sents[79] == [0]
    assert batch.tgt_sents[80] == [0]
    assert batch.tgt_sents[95] == [0]

    # fragment_size == 16
    assert batch.info['src_fragment_len'] == 8
    assert batch.info['tgt_fragment_len'] == 16

    assert batch.info['src_frags_cnt'] == 3
    assert batch.info['tgt_frags_cnt'] == 6
    assert batch.src_fragment_len == [8, 5, 3]
    assert batch.tgt_fragment_len == [16, 14, 10, 1, 8, 1]

    assert batch.info['src_doc_len_in_frags'] == 1
    assert batch.info['tgt_doc_len_in_frags'] == 2

    assert batch.src_doc_len_in_frags == [1, 1, 1]
    assert batch.tgt_doc_len_in_frags == [2, 1, 1]
