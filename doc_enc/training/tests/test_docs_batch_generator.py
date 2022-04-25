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
        with open(ds1_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("1 2 3 4 5\n" * 15)
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
            f.write("ds1,15,30,1,15,30,15hash,30hash\n")
            f.write("ds1,15,16,0,15,16,15hash,16hash\n")
            f.write("ds1,15,40,0,15,40,15hash,40hash\n")

        yield tmpdirname


def _create_gen_opts(input_dir, **kwargs):
    conf = DocsBatchGeneratorConf(
        input_dir=input_dir,
        positives_per_doc=[2, 2],
        negatives_per_doc=[2, 2],
        fragment_size=16,
        **kwargs,
    )
    tok_conf = TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED)
    return conf, tok_conf


def test_gen_basic(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(FakeTrainingData)
    gen = DocsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert len(batch.src_sents) == 18
    assert len(batch.tgt_sents) == 166

    assert batch.src_fragment_len == [3, 15]
    assert batch.tgt_fragment_len == [
        16,
        16,
        8,
        # doc16
        16,
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
    ]

    assert batch.src_doc_len_in_frags == [1, 1]
    assert batch.src_doc_len_in_sents == [3, 15]

    assert batch.tgt_doc_len_in_frags == [3, 1, 2, 4, 2]
    assert batch.tgt_doc_len_in_sents == [40, 16, 30, 50, 30]

    assert batch.positive_idxs[0] == [0]
    assert batch.positive_idxs[1] == [2, 3]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 5


def test_two_batches(FakeTrainingData):
    conf, tok_conf = _create_gen_opts(
        FakeTrainingData, batch_sent_size=15, allow_docs_without_positives=True
    )
    gen = DocsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 2
    batch1: DocsBatch = batches[0]

    assert len(batch1.src_sents) == 18
    assert len(batch1.tgt_sents) == 166

    assert batch1.src_fragment_len == [3, 15]

    assert batch1.info['src_docs_cnt'] == 2
    assert batch1.info['tgt_docs_cnt'] == 5

    batch2: DocsBatch = batches[1]

    assert len(batch2.src_sents) == 16
    assert len(batch2.tgt_sents) == 40

    assert batch2.src_fragment_len == [16]
    assert batch2.tgt_fragment_len == [16, 16, 8]

    assert batch2.info['src_docs_cnt'] == 1
    assert batch2.info['tgt_docs_cnt'] == 1


def test_iterator_two_generators(FakeTrainingData):
    gen_conf, tok_conf = _create_gen_opts(FakeTrainingData, allow_docs_without_positives=True)
    iter_conf = DocsBatchIteratorConf(batch_generator_conf=gen_conf, async_generators=2)
    biter = DocsBatchIterator(iter_conf, tok_conf, 'train')

    biter.init_epoch(1)
    res = list(biter.batches())
    assert len(res) == 2
    batch1, labels1 = res[0]
    assert batch1.info['src_docs_cnt'] == 2
    assert batch1.info['tgt_docs_cnt'] == 3
    assert labels1[0].tolist() == [1, 0, 0]
    assert labels1[1].tolist() == [0, 0, 1]

    batch2, labels2 = res[1]
    assert batch2.info['src_docs_cnt'] == 2
    assert batch2.info['tgt_docs_cnt'] == 3
    assert labels2[0].tolist() == [1, 0, 0]
    assert labels2[1].tolist() == [0, 0, 0]


def test_gen_with_dups(FakeTrainingDataWithDups):
    random.seed(1)

    conf = DocsBatchGeneratorConf(
        input_dir=FakeTrainingDataWithDups,
        positives_per_doc=[1, 1],
        negatives_per_doc=[2, 2],
    )
    tok_conf = TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED)
    gen = DocsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [3, 15]
    assert batch.tgt_ids == [30, 16, 40]
    assert len(batch.src_sents) == 18
    assert len(batch.tgt_sents) == 86

    assert batch.src_fragment_len == [3, 15]

    assert batch.src_doc_len_in_sents == [3, 15]

    assert batch.tgt_doc_len_in_sents == [30, 16, 40]

    assert batch.positive_idxs[0] == [0, 1, 2]
    assert batch.positive_idxs[1] == [0]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 3


def test_gen_with_dups2(FakeTrainingDataWithDups):
    # the same test with different seed
    random.seed(2)

    conf = DocsBatchGeneratorConf(
        input_dir=FakeTrainingDataWithDups,
        positives_per_doc=[1, 1],
        negatives_per_doc=[2, 2],
    )
    tok_conf = TokenizerConf(tokenizer_type=TokenizerType.PRETOKENIZED)
    gen = DocsBatchGenerator(conf, tok_conf=tok_conf, split='train', line_offset=0)
    batches = list(gen.batches())
    assert len(batches) == 1
    batch: DocsBatch = batches[0]
    print(batch.src_sents)
    assert batch.src_ids == [3, 15]
    assert batch.tgt_ids == [40, 30, 16]
    assert len(batch.src_sents) == 18
    assert len(batch.tgt_sents) == 86

    assert batch.src_fragment_len == [3, 15]

    assert batch.src_doc_len_in_sents == [3, 15]

    assert batch.tgt_doc_len_in_sents == [40, 30, 16]

    assert batch.positive_idxs[0] == [0, 1, 2]
    assert batch.positive_idxs[1] == [1]

    assert batch.info['src_docs_cnt'] == 2
    assert batch.info['tgt_docs_cnt'] == 3
