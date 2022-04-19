#!/usr/bin/env python3

import itertools
from pathlib import Path
import tempfile

import pytest

from doc_enc.training.combine_docs_sources import combine_docs_datasets


@pytest.fixture
def FakeTrainingData():
    with tempfile.TemporaryDirectory() as tmpdirname:

        tmpdirname = Path(tmpdirname)
        for n in ("ds1", "ds2"):
            ds = tmpdirname / n
            ds.mkdir()
            ds_docs = ds / "texts"
            ds_docs.mkdir()

        ds1_docs_dir = tmpdirname / "ds1" / "texts"
        with open(ds1_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("линия1\nлиния2\nлиния3")
        with open(ds1_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 14)
            f.write("line")
        with open(ds1_docs_dir / '16.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 14)
            f.write("line+line")
        with open(ds1_docs_dir / '1001.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 1000)
            f.write("line")
        with open(ds1_docs_dir / '500.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 499)
            f.write("line")
        with open(ds1_docs_dir / '1501.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 1500)
            f.write("line")

        with open(tmpdirname / 'ds1' / 'train.csv', 'w', encoding='utf8') as f:
            f.write("id1,t1,id2,t2,l\n")
            f.write("3,_,1001,_,1\n")
            f.write("3,_,1501,_,0\n")
            f.write("15,_,500,_,1\n")
            f.write("15,_,1501,_,0\n")
            f.write("15,_,1001,_,0\n")
            f.write("16,_,1501,_,1\n")

        ds2_docs_dir = tmpdirname / "ds2" / "texts"
        with open(ds2_docs_dir / '3.txt', 'w', encoding='utf8') as f:
            f.write("line1\nline2\nline3")
        with open(ds2_docs_dir / '15.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 14)
            f.write("line")
        with open(ds2_docs_dir / '20.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 19)
            f.write("line")
        with open(ds2_docs_dir / '1001.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 2000)
            f.write("line")
        with open(ds2_docs_dir / '600.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 599)
            f.write("line")
        with open(ds2_docs_dir / '400.txt', 'w', encoding='utf8') as f:
            f.write("line\n" * 399)
            f.write("line")

        with open(tmpdirname / 'ds2' / 'train.csv', 'w', encoding='utf8') as f:
            f.write("id1,t1,id2,t2,l\n")
            f.write("3,_,600,_,1\n")
            f.write("3,_,400,_,0\n")
            f.write("15,_,1001,_,1\n")
            f.write("20,_,1001,_,1\n")
            f.write("20,_,600,_,0\n")

        yield tmpdirname


def _parse_combined_output(l):
    ds, src, tgt, label, slen, tlen, shash, thash = l.rstrip().split(',')
    return (ds, int(src), int(tgt), int(label), int(slen), int(tlen), shash, thash)


def test_gen_basic(FakeTrainingData):
    combine_docs_datasets(FakeTrainingData, 'train')
    assert (FakeTrainingData / "combined_train.csv").exists()

    with open(FakeTrainingData / "combined_train.csv", 'r', encoding='utf8') as f:
        id1_3_list = [_parse_combined_output(l) for l in itertools.islice(f, 0, 4)]
        assert len(id1_3_list) == 4
        print(id1_3_list)
        for t in id1_3_list:
            assert t[1] == 3
            assert t[4] == 3
        for t in id1_3_list[:2]:
            assert t[0] == 'ds1'
        for t in id1_3_list[2:]:
            assert t[0] == 'ds2'

        assert id1_3_list[0][2] == 1001
        assert id1_3_list[2][2] == 600

        id1_15_list = [_parse_combined_output(l) for l in itertools.islice(f, 0, 5)]
        assert len(id1_15_list) == 5
        print(id1_15_list)
        assert id1_15_list[0][1] == 16
        assert id1_15_list[0][4] == 15
        for t in id1_15_list[1:]:
            assert t[1] == 15
            assert t[4] == 15

        src_hash = id1_15_list[1][-2]
        assert id1_15_list[0][-2] != src_hash
        for t in id1_15_list[1:]:
            assert t[-2] == src_hash
