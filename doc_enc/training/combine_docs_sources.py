#!/usr/bin/env python3

import logging
from typing import NamedTuple
from pathlib import Path
import csv
import hashlib

from doc_enc.training.base_batch_generator import find_file, open_bin_file, open_file


class Example(NamedTuple):
    dataset: int
    src_id: int
    tgt_id: int
    src_len: int
    tgt_len: int
    label: int
    src_hash: str
    tgt_hash: str


def combine_docs_datasets(
    input_dir,
    split,
    include_datasets=None,
    exclude_datasets=None,
    out_filename_prefix="combined",
):
    input_path = Path(input_dir)
    datasets = []
    for p in input_path.iterdir():
        if not p.is_dir():
            continue

        entry = p.name
        if exclude_datasets and entry in exclude_datasets:
            continue
        if include_datasets and entry not in include_datasets:
            continue
        datasets.append(p)

    with open(input_path / f"{out_filename_prefix}_{split}.csv", 'w', encoding="utf-8") as outf:
        csv_writer = csv.writer(outf)
        # csv_writer.writerow(("ds", "src", "tgt", "label", "slen", "tlen", "shash", "thash"))
        if not datasets:
            return

        all_examples = []
        for i, dsp in enumerate(datasets):
            src_info_dict, tgt_info_dict = _calc_sentence_size_and_hash(dsp)
            all_examples.extend(
                _generate_examples_from_dataset(dsp, split, i, src_info_dict, tgt_info_dict)
            )

        all_examples.sort(key=lambda t: (-t.src_len, t.src_hash, -t.label, t.tgt_len))
        for e in all_examples:
            csv_writer.writerow(
                (
                    datasets[e.dataset].name,
                    e.src_id,
                    e.tgt_id,
                    e.label,
                    e.src_len,
                    e.tgt_len,
                    e.src_hash,
                    e.tgt_hash,
                )
            )


def _calc_sentence_size_and_hash(dataset_path: Path):
    docs_path = dataset_path / "texts"
    if docs_path.exists():
        src_docs_path = docs_path
        info_dict = {}
        _calc_sentence_size_and_hash_in_dir(docs_path, info_dict)
        return info_dict, info_dict

    src_docs_path = dataset_path / "texts_1"
    tgt_docs_path = dataset_path / "texts_2"
    if src_docs_path.exists() and tgt_docs_path.exists():
        src_info_dict = {}
        _calc_sentence_size_and_hash_in_dir(src_docs_path, src_info_dict)
        tgt_info_dict = {}
        _calc_sentence_size_and_hash_in_dir(tgt_docs_path, tgt_info_dict)
        return src_info_dict, tgt_info_dict

    raise RuntimeError(f"Not found texts folder (or texts_1,texts_2 folders) in {dataset_path}")


def _calc_sentence_size_and_hash_in_dir(docs_path: Path, out_info_dict):
    # info_dict = {}
    for p in docs_path.iterdir():
        if not p.is_file() or not p.suffix in ('.gz', '.txt'):
            continue

        doc_id = p
        while doc_id.suffix:
            doc_id = doc_id.with_suffix('')

        doc_id = int(doc_id.name)
        if doc_id in out_info_dict:
            continue

        with open_bin_file(p) as f:
            i = 0
            md5hash = hashlib.md5()
            for i, l in enumerate(f, 1):
                md5hash.update(l)
            out_info_dict[doc_id] = (i, md5hash.hexdigest())


def _generate_examples_from_dataset(
    p: Path, split: str, dataset_id: int, src_info_dict, tgt_info_dict
):
    meta_path = find_file(p / f"{split}.csv")
    if not meta_path.exists():
        raise RuntimeError(f"Not found {split}.csv in {p}")
    with open_file(meta_path) as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return

        for row in reader:
            src_id, _, tgt_id, _, label, *_ = row
            src_id = int(src_id)
            tgt_id = int(tgt_id)
            label = int(label)
            src_info = src_info_dict.get(src_id)
            tgt_info = tgt_info_dict.get(tgt_id)
            if src_info is None:
                logging.warning("%s: src text is missing for id %s", p.name, src_id)
                continue
            if tgt_info is None:
                logging.warning("%s: tgt file is missing for id %s", p.name, tgt_id)
                continue
            yield Example(
                dataset_id,
                src_id,
                tgt_id,
                src_len=src_info[0],
                tgt_len=tgt_info[0],
                label=label,
                src_hash=src_info[1],
                tgt_hash=tgt_info[1],
            )
