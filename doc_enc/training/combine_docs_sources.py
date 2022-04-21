#!/usr/bin/env python3


from typing import NamedTuple
from pathlib import Path
import csv
import hashlib


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
            info_dict = _calc_sentence_size_and_hash(dsp)
            all_examples.extend(_generate_examples_from_dataset(dsp, split, i, info_dict))

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
    if not docs_path.exists():
        raise RuntimeError(f"Not found texts folder in {dataset_path}")

    info_dict = {}
    for p in docs_path.iterdir():
        if not p.is_file() or not p.name.endswith(".txt"):
            continue
        doc_id = int(p.name[:-4])
        if doc_id in info_dict:
            continue

        with open(p, 'rb') as f:
            i = 0
            md5hash = hashlib.md5()
            for i, l in enumerate(f, 1):
                md5hash.update(l)
            info_dict[doc_id] = (i, md5hash.hexdigest())
    return info_dict


def _generate_examples_from_dataset(p: Path, split: str, dataset_id: int, info_dict):
    meta_path = p / f"{split}.csv"
    docs_path = p / "texts"
    if not meta_path.exists():
        raise RuntimeError(f"Not found {split}.csv in {p}")
    if not docs_path.exists():
        raise RuntimeError(f"Not found texts folder in {p}")
    with open(meta_path, 'r', encoding='utf8') as f:
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
            yield Example(
                dataset_id,
                src_id,
                tgt_id,
                src_len=info_dict[src_id][0],
                tgt_len=info_dict[tgt_id][0],
                label=label,
                src_hash=info_dict[src_id][1],
                tgt_hash=info_dict[tgt_id][1],
            )
