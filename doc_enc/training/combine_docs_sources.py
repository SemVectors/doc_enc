#!/usr/bin/env python3

import logging
from typing import NamedTuple, Optional
from pathlib import Path
import csv
import hashlib
import concurrent.futures

from doc_enc.utils import find_file, open_bin_file, open_file
from doc_enc.text_processor import TextProcessor


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
    text_proc: Optional[TextProcessor] = None,
    include_datasets=None,
    exclude_datasets=None,
    out_filename_prefix="combined",
    min_doc_len=0,
    max_doc_len=float('inf'),
    sort_by_len=False,
    procs=4,
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
            src_info_dict, tgt_info_dict = _calc_sentence_size_and_hash(dsp, text_proc, procs=procs)
            all_examples.extend(
                _generate_examples_from_dataset(
                    dsp, split, i, src_info_dict, tgt_info_dict, min_doc_len, max_doc_len
                )
            )

        if sort_by_len:
            key = lambda t: (-t.src_len, t.src_hash, -t.label)
        else:
            key = lambda t: (t.src_hash, -t.label)
        all_examples.sort(key=key)
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


def _calc_sentence_size_and_hash(dataset_path: Path, text_proc: Optional[TextProcessor], procs=4):
    docs_path = dataset_path / "texts"
    if docs_path.exists():
        src_docs_path = docs_path
        info_dict = {}
        _calc_sentence_size_and_hash_in_dir(docs_path, text_proc, info_dict, procs=procs)
        return info_dict, info_dict

    src_docs_path = dataset_path / "texts_1"
    tgt_docs_path = dataset_path / "texts_2"
    if src_docs_path.exists() and tgt_docs_path.exists():
        src_info_dict = {}
        _calc_sentence_size_and_hash_in_dir(src_docs_path, text_proc, src_info_dict, procs=procs)
        tgt_info_dict = {}
        _calc_sentence_size_and_hash_in_dir(tgt_docs_path, text_proc, tgt_info_dict, procs=procs)
        return src_info_dict, tgt_info_dict

    raise RuntimeError(f"Not found texts folder (or texts_1,texts_2 folders) in {dataset_path}")


def _process_in_pool(pool, func, args_generator, on_result=None):
    futures = set()

    processed = 0

    def _wait_for_futures(f):
        nonlocal processed
        done, f = concurrent.futures.wait(f, return_when=concurrent.futures.FIRST_COMPLETED)
        processed += len(done)

        while done:
            done_task = done.pop()
            # TODO exception handling
            res = done_task.result()
            if on_result:
                on_result(res)

        return f

    for args in args_generator:
        if len(futures) >= pool._max_workers:
            futures = _wait_for_futures(futures)

        futures.add(pool.submit(func, *args))

    while futures:
        futures = _wait_for_futures(futures)


_PROC_TEXT_PROC = None


def _proc_init(text_proc):
    global _PROC_TEXT_PROC
    _PROC_TEXT_PROC = text_proc


def _doc_id_from_path(p):
    doc_id = p
    while doc_id.suffix:
        doc_id = doc_id.with_suffix('')
    doc_id = int(doc_id.name)
    return doc_id


def _proc_calc_sent_info_for_paths(paths: list[Path]):
    out_info_dict = {}
    for p in paths:
        md5hash = hashlib.md5()
        cnt = 0
        if _PROC_TEXT_PROC is not None:
            sent_tokens, _ = _PROC_TEXT_PROC.prepare_text_from_file(p, split_into_fragments=False)
            cnt = len(sent_tokens)
            for tokens in sent_tokens:
                md5hash.update(b''.join(t.to_bytes(4, 'little') for t in tokens))
        else:
            with open_bin_file(p) as f:
                for l in f:
                    if l.strip():
                        cnt += 1
                        md5hash.update(l)

        if cnt > 0:
            doc_id = _doc_id_from_path(p)
            out_info_dict[doc_id] = (cnt, md5hash.hexdigest())
    return out_info_dict


def _calc_sentence_size_and_hash_in_dir(
    docs_path: Path,
    text_proc: Optional[TextProcessor],
    out_info_dict,
    procs=4,
    batch_size=100,
):
    def _batch_gen():
        batch = []
        cnt = 0
        for p in docs_path.iterdir():
            if not p.is_file() or not p.suffix in ('.gz', '.txt'):
                continue

            doc_id = _doc_id_from_path(p)
            if doc_id in out_info_dict:
                continue

            batch.append(p)
            cnt += 1
            if cnt % 10_000 == 0:
                logging.info("processed 10k docs")
            if len(batch) > batch_size:
                yield (batch,)
                batch = []

        if batch:
            yield (batch,)

    logging.info("precalculating doc lengths and hash in %s", docs_path)
    pool = concurrent.futures.ProcessPoolExecutor(
        procs, initializer=_proc_init, initargs=(text_proc,)
    )
    _process_in_pool(
        pool,
        _proc_calc_sent_info_for_paths,
        _batch_gen(),
        out_info_dict.update,
    )


class Stat:
    def __init__(self) -> None:
        self.total_src_docs = 0
        self.total_src_doc_len = 0
        self.max_src_doc_len = 0

        self.total_tgt_docs = 0
        self.total_tgt_doc_len = 0
        self.max_tgt_doc_len = 0

        self.unique_srcs = set()
        self.unique_tgts = set()

        self.filter_src_min_doc_len = 0
        self.filter_tgt_min_doc_len = 0
        self.filter_src_max_doc_len = 0
        self.filter_tgt_max_doc_len = 0

    def add_src_stat(self, src_id, src_len, max_doc_len, min_doc_len):
        if src_id not in self.unique_srcs:
            self.unique_srcs.add(src_id)
            self.total_src_docs += 1
            self.total_src_doc_len += src_len
            self.max_src_doc_len = max(self.max_src_doc_len, src_len)
        if src_len < min_doc_len:
            self.filter_src_min_doc_len += 1
        elif max_doc_len <= src_len:
            self.filter_src_max_doc_len += 1

    def add_tgt_stat(self, tgt_id, tgt_len, max_doc_len, min_doc_len):
        if tgt_id not in self.unique_tgts:
            self.unique_tgts.add(tgt_id)
            self.total_tgt_docs += 1
            self.total_tgt_doc_len += tgt_len
            self.max_tgt_doc_len = max(self.max_tgt_doc_len, tgt_len)
        if tgt_len < min_doc_len:
            self.filter_tgt_min_doc_len += 1
        elif max_doc_len <= tgt_len:
            self.filter_tgt_max_doc_len += 1

    def __str__(self) -> str:
        return (
            f'avg src doc len: {self.total_src_doc_len/self.total_src_docs}, '
            f'max src doc len: {self.max_src_doc_len}, '
            f'avg tgt doc len: {self.total_tgt_doc_len/self.total_tgt_docs}, '
            f'max tgt doc len: {self.max_tgt_doc_len}\n'
            f'filtered by: src_min_doc_len: {self.filter_src_min_doc_len}, '
            f'tgt_min_doc_len: {self.filter_tgt_min_doc_len}, '
            f'src_max_doc_len: {self.filter_src_max_doc_len}, '
            f'tgt_max_doc_len: {self.filter_tgt_max_doc_len}'
        )


def _generate_examples_from_dataset(
    p: Path, split: str, dataset_id: int, src_info_dict, tgt_info_dict, min_doc_len, max_doc_len
):
    meta_path = find_file(p / f"{split}.csv")
    if not meta_path.exists():
        raise RuntimeError(f"Not found {split}.csv in {p}")

    stat = Stat()
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
            src_len = src_info[0]
            tgt_len = tgt_info[0]

            stat.add_src_stat(src_id, src_len, max_doc_len=max_doc_len, min_doc_len=min_doc_len)
            stat.add_tgt_stat(tgt_id, tgt_len, max_doc_len=max_doc_len, min_doc_len=min_doc_len)

            if min_doc_len <= src_len < max_doc_len and min_doc_len <= tgt_len < max_doc_len:
                yield Example(
                    dataset_id,
                    src_id,
                    tgt_id,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    label=label,
                    src_hash=src_info[1],
                    tgt_hash=tgt_info[1],
                )
    logging.info("stat for %s:", p)
    logging.info(str(stat))
