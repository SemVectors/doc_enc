#!/usr/bin/env python3

import logging
from pathlib import Path
import csv

from doc_enc.utils import find_file


def collect_src_tgt_ids(meta_path: Path):
    src_ids = set()
    tgt_ids = set()

    with open(meta_path, 'r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        next(reader)
        for row in reader:
            src_id, _, tgt_id, *_ = row
            src_ids.add(src_id)
            tgt_ids.add(tgt_id)

    return list(src_ids), list(tgt_ids)


def paths_from_ids(text_dir: Path, id_list):
    paths = []
    for i in id_list:
        fp = text_dir / f"{i}.txt"
        try:
            fp = find_file(fp, throw_if_not_exist=False)
            paths.append(fp)
        except RuntimeError:
            logging.warning("%s does not exist", fp)
    return paths


def collect_all_paths(text_dir: Path, meta_path):
    src_ids, tgt_ids = collect_src_tgt_ids(meta_path)
    all_ids = list(frozenset(src_ids + tgt_ids))
    return paths_from_ids(text_dir, all_ids)


def id_from_path(p: Path):
    while p.suffix:
        p = p.with_suffix('')
    return p.name


def paths_to_ids(paths):
    return [id_from_path(p) for p in paths]
