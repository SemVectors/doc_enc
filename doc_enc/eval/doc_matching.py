#!/usr/bin/env python3

import logging
from typing import List, Mapping
import dataclasses
from pathlib import Path
import csv

import torch

from doc_enc.doc_encoder import DocEncoder


@dataclasses.dataclass
class DatasetConf:
    meta: str
    texts: str


@dataclasses.dataclass
class DocMatchingConf:
    datasets: List[DatasetConf]

    threshold: float = 0.5


def _eval_impl(conf, doc_encoder, meta_path, texts_dir):
    texts_dir = Path(texts_dir)
    if not texts_dir.exists():
        raise RuntimeError(f'{texts_dir} does not exist')
    filenames, doc_embs = doc_encoder.encode_docs_from_dir(texts_dir)
    filename2idx = {f: i for i, f in enumerate(filenames)}

    total = 0
    good = 0
    not_found = 0
    Cos = torch.nn.CosineSimilarity(dim=0)
    with open(meta_path, 'r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        next(reader)
        for row in reader:
            src_id, _, tgt_id, _, label, *_ = row
            src_i = filename2idx.get(f'{src_id}.txt')
            if src_i is None:
                not_found += 1
                continue
            tgt_i = filename2idx.get(f'{tgt_id}.txt')
            if tgt_i is None:
                not_found += 1
                continue
            sim = Cos(doc_embs[src_i], doc_embs[tgt_i]).item()

            computed_label = 0
            if sim > conf.threshold:
                computed_label = 1
            good += computed_label == int(label)
            total += 1
    if not_found:
        logging.warning("%d text embeddings were missing!")
    return good / total


def doc_matching_eval(conf: DocMatchingConf, doc_encoder: DocEncoder):
    for dataset in conf.datasets:
        acc = _eval_impl(conf, doc_encoder, meta_path=dataset.meta, texts_dir=dataset.texts)
        return acc
