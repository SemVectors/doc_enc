#!/usr/bin/env python3

import logging
from typing import List, Tuple, Any
import dataclasses
from pathlib import Path
import csv
import random

import scipy.spatial.distance as scipy_dist

from doc_enc.eval.eval_utils import paths_from_ids
from doc_enc.doc_encoder import DocEncoder


@dataclasses.dataclass
class DatasetConf:
    name: str
    meta: str
    texts: str


@dataclasses.dataclass
class DocMatchingConf:
    ds_base_dir: str
    datasets: List[DatasetConf]
    enabled_ds: List = dataclasses.field(default_factory=list)

    threshold: float = 0.0
    choose_threshold: bool = True
    balance_pos_and_neg_examples: bool = True
    seed: int = 2022 * 55


def _finalize_gold(conf: DocMatchingConf, src_id, positives, negatives, stat):
    if conf.balance_pos_and_neg_examples:
        min_len = min(len(positives), len(negatives))
        if min_len == 0:
            min_len = 1
        if len(positives) > min_len:
            positives = random.sample(positives, min_len)
        if len(negatives) > min_len:
            negatives = random.sample(negatives, min_len)

    stat[0] += len(positives)
    stat[1] += len(negatives)
    examples: List[Tuple[Any, Any, Any]] = [(src_id, i, 1) for i in positives]
    examples.extend([(src_id, i, 0) for i in negatives])
    return examples


def _load_gold_data(conf: DocMatchingConf, meta_path):
    gold = []
    stat = [0, 0]
    with open(conf.ds_base_dir + '/' + meta_path, 'r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        next(reader)
        cur_src_id = ''
        positives = []
        negatives = []
        for row in reader:
            src_id, _, tgt_id, _, label, *_ = row
            if src_id != cur_src_id:
                if positives or negatives:
                    gold.extend(_finalize_gold(conf, cur_src_id, positives, negatives, stat))
                cur_src_id = src_id
                positives = []
                negatives = []

            if int(label) == 1:
                positives.append(tgt_id)
            else:
                negatives.append(tgt_id)
        if positives or negatives:
            gold.extend(_finalize_gold(conf, cur_src_id, positives, negatives, stat))
    logging.info("# of positives: %s, # of negatives: %s", stat[0], stat[1])
    return gold


def _calc_metrics(threshold, gold, inv_idx, doc_embs):
    total = 0
    good = 0
    not_found = 0

    for src_id, tgt_id, label in gold:
        src_i = inv_idx.get(src_id)
        if src_i is None:
            not_found += 1
            continue
        tgt_i = inv_idx.get(tgt_id)
        if tgt_i is None:
            not_found += 1
            continue
        sim = 1.0 - scipy_dist.cosine(doc_embs[src_i], doc_embs[tgt_i])

        computed_label = 0
        if sim > threshold:
            computed_label = 1

        good += computed_label == label
        total += 1
    if not_found:
        logging.warning("%d text embeddings were missing!", not_found)
    return good / total


def _eval_impl(conf: DocMatchingConf, doc_encoder, meta_path, texts_dir):
    base_dir = Path(conf.ds_base_dir)
    full_texts_dir = base_dir / texts_dir
    if not full_texts_dir.exists():
        raise RuntimeError(f'{full_texts_dir} does not exist')

    gold = _load_gold_data(conf, meta_path)
    all_ids = set()
    for src_id, tgt_id, _ in gold:
        all_ids.add(src_id)
        all_ids.add(tgt_id)

    paths = paths_from_ids(base_dir / texts_dir, all_ids)
    logging.info("encoding %s documents", len(paths))
    doc_embs = doc_encoder.encode_docs_from_path_list(paths)
    assert len(doc_embs) == len(paths)
    logging.info("Shape of computed embs: %s", doc_embs.shape)

    inv_idx = {}
    for i, p in enumerate(paths):
        while p.suffix:
            p = p.with_suffix('')
        inv_idx[p.name] = i

    if conf.choose_threshold:
        results = []
        for t in range(2, 9):
            t = t / 10
            acc = _calc_metrics(t, gold, inv_idx, doc_embs)
            m = {'threshold': t, 'acc': acc}
            results.append(m)
        return results
    acc = _calc_metrics(conf.threshold, meta_path, inv_idx, doc_embs)
    return {"acc": acc}


def doc_matching_eval(conf: DocMatchingConf, doc_encoder: DocEncoder):
    random.seed(conf.seed)
    results = []
    for dataset in conf.datasets:
        if conf.enabled_ds and dataset.name not in conf.enabled_ds:
            continue
        m = _eval_impl(conf, doc_encoder, meta_path=dataset.meta, texts_dir=dataset.texts)
        results.append((dataset.name, m))
    return results
