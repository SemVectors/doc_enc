#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple
import dataclasses
from pathlib import Path
import csv
import itertools

import numpy as np

from doc_enc.utils import find_file
from doc_enc.eval.eval_utils import paths_from_ids, collect_src_tgt_ids, id_from_path
from doc_enc.eval.sim_util import calc_sim, SimKind
from doc_enc.doc_encoder import DocEncoder


@dataclasses.dataclass
class DatasetConf:
    name: str
    meta: str
    texts: str
    search_over_all_texts: bool = False
    other_texts_limit: int = 0
    extra_other_dir: Optional[str] = None
    eval_for_len_groups: bool = False


@dataclasses.dataclass
class DocRetrievalConf:
    ds_base_dir: str
    datasets: List[DatasetConf]
    enabled_ds: List = dataclasses.field(default_factory=list)

    sim_kind: SimKind = SimKind.COS
    topk: List[int] = dataclasses.field(default_factory=lambda: [3, 5, 10, 20])

    use_gpu: Optional[int] = None


def _load_gold_data(meta_path, query_dir, query_data, other_dir, other_data):
    _, query_inv_idx = query_data
    _, other_inv_idx = other_data

    gold = []
    with open(meta_path, 'r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        next(reader)
        cur_src_id = ''
        cur_rel = []
        for row in reader:
            src_id, _, tgt_id, _, label, *rest = row
            if int(label) != 1:
                continue
            if src_id != cur_src_id:
                if cur_rel:
                    gold.append((query_inv_idx[_make_key(query_dir, cur_src_id)], cur_rel, rest))
                cur_src_id = src_id
                cur_rel = []
            cur_rel.append(other_inv_idx[_make_key(other_dir, tgt_id)])

        if cur_rel:
            gold.append((query_inv_idx[_make_key(query_dir, cur_src_id)], cur_rel, rest))
    return gold


def _calc_metrics(conf: DocRetrievalConf, sim_matrix, gold_data, query_data, other_data):
    if not gold_data:
        return {}

    query_keys, _ = query_data
    other_keys, _ = other_data

    metrics = {}
    for k in conf.topk:
        total_rels = 0
        found_rels = 0
        cum_ap = 0.0
        all_found_idxs = sim_matrix[:, : k + 1]
        for query_idx, rel_idxs, _ in gold_data:
            gold_rel = min(k, len(rel_idxs))
            found_rel = all_found_idxs[query_idx]

            query_key = query_keys[query_idx]
            first_rel_key = other_keys[found_rel[0]]
            if query_key == first_rel_key:
                # this is the same doc
                found_rel = found_rel[1:]
            else:
                found_rel = found_rel[:-1]

            found_rel_cnt = len(np.intersect1d(found_rel, rel_idxs))
            total_rels += gold_rel
            found_rels += found_rel_cnt

            tp = 0
            ap = 0.0
            for num, found_idx in enumerate(found_rel):
                rel = int(found_idx in rel_idxs)
                tp += rel
                ap += rel * tp / (num + 1)
            cum_ap += ap / gold_rel

        metrics[f'rec@{k}'] = found_rels / total_rels if total_rels else 0.0
        metrics[f'MAP@{k}'] = cum_ap / len(gold_data)
    return metrics


def _find_text_dirs(prefix_dir: Path) -> Tuple[Path, Path]:
    if prefix_dir.exists():
        return (prefix_dir, prefix_dir)
    t1 = Path(str(prefix_dir) + "_1")
    t2 = Path(str(prefix_dir) + "_2")
    if t1.exists() and t2.exists():
        return (t1, t2)

    raise RuntimeError(f'{prefix_dir} (or {t1}, {t2}) does not exist')


def _make_key(text_dir: Path, doc_id):
    return (text_dir, doc_id)


def _make_keys_dict(base_dir: Path, paths):
    id2idx = {}
    for i, p in enumerate(paths):
        doc_id = id_from_path(p)
        text_dir = p.parent.relative_to(base_dir)
        id2idx[_make_key(text_dir, doc_id)] = i

    return id2idx


def _add_docs_from_other_dir(dsconf: DatasetConf, base_dir: Path, other_text_dir: Path, other_keys):
    all_other_paths = list(f for f in (base_dir / other_text_dir).iterdir() if f.is_file())
    all_other_paths.sort()
    if dsconf.other_texts_limit:
        other_keys_set = frozenset(other_keys)
        for p in all_other_paths:
            i = id_from_path(p)
            key = _make_key(other_text_dir, i)
            if key in other_keys_set:
                continue
            other_keys.append(key)
            if len(other_keys) >= dsconf.other_texts_limit:
                break
        return other_keys
    return [_make_key(other_text_dir, id_from_path(p)) for p in all_other_paths]


def paths_from_keys(base_dir: Path, key_list):
    paths = []
    for text_dir, i in key_list:
        fp = base_dir / text_dir / f"{i}.txt"
        try:
            fp = find_file(fp, throw_if_not_exist=False)
            paths.append(fp)
        except RuntimeError:
            logging.warning("%s does not exist", fp)
    return paths


def _eval_each_group(conf, sim_matrix, gold_data, query_data, other_data):
    all_group_metrics = []

    if not gold_data:
        return all_group_metrics
    # check if gold data contain group info
    rest = gold_data[0][-1]
    if len(rest) != 3:
        raise RuntimeError(f"expected that extra data contains 3 fields: {rest}")

    key_func = lambda t: (t[-1][0], t[-1][2])
    gold_data.sort(key=key_func)
    for key, group in itertools.groupby(gold_data, key=key_func):
        group_gold = list(group)
        group_metrics = _calc_metrics(conf, sim_matrix, group_gold, query_data, other_data)
        avg_sent_len = sum(int(t[-1][1]) for t in group_gold) / len(group_gold)
        group_metrics['comparable'] = int(key[0])
        group_metrics['bin'] = int(key[1])
        group_metrics['avg_sent_len'] = avg_sent_len
        all_group_metrics.append(group_metrics)

    return all_group_metrics


def _eval_impl(conf: DocRetrievalConf, dsconf: DatasetConf, doc_encoder: DocEncoder):
    base_dir = Path(conf.ds_base_dir)
    abs_query_text_dir, abs_other_text_dir = _find_text_dirs(base_dir / dsconf.texts)
    query_text_dir = abs_query_text_dir.relative_to(base_dir)
    other_text_dir = abs_other_text_dir.relative_to(base_dir)

    query_ids, other_ids = collect_src_tgt_ids(base_dir / dsconf.meta)
    other_keys = [_make_key(other_text_dir, o) for o in other_ids]
    if dsconf.search_over_all_texts:
        other_keys = _add_docs_from_other_dir(dsconf, base_dir, other_text_dir, other_keys)
        if dsconf.extra_other_dir:
            other_keys = _add_docs_from_other_dir(
                dsconf, base_dir, Path(dsconf.extra_other_dir), other_keys
            )

    other_paths = paths_from_keys(base_dir, other_keys)
    logging.info("computing embeddings for %s docs", len(other_paths))
    other_doc_embs = doc_encoder.encode_docs_from_path_list(other_paths)
    assert len(other_doc_embs) == len(other_paths) == len(other_keys)
    logging.info("Shape of computed other embs: %s", other_doc_embs.shape)
    other_inv_idx = _make_keys_dict(base_dir, other_paths)
    other_data = (other_keys, other_inv_idx)

    query_paths = paths_from_ids(abs_query_text_dir, query_ids)
    query_doc_embs = doc_encoder.encode_docs_from_path_list(query_paths)
    assert len(query_paths) == len(query_doc_embs) == len(query_ids)
    logging.info("Shape of computed query embs: %s", query_doc_embs.shape)
    query_inv_idx = _make_keys_dict(base_dir, query_paths)
    query_data = ([_make_key(query_text_dir, i) for i in query_ids], query_inv_idx)

    max_k = max(conf.topk) + 1
    _, indexes = calc_sim(conf.sim_kind, max_k, query_doc_embs, other_doc_embs)

    gold_data = _load_gold_data(
        base_dir / dsconf.meta, query_text_dir, query_data, other_text_dir, other_data
    )
    metrics = _calc_metrics(conf, indexes, gold_data, query_data, other_data)
    if dsconf.eval_for_len_groups:
        group_metrics = _eval_each_group(conf, indexes, gold_data, query_data, other_data)
        metrics = [metrics] + group_metrics
    return metrics


def doc_retrieval_eval(conf: DocRetrievalConf, doc_encoder: DocEncoder):
    results = []
    for dataset in conf.datasets:
        if conf.enabled_ds and dataset.name not in conf.enabled_ds:
            continue
        logging.info("Evaling doc retrieval on %s", dataset.name)
        m = _eval_impl(conf, dataset, doc_encoder)
        results.append((dataset.name, m))
    return results
