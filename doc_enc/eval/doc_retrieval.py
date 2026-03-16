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

    # For format from MTEB {queries,corpus,qrels}.csv. corpus file is passed in
    # texts, qrels in meta.
    queries: str | None = None

    search_over_all_texts: bool = False
    other_texts_limit: int = 0
    extra_other_dir: Optional[str] = None
    eval_for_len_groups: bool = False

    # If dataset does not exist skip it w/o warning.
    optional: bool = False


@dataclasses.dataclass
class DocRetrievalConf:
    ds_base_dir: str
    datasets: List[DatasetConf]
    enabled_ds: List = dataclasses.field(default_factory=list)

    sim_kind: SimKind = SimKind.COS
    topk: List[int] = dataclasses.field(default_factory=lambda: [3, 5, 10, 20])

    use_gpu: int = -1
    fp16: bool = False

    query_instruction: str = ''

    save_predictions_prefix: str = ''


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


def _key_fmt(key: tuple):
    return f'{str(key[0])}/{key[1]}'


def _save_predictions(
    conf: DocRetrievalConf,
    ds_conf: DatasetConf,
    sim_arr: np.ndarray,
    idx_arr: np.ndarray,
    query_data,
    other_data,
):
    if not conf.save_predictions_prefix:
        return
    max_topk = max(conf.topk)

    query_keys, _ = query_data
    other_keys, _ = other_data

    p = Path(f'{conf.save_predictions_prefix}_{ds_conf.name}.csv')
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as outf:
        outf.write('rank,src,tgt,sim\n')
        for i in range(len(idx_arr)):
            query_key = query_keys[i]
            first_rel_key = other_keys[idx_arr[i][0]]
            if query_key == first_rel_key:
                # this is the same doc
                idxs = idx_arr[i][1:]
                sims = sim_arr[i][1:]
            else:
                idxs = idx_arr[i]
                sims = sim_arr[i]

            for rank, (j, sim) in enumerate(zip(idxs, sims)):
                if rank >= max_topk:
                    break
                outf.write(f'{rank+1},{_key_fmt(query_key)},{_key_fmt(other_keys[j])},{sim}\n')


def _qrel_find_col(header: list[str], col_name: str):
    try:
        return header.index(col_name)
    except ValueError:
        raise RuntimeError(f"Failed to find {col_name} in header '{header}'")


def _encode_text_from_csv(csv_path: Path, instr: str = ''):
    text_list = []
    text_id_list = []
    inv_idx = {}
    with open(csv_path, 'r') as inpf:
        reader = csv.reader(inpf)
        header = next(reader)
        id_col_num = _qrel_find_col(header, 'id')
        text_col_num = _qrel_find_col(header, 'text')
        for num, row in enumerate(reader):
            text_id = row[id_col_num]
            text_id_list.append(text_id)
            inv_idx[text_id] = num

            text = row[text_col_num]
            if instr:
                text = instr + text
            text_list.append(text)
    return (text_id_list, inv_idx), text_list


def _load_qrels(qrel_file_path: Path, query_inv_idx: dict, doc_inv_idx: dict):
    gold = []
    with open(qrel_file_path, 'r') as inpf:
        reader = csv.reader(inpf)
        header = next(reader)
        cur_query_id = ''
        cur_rel = []
        if header[:3] != ['query-id', 'corpus-id', 'score']:
            raise RuntimeError(
                "Header of qrels is expected to start from query-id,corpus-id,score!"
            )
        for qid, cid, score in reader:
            if int(score) != 1:
                continue

            if qid != cur_query_id:
                if cur_rel:
                    gold.append((query_inv_idx[cur_query_id], cur_rel, None))
                cur_query_id = qid
                cur_rel = []
            cur_rel.append(doc_inv_idx[cid])
        if cur_rel:
            gold.append((query_inv_idx[cur_query_id], cur_rel, None))
    return gold


def _eval_qrels_ds(conf: DocRetrievalConf, ds_conf: DatasetConf, doc_encoder: DocEncoder):
    # Collect docs
    base_dir = Path(conf.ds_base_dir)
    corpus_file_path = base_dir / ds_conf.texts
    doc_data, doc_list = _encode_text_from_csv(corpus_file_path)
    # Encode docs
    doc_embs = doc_encoder.encode_docs(doc_list)

    # Collect queries
    assert ds_conf.queries, 'Logic error 839110'
    queries_file_path = base_dir / ds_conf.queries
    query_data, query_list = _encode_text_from_csv(queries_file_path, conf.query_instruction)
    query_embs = doc_encoder.encode_docs(query_list)

    max_k = max(conf.topk) + 1
    sims, indexes = calc_sim(conf.sim_kind, max_k, query_embs, doc_embs, use_gpu=conf.use_gpu)

    _save_predictions(conf, ds_conf, sims, indexes, query_data, doc_data)

    # load gold data
    qrel_file_path = base_dir / ds_conf.meta
    gold = _load_qrels(qrel_file_path, query_data[1], doc_data[1])
    metrics = _calc_metrics(conf, indexes, gold, query_data, doc_data)
    return metrics


def _eval_impl(conf: DocRetrievalConf, ds_conf: DatasetConf, doc_encoder: DocEncoder):
    if ds_conf.queries is not None:
        return _eval_qrels_ds(conf, ds_conf, doc_encoder)

    if conf.query_instruction:
        raise RuntimeError('query_instruction is only supported with datasets in qrels format!')

    base_dir = Path(conf.ds_base_dir)
    abs_query_text_dir, abs_other_text_dir = _find_text_dirs(base_dir / ds_conf.texts)
    query_text_dir = abs_query_text_dir.relative_to(base_dir)
    other_text_dir = abs_other_text_dir.relative_to(base_dir)

    query_ids, other_ids = collect_src_tgt_ids(base_dir / ds_conf.meta)
    other_keys = [_make_key(other_text_dir, o) for o in other_ids]
    if ds_conf.search_over_all_texts:
        other_keys = _add_docs_from_other_dir(ds_conf, base_dir, other_text_dir, other_keys)
        if ds_conf.extra_other_dir:
            other_keys = _add_docs_from_other_dir(
                ds_conf, base_dir, Path(ds_conf.extra_other_dir), other_keys
            )

    other_paths = paths_from_keys(base_dir, other_keys)
    logging.info("computing embeddings for %s docs", len(other_paths))
    other_doc_embs = doc_encoder.encode_docs_from_path_list(other_paths)
    assert len(other_doc_embs) == len(other_paths) == len(other_keys)
    logging.info("other_embs, shape=%s, dtype=%s", other_doc_embs.shape, other_doc_embs.dtype)
    other_inv_idx = _make_keys_dict(base_dir, other_paths)
    other_data = (other_keys, other_inv_idx)

    query_paths = paths_from_ids(abs_query_text_dir, query_ids)
    query_doc_embs = doc_encoder.encode_docs_from_path_list(query_paths)
    assert len(query_paths) == len(query_doc_embs) == len(query_ids)
    logging.info("query_embs, shape=%s, dtype=%s", query_doc_embs.shape, query_doc_embs.dtype)
    if conf.fp16 and query_doc_embs.dtype != np.float16:
        logging.info("convert_embs_dtype, from=%s, to=%s", query_doc_embs.dtype, np.float16)
        query_doc_embs = query_doc_embs.astype(np.float16)
        other_doc_embs = other_doc_embs.astype(np.float16)

    query_inv_idx = _make_keys_dict(base_dir, query_paths)
    query_data = ([_make_key(query_text_dir, i) for i in query_ids], query_inv_idx)

    max_k = max(conf.topk) + 1
    sims, indexes = calc_sim(
        conf.sim_kind, max_k, query_doc_embs, other_doc_embs, use_gpu=conf.use_gpu
    )
    _save_predictions(conf, ds_conf, sims, indexes, query_data, other_data)

    gold_data = _load_gold_data(
        base_dir / ds_conf.meta, query_text_dir, query_data, other_text_dir, other_data
    )
    metrics = _calc_metrics(conf, indexes, gold_data, query_data, other_data)
    if ds_conf.eval_for_len_groups:
        group_metrics = _eval_each_group(conf, indexes, gold_data, query_data, other_data)
        metrics = [metrics] + group_metrics
    return metrics


def _check_ds(conf: DocRetrievalConf, dsconf: DatasetConf):
    base_dir = Path(conf.ds_base_dir)
    if not (base_dir / dsconf.meta).exists():
        if not dsconf.optional:
            logging.warning("%s does not exist. Skip this dataset", base_dir / dsconf.meta)
        return False
    return True


def doc_retrieval_eval(conf: DocRetrievalConf, doc_encoder: DocEncoder):
    results = []
    for dataset in conf.datasets:
        if (conf.enabled_ds and dataset.name not in conf.enabled_ds) or (
            not _check_ds(conf, dataset)
        ):
            continue

        logging.info("Evaling doc retrieval on %s", dataset.name)
        m = _eval_impl(conf, dataset, doc_encoder)
        results.append((dataset.name, m))
    return results
