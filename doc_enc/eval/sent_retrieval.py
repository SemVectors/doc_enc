#!/usr/bin/env python3

import logging
import dataclasses
from pathlib import Path
import collections

import numpy as np

from doc_enc.utils import open_file, find_file
from doc_enc.eval.sim_util import calc_sim, SimKind
from doc_enc.doc_encoder import DocEncoder


@dataclasses.dataclass
class DatasetConf:
    name: str
    meta: str
    sents: str
    optional: bool = False


@dataclasses.dataclass
class SentRetrievalConf:
    ds_base_dir: str
    datasets: list[DatasetConf]
    enabled_ds: list = dataclasses.field(default_factory=list)

    sim_kind: SimKind = SimKind.COS
    topk: list[int] = dataclasses.field(default_factory=lambda: [1, 20])
    thresholds: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.4, 0.5])

    use_gpu: int = -1

    query_instruction: str = ''
    fp16: bool = False

    save_predictions_prefix: str = ''


def _read_sents(sent_file):
    sent_ids = []
    sents = []
    with open_file(find_file(sent_file)) as fp:
        for l in fp:
            sent_id, sent = l.rstrip().split('\t', 1)
            sent_ids.append(sent_id)
            sents.append(sent)
    return sent_ids, sents


def load_gold_data(meta_path: Path):
    with open(meta_path, 'r', encoding='utf8') as f:
        return [tuple(l.rstrip().split('\t')) for l in f]


def make_predictions(threshold, msim, midx, src_ids, tgt_ids):
    predictions = []
    f = msim > threshold
    for i in range(len(midx)):
        idxs = midx[i][f[i]]
        for j in idxs:
            predictions.append((src_ids[i], tgt_ids[j]))

    return predictions


def _calc_map(predictions, gold_set):
    cum_ap = 0.0
    tp = 0
    ap = 0.0
    rank = 0

    gold_cnt_dict = collections.Counter()
    for src_id, _ in gold_set:
        gold_cnt_dict[src_id] += 1

    cur_src_id = predictions[0][0]
    for src_id, tgt_id in predictions:
        if cur_src_id != src_id:
            cum_ap += ap / gold_cnt_dict[src_id]
            tp = 0
            ap = 0.0
            rank = 0
            cur_src_id = src_id
        rel = int((src_id, tgt_id) in gold_set)
        tp += rel
        ap += rel * tp / (rank + 1)
        rank += 1
    cum_ap += ap / gold_cnt_dict[cur_src_id]

    return cum_ap / len(gold_cnt_dict)


def calc_metrics(predictions, gold):
    if not predictions:
        return {}
    gold_set = frozenset(gold)
    ncorrect = len(frozenset(predictions) & gold_set)
    if not ncorrect:
        return {}

    logging.info(
        "SentRetrieval: ngolds=%d, ncorrect=%d, npredicted=%d",
        len(gold),
        ncorrect,
        len(predictions),
    )
    recall = ncorrect / len(gold)
    precision = ncorrect / len(predictions)
    f1 = 2 * recall * precision / (recall + precision)
    return {'MAP': _calc_map(predictions, gold_set), 'rec': recall, 'prec': precision, 'f1': f1}


def _create_gen_queries_with_instr(src_file_path: Path, instr: str, sep: str = '\t'):

    def _gen():
        with open(src_file_path, 'r') as inpf:
            for line in inpf:

                sent_id, sent = line.rstrip().split(sep, 1)
                yield sent_id, instr + sent

    return _gen


def _encode_srcs(src_file_path: Path, conf: SentRetrievalConf, doc_encoder: DocEncoder):
    """This is terminology of translations mining (from MT): src == queries,
    target == docs/texts."""
    if not conf.query_instruction:
        src_sent_ids, src_embs = doc_encoder.encode_sents_from_file(
            src_file_path, first_column_is_id=True
        )
    else:
        src_sent_ids = []
        all_src_embs = []
        gens = [_create_gen_queries_with_instr(src_file_path, conf.query_instruction)]
        for ids, embs in doc_encoder.encode_sents_from_generators(gens):
            src_sent_ids.extend(ids)
            all_src_embs.append(embs)
        src_embs = np.vstack(all_src_embs)
        assert len(src_sent_ids) == src_embs.shape[0], "Missaligned data 83292"
    return src_sent_ids, src_embs


def _save_predictions(
    conf: SentRetrievalConf,
    ds_conf: DatasetConf,
    sim_arr: np.ndarray,
    idx_arr: np.ndarray,
    src_sent_ids,
    tgt_sent_ids,
):
    if not conf.save_predictions_prefix:
        return
    max_topk = max(conf.topk)
    min_sim = min(conf.thresholds)

    p = Path(f'{conf.save_predictions_prefix}_{ds_conf.name}.csv')
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as outf:
        outf.write('rank,src,tgt,sim\n')
        for i in range(len(idx_arr)):
            for rank, (j, sim) in enumerate(zip(idx_arr[i], sim_arr[i])):
                if sim < min_sim or rank >= max_topk:
                    break
                outf.write(f'{rank+1},{src_sent_ids[i]},{tgt_sent_ids[j]},{sim}\n')


def _eval_impl(conf: SentRetrievalConf, ds_conf: DatasetConf, doc_encoder: DocEncoder):
    base_dir = Path(conf.ds_base_dir)

    src_file_path = base_dir / (ds_conf.sents + ".src")
    logging.info("encoding src")
    src_sent_ids, src_embs = _encode_srcs(src_file_path, conf, doc_encoder)
    logging.info("src_encoded, shape=%s, dtype=%s", src_embs.shape, src_embs.dtype)

    tgt_file_path = base_dir / (ds_conf.sents + ".tgt")
    logging.info("encoding tgt")
    tgt_sent_ids, tgt_embs = doc_encoder.encode_sents_from_file(
        tgt_file_path, first_column_is_id=True
    )
    logging.info("tgt_encoded, shape=%s, dtype=%s", tgt_embs.shape, tgt_embs.dtype)

    if conf.fp16 and src_embs.dtype != np.float16:
        logging.info("convert_embs_dtype, from=%s, to=%s", src_embs.dtype, np.float16)
        src_embs = src_embs.astype(np.float16)
        tgt_embs = tgt_embs.astype(np.float16)

    max_k = max(conf.topk)
    msim, indexes = calc_sim(conf.sim_kind, max_k, src_embs, tgt_embs, use_gpu=conf.use_gpu)
    _save_predictions(conf, ds_conf, msim, indexes, src_sent_ids, tgt_sent_ids)
    gold = load_gold_data(base_dir / ds_conf.meta)
    metrics = []
    for k in conf.topk:
        for threshold in conf.thresholds:
            k_found_sims = msim[:, :k]
            k_found_idxs = indexes[:, :k]
            predictions = make_predictions(
                threshold, k_found_sims, k_found_idxs, src_sent_ids, tgt_sent_ids
            )
            m = calc_metrics(predictions, gold)
            m.update({'k': k, 'min_sim': threshold})
            metrics.append(m)
    return metrics


def _check_ds(conf: SentRetrievalConf, dsconf: DatasetConf):
    base_dir = Path(conf.ds_base_dir)
    if not (base_dir / dsconf.meta).exists():
        if not dsconf.optional:
            logging.warning("%s does not exist. Skip this dataset", base_dir / dsconf.meta)
        return False
    return True


def sent_retrieval_eval(conf: SentRetrievalConf, doc_encoder: DocEncoder):
    results = []
    for dataset in conf.datasets:
        if (conf.enabled_ds and dataset.name not in conf.enabled_ds) or (
            not _check_ds(conf, dataset)
        ):
            continue
        logging.info("Evaling sent retrieval on %s", dataset.name)
        m = _eval_impl(conf, dataset, doc_encoder)
        results.append((dataset.name, m))
    return results
