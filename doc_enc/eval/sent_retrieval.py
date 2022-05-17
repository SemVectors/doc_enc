#!/usr/bin/env python3

import logging
from typing import List, Optional
import dataclasses
from pathlib import Path

from doc_enc.utils import open_file, find_file
from doc_enc.eval.sim_util import calc_sim, SimKind
from doc_enc.doc_encoder import DocEncoder


@dataclasses.dataclass
class DatasetConf:
    name: str
    meta: str
    sents: str


@dataclasses.dataclass
class SentRetrievalConf:
    ds_base_dir: str
    datasets: List[DatasetConf]
    enabled_ds: List = dataclasses.field(default_factory=list)

    sim_kind: SimKind = SimKind.COS
    topk: List[int] = dataclasses.field(default_factory=lambda: [1, 20])
    thresholds: List[float] = dataclasses.field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])

    use_gpu: Optional[int] = None


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


def calc_metrics(predictions, gold):
    ncorrect = len(frozenset(predictions) & frozenset(gold))
    if not ncorrect:
        return {}
    recall = ncorrect / len(gold)
    precision = ncorrect / len(predictions)
    f1 = 2 * recall * precision / (recall + precision)
    return {'rec': recall, 'prec': precision, 'f1': f1}


def _eval_impl(conf: SentRetrievalConf, ds_conf: DatasetConf, doc_encoder: DocEncoder):
    base_dir = Path(conf.ds_base_dir)
    src_sent_ids, src_sents = _read_sents(base_dir / (ds_conf.sents + ".src"))
    logging.info("loaded %d src sents", len(src_sent_ids))
    src_embs = doc_encoder.encode_sents(src_sents)
    logging.info("shape of encoded sents: %s", src_embs.shape)

    tgt_sent_ids, tgt_sents = _read_sents(base_dir / (ds_conf.sents + ".tgt"))
    logging.info("loaded %d tgt sents", len(tgt_sent_ids))
    tgt_embs = doc_encoder.encode_sents(tgt_sents)
    logging.info("shape of encoded sents: %s", tgt_embs.shape)

    max_k = max(conf.topk)
    msim, indexes = calc_sim(conf.sim_kind, max_k, src_embs, tgt_embs)
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


def sent_retrieval_eval(conf: SentRetrievalConf, doc_encoder: DocEncoder):
    results = []
    for dataset in conf.datasets:
        if conf.enabled_ds and dataset.name not in conf.enabled_ds:
            continue
        logging.info("Evaling doc retrieval on %s", dataset.name)
        m = _eval_impl(conf, dataset, doc_encoder)
        results.append((dataset.name, m))
    return results
