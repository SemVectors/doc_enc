#!/usr/bin/env python3

import dataclasses
import logging
from typing import Optional, List
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf
from doc_enc.eval.caching_doc_encoder import CachingDocEncoder
from doc_enc.eval.doc_matching import DocMatchingConf, doc_matching_eval
from doc_enc.eval.doc_retrieval import DocRetrievalConf, doc_retrieval_eval
from doc_enc.eval.sent_retrieval import SentRetrievalConf, sent_retrieval_eval
from doc_enc.eval.bench_model import BenchConf, bench_sents_encoding, bench_docs_encoding


@dataclasses.dataclass
class Config:
    doc_encoder: DocEncoderConf
    eval_doc_matching: bool = True
    doc_matching: DocMatchingConf = MISSING

    eval_doc_retrieval: bool = True
    doc_retrieval: DocRetrievalConf = MISSING

    eval_sent_retrieval: bool = True
    sent_retrieval: SentRetrievalConf = MISSING

    print_as_csv: bool = False

    cache_embeddings: bool = True
    model_id: Optional[str] = None

    eval_checkpoints: List[str] = dataclasses.field(default_factory=list)

    bench_doc_encoding: bool = False
    bench_sent_encoding: bool = False
    bench: BenchConf = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)
cs.store(name="base_doc_matching", group="doc_matching", node=DocMatchingConf)
cs.store(name="base_doc_retrieval", group="doc_retrieval", node=DocRetrievalConf)
cs.store(name="base_sent_retrieval", group="sent_retrieval", node=SentRetrievalConf)
cs.store(name="base_bench", group="bench", node=BenchConf)


def _print_row(ds, metrics_dict, model_id=None, **extra):
    values = list(metrics_dict.values())
    first_values = []
    if model_id is not None:
        first_values = [model_id]
    first_values.extend(extra.values())
    first_values.append(ds)

    values = first_values + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
    print(*values, sep=',')


def _print_results_as_csv(conf: Config, results, **extra):
    if not results:
        return

    first_m = results[0][1]
    if isinstance(first_m, dict):
        metrics = list(first_m.keys())
    elif isinstance(first_m, list):
        metrics = first_m[0].keys()
    else:
        raise RuntimeError("Unknown metrics format")

    header_prefix = ""
    if conf.model_id is not None:
        header_prefix = "model,"
    header_prefix += ','.join(extra.keys())
    if header_prefix and extra:
        header_prefix += ','

    header = f"{header_prefix}ds,{','.join(metrics)}"
    print(header)
    for ds, maybe_list in results:
        if isinstance(maybe_list, list):
            for m in maybe_list:
                _print_row(ds, m, model_id=conf.model_id, **extra)
        else:
            _print_row(ds, maybe_list, model_id=conf.model_id, **extra)


def _print_results(conf: Config, results, **extra):
    for ds, m in results:
        logging.info("Metrics for ds: %s", ds)
        logging.info(extra)
        logging.info(m)


def _eval_and_bench(conf: Config, doc_encoder: DocEncoder, **extra):
    if conf.print_as_csv:
        printer = _print_results_as_csv
    else:
        printer = _print_results

    if conf.eval_doc_retrieval:
        results = doc_retrieval_eval(conf.doc_retrieval, doc_encoder)
        logging.info("doc retrieval results")
        printer(conf, results, **extra)

    if conf.eval_doc_matching:
        results = doc_matching_eval(conf.doc_matching, doc_encoder)
        logging.info("doc matching results")
        printer(conf, results, **extra)

    if conf.eval_sent_retrieval:
        if not doc_encoder.sent_encoding_supported():
            logging.warning(
                "Sent encoding is not supported by this model! Skip evaling sent retrieval"
            )
        else:
            results = sent_retrieval_eval(conf.sent_retrieval, doc_encoder)
            logging.info("sent retrieval results")
            printer(conf, results, **extra)

    if conf.bench_sent_encoding:
        results = bench_sents_encoding(conf.bench, doc_encoder)
        logging.info("sents encoding bench results")
        printer(conf, results, **extra)

    if conf.bench_doc_encoding:
        results = bench_docs_encoding(conf.bench, doc_encoder)
        logging.info("doc encoding bench results")
        printer(conf, results, **extra)


@hydra.main(config_path=None, config_name="config", version_base=None)
def eval_cli(conf: Config) -> None:
    if conf.cache_embeddings:
        if conf.model_id is None:
            raise RuntimeError("You need to specify model id if you use caching doc encoder")
        doc_encoder = CachingDocEncoder(conf.doc_encoder, conf.model_id)
    else:
        doc_encoder = DocEncoder(conf.doc_encoder)

    if not conf.eval_checkpoints:
        return _eval_and_bench(conf, doc_encoder)

    base_dir = Path(conf.doc_encoder.model_path).parent
    for p in conf.eval_checkpoints:
        checkpoints = base_dir.glob(p)
        for cp in checkpoints:
            logging.info("loading parameters from: %s", cp)
            doc_encoder.load_params_from_checkpoint(cp)
            _eval_and_bench(conf, doc_encoder, checkpoint=cp.with_suffix('').name)


if __name__ == "__main__":
    eval_cli()
