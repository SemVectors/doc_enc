#!/usr/bin/env python3

import dataclasses
import logging
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf
from doc_enc.eval.caching_doc_encoder import CachingDocEncoder
from doc_enc.eval.doc_matching import DocMatchingConf, doc_matching_eval
from doc_enc.eval.doc_retrieval import DocRetrievalConf, doc_retrieval_eval
from doc_enc.eval.sent_retrieval import SentRetrievalConf, sent_retrieval_eval


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


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_doc_matching", group="doc_matching", node=DocMatchingConf)
cs.store(name="base_doc_retrieval", group="doc_retrieval", node=DocRetrievalConf)
cs.store(name="base_sent_retrieval", group="sent_retrieval", node=SentRetrievalConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


def _print_row(ds, metrics_dict, model_id=None):
    values = list(metrics_dict.values())
    first_values = []
    if model_id is not None:
        first_values = [model_id]
    first_values.append(ds)

    values = first_values + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
    print(*values, sep=',')


def _print_results_as_csv(conf: Config, results):
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

    header = f"{header_prefix}ds,{','.join(metrics)}"
    print(header)
    for ds, maybe_list in results:
        if isinstance(maybe_list, list):
            for m in maybe_list:
                _print_row(ds, m, model_id=conf.model_id)
        else:
            _print_row(ds, maybe_list, model_id=conf.model_id)


def _print_results(conf: Config, results):
    for ds, m in results:
        logging.info("Metrics for ds: %s", ds)
        logging.info(m)


@hydra.main(config_path="conf", config_name="config")
def eval_cli(conf: Config) -> None:
    if conf.cache_embeddings:
        if conf.model_id is None:
            raise RuntimeError("You need to specify model id if you use caching doc encoder")
        doc_encoder = CachingDocEncoder(conf.doc_encoder, conf.model_id)
    else:
        doc_encoder = DocEncoder(conf.doc_encoder)
    if conf.print_as_csv:
        printer = _print_results_as_csv
    else:
        printer = _print_results

    if conf.eval_doc_retrieval:
        results = doc_retrieval_eval(conf.doc_retrieval, doc_encoder)
        logging.info("doc retrieval results")
        printer(conf, results)

    if conf.eval_doc_matching:
        results = doc_matching_eval(conf.doc_matching, doc_encoder)
        logging.info("doc matching results")
        printer(conf, results)

    if conf.eval_sent_retrieval:
        results = sent_retrieval_eval(conf.sent_retrieval, doc_encoder)
        logging.info("sent retrieval results")
        printer(conf, results)


if __name__ == "__main__":
    eval_cli()
