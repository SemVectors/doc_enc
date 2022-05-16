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


@dataclasses.dataclass
class Config:
    doc_encoder: DocEncoderConf
    eval_doc_matching: bool = True
    doc_matching: DocMatchingConf = MISSING

    eval_doc_retrieval: bool = True
    doc_retrieval: DocRetrievalConf = MISSING

    print_as_csv: bool = False

    cache_embeddings: bool = True
    model_id: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_doc_matching", group="doc_matching", node=DocMatchingConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


def _print_results_as_csv(results):
    if not results:
        return

    metrics = list(results[0][1].keys())
    header = f"ds,{','.join(metrics)}"
    print(header)
    for ds, m in results:
        values = list(m.values())
        values = [ds] + [f"{v:.3f}" for v in values]
        print(*values, sep=',')


def _print_results(results):
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
        printer(results)

    if conf.eval_doc_matching:
        results = doc_matching_eval(conf.doc_matching, doc_encoder)
        logging.info("doc matching results")
        printer(results)


if __name__ == "__main__":
    eval_cli()
