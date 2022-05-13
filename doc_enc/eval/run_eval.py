#!/usr/bin/env python3

from dataclasses import dataclass
import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf
from doc_enc.eval.doc_matching import DocMatchingConf, doc_matching_eval
from doc_enc.eval.doc_retrieval import DocRetrievalConf, doc_retrieval_eval


@dataclass
class Config:
    doc_encoder: DocEncoderConf
    eval_doc_matching: bool = True
    doc_matching: DocMatchingConf = MISSING

    eval_doc_retrieval: bool = True
    doc_retrieval: DocRetrievalConf = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_doc_matching", group="doc_matching", node=DocMatchingConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


@hydra.main(config_path="conf", config_name="config")
def eval_cli(conf: Config) -> None:
    doc_encoder = DocEncoder(conf.doc_encoder)
    if conf.eval_doc_matching:
        results = doc_matching_eval(conf.doc_matching, doc_encoder)
        logging.info("doc matching results")
        for ds, m in results.items():
            logging.info("Metrics for ds: %s", ds)
            logging.info(m)
    if conf.eval_doc_retrieval:
        results = doc_retrieval_eval(conf.doc_retrieval, doc_encoder)
        logging.info("doc retrieval results")
        for ds, m in results.items():
            logging.info("Metrics for ds: %s", ds)
            logging.info(m)


if __name__ == "__main__":
    eval_cli()
