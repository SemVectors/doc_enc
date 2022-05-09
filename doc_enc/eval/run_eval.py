#!/usr/bin/env python3

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf
from doc_enc.eval.doc_matching import DocMatchingConf, doc_matching_eval


@dataclass
class Config:
    doc_encoder: DocEncoderConf
    doc_matching: DocMatchingConf


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_doc_matching", group="doc_matching", node=DocMatchingConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


@hydra.main(config_path="conf", config_name="config")
def eval_cli(conf: Config) -> None:
    doc_encoder = DocEncoder(conf.doc_encoder)
    doc_matching_eval(conf.doc_matching, doc_encoder)


if __name__ == "__main__":
    eval_cli()
