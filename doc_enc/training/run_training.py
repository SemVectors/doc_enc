#!/usr/bin/env python3

import logging
import sys
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.utils import configure_log
from hydra.core.config_store import ConfigStore

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn as mp_spawn

from doc_enc.tokenizer import create_tokenizer, TokenizerConf
from doc_enc.training.batch_iterator import BatchIterator, BatchIteratorConf
from doc_enc.training.sents_batch_generator import SentsBatchIteratorConf
from doc_enc.training.docs_batch_generator import DocsBatchIteratorConf
from doc_enc.training.trainer import Trainer, TrainerConf
from doc_enc.training.models.model_conf import DocModelConf, SentModelConf
from doc_enc.encoders.enc_config import SentEncoderConf, FragmentEncoderConf, DocEncoderConf

from doc_enc.training.combine_docs_sources import combine_docs_datasets


@dataclass
class Config:
    tokenizer: TokenizerConf
    batches: BatchIteratorConf
    trainer: TrainerConf
    model: DocModelConf

    job_logging: Dict[str, Any]
    verbose: Any = False
    enable_log_for_all_procs: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_tokenizer_config", group="tokenizer", node=TokenizerConf)
cs.store(name="base_batches_config", group="batches", node=BatchIteratorConf)
# cs.store(name="base_sent_batches_config", group="batches/sent", node=SentsBatchIteratorConf)
# cs.store(name="base_doc_batches_config", group="batches/doc", node=DocsBatchIteratorConf)
cs.store(name="base_trainer_config", group="trainer", node=TrainerConf)
cs.store(name="base_model_config", group="model", node=DocModelConf)
cs.store(name="base_sent_model_config", group="model/sent", node=SentModelConf)
cs.store(name="base_sent_encoder_config", group="model/sent/encoder", node=SentEncoderConf)
cs.store(name="base_frag_encoder_config", group="model/fragment", node=FragmentEncoderConf)
cs.store(name="base_doc_encoder_config", group="model/doc", node=DocEncoderConf)


def _init_proc(rank, world_size, conf: Config):
    configure_log(conf.job_logging, conf.verbose)
    if not conf.enable_log_for_all_procs and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    torch.cuda.set_device(rank)


def _destroy_proc():
    dist.destroy_process_group()


def _run_train(rank, world_size, conf: Config):
    _init_proc(rank, world_size, conf)

    train_iter = None
    dev_iter = None
    try:
        vocab = create_tokenizer(conf.tokenizer)
        train_iter = BatchIterator(
            conf.batches,
            conf.tokenizer,
            conf.job_logging,
            split="train",
            rank=rank,
            world_size=world_size,
            pad_idx=vocab.pad_idx(),
        )

        dev_iter = BatchIterator(
            conf.batches,
            conf.tokenizer,
            conf.job_logging,
            split="dev",
            rank=rank,
            world_size=world_size,
            pad_idx=vocab.pad_idx(),
        )

        trainer = Trainer(conf.trainer, conf.model, vocab, world_size, rank, verbose=conf.verbose)
        trainer(train_iter, dev_iter)

    except Exception as e:
        logging.error(e)
        if train_iter is not None:
            train_iter.destroy()
        if dev_iter is not None:
            dev_iter.destroy()
        sys.exit(1)
    finally:
        _destroy_proc()


def _preproc(conf: Config):
    iter_conf = conf.batches.docs_batch_iterator_conf
    gen_conf = iter_conf.batch_generator_conf
    input_dir = gen_conf.input_dir
    prefix = gen_conf.meta_prefix
    try_find_existing_meta = iter_conf.use_existing_combined_meta
    if try_find_existing_meta:
        tp = Path(f"{input_dir}/{prefix}_train.csv")
        dp = Path(f"{input_dir}/{prefix}_dev.csv")
        if tp.exists() and dp.exists():
            logging.info("reusing existing %s and %s", tp, dp)
            return

    logging.info("combining docs datasets. It may take some time...")
    combine_docs_datasets(
        input_dir,
        split="train",
        include_datasets=iter_conf.include_datasets,
        exclude_datasets=iter_conf.exclude_datasets,
        out_filename_prefix=prefix,
        min_doc_len=gen_conf.min_sents_per_doc,
        max_doc_len=gen_conf.max_sents_per_doc,
    )
    logging.info("done with train")
    combine_docs_datasets(
        input_dir,
        split="dev",
        include_datasets=iter_conf.include_datasets,
        exclude_datasets=iter_conf.exclude_datasets,
        out_filename_prefix=prefix,
        min_doc_len=gen_conf.min_sents_per_doc,
        max_doc_len=gen_conf.max_sents_per_doc,
    )


@hydra.main(config_path="conf", config_name="config")
def train_cli(conf: Config) -> None:
    gpu_cnt = torch.cuda.device_count()
    if gpu_cnt <= 0:
        raise RuntimeError("No gpu was found")
    _preproc(conf)
    mp_spawn(_run_train, args=(gpu_cnt, conf), nprocs=gpu_cnt, join=True)


if __name__ == "__main__":
    train_cli()
