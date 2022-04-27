#!/usr/bin/env python3

import logging
from typing import Dict, Any
from dataclasses import dataclass

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


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_tokenizer_config", group="tokenizer", node=TokenizerConf)
cs.store(name="base_batches_config", group="batches", node=BatchIteratorConf)
cs.store(name="base_sent_batches_config", group="batches/sent", node=SentsBatchIteratorConf)
cs.store(name="base_doc_batches_config", group="batches/doc", node=DocsBatchIteratorConf)
cs.store(name="base_trainer_config", group="trainer", node=TrainerConf)
cs.store(name="base_model_config", group="model", node=DocModelConf)
cs.store(name="base_sent_model_config", group="model/sent", node=SentModelConf)
cs.store(name="base_sent_encoder_config", group="model/sent/encoder", node=SentEncoderConf)
cs.store(name="base_frag_encoder_config", group="model/fragment", node=FragmentEncoderConf)
cs.store(name="base_doc_encoder_config", group="model/doc", node=DocEncoderConf)


def _init_proc(rank, world_size, conf: Config):
    configure_log(conf.job_logging, conf.verbose)
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    torch.cuda.set_device(rank)


def _destroy_proc():
    dist.destroy_process_group()


def _run_train(rank, world_size, conf: Config):
    _init_proc(rank, world_size, conf)

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

    if rank == 0:
        dev_iter = BatchIterator(
            conf.batches,
            conf.tokenizer,
            conf.job_logging,
            split="dev",
            rank=0,
            world_size=-1,
            pad_idx=vocab.pad_idx(),
        )
    else:
        dev_iter = None
    trainer = Trainer(conf.trainer, conf.model, vocab, world_size, rank, verbose=conf.verbose)
    trainer(train_iter, dev_iter)

    train_iter.destroy()
    if dev_iter is not None:
        dev_iter.destroy()

    _destroy_proc()


def _preproc(conf: Config):
    logging.info("combining docs datasets. It may take some time...")
    input_dir = conf.batches.docs_batch_iterator_conf.batch_generator_conf.input_dir
    include = conf.batches.docs_batch_iterator_conf.include_datasets
    exclude = conf.batches.docs_batch_iterator_conf.exclude_datasets
    prefix = conf.batches.docs_batch_iterator_conf.batch_generator_conf.meta_prefix
    combine_docs_datasets(
        input_dir,
        split="train",
        include_datasets=include,
        exclude_datasets=exclude,
        out_filename_prefix=prefix,
    )
    logging.info("done with train")
    combine_docs_datasets(
        input_dir,
        split="dev",
        include_datasets=include,
        exclude_datasets=exclude,
        out_filename_prefix=prefix,
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
