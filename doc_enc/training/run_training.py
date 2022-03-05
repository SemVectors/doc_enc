#!/usr/bin/env python3

import logging
from dataclasses import dataclass
import os

import hydra
from hydra.core.config_store import ConfigStore

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from doc_enc.tokenizer import create_tokenizer, TokenizerConf
from doc_enc.training.sents_batch_generator import SentsBatchIterator, SentsBatchIteratorConf
from doc_enc.training.trainer import Trainer, TrainerConf
from doc_enc.training.models.model_conf import ModelConf, SentModelConf
from doc_enc.encoders.enc_config import SentEncoderConf


@dataclass
class Config:
    tokenizer: TokenizerConf
    sent_batches: SentsBatchIteratorConf
    trainer: TrainerConf
    model: ModelConf

    verbose: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_tokenizer_config", group="tokenizer", node=TokenizerConf)
cs.store(name="base_sent_batches_config", group="sent_batches", node=SentsBatchIteratorConf)
cs.store(name="base_trainer_config", group="trainer", node=TrainerConf)
cs.store(name="base_model_config", group="model", node=ModelConf)
cs.store(name="base_sent_model_config", group="model/sent", node=SentModelConf)
cs.store(name="base_sent_encoder_config", group="model/sent/encoder", node=SentEncoderConf)


def _init_proc(rank, world_size, opts, port='29500'):
    if rank == 0:
        logging.basicConfig(
            level=logging.DEBUG if opts.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
        )
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _destroy_proc():
    dist.destroy_process_group()


def _run_train(rank, world_size, conf: Config):
    _init_proc(rank, world_size, conf)

    vocab = create_tokenizer(conf.tokenizer)
    train_iter = SentsBatchIterator(
        conf.sent_batches,
        conf.tokenizer,
        split="train",
        rank=rank,
        world_size=world_size,
        pad_idx=vocab.pad_idx(),
    )

    if rank == 0:
        dev_iter = SentsBatchIterator(
            conf.sent_batches,
            conf.tokenizer,
            split="dev",
            rank=0,
            world_size=-1,
            pad_idx=vocab.pad_idx(),
        )
    else:
        dev_iter = None
    trainer = Trainer(conf.trainer, conf.model, vocab, world_size, rank)
    trainer(train_iter, dev_iter)
    _destroy_proc()


@hydra.main(config_path="conf", config_name="config")
def train_cli(opts: Config) -> None:
    gpu_cnt = torch.cuda.device_count()
    if gpu_cnt <= 0:
        raise RuntimeError("No gpu was found")
    mp.spawn(_run_train, args=(gpu_cnt, opts), nprocs=gpu_cnt, join=True)


if __name__ == "__main__":
    train_cli()
