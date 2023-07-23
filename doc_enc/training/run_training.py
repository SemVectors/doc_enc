#!/usr/bin/env python3

import logging
import sys
import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import random
import datetime

import hydra
from hydra.core.utils import configure_log
from hydra.core.config_store import ConfigStore

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn as mp_spawn

from doc_enc.common_types import EncoderKind
from doc_enc.text_processor import TextProcessorConf, TextProcessor
from doc_enc.tokenizer import TokenizerConf, TokenizerType
from doc_enc.training.batch_iterator import BatchIterator, BatchIteratorConf

from doc_enc.training.train_conf import TrainerConf, OptimConf
from doc_enc.training.trainer import Trainer, BaseTrainerUtils
from doc_enc.training.models.model_conf import DocModelConf, SentModelConf
from doc_enc.encoders.enc_config import SentEncoderConf, SeqEncoderConf, BaseEncoderConf
from doc_enc.training.index.prepare_index_util import prepare_sent_index

from doc_enc.training.combine_docs_sources import combine_docs_datasets
from doc_enc.training.types import TaskType


@dataclass
class Config:
    text_proc: TextProcessorConf
    batches: BatchIteratorConf
    trainer: TrainerConf
    model: DocModelConf

    job_logging: Dict[str, Any]
    verbose: Any = False
    enable_log_for_all_procs: bool = False

    combine_datasets_use_text_proc: bool = False
    force_determinism: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_text_processor_config", group="text_proc", node=TextProcessorConf)
cs.store(name="base_tokenizer_config", group="text_proc/tokenizer", node=TokenizerConf)
cs.store(name="base_batches_config", group="batches", node=BatchIteratorConf)
cs.store(name="base_trainer_config", group="trainer", node=TrainerConf)
cs.store(name="base_optim_config", group="trainer/optim", node=OptimConf)
cs.store(name="base_model_config", group="model", node=DocModelConf)
cs.store(name="base_sent_model_config", group="model/sent", node=SentModelConf)
cs.store(name="base_sent_for_doc_config", group="model/sent_for_doc", node=BaseEncoderConf)
cs.store(name="base_sent_encoder_config", group="model/sent/encoder", node=SentEncoderConf)
cs.store(name="base_frag_encoder_config", group="model/fragment", node=SeqEncoderConf)
cs.store(name="base_doc_encoder_config", group="model/doc", node=SeqEncoderConf)


def _init_dist_default_group(local_rank, local_world_size, port='29500'):
    world_size = int(os.environ.get('TORCH_DIST_WORLD_SIZE', local_world_size))

    if (rank := os.environ.get('TORCH_DIST_RANK')) is not None:
        rank = int(rank)
        rank += local_rank
    else:
        rank = local_rank

    if 'MASTER_ADDR' not in os.environ:
        if rank == 0:
            os.environ['MASTER_ADDR'] = '0.0.0.0'
        else:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = port

    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    timeout = int(os.environ.get("TORCH_DIST_TIMEOUT_MIN", "5"))
    dist.init_process_group(
        'nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=timeout)
    )
    return rank, world_size


def _init_proc(local_rank, local_world_size, conf: Config):
    rank, world_size = _init_dist_default_group(local_rank, local_world_size)
    configure_log(conf.job_logging, conf.verbose)
    if not conf.enable_log_for_all_procs and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if conf.force_determinism:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if (
            conf.batches.docs_batch_iterator_conf.async_generators > 1
            or conf.batches.sents_batch_iterator_conf.async_generators > 1
        ):
            logging.error("set async_generators==1 in all batch_iterators")
            raise RuntimeError("force_determinism is true but some async_generators > 1")

    torch.manual_seed(2022 * 8)
    random.seed(2022 * 9)

    logging.info("Inited proc with rank=%d (ws=%d), local_rank=%d", rank, world_size, local_rank)
    return rank, world_size


def _destroy_proc(world_size):
    if world_size > 1:
        dist.destroy_process_group()


def _is_training_required(conf: Config):
    def _train_enc(c: BaseEncoderConf | None):
        if c is not None and (
            c.encoder_kind != EncoderKind.TRANSFORMERS_AUTO
            or not c.transformers_fix_pretrained_params
        ):
            return True
        return False

    cm = conf.model
    return (
        _train_enc(cm.sent.encoder if cm.sent else None)
        or _train_enc(cm.sent_for_doc)
        or _train_enc(cm.fragment)
        or _train_enc(cm.doc)
    )


def _adjust_config(conf: Config):
    def _get_auto_model_name(c: BaseEncoderConf | None):
        if c is None:
            return None
        if c.encoder_kind == EncoderKind.TRANSFORMERS_AUTO:
            if not c.transformers_auto_name:
                raise RuntimeError(
                    "Encoder kind is TRANSFORMERS_AUTO, but transformers_auto_name is not set"
                )
            return (c.transformers_auto_name, c.transformers_cache_dir)
        raise RuntimeError(
            "TokenizerType is TRANSFORMERS_AUTO, but first encoder is not transformers_auto"
        )

    # adjust text processor options based on model architecture
    if conf.model.sent is None:
        conf.text_proc.split_into_sents = False
        if conf.model.fragment is None:
            conf.text_proc.split_into_fragments = False

    # propagate configs of a model to tokenizer if tokenizer is TRANSFORMERS_AUTO
    tok_conf = conf.text_proc.tokenizer
    if (
        tok_conf.tokenizer_type == TokenizerType.TRANSFORMERS_AUTO
        and not tok_conf.transformers_auto_name
    ):
        # find first transformes auto model and use its name

        cm = conf.model
        for c in [cm.sent.encoder if cm.sent else None, cm.sent_for_doc, cm.fragment, cm.doc]:
            t = _get_auto_model_name(c)
            if t is not None:
                name, cache_dir = t
                tok_conf.transformers_auto_name = name
                tok_conf.transformers_cache_dir = cache_dir
                break
        else:
            raise RuntimeError(
                "TokenizerType is TRANSFORMERS_AUTO, but none of encoders is transformers_auto"
            )

    # check batch options
    docs_gen_conf = conf.batches.docs_batch_iterator_conf.batch_generator_conf
    if (
        not docs_gen_conf.batch_docs_cnt
        and not docs_gen_conf.batch_total_sents_cnt
        and not docs_gen_conf.batch_total_tokens_cnt
    ):
        raise RuntimeError("batch size limits weren't set!")


def _check_training_tasks(conf: Config):
    if not conf.trainer.tasks:
        conf.trainer.tasks = [TaskType.SENT_RETR, TaskType.DOC_RETR]
    if not conf.trainer.eval_tasks:
        conf.trainer.eval_tasks = conf.trainer.tasks

    if conf.model.sent is None:
        if (
            TaskType.SENT_RETR in conf.trainer.tasks
            or TaskType.SENT_RETR in conf.trainer.eval_tasks
        ):
            logging.warning(
                "Sent layer is not presented but SENT_RETR task is set for training."
                " It will be excluded from training tasks."
            )
            conf.trainer.tasks = [t for t in conf.trainer.tasks if t != TaskType.SENT_RETR]
            conf.trainer.eval_tasks = [
                t for t in conf.trainer.eval_tasks if t != TaskType.SENT_RETR
            ]


def _run_train(local_rank, local_world_size, conf: Config):
    if not _is_training_required(conf):
        logging.info("Training is not required (using all pretrained models), just saving model")
        # just export model so it can be loaded by DocEncdoer
        BaseTrainerUtils(conf.trainer, conf.model, conf.text_proc).save_model()
        return

    _check_training_tasks(conf)

    logging.info("local_rank=%s", local_rank)
    rank, world_size = _init_proc(local_rank, local_world_size, conf)

    train_iter = None
    dev_iter = None
    try:
        trainer = Trainer(
            conf.trainer,
            conf.model,
            conf.text_proc,
            world_size=world_size,
            is_master=(rank == 0),
            local_rank=local_rank,
            verbose=conf.verbose,
        )
        if conf.model.fragment is None:
            conf.batches.docs_batch_iterator_conf.batch_generator_conf.pad_fragments_level = False
        pad_to_multiple_of = 0
        if conf.model.sent is not None and conf.model.sent.encoder.attention_window:
            pad_to_multiple_of = max(conf.model.sent.encoder.attention_window)

        batch_device = torch.device(f'cuda:{local_rank}')
        train_iter = BatchIterator(
            conf.batches,
            conf.text_proc,
            (conf.job_logging, conf.verbose),
            split="train",
            rank=rank,
            world_size=world_size,
            device=batch_device,
            pad_idx=trainer.vocab().pad_idx(),
            pad_to_multiple_of=pad_to_multiple_of,
        )

        dev_iter = BatchIterator(
            conf.batches,
            conf.text_proc,
            (conf.job_logging, conf.verbose),
            split="dev",
            rank=rank,
            world_size=world_size,
            device=batch_device,
            pad_idx=trainer.vocab().pad_idx(),
            pad_to_multiple_of=pad_to_multiple_of,
        )

        trainer(train_iter, dev_iter)

    except Exception as e:
        logging.exception(e)
        if train_iter is not None:
            train_iter.destroy()
        if dev_iter is not None:
            dev_iter.destroy()
        sys.exit(1)
    finally:
        _destroy_proc(world_size)


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

    text_proc = None
    min_doc_len = gen_conf.min_sents_per_doc
    max_doc_len = gen_conf.max_sents_per_doc
    if conf.combine_datasets_use_text_proc:
        text_proc = TextProcessor(conf.text_proc)
        if not text_proc.conf().split_into_sents:
            # Do not filter by doc length
            min_doc_len = 0
            max_doc_len = float('inf')

    combine_docs_datasets(
        input_dir,
        split="train",
        text_proc=text_proc,
        include_datasets=iter_conf.include_datasets,
        exclude_datasets=iter_conf.exclude_datasets,
        out_filename_prefix=prefix,
        min_doc_len=min_doc_len,
        max_doc_len=max_doc_len,
        procs=iter_conf.combine_procs_cnt,
    )
    logging.info("done with train")
    combine_docs_datasets(
        input_dir,
        split="dev",
        text_proc=text_proc,
        include_datasets=iter_conf.include_datasets,
        exclude_datasets=iter_conf.exclude_datasets,
        out_filename_prefix=prefix,
        min_doc_len=min_doc_len,
        max_doc_len=max_doc_len,
        procs=iter_conf.combine_procs_cnt,
    )
    logging.info("done with dev")


def _prepare_indexes(conf: Config):
    if conf.model.sent is not None and conf.model.sent.index.enable:
        prepare_sent_index(
            conf.model.sent,
            conf.batches.sents_batch_iterator_conf.batch_generator_conf,
        )


@hydra.main(config_path=None, config_name="config", version_base=None)
def train_cli(conf: Config) -> None:
    gpu_cnt = torch.cuda.device_count()
    if gpu_cnt <= 0:
        raise RuntimeError("No gpu was found")

    _adjust_config(conf)
    _preproc(conf)
    _prepare_indexes(conf)
    try:
        mp_spawn(_run_train, args=(gpu_cnt, conf), nprocs=gpu_cnt, join=True)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


@hydra.main(config_path=None, config_name="config", version_base=None)
def preproc_cli(conf: Config) -> None:
    _preproc(conf)
    _prepare_indexes(conf)


@hydra.main(config_path=None, config_name="config", version_base=None)
def repack_cli(conf: Config) -> None:
    _adjust_config(conf)
    if _is_training_required(conf):
        raise RuntimeError("Training is required for a given config")

    BaseTrainerUtils(conf.trainer, conf.model, conf.text_proc).save_model()


if __name__ == "__main__":
    train_cli()
