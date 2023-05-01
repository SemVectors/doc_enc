#!/usr/bin/env python3

import copy
import os
import datetime
import contextlib
from enum import Enum
from typing import List, Optional, Dict, NamedTuple
import dataclasses
import json
from pathlib import Path
import multiprocessing
import logging

import pkg_resources  # part of setuptools
from omegaconf import MISSING

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.checkpoint

from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRoOptim
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

from doc_enc.training.batch_iterator import BatchIterator
from doc_enc.text_processor import TextProcessorConf, TextProcessor
from doc_enc.training.types import DocRetrLossType, TaskType, SentRetrLossType
from doc_enc.training.models.model_factory import create_models
from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.models.base_sent_model import BaseSentModel
from doc_enc.training.models.base_doc_model import BaseDocModel
from doc_enc.training.models.base_model import DualEncModelOutput

from doc_enc.training.index.train_util import (
    calculate_ivf_pq_loss,
    combine_dense_and_index_loss,
    update_doc_index_model,
    update_sent_index_model,
)
from doc_enc.training.index.prepare_index_util import re_add_sent_vectors

from doc_enc.training.metrics import create_metrics


class LRSchedulerKind(Enum):
    NONE = 0
    LINEAR = 1
    MULT = 2
    CYCLIC = 3
    ONE_CYCLE = 4


class OptimKind(Enum):
    SGD = 1
    ADAM = 2
    RADAM = 3
    NADAM = 4
    ADAMW = 5


class Models(NamedTuple):
    sent_model: BaseSentModel
    doc_model: BaseDocModel


@dataclasses.dataclass
class ParamGroupConf:
    lr: Optional[float] = None
    final_lr: Optional[float] = None
    weight_decay: Optional[float] = None
    momentum: Optional[float] = None

    max_grad_norm: Optional[float] = None


@dataclasses.dataclass
class OptimConf:
    max_grad_norm: float = 0.0

    optim_kind: OptimKind = OptimKind.ADAM
    weight_decay: float = 0.0
    # for SGD
    momentum: float = 0.9
    use_zero_optim: bool = False

    # LR
    lr: float = MISSING
    final_lr: float = 0.0
    emb: ParamGroupConf = ParamGroupConf()
    sent: ParamGroupConf = ParamGroupConf()
    sent_for_doc: ParamGroupConf = ParamGroupConf()
    sent_index: ParamGroupConf = ParamGroupConf()
    fragment: ParamGroupConf = ParamGroupConf()
    doc: ParamGroupConf = ParamGroupConf()
    doc_index: ParamGroupConf = ParamGroupConf()

    lr_scheduler: LRSchedulerKind = LRSchedulerKind.NONE
    lr_scheduler_kwargs: Dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainerConf:
    tasks: List[TaskType] = dataclasses.field(default_factory=list)
    eval_tasks: List[TaskType] = dataclasses.field(default_factory=list)

    optim: OptimConf = MISSING
    max_updates: int = MISSING

    sent_retr_loss_type: SentRetrLossType = SentRetrLossType.BICE
    doc_retr_loss_type: DocRetrLossType = DocRetrLossType.CE

    save_path: str = ''
    resume_checkpoint: str = ''
    restore_lr_scheduler_state: bool = True

    use_grad_checkpoint: bool = False
    emb_grad_scale: float = 0.0

    switch_tasks_every: int = 10
    log_every: int = 100
    eval_every: int = 300_000
    checkpoint_every: int = 200_000

    debug_iters: List[int] = dataclasses.field(default_factory=list)
    print_batches: bool = False
    print_gpu_memory_stat_every: int = 0
    spam_allocated_memory_info: bool = False


def _create_lr_scheduler(conf: TrainerConf, optimizer):
    optim_conf = conf.optim
    if optim_conf.lr_scheduler == LRSchedulerKind.NONE:
        return None
    approx_iters = conf.max_updates // conf.switch_tasks_every
    max_lr, base_lr = _create_lr_lists(optim_conf, optimizer.param_groups)
    if optim_conf.lr_scheduler == LRSchedulerKind.MULT:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1 / 3, total_iters=approx_iters
        )
    if optim_conf.lr_scheduler == LRSchedulerKind.LINEAR:
        return LinearLRSchedule(optim_conf, optimizer, approx_iters)

    if optim_conf.lr_scheduler == LRSchedulerKind.CYCLIC:
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr, max_lr, **optim_conf.lr_scheduler_kwargs
        )
    if optim_conf.lr_scheduler == LRSchedulerKind.ONE_CYCLE:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, total_steps=approx_iters, **optim_conf.lr_scheduler_kwargs
        )

    raise RuntimeError(f"Unknown lr scheduler: {optim_conf.lr_scheduler}")


def _create_optimizer(conf: OptimConf, models: Models, world_size):
    def _init_gr(group, default, conf, no_decay=False):
        if no_decay:
            weight_decay = 0
        else:
            weight_decay = default.weight_decay if conf.weight_decay is None else conf.weight_decay

        group.update(
            {
                'lr': default.lr if conf.lr is None else conf.lr,
                'weight_decay': weight_decay,
                'momentum': default.momentum if conf.momentum is None else conf.momentum,
                'max_grad_norm': default.max_grad_norm
                if conf.max_grad_norm is None
                else conf.max_grad_norm,
            }
        )
        return group

    logging.info("create optimizer %s", conf.optim_kind)

    # https://deci.ai/deep-learning-glossary/zero-weight-decay-on-batchnorm-and-bias/
    no_decay = ['bias', 'norm.weight', 'LayerNorm.weight']

    def _no_decay_params(named_params, inverse=False):
        for k, p in named_params:
            is_no_decay = any(nd in k for nd in no_decay)
            if inverse:
                is_no_decay = not is_no_decay
            if is_no_decay:
                yield p

    param_groups = [
        _init_gr(
            {
                'name': 'emb',
                'params': _no_decay_params(models.sent_model.encoder.emb_named_params(), True),
            },
            conf,
            conf.emb,
        ),
        _init_gr(
            {
                'name': 'emb.no_decay',
                'params': _no_decay_params(models.sent_model.encoder.emb_named_params()),
            },
            conf,
            conf.emb,
            no_decay=True,
        ),
        _init_gr(
            {
                'name': 'sent',
                'params': _no_decay_params(models.sent_model.encoder.enc_named_params(), True),
            },
            conf,
            conf.sent,
        ),
        _init_gr(
            {
                'name': 'sent.no_decay',
                'params': _no_decay_params(models.sent_model.encoder.enc_named_params()),
            },
            conf,
            conf.sent,
            no_decay=True,
        ),
        _init_gr(
            {
                'name': 'doc',
                'params': _no_decay_params(models.doc_model.doc_encoder.named_parameters(), True),
            },
            conf,
            conf.doc,
        ),
        _init_gr(
            {
                'name': 'doc.no_decay',
                'params': _no_decay_params(models.doc_model.doc_encoder.named_parameters()),
            },
            conf,
            conf.doc,
            no_decay=True,
        ),
    ]

    if models.doc_model.frag_encoder is not None:
        param_groups.append(
            _init_gr(
                {
                    'name': 'frag',
                    'params': _no_decay_params(
                        models.doc_model.frag_encoder.named_parameters(), True
                    ),
                },
                conf,
                conf.fragment,
            )
        )
        param_groups.append(
            _init_gr(
                {
                    'name': 'frag.no_decay',
                    'params': _no_decay_params(models.doc_model.frag_encoder.named_parameters()),
                },
                conf,
                conf.fragment,
                no_decay=True,
            )
        )
    sent_for_doc = models.doc_model.sent_encoder.doc_mode_encoder
    if sent_for_doc is not None:
        param_groups.append(
            _init_gr(
                {
                    'name': 'sent_for_doc',
                    'params': _no_decay_params(sent_for_doc.named_parameters(), True),
                },
                conf,
                conf.sent_for_doc,
            )
        )
        param_groups.append(
            _init_gr(
                {
                    'name': 'sent_for_doc.no_decay',
                    'params': _no_decay_params(sent_for_doc.named_parameters()),
                },
                conf,
                conf.sent_for_doc,
                no_decay=True,
            )
        )

    if models.sent_model.index is not None:
        param_groups.append(
            _init_gr(
                {
                    'name': 'sent.index',
                    'params': models.sent_model.index.parameters(),
                },
                conf,
                conf.sent_index,
                no_decay=True,
            )
        )
    if models.doc_model.index is not None:
        param_groups.append(
            _init_gr(
                {
                    'name': 'doc.index',
                    'params': models.doc_model.index.parameters(),
                },
                conf,
                conf.doc_index,
                no_decay=True,
            )
        )

    if conf.optim_kind == OptimKind.SGD:
        return torch.optim.SGD(
            param_groups, lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay
        )

    if conf.optim_kind == OptimKind.ADAM:
        adam_cls = torch.optim.Adam
    elif conf.optim_kind == OptimKind.RADAM:
        adam_cls = torch.optim.RAdam
    elif conf.optim_kind == OptimKind.NADAM:
        adam_cls = torch.optim.NAdam
    elif conf.optim_kind == OptimKind.ADAMW:
        adam_cls = torch.optim.AdamW
    else:
        raise RuntimeError(f"Unknown optimizer kind: {conf.optim_kind}")
    # TODO ZeRoOptim does not support param_groups in 1.11
    # Its fixed in 1.12 (no)
    if world_size > 1 and conf.use_zero_optim:
        logging.info("creating ZeRoOptim optimizer")
        return ZeRoOptim(param_groups, adam_cls, lr=conf.lr)

    return adam_cls(param_groups, lr=conf.lr, weight_decay=conf.weight_decay)


def _create_lr_lists(conf: OptimConf, param_groups):
    def _v(v, d):
        return v if v is not None else d

    n = len(param_groups)

    if n == 1:
        # its only one lr
        final_lr = [conf.final_lr]
        lr = [conf.lr]
    elif 1 < n < 13:
        final_lr = [
            _v(conf.emb.final_lr, conf.final_lr),
            _v(conf.emb.final_lr, conf.final_lr),
            _v(conf.sent.final_lr, conf.final_lr),
            _v(conf.sent.final_lr, conf.final_lr),
            _v(conf.doc.final_lr, conf.final_lr),
            _v(conf.doc.final_lr, conf.final_lr),
        ]
        lr = [
            _v(conf.emb.lr, conf.lr),
            _v(conf.emb.lr, conf.lr),
            _v(conf.sent.lr, conf.lr),
            _v(conf.sent.lr, conf.lr),
            _v(conf.doc.lr, conf.lr),
            _v(conf.doc.lr, conf.lr),
        ]

        other_n = n - len(lr)
        while other_n:
            pg = param_groups[-other_n]
            if pg['name'].startswith('frag'):
                final_lr.append(_v(conf.fragment.final_lr, conf.final_lr))
                lr.append(_v(conf.fragment.lr, conf.lr))
            elif pg['name'].startswith('sent_for_doc'):
                final_lr.append(_v(conf.sent_for_doc.final_lr, conf.final_lr))
                lr.append(_v(conf.sent_for_doc.lr, conf.lr))
            elif pg['name'] == 'sent.index':
                final_lr.append(_v(conf.sent_index.final_lr, conf.final_lr))
                lr.append(_v(conf.sent_index.lr, conf.lr))
            elif pg['name'] == 'doc.index':
                final_lr.append(_v(conf.doc_index.final_lr, conf.final_lr))
                lr.append(_v(conf.doc_index.lr, conf.lr))
            else:
                raise RuntimeError(f"Unknown name of param group {pg['name']}")
            other_n -= 1

    else:
        raise RuntimeError(f"Unsupported # of param groups: {n}")

    return lr, final_lr


class LinearLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opts: OptimConf, optimizer, total_iters, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

        self._conf = opts

        b = self.base_lrs
        _, self._final_lrs = _create_lr_lists(opts, optimizer.param_groups)
        self._steps = [(l - f) / total_iters for l, f in zip(b, self._final_lrs)]
        logging.info("step lrs: %s", self._steps)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logging.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] - s for group, s in zip(self.optimizer.param_groups, self._steps)]


class Trainer:
    def __init__(
        self,
        opts: TrainerConf,
        model_conf: DocModelConf,
        tp_conf: TextProcessorConf,
        world_size,
        is_master,
        local_rank,
        amp=True,
        verbose=False,
    ):
        self._conf = opts
        if not self._conf.save_path:
            self._conf.save_path = os.getcwd()

        self._model_conf = model_conf
        self._is_master = is_master
        self._world_size = world_size
        self._device = torch.device(f'cuda:{local_rank}')
        self._amp_enabled = amp
        self._verbose = verbose

        self._run_id = self._create_run_id()

        self._tp_conf = tp_conf
        self._tp = TextProcessor(tp_conf)
        self._local_models = self._create_models(model_conf)

        if world_size > 1:
            logging.info("Creating DistributedDataParallel instance")
            # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113/2
            torch.cuda.set_device(local_rank)
            self._sync_group = dist.new_group(backend='gloo', timeout=datetime.timedelta(minutes=1))
            self._sent_model: nn.Module = DDP(
                self._local_models.sent_model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
                static_graph=True,
            )
            self._doc_model: nn.Module = DDP(
                self._local_models.doc_model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
                static_graph=True,
            )
            self._uneven_input_handling = Join
        else:
            logging.info("Skip creating DistributedDataParallel since world size == 1")
            self._sync_group = None
            self._doc_model: nn.Module = self._local_models.doc_model
            self._sent_model: nn.Module = self._local_models.sent_model
            self._uneven_input_handling = contextlib.nullcontext

        self._sent_retr_criterion = torch.nn.CrossEntropyLoss()
        self._doc_retr_criterion = torch.nn.CrossEntropyLoss()

        self._optimizer = _create_optimizer(opts.optim, self._local_models, world_size)

        self._scheduler = _create_lr_scheduler(opts, self._optimizer)
        self._scaler = GradScaler(enabled=amp)
        self._num_updates = 0
        self._last_eval_update = 0
        self._last_checkpoint_update = 0
        self._best_metric = 0.0

        self._init_epoch = 1
        if self._conf.resume_checkpoint:
            self._init_epoch = self._load_from_checkpoint()

    def _log_current_mem_usage(self, prefix=''):
        if self._conf.spam_allocated_memory_info:
            mem = torch.cuda.memory_allocated(self._device)
            logging.info("%s mem usage: %.3fMb", prefix, mem / 1024 / 1024)

    def vocab(self):
        return self._tp.vocab()

    def _create_run_id(self):
        # hydra sets working dir to outputs/<date>/<time> by default
        cwd = os.getcwd()
        bn = os.path.basename
        dn = os.path.dirname
        return f"{bn(dn(cwd))}_{bn(cwd)}"

    def _create_models(self, model_conf: DocModelConf) -> Models:
        sent_model, doc_model = create_models(model_conf, self.vocab(), self._device)
        self._log_current_mem_usage('start')
        sent_model = sent_model.to(self._device)
        doc_model = doc_model.to(self._device)
        self._log_current_mem_usage('after loading')
        logging.info("created model %s", doc_model)
        logging.info("model loaded to %s", self._device)
        mem_params = sum(
            param.nelement() * param.element_size() for param in doc_model.parameters()
        )
        mem_bufs = sum(buf.nelement() * buf.element_size() for buf in doc_model.buffers())
        logging.info("Model params mem usage %.3f Mb", mem_params / 1024 / 1024)
        logging.info("Model buf mem usage %.3f Kb", mem_bufs / 1024)
        return Models(sent_model=sent_model, doc_model=doc_model)

    def _calc_sent_retr_loss(self, output: DualEncModelOutput, labels, batch_size):
        sim_matrix = output.dense_score_matrix
        if self._conf.sent_retr_loss_type == SentRetrLossType.CE:
            dense_loss = self._sent_retr_criterion(sim_matrix, labels)
        elif self._conf.sent_retr_loss_type == SentRetrLossType.BICE:
            loss_src = self._sent_retr_criterion(sim_matrix, labels)
            loss_tgt = self._sent_retr_criterion(sim_matrix.t()[:batch_size], labels)
            dense_loss = loss_src + loss_tgt
        else:
            raise RuntimeError("Logic error 987")

        conf = self._local_models.sent_model.conf.index
        ivf_loss, pq_loss = calculate_ivf_pq_loss(conf, output, labels)
        if ivf_loss is not None:
            loss = combine_dense_and_index_loss(
                conf, dense_loss, ivf_loss, pq_loss, self._world_size
            )
        else:
            loss = dense_loss

        return loss, (dense_loss, ivf_loss, pq_loss)

    def _calc_doc_retr_loss(self, output, labels, batch_size):
        if self._conf.doc_retr_loss_type == DocRetrLossType.CE:
            dense_loss = self._doc_retr_criterion(output, labels)
        else:
            raise RuntimeError("Logic error 988")

        conf = self._local_models.doc_model.conf.index
        ivf_loss, pq_loss = calculate_ivf_pq_loss(conf, output, labels)
        if ivf_loss is not None:
            loss = combine_dense_and_index_loss(
                conf, dense_loss, ivf_loss, pq_loss, self._world_size
            )
        else:
            loss = dense_loss
        return loss, (dense_loss, ivf_loss, pq_loss)

    def _calc_loss_and_metrics(self, task: TaskType, output: DualEncModelOutput, labels, batch):
        if task == TaskType.SENT_RETR:
            loss, losses_tuple = self._calc_sent_retr_loss(output, labels, batch.info['bs'])
        elif task == TaskType.DOC_RETR:
            loss, losses_tuple = self._calc_doc_retr_loss(output, labels, batch.info['bs'])
        else:
            raise RuntimeError(f"Unknown task in calc loss: {task}")

        m = create_metrics(task)
        m.update_metrics(loss.item(), losses_tuple, output, labels, batch)
        return loss, m

    def _save_debug_info(self, batch, output: DualEncModelOutput, labels, meta):
        if meta['task'] == TaskType.SENT_RETR:
            meta['task'] = "sent_retr"
            self._save_retr_debug_info(batch, output, labels, meta)
        elif meta['task'] == TaskType.DOC_RETR:
            meta['task'] = "doc_retr"
            self._save_doc_retr_debug_info(batch, output, labels, meta)
        else:
            raise RuntimeError("Logic error 342")

    def _save_doc_retr_debug_info(self, batch, output: DualEncModelOutput, labels, meta):
        score_matrix = output.dense_score_matrix
        maxk = min(5, score_matrix.size(1))
        values, indices = torch.topk(score_matrix, maxk, 1)
        meta['num_updates'] = self._num_updates
        meta['avg_src_len'] = meta['asl']
        del meta['asl']
        meta['avg_tgt_len'] = meta['atl']
        del meta['atl']

        unscale_factor = 1 / self._model_conf.scale if self._model_conf.scale else 1.0
        examples = []
        for i in range(batch.info['bs']):
            obj = {'src_batch_num': i, 'src_id': batch.src_ids[i], 'found': []}

            for v, idx in zip(values[i], indices[i]):
                if idx >= len(batch.tgt_ids):
                    # idx from other device
                    tgt_id = -1
                else:
                    tgt_id = batch.tgt_ids[idx]

                obj['found'].append(
                    {
                        'tgt_batch_num': idx.item(),
                        'tgt_id': tgt_id,
                        'sim': v.item(),
                        'sim_unscaled': v.item() * unscale_factor,
                    }
                )
            gold = []
            for pidx in batch.positive_idxs[i]:
                usim = score_matrix[i][pidx].item() * unscale_factor

                gold.append(
                    {
                        'tgt_batch_num': pidx,
                        'tgt_id': batch.tgt_ids[pidx],
                        'sim': score_matrix[i][pidx].item(),
                        'sim_unscaled': usim,
                        'sim_unscaled_wo_margin': usim + self._model_conf.margin,
                    }
                )
            obj['gold'] = gold
            examples.append(obj)
        meta['examples'] = examples
        meta_path = Path(self._conf.save_path) / 'doc_retr_debug_batches.jsonl'
        with open(meta_path, 'a', encoding='utf8') as f:
            f.write(json.dumps(meta))
            f.write('\n')

    def _save_retr_debug_info(self, batch, output: DualEncModelOutput, _, meta):
        self._save_sent_retr_debug_info_impl(
            self._model_conf.sent,
            batch,
            output.dense_score_matrix,
            copy.copy(meta),
            Path(self._conf.save_path) / 'sent_dense_retr_debug_batches.jsonl',
        )
        if output.ivf_score_matrix is not None:
            self._save_sent_retr_debug_info_impl(
                self._model_conf.sent.index.ivf,
                batch,
                output.ivf_score_matrix,
                copy.copy(meta),
                Path(self._conf.save_path) / 'sent_ivf_retr_debug_batches.jsonl',
            )
        if output.pq_score_matrix is not None:
            self._save_sent_retr_debug_info_impl(
                self._model_conf.sent.index.pq,
                batch,
                output.pq_score_matrix,
                copy.copy(meta),
                Path(self._conf.save_path) / 'sent_pq_retr_debug_batches.jsonl',
            )

    def _save_sent_retr_debug_info_impl(self, conf, batch, score_matrix, meta, outpath):
        values, indices = torch.topk(score_matrix, 3, 1)

        meta['num_updates'] = self._num_updates
        meta['avg_src_len'] = meta['asl']
        del meta['asl']
        meta['avg_tgt_len'] = meta['atl']
        del meta['atl']

        examples = []
        unscale_factor = 1 / conf.scale if conf.scale else 1.0
        for i in range(len(batch.src)):
            obj = {'src_batch_num': i, 'src_id': batch.src_id[i], 'found': []}
            for v, idx in zip(values[i], indices[i]):
                if idx >= len(batch.tgt_id):
                    # idx from other device
                    tgt_id = -1
                else:
                    tgt_id = batch.tgt_id[idx]
                obj['found'].append(
                    {
                        'tgt_batch_num': idx.item(),
                        'tgt_id': tgt_id,
                        'sim': v.item(),
                        'sim_unscaled': v.item() * unscale_factor,
                    }
                )

            wo_margin = score_matrix[i][i].item()
            wo_margin *= unscale_factor
            wo_margin += conf.margin
            wo_margin /= unscale_factor

            obj['gold'] = {
                'tgt_batch_num': i,
                'tgt_id': batch.tgt_id[i],
                'sim': score_matrix[i][i].item(),
                'sim_unscaled': score_matrix[i][i].item() * unscale_factor,
                'sim_wo_margin': wo_margin,
            }
            examples.append(obj)
        meta['examples'] = examples

        with open(outpath, 'a', encoding='utf8') as f:
            f.write(json.dumps(meta))
            f.write('\n')

    def _debug_batch(self, task, batch, labels):
        if not self._verbose:
            return
        if task == TaskType.SENT_RETR:
            logging.debug('src sents shape: %s', batch.src.size())
            logging.debug('tgt sents shape: %s', batch.tgt.size())
            if self._conf.print_batches:
                logging.debug('src_len: %s', batch.src_len)
                logging.debug('tgt_len: %s', batch.tgt_len)
        elif task == TaskType.DOC_RETR:
            logging.debug(
                "src docs cnt: %s; frags cnt: %s; sents cnt: %s",
                len(batch.src_ids),
                len(batch.src_fragment_len) if batch.src_fragment_len else '-',
                len(batch.src_sents),
            )
            logging.debug(
                "tgt docs cnt: %s; frags cnt: %s; sents cnt: %s",
                len(batch.tgt_ids),
                len(batch.tgt_fragment_len) if batch.tgt_fragment_len else '-',
                len(batch.tgt_sents),
            )
            src_sent_sum = batch.src_sent_len.sum().item()
            logging.debug(
                "src sents stat: max: %s; min: %s; avg: %s; sum: %s",
                batch.src_sent_len.max().item(),
                batch.src_sent_len.min().item(),
                src_sent_sum / len(batch.src_sent_len),
                src_sent_sum,
            )
            tgt_sent_sum = batch.tgt_sent_len.sum().item()
            logging.debug(
                "tgt sents stat: max: %s; min: %s; avg: %s; sum: %s",
                batch.tgt_sent_len.max().item(),
                batch.tgt_sent_len.min().item(),
                tgt_sent_sum / len(batch.tgt_sent_len),
                tgt_sent_sum,
            )

            if self._conf.print_batches:
                logging.debug(
                    'src ids: %s\nsrc sents cnt: %s\n src_len: %s\nsrc_fragment_len:%s\n'
                    'src_doc_len_in_sents: %s\nsrc_doc_len_in_frags: %s',
                    batch.src_ids,
                    len(batch.src_sents),
                    batch.src_sent_len,
                    batch.src_fragment_len,
                    batch.src_doc_len_in_sents,
                    batch.src_doc_len_in_frags,
                )
                logging.debug(
                    'tgt ids: %s\ntgt sents cnt: %s\n tgt_len: %s\ntgt_fragment_len:%s\n'
                    'tgt_doc_len_in_sents: %s\ntgt_doc_len_in_frags: %s',
                    batch.tgt_ids,
                    len(batch.tgt_sents),
                    batch.tgt_sent_len,
                    batch.tgt_fragment_len,
                    batch.tgt_doc_len_in_sents,
                    batch.tgt_doc_len_in_frags,
                )
                logging.debug("labels: %s", labels)

    def _save_gpu_memory_stat(self):
        summary = torch.cuda.memory_summary(self._device)
        process_name = multiprocessing.current_process().name
        out_dir = Path(self._conf.save_path) / 'gpu_memory_stat'
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / process_name, 'a', encoding='ascii') as f:
            f.write(f'#UP {self._num_updates}\n')
            f.write(summary)
            f.write('\n\n')

    def _make_update(self, task):
        any_max_grad_norm = any(g['max_grad_norm'] for g in self._optimizer.param_groups)

        if any_max_grad_norm or self._conf.emb_grad_scale:
            self._scaler.unscale_(self._optimizer)

            if self._conf.emb_grad_scale:
                wgrad = self._local_models.sent_model.encoder.embed.embed_tokens.weight.grad
                if wgrad is not None:
                    wgrad *= self._conf.emb_grad_scale

            if any_max_grad_norm:
                for group in self._optimizer.param_groups:
                    max_grad_norm = group['max_grad_norm']
                    if max_grad_norm:
                        clip_grad_norm_(group['params'], max_grad_norm)

        self._scaler.step(self._optimizer)
        self._scaler.update()

        update_sent_index_model(self._local_models.sent_model, self._world_size)
        update_doc_index_model(self._local_models.doc_model, self._world_size)

        self._optimizer.zero_grad()

    def _run_forward(self, task, batch, labels) -> DualEncModelOutput:
        if task == TaskType.SENT_RETR:
            output = self._sent_model(batch)
        elif task == TaskType.DOC_RETR:
            if not self._conf.use_grad_checkpoint:
                output = self._doc_model(batch, labels)
            else:
                # use_reentrant=True requires any input tensor has requires_grad field set to true
                labels.requires_grad_(True)
                output = torch.utils.checkpoint.checkpoint(
                    self._doc_model, batch, labels, use_reentrant=True
                )
        else:
            raise RuntimeError("Logic error 89837")
        return output

    def _process_task_batches(self, task, task_batches):
        running_metrics = create_metrics(task)
        for batch, labels in task_batches:
            self._debug_batch(task, batch, labels)

            # forward pass
            with autocast(enabled=self._amp_enabled):
                self._log_current_mem_usage('before forward')
                output = self._run_forward(task, batch, labels)
                self._log_current_mem_usage('after forward')

                logging.debug("dense output of model shape: %s", output.dense_score_matrix.size())
                # output matrix size is bsz x tgt_size for retrieval task
                loss, m = self._calc_loss_and_metrics(task, output, labels, batch)
                logging.debug("loss: %s; metrics: %s", loss.item(), m)

                running_metrics += m

            # backpropagate and update optimizer learning rate
            self._log_current_mem_usage('before backward')
            self._scaler.scale(loss).backward()
            self._log_current_mem_usage('after backward')
            logging.debug("backward step done")

            self._make_update(task)
            self._num_updates += 1
            logging.debug("update done")

            if self._is_master and self._conf.debug_iters:
                l = [self._num_updates % int(v) for v in self._conf.debug_iters]
                if not all(l):
                    meta = {'task': task, 'loss': loss.item()}
                    meta.update(m.metrics())
                    meta.update(m.stats())
                    self._save_debug_info(batch, output, labels, meta)
            if (
                self._conf.print_gpu_memory_stat_every
                and self._num_updates % self._conf.print_gpu_memory_stat_every == 0
            ):
                self._save_gpu_memory_stat()

        return running_metrics

    def _sync_epoch_updates(self, n):
        # When data is unevenly split, processes may have done different number of updates
        if self._sync_group is None:
            return n
        t = torch.tensor([n, self._num_updates])
        dist.broadcast(t, 0, group=self._sync_group, async_op=False)
        self._num_updates = t[1].item()
        return t[0].item()

    def _sync_quiting(self, done):
        # Quit from epoch with all processes at once
        if self._sync_group is None:
            return done
        t = torch.tensor([int(done)])
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self._sync_group)
        return t.item() == self._world_size

    def _collect_metrics(self, metrics_dict):
        if self._sync_group is None:
            return metrics_dict
        tasks = []
        metrics_list = []
        for t, m in metrics_dict.items():
            tasks.append(t)
            metrics_list.append(m.tolist())
        t = torch.tensor(metrics_list, dtype=torch.float32)
        dist.reduce(t, 0, op=dist.ReduceOp.SUM, group=self._sync_group)
        return {k: create_metrics(k, v) for k, v in zip(tasks, t.tolist())}

    def _format_lr(self):
        if self._scheduler is None:
            return '-'
        lrs = self._scheduler.get_last_lr()
        lrstr = ','.join(f"{l:.4e}" for l in lrs)
        return f"[{lrstr}]"

    def _train_epoch(self, epoch, train_iter: BatchIterator, dev_iter: BatchIterator):
        epoch_updates = 0
        running_metrics = {}
        last_log_update = 0

        def _reset():
            nonlocal running_metrics
            running_metrics = {t: create_metrics(t) for t in train_iter.supported_tasks()}

        _reset()

        while True:
            task = train_iter.current_task()
            logging.debug("current task is %s", task)
            gen = train_iter.batches(self._conf.switch_tasks_every)

            join_modules = []
            if task == TaskType.SENT_RETR:
                self._sent_model.train()
                join_modules.append(self._sent_model)
            else:
                self._doc_model.train()
                join_modules.append(self._doc_model)
            if isinstance(self._optimizer, ZeRoOptim):
                join_modules.append(self._optimizer)

            with self._uneven_input_handling(join_modules):
                metrics = self._process_task_batches(task, gen)

            epoch_updates += metrics.updates_num()
            epoch_updates = self._sync_epoch_updates(epoch_updates)

            running_metrics[task] += metrics

            if epoch_updates - last_log_update >= self._conf.log_every:
                last_log_update = epoch_updates
                running_metrics = self._collect_metrics(running_metrics)
                if self._is_master:
                    sm = ''
                    for t, m in running_metrics.items():
                        tstr = "SR" if t == TaskType.SENT_RETR else "DR"
                        sm += f"\nTask: {tstr} #up: {m.updates_num()/self._world_size}{m}"

                    logging.info(
                        "#%d %d/%d, lr %s%s",
                        self._num_updates,
                        epoch,
                        epoch_updates,
                        self._format_lr(),
                        sm,
                    )
                _reset()

            if self._num_updates - self._last_eval_update >= self._conf.eval_every:
                self._last_eval_update = self._num_updates
                self._eval_and_save(epoch, dev_iter)

            if self._num_updates - self._last_checkpoint_update >= self._conf.checkpoint_every:
                self._last_checkpoint_update = self._num_updates
                self._save_checkpoint(epoch)
                self._save_indexes(train_iter)

            if self._num_updates >= self._conf.max_updates:
                break

            if self._scheduler is not None:
                self._scheduler.step()

            if self._sync_quiting(train_iter.empty()):
                break

            train_iter.switch_task()

        if self._num_updates % self._conf.switch_tasks_every != 0:
            self._num_updates = (
                self._num_updates // self._conf.switch_tasks_every + 1
            ) * self._conf.switch_tasks_every
            self._sync_epoch_updates(epoch_updates)

        if self._sync_group is not None:
            dist.barrier(group=self._sync_group)

    def _load_from_checkpoint(self):
        logging.info("loading state from %s", self._conf.resume_checkpoint)
        state = torch.load(self._conf.resume_checkpoint, map_location=self._device)
        self._num_updates = state['num_updates']
        self._optimizer.load_state_dict(state['optimizer'])
        if self._scheduler is not None and self._conf.restore_lr_scheduler_state:
            self._scheduler.load_state_dict(state['scheduler'])
        self._scaler.load_state_dict(state['scaler'])
        self._best_metric = state['best_metric']
        self._local_models.doc_model.load_state_dict(state['model'])
        self._local_models.sent_model.encoder = (
            self._local_models.doc_model.sent_encoder.cast_to_base()
        )
        if isinstance(self._doc_model, DDP):
            self._doc_model.module = self._local_models.doc_model
            self._sent_model.module = self._local_models.sent_model
        else:
            self._doc_model = self._local_models.doc_model
            self._sent_model = self._local_models.sent_model

        return state['epoch']

    def _save_indexes(self, train_iter: BatchIterator):
        if self._local_models.sent_model.index is not None:
            if self._is_master:
                logging.info("Saving sent faiss index")
                index_path = self._local_models.sent_model.index.save_as_faiss_index(
                    self._conf.save_path, 'sent'
                )
            else:
                index_path = self._local_models.sent_model.index.index_path(
                    self._conf.save_path, 'sent'
                )
            index_conf = self._local_models.sent_model.conf.index
            if index_conf.readd_vectors_while_training:
                if self._is_master:
                    re_add_sent_vectors(
                        index_path,
                        Path(self._conf.save_path) / 'model.pt',
                        train_iter.get_config().sents_batch_iterator_conf.batch_generator_conf,
                    )

                if self._sync_group is not None:
                    # FIXME torch 1.13.1 timeout in monitored barrier is ignored
                    # dist.monitored_barrier(
                    #     group=self._sync_group, timeout=datetime.timedelta(minutes=30)
                    # )
                    dist.barrier()

                self._local_models.sent_model.index.ivf.reassign_id2center(index_path)
        if self._local_models.doc_model.index is not None:
            if self._is_master:
                logging.info("saving doc faiss index")
                index_path = self._local_models.doc_model.index.save_as_faiss_index(
                    self._conf.save_path, 'doc'
                )

    def _save_checkpoint(self, epoch):
        snapshot_path = Path(self._conf.save_path) / f'checkpoint_{epoch}_{self._num_updates}.pt'
        if isinstance(self._optimizer, ZeRoOptim):
            self._optimizer.consolidate_state_dict(to=0)
        if self._is_master:
            logging.info("Saving new checkpoint")

            state_dict = {
                'num_updates': self._num_updates,
                'epoch': epoch,
                'model': self._local_models.doc_model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'scaler': self._scaler.state_dict(),
                'best_metric': self._best_metric,
            }
            if self._scheduler is not None:
                state_dict['scheduler'] = self._scheduler.state_dict()

            torch.save(state_dict, snapshot_path)

    def _eval_on_dev(self, epoch, dev_iter: BatchIterator):
        with torch.no_grad():
            self._local_models.sent_model.eval()
            self._local_models.doc_model.eval()
            all_tasks = self._conf.eval_tasks
            if not all_tasks:
                all_tasks = dev_iter.supported_tasks()

            metrics_per_task = {}
            for task in all_tasks:
                batches = 0
                cum_metrics = create_metrics(task)
                dev_iter.init_epoch(epoch, [task])
                if task == TaskType.SENT_RETR:
                    model = self._local_models.sent_model
                else:
                    model = self._local_models.doc_model

                for batch, labels in dev_iter.batches(batches_cnt=0):
                    output = model.calc_sim_matrix(batch, dont_cross_device_sample=True)
                    _, m = self._calc_loss_and_metrics(task, output, labels, batch)
                    cum_metrics += m
                    batches += 1
                metrics_per_task[task] = cum_metrics

            return metrics_per_task

    def _eval_and_save(self, epoch, dev_iter: BatchIterator):
        logging.info("Evaling on dev...")
        m_per_task = self._eval_on_dev(epoch, dev_iter)
        m_per_task = self._collect_metrics(m_per_task)
        if self._is_master:
            logging.info("Results on dev data")
            for task, m in m_per_task.items():
                logging.info(
                    "Task %s; Epoch %s; #up %d%s",
                    task,
                    epoch,
                    self._num_updates,
                    m,
                )

            best_m = 0.0
            for task, m in m_per_task.items():
                _, v = m.best_metric_for_task()
                best_m += v
            best_m /= len(m_per_task)

            logging.info("best %s dev %s", self._best_metric, best_m)
            if best_m > self._best_metric:
                self._best_metric = best_m
                self._save_model(Path(self._conf.save_path) / 'model.pt')
                logging.info(
                    "new best model was saved in %s", Path(self._conf.save_path) / 'model.pt'
                )

    def _save_model(self, out_path):
        try:
            version = pkg_resources.require("doc_enc")[0].version
        except pkg_resources.DistributionNotFound:
            version = 0
        state_dict = {
            'version': version,
            'trainer_conf': self._conf,
            'model_conf': self._model_conf,
            'tp_conf': self._tp_conf,
            'tp': self._tp.state_dict(),
            'sent_enc': self._local_models.sent_model.encoder.state_dict(),
            'doc_enc': self._local_models.doc_model.doc_encoder.state_dict(),
        }
        if self._local_models.doc_model.frag_encoder is not None:
            state_dict['frag_enc'] = self._local_models.doc_model.frag_encoder.state_dict()

        sent_for_doc = self._local_models.doc_model.sent_encoder.doc_mode_encoder
        if sent_for_doc is not None:
            state_dict['sent_for_doc'] = sent_for_doc.state_dict()

        torch.save(state_dict, out_path)

    def __call__(self, train_iter: BatchIterator, dev_iter: BatchIterator):
        epoch = self._init_epoch

        while True:
            train_iter.init_epoch(epoch, self._conf.tasks)
            logging.info("Start epoch %d", epoch)
            self._train_epoch(epoch, train_iter, dev_iter)
            train_iter.end_epoch()
            logging.info("End epoch %d", epoch)
            self._save_indexes(train_iter)
            if self._num_updates >= self._conf.max_updates:
                break
            epoch += 1
