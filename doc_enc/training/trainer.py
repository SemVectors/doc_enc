#!/usr/bin/env python3

import os
from typing import List
import dataclasses
import json
from pathlib import Path
import logging


from omegaconf import MISSING

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from torch.nn.parallel import DistributedDataParallel as DDP

from doc_enc.tokenizer import AbcTokenizer
from doc_enc.training.types import DocRetrLossType, TaskType, SentRetrLossType
from doc_enc.training.models.model_factory import create_model
from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.training.metrics import create_metrics


@dataclasses.dataclass
class TrainerConf:
    save_path: str = ''
    sent_retr_loss_type: SentRetrLossType = SentRetrLossType.BICE
    doc_retr_loss_type: DocRetrLossType = DocRetrLossType.CE
    max_updates: int = MISSING
    lr: float = MISSING
    warmup_updates: int = 0
    warmup_init_lr: float = -1.0
    final_lr: float = -1.0
    resume_snapshot: str = ''
    emb_grad_scale: float = 0.0
    max_grad_norm: float = 0.0

    log_every: int = 100
    eval_every: int = 300_000
    debug_iters: List[int] = dataclasses.field(default_factory=list)


# from fairseq
class InverseSquareRootSchedule:
    def __init__(self, opts: TrainerConf, optimizer):
        self._opts = opts
        self._optimizer = optimizer
        self._decay_factor = opts.lr * opts.warmup_updates**0.5

        # initial learning rate
        if opts.warmup_init_lr >= 0:
            self._lr_warmup_step = (opts.lr - opts.warmup_init_lr) / opts.warmup_updates
            self._lr = opts.warmup_init_lr
        else:
            self._lr_warmup_step = 0.0
            self._lr = opts.lr

        self._set_lr()

    def _set_lr(self):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self._lr

    def get_lr(self):
        return self._lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self._opts.warmup_updates:
            if self._lr_warmup_step:
                self._lr = self._opts.warmup_init_lr + num_updates * self._lr_warmup_step
                self._set_lr()
            return self._lr

        self._lr = self._decay_factor * num_updates**-0.5
        self._set_lr()
        return self._lr


class LinearLRSchedule:
    def __init__(self, opts: TrainerConf, optimizer):
        self._opts = opts
        self._optimizer = optimizer
        self._lr = opts.lr
        self._set_lr()
        self._step = (self._lr - opts.final_lr) / (opts.max_updates - opts.warmup_updates)
        logging.info("step lr: %e", self._step)

    def _set_lr(self):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self._lr

    def get_lr(self):
        return self._lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self._opts.warmup_updates:
            return self._lr
        self._lr -= self._step
        self._set_lr()
        return self._lr


class Trainer:
    def __init__(
        self,
        opts: TrainerConf,
        model_conf: DocModelConf,
        vocab: AbcTokenizer,
        world_size,
        rank,
        amp=True,
        verbose=False,
    ):
        self._opts = opts
        if not self._opts.save_path:
            self._opts.save_path = os.getcwd()

        self._model_conf = model_conf
        self._rank = rank
        self._device = torch.device(f'cuda:{rank}')
        self._amp_enabled = amp
        self._verbose = verbose

        self._local_model = self._create_model(vocab, model_conf)
        if world_size > 1:
            logging.info("Creating DistributedDataParallel instance")
            self._model = DDP(
                self._local_model,
                device_ids=[rank],
                output_device=rank,
                bucket_cap_mb=200,
                find_unused_parameters=False,
            )
        else:
            logging.info("Skip creating DistributedDataParallel since world size == 1")
            self._model = self._local_model

        self._sent_retr_criterion = torch.nn.CrossEntropyLoss()
        self._doc_retr_criterion = torch.nn.CrossEntropyLoss()

        self._optimizer = torch.optim.Adam(self._model.parameters())
        if opts.final_lr != -1.0:
            self._scheduler = LinearLRSchedule(opts, self._optimizer)
        else:
            self._scheduler = InverseSquareRootSchedule(opts, self._optimizer)
        self._scaler = GradScaler(enabled=amp)
        self._num_updates = 1
        self._best_metric = 0.0

        self._init_epoch = 1
        if self._opts.resume_snapshot:
            self._init_epoch = self._load_from_checkpoint()

    def _create_model(self, vocab: AbcTokenizer, model_conf: DocModelConf):
        model = create_model(model_conf, vocab.vocab_size(), vocab.pad_idx())
        model = model.to(self._device)
        logging.info("created model %s", model)
        logging.info("model loaded to %s", self._device)
        return model

    def _calc_sent_retr_loss(self, output, labels, batch_size):
        if self._opts.sent_retr_loss_type == SentRetrLossType.CE:
            loss = self._sent_retr_criterion(output, labels)
        elif self._opts.sent_retr_loss_type == SentRetrLossType.BICE:
            loss_src = self._sent_retr_criterion(output, labels)
            loss_tgt = self._sent_retr_criterion(output.t()[:batch_size], labels)
            loss = loss_src + loss_tgt
        else:
            raise RuntimeError("Logic error 987")
        return loss

    def _calc_doc_retr_loss(self, output, labels, batch_size):
        if self._opts.doc_retr_loss_type == DocRetrLossType.CE:
            loss = self._doc_retr_criterion(output, labels)
        else:
            raise RuntimeError("Logic error 988")
        return loss

    def _calc_loss_and_metrics(self, task, output, labels, batch):
        if task == TaskType.SENT_RETR:
            loss = self._calc_sent_retr_loss(output, labels, batch.info['bs'])
        elif task == TaskType.DOC_RETR:
            loss = self._calc_doc_retr_loss(output, labels, batch.info['bs'])
        else:
            raise RuntimeError(f"Unknown task in calc loss: {task}")

        m = create_metrics(task)
        m.update_metrics(output, labels, batch)
        return loss, m

    def _save_debug_info(self, batch, output, labels, meta):
        if meta['task'] == TaskType.SENT_RETR:
            self._save_retr_debug_info(batch, output, labels, meta)
        else:
            raise RuntimeError("Logic error 342")

    def _save_retr_debug_info(self, batch, output, _, meta):
        values, indices = torch.topk(output, 3, 1)

        meta['num_updates'] = self._num_updates
        meta['avg_src_len'] = batch.src_len.float().mean().item()
        meta['avg_tgt_len'] = batch.tgt_len.float().mean().item()

        examples = []
        unscale_factor = 1 / self._model_conf.sent.scale if self._model_conf.sent.scale else 1.0
        for i in range(len(batch.src)):
            obj = {'src_batch_num': i, 'src_id': batch.src_id[i], 'found': []}
            for v, idx in zip(values[i], indices[i]):
                if v < 0.7:
                    continue
                obj['found'].append(
                    {
                        'tgt_batch_num': idx.item(),
                        'tgt_id': batch.tgt_id[idx],
                        'sim': v.item(),
                        'sim_unscaled': v.item() * unscale_factor,
                    }
                )

            wo_margin = output[i][i].item()
            wo_margin *= unscale_factor
            wo_margin += self._model_conf.sent.margin
            wo_margin /= unscale_factor

            obj['etal'] = {
                'tgt_batch_num': i,
                'tgt_id': batch.tgt_id[i],
                'sim': output[i][i].item(),
                'sim_unscaled': output[i][i].item() * unscale_factor,
                'sim_wo_margin': wo_margin,
            }
            examples.append(obj)
        meta['examples'] = examples
        meta_path = Path(self._opts.save_path) / 'samples.jsonl'
        with open(meta_path, 'a', encoding='utf8') as f:
            f.write(json.dumps(meta))
            f.write('\n')

    def _debug_batch(self, task, batch, labels):
        if not self._verbose:
            return
        if task == TaskType.SENT_RETR:
            logging.debug('src shape: %s\nsrc_len: %s', batch.src.size(), batch.src_len)
            logging.debug('tgt shape: %s\ntgt_len: %s', batch.tgt.size(), batch.tgt_len)
        elif task == TaskType.DOC_RETR:
            logging.debug(
                'src sents cnt: %s\n src_len: %s\nsrc_fragment_len:%s\n'
                'src_doc_len_in_sents: %s\nsrc_doc_len_in_frags: %s',
                len(batch.src_sents),
                batch.src_sent_len,
                batch.src_fragment_len,
                batch.src_doc_len_in_sents,
                batch.src_doc_len_in_frags,
            )
            logging.debug(
                'tgt sents cnt: %s\n tgt_len: %s\ntgt_fragment_len:%s\n'
                'tgt_doc_len_in_sents: %s\ntgt_doc_len_in_frags: %s',
                len(batch.tgt_sents),
                batch.tgt_sent_len,
                batch.tgt_fragment_len,
                batch.tgt_doc_len_in_sents,
                batch.tgt_doc_len_in_frags,
            )
            logging.debug("labels: %s", labels)

    # def _accumulation_steps(self):
    #     # TODO
    #     return 5

    def _make_update(self, task):
        if self._opts.max_grad_norm or self._opts.emb_grad_scale:
            self._scaler.unscale_(self._optimizer)

            if self._opts.emb_grad_scale:
                wgrad = self._local_model.sent_model.encoder.embed_tokens.weight.grad
                if wgrad is not None:
                    wgrad *= self._opts.emb_grad_scale

            if self._opts.max_grad_norm:
                clip_grad_norm_(self._local_model.parameters(), self._opts.max_grad_norm)

        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._optimizer.zero_grad()

        self._num_updates += 1
        self._scheduler.step_update(self._num_updates)

    def _train_epoch(self, epoch, train_iter, dev_iter):
        epoch_updates = 1
        running_loss = 0.0
        task_num_updates = 0
        running_metrics = create_metrics(train_iter.initial_task())
        prev_task = None
        self._model.train()

        def _reset(new_task):
            nonlocal running_loss
            nonlocal task_num_updates
            nonlocal running_metrics
            running_loss = 0.0
            task_num_updates = 0
            running_metrics = create_metrics(new_task)

        for task, batch, labels in train_iter.batches():
            self._debug_batch(task, batch, labels)

            if prev_task != task:
                logging.info("switching to %s task", task)
                prev_task = task
                _reset(task)

            # forward pass
            with autocast(enabled=self._amp_enabled):
                output = self._model(task, batch, labels)
                # output size is bsz x tgt_size for retrieval task
                loss, m = self._calc_loss_and_metrics(task, output, labels, batch)

                running_loss += loss.item()
                running_metrics += m

            # backpropagate and update optimizer learning rate
            self._scaler.scale(loss).backward()

            self._make_update(task)
            epoch_updates += 1
            task_num_updates += 1
            if self._rank == 0 and self._num_updates % self._opts.log_every == 0:
                logging.info(
                    "#%d %d/%d, lr %.4e, loss %.5f%s",
                    self._num_updates,
                    epoch,
                    epoch_updates,
                    self._scheduler.get_lr(),
                    running_loss / task_num_updates,
                    running_metrics,
                )
                _reset(task)

            if self._rank == 0 and self._num_updates % self._opts.eval_every == 0:
                self._save_and_eval(epoch, dev_iter)
                self._model.train()

            if self._rank == 0 and self._opts.debug_iters:
                l = [self._num_updates % int(v) for v in self._opts.debug_iters]
                if not all(l):
                    meta = {'task': task, 'loss': loss.item()}
                    meta.update(m.metrics())
                    self._save_debug_info(batch, output, labels, meta)

    def _load_from_checkpoint(self):
        logging.info("loading state from %s", self._opts.resume_snapshot)
        state = torch.load(self._opts.resume_snapshot, map_location=self._device)
        self._num_updates = state['num_updates']
        self._optimizer = state['optimizer']
        self._scheduler = state['scheduler']
        self._best_metric = state['best_metric']
        # TODO support ddp resuming
        self._model = self._local_model = state['model']
        return state['epoch']

    def _save_checkpoint(self, epoch):
        snapshot_path = Path(self._opts.save_path) / f'checkpoint_{epoch}_{self._num_updates}.pt'
        torch.save(
            {
                'num_updates': self._num_updates,
                'epoch': epoch,
                'model': self._model,
                'optimizer': self._optimizer,
                'scheduler': self._scheduler,
                'best_metric': self._best_metric,
            },
            snapshot_path,
        )

    def _eval_on_dev(self, dev_iter):
        with torch.no_grad():
            self._model.eval()
            tasks = dev_iter.supported_tasks()

            metrics_per_task = {}
            for task in tasks:
                batches = 0
                cum_loss = 0.0
                cum_metrics = create_metrics(task)
                for _, batch, labels in dev_iter.batches(task):
                    output = self._model(task, batch, labels)
                    loss, m = self._calc_loss_and_metrics(task, output, labels, batch)
                    cum_loss += loss.item()
                    cum_metrics += m
                    batches += 1
                metrics_per_task[task] = (cum_loss / batches, cum_metrics)

            return metrics_per_task

    def _save_and_eval(self, epoch, dev_iter):
        self._save_checkpoint(epoch)
        dev_iter.init_epoch(epoch)
        m_per_task = self._eval_on_dev(dev_iter)
        for task, (loss, m) in m_per_task.items():
            logging.info(
                "Task %s; Epoch %s; #up %d; Loss on dev %.5f%s",
                task,
                epoch,
                self._num_updates,
                loss,
                m,
            )

        best_m = 0.0
        for task, (loss, m) in m_per_task.items():
            _, v = m.best_metric_for_task()
            best_m += v
        best_m /= len(m_per_task)

        logging.info("best %s dev %s", self._best_metric, best_m)
        if best_m > self._best_metric:
            self._best_metric = best_m
            self._save_model(Path(self._opts.save_path) / 'model.pt')
            logging.info("new best model was saved in %s", Path(self._opts.save_path) / 'model.pt')

    def _save_model(self, out_path):
        # opts_dict = vars(self._opts)
        # if 'func' in opts_dict:
        #     del opts_dict['func']
        state_dict = {
            'args': self._opts,
            'sent_enc': self._local_model.sent_model.encoder.state_dict(),
            'doc_enc': self._local_model.doc_encoder.state_dict(),
        }
        if self._local_model.frag_encoder is not None:
            state_dict['frag_enc'] = self._local_model.frag_encoder.state_dict()

        torch.save(state_dict, out_path)

    def __call__(self, train_iter, dev_iter):
        epoch = self._init_epoch
        while True:
            train_iter.init_epoch(epoch)
            logging.info("start epoch %d", epoch)
            self._train_epoch(epoch, train_iter, dev_iter)
            logging.info("end epoch %d", epoch)
            epoch += 1
            if self._num_updates >= self._opts.max_updates:
                break
