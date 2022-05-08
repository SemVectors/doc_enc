#!/usr/bin/env python3

from enum import Enum
from typing import Optional, Generator, List
import dataclasses
import logging
import copy

from doc_enc.training.types import TaskType
from doc_enc.training.sents_batch_generator import SentsBatchIterator, SentsBatchIteratorConf
from doc_enc.training.docs_batch_generator import DocsBatchIteratorConf, DocsBatchIterator
from doc_enc.text_processor import TextProcessorConf


class EarlyIterEndPolicy(Enum):
    LONGEST = 1
    REITER = 2


@dataclasses.dataclass
class BatchIteratorConf:
    sents_batch_iterator_conf: SentsBatchIteratorConf
    docs_batch_iterator_conf: DocsBatchIteratorConf

    early_iter_end_policy: EarlyIterEndPolicy = EarlyIterEndPolicy.REITER
    reinit_last_iter: bool = False


class BatchIterator:
    def __init__(
        self,
        opts: BatchIteratorConf,
        tp_conf: TextProcessorConf,
        logging_conf,
        split,
        rank=0,
        world_size=-1,
        pad_idx=0,
    ):

        self._sents_batch_iterator = SentsBatchIterator(
            opts.sents_batch_iterator_conf,
            tp_conf.tokenizer,
            logging_conf,
            split=split,
            rank=rank,
            world_size=world_size,
            pad_idx=pad_idx,
        )

        self._docs_batch_iterator = DocsBatchIterator(
            opts.docs_batch_iterator_conf,
            tp_conf,
            logging_conf,
            split=split,
            rank=rank,
            world_size=world_size,
            pad_idx=pad_idx,
        )

        self._opts = opts
        self._epoch = 0

        self._task_idx = 0
        self._current_tasks = None
        self._iterators: Optional[List[Optional[Generator]]] = None
        self._done = None
        self._early_end_policy = self._opts.early_iter_end_policy

    def init_epoch(self, epoch, tasks=None):
        self._epoch = epoch - 1

        if not tasks:
            self._current_tasks = copy.copy(self.supported_tasks())
        else:
            self._current_tasks = copy.copy(tasks)

        self._task_idx = 0

        logging.info("Tasks for the %d epoch: %s", epoch, self._current_tasks)
        self._iterators = [self._init_iterator(t, epoch) for t in self._current_tasks]
        self._done = [False] * len(self._iterators)

        self._early_end_policy = self._opts.early_iter_end_policy

    def end_epoch(self):
        # cleanup previous epoch
        if self._iterators is None or self._current_tasks is None:
            return
        for task, it in zip(self._current_tasks, self._iterators):
            if it is None:
                continue

            logging.info("Destroying iterator for task: %s", task)
            if task == TaskType.SENT_RETR:
                self._sents_batch_iterator.end_epoch()
            elif task == TaskType.DOC_RETR:
                self._docs_batch_iterator.end_epoch()

    def destroy(self):
        self._sents_batch_iterator.destroy()
        self._docs_batch_iterator.destroy()

    def initial_task(self):
        if self._current_tasks is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        return self._current_tasks[0]

    def current_task(self):
        if self._current_tasks is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        return self._current_tasks[self._task_idx]

    def empty(self):
        if self._done is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        # return all(it is None for it in self._iterators)
        return all(self._done)

    def supported_tasks(self):
        return [TaskType.SENT_RETR, TaskType.DOC_RETR]

    def _init_iterator(self, task, epoch):
        if task == TaskType.SENT_RETR:
            self._sents_batch_iterator.init_epoch(epoch)
            return self._sents_batch_iterator.batches()

        if task == TaskType.DOC_RETR:
            self._docs_batch_iterator.init_epoch(epoch)
            return self._docs_batch_iterator.batches()
        raise RuntimeError(f"Unsupported task: {task} 34289")

    def batches(self, batches_cnt: int):
        if self._iterators is None or self._current_tasks is None or self._done is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        if (
            not batches_cnt
            and len(self._current_tasks) == 1
            and self._early_end_policy == EarlyIterEndPolicy.REITER
        ):
            logging.info(
                "Set early_iter_end_policy to LONGEST since there is only one task "
                "and no limit for batch count is specified"
            )
            self._early_end_policy = EarlyIterEndPolicy.LONGEST

        batch_num = 0
        iterator = self._iterators[self._task_idx]
        if iterator is None:
            return
        while True:
            try:
                b = next(iterator)
                yield b
            except StopIteration:
                self._done[self._task_idx] = True

                if self._early_end_policy == EarlyIterEndPolicy.LONGEST:
                    self._iterators[self._task_idx] = None
                    break
                if self._early_end_policy == EarlyIterEndPolicy.REITER:
                    if not self.empty() or self._opts.reinit_last_iter:
                        task = self._current_tasks[self._task_idx]
                        logging.info("Reinit iterator for task: %s", task)
                        iterator = self._init_iterator(task, self._epoch)
                        self._iterators[self._task_idx] = iterator
                    else:
                        self._iterators[self._task_idx] = None
                        break

                else:
                    raise RuntimeError(
                        f"unsupported early iter end policy {self._opts.early_iter_end_policy}"
                    )

            else:
                batch_num += 1
                if batches_cnt and batch_num >= batches_cnt:
                    break

    def switch_task(self):
        if self._current_tasks is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        self._task_idx += 1
        self._task_idx = self._task_idx % len(self._current_tasks)
