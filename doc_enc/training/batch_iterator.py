#!/usr/bin/env python3

from typing import Optional, Generator, List
import dataclasses
import logging
import copy

from doc_enc.training.types import TaskType
from doc_enc.training.sents_batch_generator import SentsBatchIterator, SentsBatchIteratorConf
from doc_enc.training.docs_batch_generator import DocsBatchIteratorConf, DocsBatchIterator
from doc_enc.tokenizer import TokenizerConf


@dataclasses.dataclass
class BatchIteratorConf:
    sents_batch_iterator_conf: SentsBatchIteratorConf
    docs_batch_iterator_conf: DocsBatchIteratorConf


class BatchIterator:
    def __init__(
        self,
        opts: BatchIteratorConf,
        tok_conf: TokenizerConf,
        logging_conf,
        split,
        rank=0,
        world_size=-1,
        pad_idx=0,
    ):

        self._sents_batch_iterator = SentsBatchIterator(
            opts.sents_batch_iterator_conf,
            tok_conf,
            logging_conf,
            split=split,
            rank=rank,
            world_size=world_size,
            pad_idx=pad_idx,
        )

        self._docs_batch_iterator = DocsBatchIterator(
            opts.docs_batch_iterator_conf,
            tok_conf,
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

    def init_epoch(self, epoch, tasks=None):
        self._epoch = epoch - 1

        if tasks is None:
            self._current_tasks = copy.copy(self.supported_tasks())
        else:
            self._current_tasks = copy.copy(tasks)

        self._task_idx = 0
        self._iterators = self._make_iterators(self._current_tasks)

        if TaskType.SENT_RETR in self._current_tasks:
            self._sents_batch_iterator.init_epoch(epoch)
        if TaskType.DOC_RETR in self._current_tasks:
            self._docs_batch_iterator.init_epoch(epoch)

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
        if self._iterators is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        return all(it is None for it in self._iterators)

    def supported_tasks(self):
        return [TaskType.SENT_RETR, TaskType.DOC_RETR]

    def _make_iterators(self, tasks):
        iterators = (self._sents_batch_iterator.batches(), self._docs_batch_iterator.batches())
        return [i for t, i in zip(self.supported_tasks(), iterators) if t in tasks]

    def batches(self, batches_cnt: int):
        if self._iterators is None or self._current_tasks is None:
            raise RuntimeError("Batch iterator is not Initialized!")

        batch_num = 0
        iterator = self._iterators[self._task_idx]
        if iterator is None:
            return
        while True:
            try:
                b = next(iterator)
                yield b
            except StopIteration:
                self._iterators[self._task_idx] = None
                break
            else:
                batch_num += 1
                if batches_cnt and batch_num >= batches_cnt:
                    break

    def switch_task(self):
        if self._current_tasks is None:
            raise RuntimeError("Batch iterator is not Initialized!")
        self._task_idx += 1
        self._task_idx = self._task_idx % len(self._current_tasks)
