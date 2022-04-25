#!/usr/bin/env python3

import dataclasses
import copy

from doc_enc.training.types import TaskType
from doc_enc.training.sents_batch_generator import SentsBatchIterator, SentsBatchIteratorConf
from doc_enc.training.docs_batch_generator import DocsBatchIteratorConf, DocsBatchIterator
from doc_enc.tokenizer import TokenizerConf


@dataclasses.dataclass
class BatchIteratorConf:
    sents_batch_iterator_conf: SentsBatchIteratorConf
    docs_batch_iterator_conf: DocsBatchIteratorConf
    initial_task: TaskType = TaskType.SENT_RETR
    switch_every: int = 10


class BatchIterator:
    def __init__(
        self,
        opts: BatchIteratorConf,
        tok_conf: TokenizerConf,
        split,
        rank=0,
        world_size=-1,
        pad_idx=0,
    ):

        self._sents_batch_iterator = SentsBatchIterator(
            opts.sents_batch_iterator_conf,
            tok_conf,
            split=split,
            rank=rank,
            world_size=world_size,
            pad_idx=pad_idx,
        )

        self._docs_batch_iterator = DocsBatchIterator(
            opts.docs_batch_iterator_conf,
            tok_conf,
            split=split,
            rank=rank,
            world_size=world_size,
            pad_idx=pad_idx,
        )

        self._opts = opts
        self._epoch = 0

    def init_epoch(self, epoch):
        self._epoch = epoch - 1
        self._sents_batch_iterator.init_epoch(epoch)
        self._docs_batch_iterator.init_epoch(epoch)

    def destroy(self):
        self._sents_batch_iterator.destroy()
        self._docs_batch_iterator.destroy()

    def initial_task(self):
        return self._opts.initial_task

    def supported_tasks(self):
        return [TaskType.SENT_RETR, TaskType.DOC_RETR]

    def _make_iterators(self, tasks):
        iterators = (self._sents_batch_iterator.batches(), self._docs_batch_iterator.batches())
        return [i for t, i in zip(self.supported_tasks(), iterators) if t in tasks]

    def batches(self, task=None):

        tasks = copy.copy(self.supported_tasks())
        switch_every = self._opts.switch_every
        if task is None:
            task = self._opts.initial_task
        else:
            if task not in self.supported_tasks():
                raise RuntimeError(f"Unsupported task {task}")
            switch_every = 0
            tasks = [task]
        task_idx = tasks.index(task)

        def _switch_task(inc=1):
            nonlocal task_idx
            if switch_every:
                task_idx += inc
                task_idx = task_idx % len(tasks)

        # change task at the beggining of each epoch except the first one
        # self._epoch == 0 in the beggining
        _switch_task(self._epoch)

        iterators = self._make_iterators(tasks)

        batch_num = 0

        while True:
            task = tasks[task_idx]
            iterator = iterators[task_idx]

            try:
                b = next(iterator)
                yield task, *b
            except StopIteration:
                del tasks[task_idx]
                del iterators[task_idx]
                if not tasks:
                    break
                _switch_task()
                batch_num = 0
            else:
                batch_num += 1
                if switch_every and batch_num % switch_every == 0:
                    _switch_task()
