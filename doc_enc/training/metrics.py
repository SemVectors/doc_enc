#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import TaskType


class Metrics:
    def __init__(self, task: TaskType = TaskType.UNDEFINED):
        self._task = task
        self._ncorrect = 0
        self._batch_size = 0

    def _check_task(self, task):
        if self._task == TaskType.UNDEFINED:
            self._task = task
        if self._task != task:
            raise RuntimeError("Different tasks in add operation!!")

    def update_metrics(self, task, output, labels, batch_size):
        self._check_task(task)

        if self._task == TaskType.SENT_RETR:
            _, ypredicted = torch.max(output, 1)
            self._ncorrect += (ypredicted == labels).sum().item()
            self._batch_size += batch_size
            return

        raise RuntimeError("Logic error 5643")

    def __iadd__(self, other):
        self._check_task(other._task)

        self._ncorrect += other._ncorrect
        self._batch_size += other._batch_size
        return self

    def metrics(self):
        if self._task == TaskType.SENT_RETR:
            acc = self._ncorrect / self._batch_size
            return {'acc': acc}

        raise RuntimeError("Logic error 5644")

    def best_metric_for_task(self):
        if self._task == TaskType.SENT_RETR:
            m = self.metrics()
            return 'acc', m['acc']

        raise RuntimeError("Logic error 5645")

    def __str__(self):

        m = self.metrics()
        fmt = '; %s: %.3f' * len(m)
        return fmt % tuple(itertools.chain.from_iterable(m.items()))
