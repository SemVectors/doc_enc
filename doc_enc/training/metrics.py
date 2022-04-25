#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import TaskType


class BaseMetrics:
    def __init__(self):
        self._ncorrect = 0
        self._total = 0

    def update_metrics(self, output, labels, batch):
        raise NotImplementedError("implement in subclass")

    def __iadd__(self, other):
        self._ncorrect += other._ncorrect
        self._total += other._total
        return self

    def metrics(self):
        rec = self._ncorrect / self._total
        return {'rec': rec}

    def best_metric_for_task(self):
        m = self.metrics()
        return 'rec', m['rec']

    def __str__(self):
        m = self.metrics()
        fmt = '; %s: %.3f' * len(m)
        return fmt % tuple(itertools.chain.from_iterable(m.items()))


class SentRetrMetrics(BaseMetrics):
    def update_metrics(self, output, labels, batch):
        _, ypredicted = torch.max(output, 1)
        self._ncorrect += (ypredicted == labels).sum().item()
        self._total += output.size(0)


class DocRetrMetrics(BaseMetrics):
    def update_metrics(self, output, labels, batch):
        k = batch.info['max_positives_per_doc']

        pidxs = batch.positive_idxs
        if k == 1:
            _, ypredicted = torch.max(output, 1)
            ll = torch.tensor(pidxs, device=ypredicted.device).squeeze()
            self._ncorrect += (ypredicted == ll).sum().item()
        else:
            for i, gold_idxs in enumerate(pidxs):
                k = len(gold_idxs)
                if k == 0:
                    raise RuntimeError("Imposibru!")
                if k == 1:
                    idx = torch.max(output[i], 0)[1].item()
                    self._ncorrect += idx == gold_idxs[0]
                else:
                    _, idxs = torch.topk(output[i], k, 0)
                    self._ncorrect += sum(1 for j in idxs if j in gold_idxs)

        self._total += labels.sum().item()


def create_metrics(task: TaskType) -> BaseMetrics:
    if task == TaskType.SENT_RETR:
        return SentRetrMetrics()
    if task == TaskType.DOC_RETR:
        return DocRetrMetrics()
    raise RuntimeError(f"Unknown task type: {task} in metrics factory")
