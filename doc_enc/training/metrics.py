#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import TaskType


class BaseMetrics:
    def __init__(self):
        self._cnt = 0
        self._ncorrect = 0
        self._total = 0
        self._loss = 0.0

    def update_metrics(self, loss, output, labels, batch):
        self._cnt += 1
        self._loss += loss
        self._update_metrics_impl(output, labels, batch)

    def _update_metrics_impl(self, output, labels, batch):
        raise NotImplementedError("implement in subclass")

    def updates_num(self):
        return self._cnt

    def loss(self):
        return self._loss

    def __iadd__(self, other):
        self._cnt += other._cnt
        self._loss += other._loss
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
        prefix = "; loss %.5f" % (self._loss / self._cnt)

        m = self.metrics()
        fmt = '; %s: %.3f' * len(m)
        metrics_str = fmt % tuple(itertools.chain.from_iterable(m.items()))
        return prefix + metrics_str


class SentRetrMetrics(BaseMetrics):
    def _update_metrics_impl(self, output, labels, batch):
        _, ypredicted = torch.max(output, 1)
        self._ncorrect += (ypredicted == labels).sum().item()
        self._total += output.size(0)


class DocRetrMetrics(BaseMetrics):
    def _update_metrics_impl(self, output, labels, batch):
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
