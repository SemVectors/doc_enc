#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import TaskType


class BaseMetrics:
    """Quality metrics and other stats"""

    def __init__(self):
        self._cnt = 0
        self._ncorrect = 0
        self._total = 0

        self._loss = 0.0

        self._src_item_len_sum = 0
        self._src_item_cnt = 0
        self._tgt_item_len_sum = 0
        self._tgt_item_cnt = 0

    def tolist(self):
        return [
            self._cnt,
            self._ncorrect,
            self._total,
            self._src_item_len_sum,
            self._src_item_cnt,
            self._tgt_item_len_sum,
            self._tgt_item_cnt,
            self._loss,
        ]

    @classmethod
    def fromlist(cls, l):
        m = cls()
        assert len(l) == 8
        fields = (
            '_cnt',
            '_ncorrect',
            '_total',
            '_src_item_len_sum',
            '_src_item_cnt',
            '_tgt_item_len_sum',
            '_tgt_item_cnt',
        )
        for f, v in zip(fields, l[:7]):
            m.__dict__[f] = int(v)

        m._loss = l[7]
        return m

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
        self._src_item_len_sum += other._src_item_len_sum
        self._src_item_cnt += other._src_item_cnt
        self._tgt_item_len_sum += other._tgt_item_len_sum
        self._tgt_item_cnt += other._tgt_item_cnt
        return self

    def metrics(self):
        rec = self._ncorrect / self._total if self._total else 0.0
        return {'rec': rec}

    def best_metric_for_task(self):
        m = self.metrics()
        return 'rec', m['rec']

    def stats(self):
        avg_src_item_len = (
            self._src_item_len_sum / self._src_item_cnt if self._src_item_cnt else 0.0
        )
        avg_tgt_item_len = (
            self._tgt_item_len_sum / self._tgt_item_cnt if self._tgt_item_cnt else 0.0
        )
        return {'asl': avg_src_item_len, 'atl': avg_tgt_item_len}

    def __str__(self):
        prefix = "; loss %.5f" % (self._loss / self._cnt if self._cnt else 0.0)

        m = self.metrics()
        s = self.stats()
        fmt = '; %s: %.3f' * (len(m) + len(s))
        m_and_s = itertools.chain(m.items(), s.items())
        metrics_str = fmt % tuple(itertools.chain.from_iterable(m_and_s))
        return prefix + metrics_str


class SentRetrMetrics(BaseMetrics):
    def _update_metrics_impl(self, output, labels, batch):
        _, ypredicted = torch.max(output, 1)
        self._ncorrect += (ypredicted == labels).sum().item()
        self._total += output.size(0)

        self._src_item_len_sum += batch.src_len.sum().item()
        self._src_item_cnt += len(batch.src_len)

        self._tgt_item_len_sum += batch.tgt_len.sum().item()
        self._tgt_item_cnt += len(batch.tgt_len)


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

        self._src_item_len_sum += sum(batch.src_doc_len_in_sents)
        self._src_item_cnt += len(batch.src_doc_len_in_sents)

        self._tgt_item_len_sum += sum(batch.tgt_doc_len_in_sents)
        self._tgt_item_cnt += len(batch.tgt_doc_len_in_sents)

    def stats(self):
        s = super().stats()
        s['asdb'] = self._src_item_cnt / self._cnt if self._cnt else 0.0
        s['atdb'] = self._tgt_item_cnt / self._cnt if self._cnt else 0.0
        return s


def create_metrics(task: TaskType, metrics_list=None) -> BaseMetrics:
    if task == TaskType.SENT_RETR:
        cls = SentRetrMetrics
    elif task == TaskType.DOC_RETR:
        cls = DocRetrMetrics
    else:
        raise RuntimeError(f"Unknown task type: {task} in metrics factory")

    if metrics_list is None:
        return cls()
    return cls.fromlist(metrics_list)
