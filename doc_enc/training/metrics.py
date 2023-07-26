#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import DocsBatch, TaskType
from doc_enc.training.models.base_model import DualEncModelOutput


class BaseMetrics:
    """Quality metrics and other stats"""

    def __init__(self):
        self._cnt = 0
        self._ncorrect = 0
        self._total = 0

        self._loss = 0.0
        self._dense_loss = 0.0
        self._ivf_loss = 0.0
        self._pq_loss = 0.0

        self._src_item_cnt = 0
        self._src_len_in_sents = 0
        self._src_len_in_frags = 0
        self._src_len_in_tokens = 0

        self._tgt_item_cnt = 0
        self._tgt_len_in_sents = 0
        self._tgt_len_in_frags = 0
        self._tgt_len_in_tokens = 0

    def tolist(self):
        return [
            self._cnt,
            self._ncorrect,
            self._total,
            self._src_item_cnt,
            self._tgt_item_cnt,
            self._src_len_in_sents,
            self._src_len_in_frags,
            self._src_len_in_tokens,
            self._tgt_len_in_sents,
            self._tgt_len_in_frags,
            self._tgt_len_in_tokens,
            self._loss,
            self._dense_loss,
            self._ivf_loss,
            self._pq_loss,
        ]

    @classmethod
    def fromlist(cls, l, inst=None):
        if inst is None:
            m = cls()
        else:
            m = inst
        assert len(l) == 15, "Logic error 23822"
        fields = (
            '_cnt',
            '_ncorrect',
            '_total',
            '_src_item_cnt',
            '_tgt_item_cnt',
            '_src_len_in_sents',
            '_src_len_in_frags',
            '_src_len_in_tokens',
            '_tgt_len_in_sents',
            '_tgt_len_in_frags',
            '_tgt_len_in_tokens',
        )
        for f, v in zip(fields, l[:11]):
            m.__dict__[f] = int(v)

        loss_fields = ('_loss', '_dense_loss', '_ivf_loss', '_pq_loss')
        for f, v in zip(loss_fields, l[11:]):
            m.__dict__[f] = v

        return m

    def update_metrics(self, loss, losses_tuple, output: DualEncModelOutput, labels, batch):
        self._cnt += 1
        self._loss += loss
        self._dense_loss += losses_tuple[0].item()
        if (ivf_loss := losses_tuple[1]) is not None:
            self._ivf_loss += ivf_loss.item()
        if (pq_loss := losses_tuple[2]) is not None:
            self._pq_loss += pq_loss.item()

        self._update_metrics_impl(output, labels, batch)

    def _update_metrics_impl(self, output: DualEncModelOutput, labels, batch):
        raise NotImplementedError("implement in subclass")

    def updates_num(self):
        return self._cnt

    def loss(self):
        return self._loss

    def __iadd__(self, other):
        self._cnt += other._cnt
        self._loss += other._loss
        self._dense_loss += other._dense_loss
        self._ivf_loss += other._ivf_loss
        self._pq_loss += other._pq_loss
        self._ncorrect += other._ncorrect
        self._total += other._total
        self._src_item_cnt += other._src_item_cnt
        self._tgt_item_cnt += other._tgt_item_cnt

        self._src_len_in_sents += other._src_len_in_sents
        self._src_len_in_frags += other._src_len_in_frags
        self._src_len_in_tokens += other._src_len_in_tokens

        self._tgt_len_in_sents += other._tgt_len_in_sents
        self._tgt_len_in_frags += other._tgt_len_in_frags
        self._tgt_len_in_tokens += other._tgt_len_in_tokens
        return self

    def metrics(self):
        rec = self._ncorrect / self._total if self._total else 0.0
        return {'rec': rec}

    def best_metric_for_task(self):
        m = self.metrics()
        return 'rec', m['rec']

    def stats(self):
        avg_src_item_cnt = self._src_item_cnt / self._cnt if self._cnt else 0.0
        avg_tgt_item_cnt = self._tgt_item_cnt / self._cnt if self._cnt else 0.0
        stat = {
            'as': avg_src_item_cnt,
            'at': avg_tgt_item_cnt,
        }

        if self._src_len_in_sents and self._src_item_cnt:
            stat['asls'] = self._src_len_in_sents / self._src_item_cnt
            stat['atls'] = self._tgt_len_in_sents / self._tgt_item_cnt
        if self._src_len_in_frags and self._src_item_cnt:
            stat['aslf'] = self._src_len_in_frags / self._src_item_cnt
            stat['atlf'] = self._tgt_len_in_frags / self._tgt_item_cnt
        if self._src_len_in_tokens and self._src_item_cnt:
            stat['aslt'] = self._src_len_in_tokens / self._src_item_cnt
            stat['atlt'] = self._tgt_len_in_tokens / self._tgt_item_cnt

        return stat

    def __str__(self):
        prefix = "; loss %.5f" % (self._loss / self._cnt if self._cnt else 0.0)
        if self._ivf_loss and self._pq_loss:
            prefix += "(dl: %.3f," % (self._dense_loss / self._cnt if self._cnt else 0.0)
            prefix += "ivf: %.3f," % (self._ivf_loss / self._cnt if self._cnt else 0.0)
            prefix += "pq: %.3f)" % (self._pq_loss / self._cnt if self._cnt else 0.0)

        m = self.metrics()
        s = self.stats()
        fmt = '; %s: %.3f' * (len(m) + len(s))
        m_and_s = itertools.chain(m.items(), s.items())
        metrics_str = fmt % tuple(itertools.chain.from_iterable(m_and_s))
        return prefix + metrics_str


class SentRetrMetrics(BaseMetrics):
    def _update_metrics_impl(self, output: DualEncModelOutput, labels, batch):
        _, ypredicted = torch.max(output.dense_score_matrix, 1)
        self._ncorrect += (ypredicted == labels).sum().item()
        self._total += output.dense_score_matrix.size(0)

        self._src_len_in_tokens += sum(batch.src_len)
        self._src_item_cnt += len(batch.src_len)

        self._tgt_len_in_tokens += sum(batch.tgt_len)
        self._tgt_item_cnt += len(batch.tgt_len)


class DocRetrMetrics(BaseMetrics):
    def _update_metrics_impl(self, output: DualEncModelOutput, labels, batch: DocsBatch):
        k = batch.info['max_positives_per_doc']

        pidxs = batch.positive_idxs
        if k == 1:
            _, ypredicted = torch.max(output.dense_score_matrix, 1)
            ll = torch.tensor(pidxs, device=ypredicted.device).squeeze()
            self._ncorrect += (ypredicted == ll).sum().item()
        else:
            for i, gold_idxs in enumerate(pidxs):
                k = len(gold_idxs)
                if k == 0:
                    raise RuntimeError("Imposibru!")
                if k == 1:
                    idx = torch.max(output.dense_score_matrix[i], 0)[1].item()
                    self._ncorrect += idx == gold_idxs[0]
                else:
                    _, idxs = torch.topk(output.dense_score_matrix[i], k, 0)
                    self._ncorrect += sum(1 for j in idxs if j in gold_idxs)

        self._total += labels.sum().item()

        self._src_item_cnt += len(batch.src_ids)
        self._src_len_in_sents += sum(batch.src_doc_len_in_sents)
        self._src_len_in_frags += sum(batch.src_doc_len_in_frags)
        self._src_len_in_tokens += sum(len(t) for t in batch.src_texts)

        self._tgt_item_cnt += len(batch.tgt_ids)
        self._tgt_len_in_sents += sum(batch.tgt_doc_len_in_sents)
        self._tgt_len_in_frags += sum(batch.tgt_doc_len_in_frags)
        self._tgt_len_in_tokens += sum(len(t) for t in batch.tgt_texts)


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
