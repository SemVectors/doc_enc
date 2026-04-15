#!/usr/bin/env python3

import itertools

import torch

from doc_enc.training.types import DocsBatch, SentsBatch, TaskType
from doc_enc.training.models.base_model import DualEncModelOutput


class BaseMetrics:
    """Quality metrics and other stats"""

    _metric_name = 'rec'

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
    def fromlist(cls, ml, inst=None):
        if inst is None:
            m = cls()
        else:
            m = inst
        assert len(ml) == 15, "Logic error 23822"
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
        for f, v in zip(fields, ml[:11]):
            m.__dict__[f] = int(v)

        loss_fields = ('_loss', '_dense_loss', '_ivf_loss', '_pq_loss')
        for f, v in zip(loss_fields, ml[11:]):
            m.__dict__[f] = v

        return m

    def update_metrics(
        self,
        loss,
        losses_tuple,
        output: DualEncModelOutput,
        labels: torch.Tensor,
        batch: DocsBatch | SentsBatch,
    ):
        self._cnt += 1
        self._loss += loss
        self._dense_loss += losses_tuple[0].item()
        if (ivf_loss := losses_tuple[1]) is not None:
            self._ivf_loss += ivf_loss.item()
        if (pq_loss := losses_tuple[2]) is not None:
            self._pq_loss += pq_loss.item()

        self._update_metrics_impl(output, labels, batch)

    def _update_metrics_impl(
        self,
        output: DualEncModelOutput,
        labels: torch.Tensor,
        batch: DocsBatch | SentsBatch,
    ):
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
        return {self._metric_name: rec}

    def best_metric_for_task(self):
        m = self.metrics()
        return self._metric_name, m[self._metric_name]

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
    def _update_metrics_impl(
        self,
        output: DualEncModelOutput,
        labels: torch.Tensor,
        batch: DocsBatch | SentsBatch,
    ):
        assert isinstance(
            batch, SentsBatch
        ), "SentRetrMetrics:_update_metrics_impl batch is not instance of SentsBatch"
        _, ypredicted = torch.max(output.dense_score_matrix, 1)
        self._ncorrect += (ypredicted == labels).sum().item()
        self._total += output.dense_score_matrix.size(0)

        sd = batch.src_data
        self._src_len_in_tokens += sd.seq_encoder_input.ntokens()
        self._src_item_cnt += len(sd.text_ids)

        td = batch.tgt_data
        self._tgt_len_in_tokens += td.seq_encoder_input.ntokens()
        self._tgt_item_cnt += len(td.text_ids)


class DocRetrMetrics(BaseMetrics):
    _metric_name = 'rec@10'

    def _update_metrics_impl(
        self,
        output: DualEncModelOutput,
        labels: torch.Tensor,
        batch: DocsBatch | SentsBatch,
    ):
        assert isinstance(
            batch, DocsBatch
        ), "DocRetrMetrics:_update_metrics_impl batch is not instance of DocsBatch"

        k = min(10, batch.get_tgt_docs_cnt())

        m = output.dense_score_matrix
        vals, _ = torch.topk(m, k, 1)
        preds = labels * torch.where(
            m >= vals[:, -1].unsqueeze(-1), m, torch.zeros(1, device=m.device)
        )
        self._ncorrect += torch.sum(preds != 0).item()
        self._total += labels.sum().item()

        sd = batch.src_data
        self._src_item_cnt += len(sd.text_ids)
        self._src_len_in_sents += sum(sd.texts_repr.text_lengths_in_sents())
        self._src_len_in_frags += sum(sd.texts_repr.text_lengths_in_fragments())
        self._src_len_in_tokens += sd.seq_encoder_input.ntokens()

        td = batch.tgt_data
        self._tgt_item_cnt += len(td.text_ids)
        self._tgt_len_in_sents += sum(td.texts_repr.text_lengths_in_sents())
        self._tgt_len_in_frags += sum(td.texts_repr.text_lengths_in_fragments())
        self._tgt_len_in_tokens += td.seq_encoder_input.ntokens()


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
