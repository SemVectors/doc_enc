#!/usr/bin/env python3


import logging
import dataclasses
import csv
import os
from pathlib import Path
import collections
import re
import random
from enum import IntEnum
from typing import Any, List, Optional


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import (
    MISSING,
    OmegaConf,
)
import omegaconf

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F


from doc_enc.doc_encoder import DocEncoderConf, EncodeModule, BatchIterator, file_path_fetcher
from doc_enc.utils import global_init

# * Configs


class LabelsData:
    def __init__(self, labels_index: list[str], labels_mapping: dict[str, int]) -> None:
        self.labels_index = labels_index
        self.labels_mapping = labels_mapping


class LabelStat:
    def __init__(self, nlbls):
        self.seen_labels = torch.zeros(nlbls, dtype=torch.long)

    def update(self, labels):
        self.seen_labels = self.seen_labels + labels.sum(dim=0).cpu()


@dataclasses.dataclass
class FineTuneConf(DocEncoderConf):
    train_meta_path: str = MISSING
    dev_meta_path: str = MISSING
    test_meta_path: str = MISSING
    data_dir: str = MISSING
    save_path: str = ''

    eval_every: int = 1000
    lr: float = 0.0003
    lr_scheduler_kwargs: dict = dataclasses.field(default_factory=dict)
    max_updates: int = 5000

    validation_metric: str = 'micro_F1'
    save_dev_eval_stat: bool = False

    only_eval_test: bool = False


class ClassifHeadType(IntEnum):
    LINEAR = 0
    MLP = 1


@dataclasses.dataclass
class ClassifModuleConf:
    head: ClassifHeadType = ClassifHeadType.LINEAR
    dropout: float = 0.5


@dataclasses.dataclass
class ThresholdPredictorOpts:
    intervals: List[List[float]] = dataclasses.field(default_factory=list)
    cls_intervals: List[List[float]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TopkPredictorOpts:
    topks: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TopkWithThresholdPredictorOpts:
    intervals: List[List[float]] = dataclasses.field(default_factory=list)
    topks: List[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GapPredictorOpts:
    max_gaps: List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ClassifFineTuneConf(FineTuneConf):
    nlabels: int = MISSING
    classif_module: ClassifModuleConf = dataclasses.field(default_factory=ClassifModuleConf)

    try_predictors_on_dev: List[str] = dataclasses.field(default_factory=list)
    threshold_pred_opts: Optional[ThresholdPredictorOpts] = None
    topk_pred_opts: Optional[TopkPredictorOpts] = None
    topk_with_threshold_pred_opts: Optional[TopkWithThresholdPredictorOpts] = None
    gap_pred_opts: Optional[GapPredictorOpts] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=FineTuneConf)
cs.store(name="base_classif_config", node=ClassifFineTuneConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


# * DocClassifier
# ** Classifier heads


class LinearClassifierHead(nn.Module):
    def __init__(self, nlabels: int, in_dim: int, device: torch.device, conf: ClassifModuleConf):
        super().__init__()

        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)
        self.classif_layer = nn.Linear(in_dim, nlabels)
        self.classif_layer = self.classif_layer.to(device=device)

    def forward(self, embeddings: torch.Tensor):
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return self.classif_layer(embeddings)


class MLPClassifierHead(nn.Module):
    def __init__(self, nlabels: int, in_dim: int, device: torch.device, conf: ClassifModuleConf):
        super().__init__()

        self.dense = nn.Linear(in_dim, in_dim, device=device)
        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)

        self.out_proj = nn.Linear(in_dim, nlabels, device=device)

    def forward(self, embeddings: torch.Tensor):
        x = embeddings
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ** Classifier


class DocClassifierModule(nn.Module):
    def __init__(self, nlabels: int, encoder: EncodeModule, conf: ClassifModuleConf):
        super().__init__()
        self.nlabels = nlabels
        self.encoder = encoder

        if conf.head == ClassifHeadType.LINEAR:
            self.cls_head = LinearClassifierHead(
                nlabels, encoder.doc_embs_dim(), self.encoder.device, conf
            )
        elif conf.head == ClassifHeadType.MLP:
            self.cls_head = MLPClassifierHead(
                nlabels, encoder.doc_embs_dim(), self.encoder.device, conf
            )
        else:
            raise RuntimeError(f'Unknown classif head: {conf.head}')

        self.predictor: AbcPredictor | None = None

    def forward(self, docs, doc_fragments):
        embeddings = self.encoder(docs, doc_fragments)
        if self.encoder.last_encode_layer().conf.transformers_torch_fp16:
            embeddings = embeddings.to(dtype=torch.float32)
        return self.cls_head(embeddings)

    def predictions_with_weights(self, docs, doc_fragments) -> list[list[tuple[str, float]]]:
        assert self.predictor is not None, "DocClassifierModule: predictor is not inited!"
        outputs = self.forward(docs, doc_fragments)
        return self.predictor.predictions_with_weights(outputs)


def _create_clsf_module(conf: ClassifFineTuneConf, eval_mode=False) -> DocClassifierModule:
    doc_encoder = EncodeModule(conf, eval_mode=eval_mode)
    model = DocClassifierModule(conf.nlabels, doc_encoder, conf.classif_module)
    return model


def load_clsf_module(conf: DocEncoderConf, eval_mode=True) -> DocClassifierModule:
    doc_encoder = EncodeModule(conf, eval_mode=eval_mode)
    state_dict = doc_encoder._state_dict

    ft_cfg: ClassifFineTuneConf | None = None
    if (ft_cfg := state_dict.get('fine_tune_cfg')) is None:
        raise RuntimeError("Failed to load clsf model: fine_tune_cfg is missing in state_dict")

    nlabels = ft_cfg.nlabels
    try:
        mod_conf = ft_cfg.classif_module
    except omegaconf.errors.ConfigAttributeError:
        head = getattr(ft_cfg, 'classif_head', ClassifHeadType.LINEAR)
        mod_conf = ClassifModuleConf(head=head)

    model = DocClassifierModule(nlabels, doc_encoder, mod_conf)
    labels_index = state_dict['labels_index']
    predictor = None
    # Compatibility with previous versions
    if thresholds := state_dict.get('thresholds'):
        predictor = ThresholdPredictor(nlabels, thresholds, doc_encoder.device, labels_index)
    elif pred_data := doc_encoder._state_dict.get('predictor_data', {}):
        name = pred_data['name']
        predictor = create_predictor(name, nlabels, doc_encoder.device, pred_data, labels_index)
    model.predictor = predictor
    # Compatibility with previous versions
    if 'classif_layer' in doc_encoder._state_dict:
        assert isinstance(
            model.cls_head, LinearClassifierHead
        ), f"Compat error: expected LinearClassifierHead but ({type(model.cls_head)}) "
        model.cls_head.classif_layer.load_state_dict(doc_encoder._state_dict['classif_layer'])
    elif 'cls_head' in doc_encoder._state_dict:
        model.cls_head.load_state_dict(doc_encoder._state_dict['cls_head'])

    return model


# * Data Iterators


class ClassifDataStat:
    def __init__(self):
        self.skipped = 0
        self.labels_per_file = collections.Counter()
        self.cumul_labels_per_text = 0


class ClassifBatchIterator:
    def __init__(
        self, meta_path, base_data_dir, docs_iter: BatchIterator, labels_mapping, device=None
    ) -> None:
        self._docs_iter = docs_iter

        self._path_list = []
        self._labels_list: list[list[int]] = []
        self._device = device
        self._labels_mapping = labels_mapping
        self._multi_label = False

        stat = ClassifDataStat()
        with open(meta_path, 'r', encoding='utf8') as infp:
            reader = csv.reader(infp)
            prev_filename = ''
            cur_file_labels = []
            for row in reader:
                filename, label = row

                if label not in labels_mapping:
                    stat.skipped += 1
                    continue

                stat.labels_per_file[filename] += 1

                if prev_filename != filename:
                    if stat.labels_per_file[filename] != 1:
                        raise RuntimeError(
                            "If it is multi-label classification sort meta file on the filename column. "
                            f"Otherwise it is error in data: duplicating of filename - {filename}"
                        )
                    fp = Path(base_data_dir) / filename
                    self._path_list.append(fp)
                    prev_filename = filename

                    cur_file_labels = [labels_mapping[label]]
                    self._labels_list.append(cur_file_labels)
                else:
                    cur_file_labels.append(labels_mapping[label])

        max_labels_per_file = max(stat.labels_per_file.values())
        self._multi_label = max_labels_per_file > 1

        logging.info(
            "Data stat: texts=%d, skipped_unk_labels=%d, avg_labels_per_text=%.1f, max_labels_per_text=%d",
            len(self._path_list),
            stat.skipped,
            sum(len(lab) for lab in self._labels_list) / len(self._path_list),
            max_labels_per_file,
        )

    def examples_cnt(self):
        return len(self._labels_list)

    def destroy(self):
        self._docs_iter.destroy()

    def multi_label(self):
        return self._multi_label

    def device(self) -> torch.device:
        return self._device if self._device is not None else torch.device('cpu')

    def batches(self):
        self._docs_iter.start_workers_for_item_list(self._path_list, file_path_fetcher)

        for docs, doc_fragments, idxs in self._docs_iter.batches():
            if self._multi_label:
                # Loss function (BCE) expects target to have float type
                labels = torch.full(
                    (len(idxs), len(self._labels_mapping)),
                    0.0,
                    dtype=torch.float32,
                    device=self._device,
                )
                for idx, li in enumerate(idxs):
                    labels[idx][self._labels_list[li]] = 1

            else:
                labels = torch.as_tensor(
                    [self._labels_list[i][0] for i in idxs], dtype=torch.long, device=self._device
                )
            yield docs, doc_fragments, labels


# * Predictors


# ** Basedev predictor


class BaseDevMetricsComputer:
    def __init__(self, nlbl: int, nvars: int, decider_metric='micro_F1'):
        self.nlbl = nlbl

        # per class aggregations
        self.cls_total = torch.zeros(self.nlbl, dtype=torch.int32)
        self.cls_tp = torch.zeros(nvars, self.nlbl, dtype=torch.int32)
        self.cls_predicted = torch.zeros(nvars, self.nlbl, dtype=torch.int32)

        # per example aggregation
        self.decider_metric = decider_metric
        self.decider_tensor: None | torch.Tensor = None
        self.max_idx: int = -1

        self.xmpl_tp = torch.zeros(nvars, dtype=torch.int32)
        self.xmpl_rel = torch.zeros(1, dtype=torch.int32)
        self.xmpl_predicted = torch.zeros(nvars, dtype=torch.int32)

        self.xmpl_rec = torch.zeros(nvars, dtype=torch.float32)
        self.xmpl_prec = torch.zeros(nvars, dtype=torch.float32)
        self.xmpl_ap = torch.zeros(nvars, dtype=torch.float32)
        self.total_examples = 0

    def _update_per_class(
        self, predictions: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ):
        # predictions [ N x nvars x L ]
        # outputs [ N x L ]
        # labels [ N x L ]

        assert self.nlbl == labels.size(1), "Input data has wrong num of labels!"

        self.cls_tp += (predictions & labels.unsqueeze(1).bool()).sum(dim=0).cpu()
        self.cls_predicted += predictions.sum(dim=0).cpu()
        self.cls_total += labels.int().sum(dim=0).cpu()

    def _update_per_example(
        self, predictions: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ):
        # predictions [ N x nvars x L ]
        # outputs [ N x L ]
        # labels [ N x L ]

        # N x V
        tp = (predictions * labels.unsqueeze(1)).sum(dim=2)
        # N
        rel = labels.sum(dim=1)
        # [N x V]
        predicted = predictions.sum(dim=2)
        # [V]
        self.xmpl_tp += tp.sum(dim=0).int().cpu()
        self.xmpl_rel += rel.sum().int().cpu()
        self.xmpl_predicted += predicted.sum(dim=0).int().cpu()

        self.total_examples += labels.size(0)
        self.xmpl_rec += (tp / rel[:, None]).nan_to_num().sum(dim=0).cpu()
        self.xmpl_prec += (tp / predicted).nan_to_num().sum(dim=0).cpu()

        # calc AP
        # N x L
        sort_idx = torch.argsort(outputs, dim=-1, descending=True)
        # N x V x L
        vars_cnt = self.xmpl_tp.size(0)
        sorted_tp = torch.gather(
            predictions * labels.unsqueeze(1),
            2,
            sort_idx.unsqueeze(1).expand(-1, vars_cnt, -1),
        )

        # N x V x L
        prec_at_i = torch.cumsum(sorted_tp, dim=-1) / torch.arange(
            1, labels.size(1) + 1, device=sorted_tp.device
        )
        self.xmpl_ap += ((prec_at_i * sorted_tp).sum(dim=-1) / rel[:, None]).sum(dim=0).cpu()

    def _compute_per_class_metrics(self, var_idx: int):
        # [L]
        tp = self.cls_tp[var_idx]
        predicted = self.cls_predicted[var_idx]
        rec = (tp / self.cls_total).nan_to_num()
        prec = (tp / predicted).nan_to_num()
        f1 = 2 * rec * prec / (rec + prec)
        f1 = f1.nan_to_num()
        return rec, prec, f1

    def _compute_per_class_macro_F1(self, var_idx: int):
        _, _, f1 = self._compute_per_class_metrics(var_idx)
        macro_F1 = f1.sum().item() / self.nlbl
        return macro_F1

    def _compute_per_example_metrics(self) -> dict[str, Any]:
        # V
        micro_rec = (self.xmpl_tp / self.xmpl_rel).nan_to_num()
        micro_prec = (self.xmpl_tp / self.xmpl_predicted).nan_to_num()
        micro_f1 = (2 * micro_rec * micro_prec / (micro_rec + micro_prec)).nan_to_num()

        macro_rec = (self.xmpl_rec / self.total_examples).nan_to_num()
        macro_prec = (self.xmpl_prec / self.total_examples).nan_to_num()
        macro_f1 = (2 * macro_rec * macro_prec / (macro_rec + macro_prec)).nan_to_num()
        mean_ap = (self.xmpl_ap / self.total_examples).nan_to_num()

        return {
            'micro_F1': micro_f1,
            'micro_recall': micro_rec,
            'micro_precision': micro_prec,
            'macro_F1': macro_f1,
            'macro_recall': macro_rec,
            'macro_precision': macro_prec,
            'MAP': mean_ap,
        }

    def _select_best_metric(self) -> tuple[int, dict[str, Any]]:
        metrics = self._compute_per_example_metrics()

        decider_m_tensor = None
        if self.decider_metric not in ('micro_F1', 'macro_F1'):
            raise RuntimeError(f"Unsupported decider metric: {self.decider_metric}!")
        decider_m_tensor = metrics.get(self.decider_metric)
        assert decider_m_tensor is not None, "Logic error comp 32892"
        self.decider_tensor = decider_m_tensor

        max_m_idx = int(torch.argmax(decider_m_tensor).item())
        self.max_idx = max_m_idx

        micro_f1 = metrics['micro_F1']
        micro_rec = metrics['micro_recall']
        micro_prec = metrics['micro_precision']
        macro_f1 = metrics['macro_F1']
        macro_rec = metrics['macro_recall']
        macro_prec = metrics['macro_precision']
        mean_ap = metrics['MAP']
        return max_m_idx, {
            'micro_F1': micro_f1[max_m_idx].item(),
            'micro_recall': micro_rec[max_m_idx].item(),
            'micro_precision': micro_prec[max_m_idx].item(),
            'macro_F1': macro_f1[max_m_idx].item(),
            'macro_recall': macro_rec[max_m_idx].item(),
            'macro_precision': macro_prec[max_m_idx].item(),
            'MAP': mean_ap[max_m_idx].item(),
        }

    def _update(self, predictions: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor):
        self._update_per_class(predictions, outputs, labels)
        self._update_per_example(predictions, outputs, labels)

    def _compute(self):
        best_var_idx, m = self._select_best_metric()
        cls_f1 = self._compute_per_class_macro_F1(best_var_idx)
        m['cls_macro_F1'] = cls_f1
        return best_var_idx, m


# ** Predictors for tunning on dev data


class AbcDevPredictor:
    def name(self):
        raise NotImplementedError("impl name")

    def enumerate_variants(self):
        raise NotImplementedError("impl enumerate_variants")

    def metrics_for_all_variants(self):
        raise NotImplementedError("impl metrics_for_all_variants")

    def cls_metrics_for_best_variant(self):
        raise NotImplementedError("impl cls_metrics_for_best_variant")


def _create_bins(inters_decl: list[list[float]]):
    bins = []
    for inter in inters_decl:
        val, end, step = inter
        while val <= end + 1e-6:
            bins.append(val)
            val += step
    return bins


class DevThresholdPredictor(AbcDevPredictor, BaseDevMetricsComputer):
    def __init__(
        self,
        conf: ClassifFineTuneConf,
        device: torch.device,
        last_eval_results: dict,
    ):
        nlbl = conf.nlabels
        self.last_eval_results = last_eval_results

        # per class aggregations
        self.cls_metric_for_threshold_selecting = 'cls_F1'

        if conf.threshold_pred_opts is not None and (
            cls_inters := conf.threshold_pred_opts.cls_intervals
        ):
            bins = _create_bins(cls_inters)
            self.cls_thresholds = torch.tensor(bins, device=device)
            self.cls_thresh_bins_cnt = len(bins)
        else:
            self.cls_thresh_bins_cnt = 10
            self.cls_thresholds = torch.linspace(0.0, 0.9, self.cls_thresh_bins_cnt).to(
                device=device
            )

        self.per_cls_predicted_vars = [
            torch.zeros(self.cls_thresh_bins_cnt, dtype=torch.int32) for _ in range(nlbl)
        ]
        self.per_cls_tp_vars = [
            torch.zeros(self.cls_thresh_bins_cnt, dtype=torch.int32) for _ in range(nlbl)
        ]

        # per example aggregation
        self.xmpl_max_threshold_vars = 30

        if tvars := self.last_eval_results.get('_threshold_vars'):
            threshold_vars_cnt = len(tvars)
        else:
            if conf.threshold_pred_opts is not None and (
                inters := conf.threshold_pred_opts.intervals
            ):
                bins = _create_bins(inters)
                tvars = [[b] * nlbl for b in bins]
                threshold_vars_cnt = len(bins)

            else:
                threshold_vars_cnt = 10
                tvars = [[v / threshold_vars_cnt] * nlbl for v in range(0, threshold_vars_cnt)]

        self.xmpl_thresholds = torch.tensor(tvars, device=device)

        super().__init__(conf.nlabels, threshold_vars_cnt, conf.validation_metric)

    def _update_per_class_thresh(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]

        assert self.nlbl == labels.size(1), "Input data has wrong num of labels!"

        for i in range(self.nlbl):
            labels_i = labels[:, i].squeeze()
            out_i = outputs[:, i].squeeze()

            # out_i      [ 1    x n ]
            # thresholds [ bins x 1 ]
            # pred_i     [ bins x n ]
            pred_i = out_i >= self.cls_thresholds[:, None]
            # labels_i    [ n ]
            self.per_cls_tp_vars[i] += (pred_i & labels_i.bool()).sum(dim=1).cpu()
            self.per_cls_predicted_vars[i] += pred_i.sum(dim=1).cpu()

    def _select_best_cls_thresholds(self):
        cls_thresholds = []
        for i in range(self.nlbl):
            # cls_total [ nlbl ]
            # tp        [ nbins ]
            rec = self.per_cls_tp_vars[i] / self.cls_total[i]
            prec = self.per_cls_tp_vars[i] / self.per_cls_predicted_vars[i]
            f1 = 2 * rec * prec / (rec + prec)
            metrics_i = {
                'cls_recall': rec.nan_to_num(),
                'cls_precision': prec.nan_to_num(),
                'cls_F1': f1.nan_to_num(),
            }
            assert (
                self.cls_metric_for_threshold_selecting in metrics_i
            ), f'Unknown decider metric {self.cls_metric_for_threshold_selecting}!'
            decider_m = metrics_i[self.cls_metric_for_threshold_selecting]
            if torch.all(decider_m < 0.05):
                # No information to select threshold for this class.
                # Do not predict it for now.
                thresh = 1.0
            else:
                max_m_idx = torch.argmax(decider_m)
                thresh = self.cls_thresholds[max_m_idx].cpu().item()

            cls_thresholds.append(thresh)

        return cls_thresholds

    def name(self):
        return 'threshold'

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        # sig_out = F.sigmoid(outputs)
        self._update_per_class_thresh(outputs, labels)

        # outputs [ N x L ]
        # labels [ N x L ]
        # thresholds [ V x L ]

        # outputs     [ N x 1 x L ]
        # thresholds  [ 1 x V x L ]
        # predictions [ N x V x L ]
        predictions = outputs.unsqueeze(1) >= self.xmpl_thresholds
        self._update(predictions, outputs, labels)

    def compute(self):
        cls_thresholds = self._select_best_cls_thresholds()

        best_var_idx, m = self._compute()

        assert self.decider_tensor is not None, "Logic error 383299"
        if self.decider_tensor.size(0) > self.xmpl_max_threshold_vars:
            _, ind = self.decider_tensor.topk(self.xmpl_max_threshold_vars, sorted=False)
            thresh_vars = self.xmpl_thresholds[ind].tolist()
        else:
            thresh_vars = self.xmpl_thresholds.tolist()

        thresh_vars.append(cls_thresholds)
        m['_threshold_vars'] = thresh_vars
        m['_predictor_data'] = {
            'thresholds': self.xmpl_thresholds[best_var_idx].tolist(),
            'name': self.name(),
        }
        return m

    def enumerate_variants(self):
        vars = []
        for i, var in enumerate(self.xmpl_thresholds.tolist()):
            str_var = ','.join('%.2f' % f for f in var)
            vars.append((i, str_var))
        return vars

    def metrics_for_all_variants(self):
        metrics = self._compute_per_example_metrics()
        assert self.xmpl_thresholds.size(0) == metrics['micro_F1'].size(
            0
        ), "Misaligned thresholds and metrics, 187200"

        return metrics

    def cls_metrics_for_best_variant(self):
        rec, prec, f1 = self._compute_per_class_metrics(self.max_idx)
        assert self.nlbl == rec.size(0), "Misaligned nlbl and class metrics, 498756"
        return {'rec': rec, 'prec': prec, 'f1': f1}


class DevTopkPredictor(AbcDevPredictor, BaseDevMetricsComputer):
    def __init__(self, conf: ClassifFineTuneConf):
        if conf.topk_pred_opts is not None and conf.topk_pred_opts.topks:
            self.topks = conf.topk_pred_opts.topks
        else:
            self.topks = [1, 2, 3, 4, 5]

        super().__init__(conf.nlabels, len(self.topks), conf.validation_metric)

    def name(self):
        return 'topk'

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]

        # predictions [ N x V x L ]
        szs = (outputs.size(0), len(self.topks), outputs.size(1))
        predictions = torch.full(szs, False, dtype=torch.bool, device=outputs.device)
        for i, k in enumerate(self.topks):
            _, ind = torch.topk(outputs, k, sorted=False)
            predictions[:, i, :].scatter_(1, ind, torch.full_like(ind, True, dtype=torch.bool))

        self._update(predictions, outputs, labels)

    def compute(self):
        best_var_idx, m = self._compute()
        m['_predictor_data'] = {'topk': self.topks[best_var_idx], 'name': self.name()}
        return m

    def enumerate_variants(self):
        vars = []
        for i, var in enumerate(self.topks):
            vars.append((i, var))
        return vars

    def metrics_for_all_variants(self):
        metrics = self._compute_per_example_metrics()
        assert len(self.topks) == metrics['micro_F1'].size(
            0
        ), "Misaligned thresholds and metrics, 187200"

        return metrics

    def cls_metrics_for_best_variant(self):
        rec, prec, f1 = self._compute_per_class_metrics(self.max_idx)
        assert self.nlbl == rec.size(0), "Misaligned nlbl and class metrics, 498756"
        return {'rec': rec, 'prec': prec, 'f1': f1}


class DevTopkWithThresholdPredictor(AbcDevPredictor, BaseDevMetricsComputer):
    def __init__(self, conf: ClassifFineTuneConf, device: torch.device):

        if (
            conf.topk_with_threshold_pred_opts is not None
            and conf.topk_with_threshold_pred_opts.topks
        ):
            self.topks = conf.topk_with_threshold_pred_opts.topks
        else:
            self.topks = [2, 3, 4]

        if conf.topk_with_threshold_pred_opts is not None and (
            inters := conf.topk_with_threshold_pred_opts.intervals
        ):
            bins = _create_bins(inters)
            tvars = [[b] * conf.nlabels for b in bins]
            threshold_vars_cnt = len(bins)

        else:
            threshold_vars_cnt = 10
            tvars = [[v / threshold_vars_cnt] * conf.nlabels for v in range(0, threshold_vars_cnt)]

        self.thresholds = torch.tensor(tvars, device=device)

        self.vars_cnt = len(self.topks) * threshold_vars_cnt

        super().__init__(conf.nlabels, self.vars_cnt, conf.validation_metric)

    def name(self):
        return 'topk_with_threshold'

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]

        # predictions [ N x V x L ]
        n = outputs.size(0)
        thresh_cnt = self.thresholds.size(0)
        szs = (n, len(self.topks), thresh_cnt, outputs.size(1))
        predictions = torch.full(szs, False, dtype=torch.bool, device=outputs.device)
        for i, k in enumerate(self.topks):
            _, ind = torch.topk(outputs, k, sorted=False)
            topk_view = predictions[:, i, ...]
            ind = ind.unsqueeze(1).expand(-1, thresh_cnt, -1)
            topk_view.scatter_(2, ind, torch.full_like(ind, True, dtype=torch.bool))
            topk_view.logical_and_(outputs.unsqueeze(1) >= self.thresholds)

        self._update(predictions.reshape(n, self.vars_cnt, -1), outputs, labels)

    def get_var_by_idx(self, idx: int):
        ts = self.thresholds.size(0)
        topk = self.topks[idx // ts]
        threshold = self.thresholds[idx % ts][0].item()
        return topk, threshold

    def compute(self):
        best_var_idx, m = self._compute()
        topk, threshold = self.get_var_by_idx(best_var_idx)
        m['_predictor_data'] = {'topk': topk, 'threshold': threshold, 'name': self.name()}
        return m

    def enumerate_variants(self):
        vars = []
        for i in range(self.vars_cnt):
            topk, threshold = self.get_var_by_idx(i)
            str_var = f'topk={topk}, threshold={threshold}'
            vars.append((i, str_var))
        return vars

    def metrics_for_all_variants(self):
        metrics = self._compute_per_example_metrics()
        assert self.vars_cnt == metrics['micro_F1'].size(
            0
        ), "Misaligned thresholds and metrics, 187200"

        return metrics

    def cls_metrics_for_best_variant(self):
        rec, prec, f1 = self._compute_per_class_metrics(self.max_idx)
        assert self.nlbl == rec.size(0), "Misaligned nlbl and class metrics, 498756"
        return {'rec': rec, 'prec': prec, 'f1': f1}


class DevGapPredictor(AbcDevPredictor, BaseDevMetricsComputer):
    def __init__(self, conf: ClassifFineTuneConf, device: torch.device):
        if conf.gap_pred_opts is not None and conf.gap_pred_opts.max_gaps:
            self.max_gaps = conf.gap_pred_opts.max_gaps
        else:
            self.max_gaps = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]

        self.gaps_tensor = torch.tensor(self.max_gaps, device=device)

        super().__init__(conf.nlabels, len(self.max_gaps), conf.validation_metric)

    def name(self):
        return 'gap'

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]
        # The main idea is find the first gap that is boundary between relevant classes and the rest.
        # For example, outputs are in sorted order:
        # 0.99, 0.95, 0.9, 0.7, 0.6, 0.2
        #                 ^gap
        # So the first three classes will be marked as relevant.

        gap_vars_cnt = len(self.max_gaps)
        sorted, ind = torch.sort(outputs, descending=True)
        gaps = -torch.diff(sorted, prepend=sorted[:, 0].unsqueeze(-1))
        boundaries = gaps.unsqueeze(1) > self.gaps_tensor[None, :, None]
        cs = torch.cumsum(boundaries, -1)
        ind = ind.unsqueeze(1).expand((-1, gap_vars_cnt, -1))
        predictions = torch.gather(cs, 2, ind) == 0

        self._update(predictions, outputs, labels)

    def compute(self):
        best_var_idx, m = self._compute()
        m['_predictor_data'] = {'gap': self.max_gaps[best_var_idx], 'name': self.name()}
        return m

    def enumerate_variants(self):
        vars = []
        for i, var in enumerate(self.max_gaps):
            vars.append((i, var))
        return vars

    def metrics_for_all_variants(self):
        metrics = self._compute_per_example_metrics()
        assert len(self.max_gaps) == metrics['micro_F1'].size(
            0
        ), "Misaligned thresholds and metrics, 187200"

        return metrics

    def cls_metrics_for_best_variant(self):
        rec, prec, f1 = self._compute_per_class_metrics(self.max_idx)
        assert self.nlbl == rec.size(0), "Misaligned nlbl and class metrics, 498756"
        return {'rec': rec, 'prec': prec, 'f1': f1}


# ** Predictors


class AbcPredictor:

    def preds_as_tensor(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Impl preds_as_tensor")

    def predictions(self, outputs: torch.Tensor) -> list[list[str]]:
        raise NotImplementedError("Impl predictions")

    def predictions_with_weights(self, outputs: torch.Tensor) -> list[list[tuple[str, float]]]:
        raise NotImplementedError("Impl predictions_with_weights")


class _BasePredictor(AbcPredictor):
    def __init__(self, labels_index: list):
        self._labels_index = labels_index

    def predictions(self, outputs: torch.Tensor) -> list[list[str]]:
        preds_with_weights = self.predictions_with_weights(outputs)
        return [[t[0] for t in r] for r in preds_with_weights]

    def predictions_with_weights(self, outputs: torch.Tensor) -> list[list[tuple[str, float]]]:
        preds = self.preds_as_tensor(outputs)
        row_indices, column_indices = torch.nonzero(preds, as_tuple=True)
        out = []
        cur_i = 0

        outputs = F.sigmoid(outputs)
        for i in range(outputs.size(0)):
            results = []
            out.append(results)
            if cur_i >= row_indices.size(0) or i != (r := int(row_indices[cur_i].item())):
                continue

            while cur_i < row_indices.size(0) and i == int(row_indices[cur_i].item()):
                c = int(column_indices[cur_i].item())
                results.append((self._labels_index[c], outputs[r, c].item()))
                cur_i += 1
        return out


class ThresholdPredictor(_BasePredictor):
    def __init__(
        self,
        nlbl: int,
        thresholds: list[float] | None,
        device: torch.device,
        labels_index: list,
    ):
        super().__init__(labels_index)

        self.nlbl = nlbl
        if thresholds is None:
            thresholds = [0.5] * nlbl
        else:
            if len(thresholds) != nlbl:
                raise RuntimeError(
                    f"Len of thresholds list should be equal to number of labels: {len(thresholds)} != {nlbl}!"
                )

        self.thresholds = torch.tensor(thresholds, dtype=torch.float, device=device)

    def preds_as_tensor(self, outputs: torch.Tensor):
        predictions = F.sigmoid(outputs)
        predictions = torch.where(torch.lt(predictions, self.thresholds), 0, 1)
        return predictions


class TopkPredictor(_BasePredictor):
    def __init__(self, nlbl: int, topk: int, labels_index: list):
        super().__init__(labels_index)
        self.nlbl = nlbl
        self.topk = topk

    def preds_as_tensor(self, outputs: torch.Tensor):
        outputs = F.sigmoid(outputs)
        _, ind = torch.topk(outputs, self.topk, sorted=False)
        predictions = torch.full(outputs.size(), 0, dtype=torch.long, device=outputs.device)
        # , device=outputs.device
        predictions.scatter_(1, ind, torch.full_like(ind, 1))
        return predictions


class TopkWithThresholdPredictor(_BasePredictor):
    def __init__(self, nlbl: int, topk: int, threshold: float, labels_index: list):
        super().__init__(labels_index)

        self.nlbl = nlbl
        self.topk = topk
        self.threshold = threshold

    def preds_as_tensor(self, outputs: torch.Tensor):
        outputs = F.sigmoid(outputs)
        _, ind = torch.topk(outputs, self.topk, sorted=False)
        predictions = torch.full(outputs.size(), 0, dtype=torch.long, device=outputs.device)
        predictions.scatter_(1, ind, torch.full_like(ind, 1))
        predictions.logical_and_(outputs >= self.threshold)
        return predictions


class GapPredictor(_BasePredictor):
    def __init__(self, nlbl: int, gap: float, labels_index: list):
        super().__init__(labels_index)
        self.nlbl = nlbl
        self.gap = gap

    def preds_as_tensor(self, outputs: torch.Tensor):
        # See DevGapPredictor.__call__
        outputs = F.sigmoid(outputs)

        sorted, ind = torch.sort(outputs, descending=True)
        gaps = -torch.diff(sorted, prepend=sorted[:, 0].unsqueeze(-1))
        boundaries = gaps > self.gap
        cs = torch.cumsum(boundaries, -1)
        predictions = torch.gather(cs, 1, ind) == 0
        return predictions.int()


def create_predictor(
    name: str,
    nlbl: int,
    device: torch.device,
    predictor_data: dict,
    labels_index: list,
):
    if name == 'threshold':
        return ThresholdPredictor(nlbl, predictor_data.get('thresholds'), device, labels_index)
    elif name == 'topk':
        return TopkPredictor(nlbl, predictor_data.get('topk', 3), labels_index)
    elif name == 'topk_with_threshold':
        return TopkWithThresholdPredictor(
            nlbl, predictor_data.get('topk', 3), predictor_data.get('threshold', 0.5), labels_index
        )
    elif name == 'gap':
        return GapPredictor(nlbl, predictor_data.get('gap', 3), labels_index)
    else:
        raise RuntimeError(f'Unknown predictor: {name}')


# * Train loop


class RunningStat:
    def __init__(self):
        self.loss = 0.0
        self.docs = 0
        # metrics
        self.correct = 0
        self.total = 0
        # multi-label
        self.correct_exact = 0
        self.correct_belong = 0
        self.correct_contain = 0

        self._multi_label = False

    def reset(self):
        self.loss = 0.0
        self.docs = 0
        self.correct = 0
        self.total = 0
        self.correct_exact = 0
        self.correct_belong = 0
        self.correct_contain = 0

    def update_running_metric(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        multi_label: bool,
        predictor: AbcPredictor,
    ):
        self._multi_label = multi_label
        with torch.no_grad():
            if multi_label:
                predicted = predictor.preds_as_tensor(outputs)

                self.correct += (predicted == labels).int().sum().cpu().item()
                self.total += labels.numel()
                # Exact match
                self.correct_exact += torch.all(predicted == labels, dim=1).sum()
                # contain - have to find all true classes, but don't mind FP.
                self.correct_contain += torch.all((predicted - labels) >= 0, dim=1).sum()
                # belong - have to be very precise but it's ok to skip some true classes.
                self.correct_belong += torch.all((predicted - labels) <= 0, dim=1).sum()
            else:
                _, predicted = torch.max(outputs, 1)
                self.correct += (predicted == labels).int().sum().cpu().item()
                self.total += labels.size(0)

    def metrics_str(self):
        acc = self.correct / self.total
        mlm = ''
        if self._multi_label:
            mlm = (
                f", acc_contain: {self.correct_contain/self.docs:.2f}, acc_belong: {self.correct_belong/self.docs:.2f}, "
                f"acc_exact: {self.correct_exact/self.docs:.3f}"
            )

        return f"acc: {acc:.2f}" + mlm


def _create_label_mapping(meta_path):
    unique_labels = []
    labels_mapping = {}
    with open(meta_path, 'r', encoding='utf8') as infp:
        reader = csv.reader(infp)
        for row in reader:
            label = row[1]
            if label not in labels_mapping:
                labels_mapping[label] = len(unique_labels)
                unique_labels.append(label)
    return LabelsData(unique_labels, labels_mapping)


def _train_classif(conf: ClassifFineTuneConf):
    torch.manual_seed(2025 * 8)
    random.seed(2025 * 9)

    labels_data = _create_label_mapping(conf.train_meta_path)
    if OmegaConf.is_missing(conf, 'nlabels'):
        conf.nlabels = len(labels_data.labels_index)
    logging.info(
        "Found %s unique labels, first 20: %s",
        len(labels_data.labels_index),
        labels_data.labels_index[:20],
    )

    model = _create_clsf_module(conf)
    logging.info("Model:\n%s", model)

    train_iter = ClassifBatchIterator(
        conf.train_meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=False),
        labels_mapping=labels_data.labels_mapping,
        device=model.encoder.device,
    )
    try:
        _train_loop(conf, train_iter, model, labels_data=labels_data)
    finally:
        train_iter.destroy()


def classif_fine_tune(conf: ClassifFineTuneConf):
    if not conf.only_eval_test:
        _train_classif(conf)

    if not OmegaConf.is_missing(conf, 'test_meta_path'):
        if not conf.only_eval_test:
            conf.model_path = conf.save_path

        logging.info("Evaling on test:")
        logging.info("Loading saved model from %s", conf.model_path)

        model = load_clsf_module(conf)
        conf.nlabels = model.nlabels
        labels_mapping = model.encoder._state_dict['labels_mapping']
        labels_index = model.encoder._state_dict['labels_index']
        with torch.inference_mode():
            metrics = eval_on_dataset(
                conf, conf.test_meta_path, model, LabelsData(labels_index, labels_mapping)
            )
        logging.info("Test metrics:\n%s", metrics)
        print('micro_F1,macro_F1')
        print(f'{metrics["micro_F1"]:.3f},{metrics["macro_F1"]:.3f}')


def _eval_on_dev_and_maybe_save(
    conf, model, prev_best_metrics, labels_data: LabelsData, labels_stat: LabelStat
):
    with torch.no_grad():
        eval_result = eval_on_dataset(
            conf,
            conf.dev_meta_path,
            model,
            labels_data=labels_data,
            last_eval_results=prev_best_metrics,
            labels_stat=labels_stat,
        )

    logging.info(
        "\nEval metrics: %s\n",
        '\n'.join(f'{m}: {v}' for m, v in eval_result.items() if not m.startswith('_')),
    )

    if (m := re.match(r'F1_(\d+)', conf.validation_metric)) is not None:
        cls_num = int(m.group(1))
        if f1m := eval_result.get('F1'):
            m = f1m[cls_num]
        else:
            raise RuntimeError(
                f'Validation metric (F1) not found among metrics: {list(eval_result.keys())}'
            )
        prev_best_metric = 0
        if prev_f1m := prev_best_metrics.get('F1'):
            prev_best_metric = prev_f1m[cls_num]
    elif (m := eval_result.get(conf.validation_metric)) is not None:
        prev_best_metric = prev_best_metrics.get(conf.validation_metric, 0.0)
    else:
        raise RuntimeError(
            f'Validation metric ({conf.validation_metric}) not found among metrics: {list(eval_result.keys())}'
        )

    logging.info(
        "%s on dev %.3f; best %s %.3f; per classes: %s",
        conf.validation_metric,
        m,
        conf.validation_metric,
        prev_best_metric,
        eval_result.get('predictions_per_cls', 'N/A'),
    )
    if m >= prev_best_metric:
        _save_model(conf, model, eval_result, labels_data)
        logging.info("new best model was saved")
        prev_best_metrics = eval_result
    return prev_best_metrics


def _calc_loss(outputs: torch.Tensor, labels: torch.Tensor, multi_label: bool):
    if multi_label:
        pos_cnt_t = labels.sum(dim=0).to(dtype=torch.float32)
        pos_weight = (outputs.new_tensor(labels.size(0)) - pos_cnt_t) / pos_cnt_t
        pos_weight[pos_weight == 0] = 1

        pos_weight = pos_weight.nan_to_num(nan=1, posinf=1)
        return F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)
    else:
        return F.cross_entropy(outputs, labels)


def _train_loop(
    conf: ClassifFineTuneConf,
    train_iter: ClassifBatchIterator,
    model: DocClassifierModule,
    labels_data: LabelsData,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        conf.lr,
        total_steps=conf.max_updates,
        **conf.lr_scheduler_kwargs,
    )

    scaler = GradScaler(enabled=conf.enable_amp)
    last_eval_result = {}
    update_nums = 0
    epoch = 0
    running_stat = RunningStat()
    labels_stat = LabelStat(conf.nlabels)
    predictor = create_predictor('threshold', conf.nlabels, train_iter.device(), {}, [])

    while update_nums < conf.max_updates:
        epoch += 1
        logging.info("Starting %s epoch", epoch)
        for docs, doc_fragments, labels in train_iter.batches():
            # zero the parameter gradients
            model.train(mode=True)
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast(train_iter.device().type, enabled=conf.enable_amp):
                outputs = model(docs, doc_fragments)
                loss = _calc_loss(outputs, labels, train_iter.multi_label())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_stat.loss += loss.item()
            running_stat.docs += labels.size(0)
            running_stat.update_running_metric(outputs, labels, train_iter.multi_label(), predictor)
            labels_stat.update(labels)
            update_nums += 1

            rep_every = 10
            if update_nums % rep_every == 0:
                logging.info(
                    "#%d, docs: %.1f, avg loss: %.3f, lr:%.5e, %s",
                    update_nums,
                    running_stat.docs,
                    running_stat.loss / rep_every,
                    scheduler.get_last_lr()[0],
                    running_stat.metrics_str(),
                )
                running_stat.reset()

            if update_nums % conf.eval_every == 0:
                last_eval_result = _eval_on_dev_and_maybe_save(
                    conf, model, last_eval_result, labels_data, labels_stat
                )
                predictor = create_predictor(
                    last_eval_result['predictor'],
                    conf.nlabels,
                    train_iter.device(),
                    last_eval_result['_predictor_data'],
                    [],
                )
            if update_nums >= conf.max_updates:
                break


# * Eval


class MultiClassEvaluator:
    def __init__(self, nlbl: int):
        self.nlbl = nlbl
        self.correct = torch.zeros(1)
        self.total = 0
        self.cls_total = torch.zeros(nlbl, dtype=torch.int32)
        self.cls_predicted = torch.zeros(nlbl, dtype=torch.int32)
        self.tp = torch.zeros(nlbl, dtype=torch.int32)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).int().sum().cpu()
        for i in range(self.nlbl):
            self.cls_predicted[i] += (predicted == i).int().sum().cpu()
            self.cls_total[i] += (labels == i).int().sum().cpu()
            self.tp[i] += (predicted[(labels == i)] == i).int().sum().cpu()

    def compute(self):
        if self.total == 0:
            return {}

        rec = self.tp / self.cls_total
        prec = self.tp / self.cls_predicted
        f1 = 2 * rec * prec / (rec + prec)
        f1 = f1.nan_to_num()
        metrics = {
            'micro_F1': self.correct.item() / self.total,
            'macro_F1': f1.mean().item(),
            'recall': rec.nan_to_num().tolist(),
            'precision': prec.nan_to_num().tolist(),
            'F1': f1.tolist(),
            'predictions_per_cls': [c.item() / self.total for c in self.cls_predicted],
        }
        return metrics


class MultiLabelTestEvaluator:
    def __init__(self, nlbl: int, predictor: AbcPredictor):
        self.nlbl = nlbl
        self.predictor = predictor

        self.metric_computer = BaseDevMetricsComputer(nlbl, 1)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        predictions = self.predictor.preds_as_tensor(outputs)
        self.metric_computer._update(predictions.unsqueeze(1), outputs, labels)

    def compute(self):
        _, m = self.metric_computer._compute()
        return m


class MultiLabelDevEvaluator:
    """MultiLabelTestEvaluator is used for evaluation at test time for given thresholds,
    whereas this class is used to tune thresholds on dev data."""

    def __init__(
        self,
        conf: ClassifFineTuneConf,
        device: torch.device,
        labels_data: LabelsData,
        last_eval_results: dict,
        labels_stat: LabelStat | None,
    ):
        self.conf = conf
        self.nlbl = conf.nlabels
        self.labels_data = labels_data
        self.last_eval_results = last_eval_results
        self.labels_stat = labels_stat
        self.eval_num = last_eval_results.get('_upd_num', 1)
        self.decider_metric = conf.validation_metric

        self.predictors = []

        if conf.try_predictors_on_dev:
            for pn in conf.try_predictors_on_dev:
                if pn == 'topk':
                    self.predictors.append(DevTopkPredictor(conf))
                elif pn == 'threshold':
                    self.predictors.append(
                        DevThresholdPredictor(
                            conf, device, last_eval_results.get(f'_{pn}-pred', {})
                        )
                    )
                elif pn == 'topk_with_threshold':
                    self.predictors.append(DevTopkWithThresholdPredictor(conf, device))
                elif pn == 'gap':
                    self.predictors.append(DevGapPredictor(conf, device))
                else:
                    raise RuntimeError(f'Unknown predictor: {pn}')

        else:
            self.predictors = [
                DevThresholdPredictor(conf, device, last_eval_results.get('_thresh-pred', {})),
            ]

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        sig_out = F.sigmoid(outputs)
        for predictor in self.predictors:
            predictor(sig_out, labels)

    def _save_predictor_stat(self, predictor: AbcDevPredictor):
        stat_dir = Path(f'./eval_stat/{self.eval_num:03}/{predictor.name()}')
        stat_dir.mkdir(parents=True, exist_ok=True)

        vars = predictor.enumerate_variants()
        with open(stat_dir / 'variants.csv', 'w') as outf:
            wrtr = csv.writer(outf)
            wrtr.writerow(('num', 'var_repr'))
            for var_num, var in vars:
                wrtr.writerow((var_num, var))

        def _fmt_v(val):
            return '%.5f' % val.item()

        with open(stat_dir / 'variant_metrics.csv', 'w') as outf:
            wrtr = csv.writer(outf)
            metrics = predictor.metrics_for_all_variants()
            mks = tuple(metrics.keys())
            wrtr.writerow(('num',) + mks)
            for var_num in range(len(vars)):
                row = [var_num]
                for mk in mks:
                    row.append(_fmt_v(metrics[mk][var_num]))
                wrtr.writerow(row)

        with open(stat_dir / 'cls_metrics_for_best_variant.csv', 'w') as outf:
            wrtr = csv.writer(outf)
            metrics = predictor.cls_metrics_for_best_variant()
            mks = tuple(metrics.keys())
            wrtr.writerow(('lbl', 'seen') + mks)
            seen_nums = 0
            for lbl_num in range(self.nlbl):
                if self.labels_stat is not None:
                    seen_nums = int(self.labels_stat.seen_labels[lbl_num].item())
                row = [self.labels_data.labels_index[lbl_num], seen_nums]
                for mk in mks:
                    row.append(_fmt_v(metrics[mk][lbl_num]))
                wrtr.writerow(row)

    def compute(self):
        pred_results = []
        eval_result = {}
        for predictor in self.predictors:
            m = predictor.compute()
            m['predictor'] = predictor.name()
            eval_result[f'_{predictor.name()}-pred'] = m
            pred_results.append((m[self.decider_metric], m))
            if self.conf.save_dev_eval_stat:
                self._save_predictor_stat(predictor)

        _, m = max(pred_results, key=lambda t: t[0])
        for k, v in m.items():
            if k.startswith('_') and k != '_predictor_data':
                continue
            eval_result[k] = v

        eval_result['_upd_num'] = self.eval_num + 1
        return eval_result


def eval_on_dataset(
    conf: ClassifFineTuneConf,
    meta_path,
    model: DocClassifierModule,
    labels_data: LabelsData,
    last_eval_results: dict | None = None,
    labels_stat: LabelStat | None = None,
):
    model.eval()
    device = model.encoder.device
    test_iter = ClassifBatchIterator(
        meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=True),
        labels_mapping=labels_data.labels_mapping,
        device=device,
    )

    if test_iter.multi_label():
        if model.predictor is not None:
            evalor = MultiLabelTestEvaluator(conf.nlabels, predictor=model.predictor)
        else:
            assert last_eval_results is not None, "Logic error dev 821"
            evalor = MultiLabelDevEvaluator(
                conf,
                device=device,
                labels_data=labels_data,
                last_eval_results=last_eval_results,
                labels_stat=labels_stat,
            )
    else:
        evalor = MultiClassEvaluator(conf.nlabels)

    for docs, doc_fragments, labels in test_iter.batches():

        with autocast(device.type, enabled=conf.enable_amp):
            output = model(docs, doc_fragments)

        evalor(output, labels)

    return evalor.compute()


def _save_model(
    conf: FineTuneConf, model: DocClassifierModule, eval_results: dict, labels_data: LabelsData
):
    d = model.encoder.to_dict()
    d['fine_tune_cfg'] = OmegaConf.to_container(
        conf, structured_config_mode=omegaconf.SCMode.INSTANTIATE
    )
    d['cls_head'] = model.cls_head.state_dict()
    d['labels_index'] = labels_data.labels_index
    d['labels_mapping'] = labels_data.labels_mapping
    if '_predictor_data' in eval_results:
        d['predictor_data'] = eval_results['_predictor_data']

    if not conf.save_path:
        conf.save_path = os.path.join(os.getcwd(), 'model.pt')

    if not (p := Path(conf.save_path).parent).exists():
        p.mkdir(parents=True)

    torch.save(d, conf.save_path)


@hydra.main(config_path=None, config_name="config", version_base=None)
def fine_tune_classif_cli(conf: ClassifFineTuneConf) -> None:
    classif_fine_tune(conf)


if __name__ == "__main__":
    global_init()
    fine_tune_classif_cli()
