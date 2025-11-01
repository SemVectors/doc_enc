#!/usr/bin/env python3


import logging
import dataclasses
import csv
import os
from pathlib import Path
import collections
import re
import random
from enum import Enum


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F


from doc_enc.doc_encoder import DocEncoderConf, EncodeModule, BatchIterator, file_path_fetcher
from doc_enc.utils import global_init

# * Configs


@dataclasses.dataclass
class FineTuneConf(DocEncoderConf):
    train_meta_path: str = MISSING
    dev_meta_path: str = MISSING
    test_meta_path: str = MISSING
    data_dir: str = MISSING
    save_path: str = ''

    eval_every: int = 1000
    dropout: float = 0.5
    lr: float = 0.0003
    lr_scheduler_kwargs: dict = dataclasses.field(default_factory=dict)
    max_updates: int = 5000

    validation_metric: str = 'micro_F1'

    only_eval_test: bool = False


class ClassifHeadType(Enum):
    LINEAR = 0
    MLP = 1


@dataclasses.dataclass
class ClassifFineTuneConf(FineTuneConf):
    nlabels: int = MISSING
    classif_head: ClassifHeadType = ClassifHeadType.LINEAR


cs = ConfigStore.instance()
cs.store(name="base_config", node=FineTuneConf)
cs.store(name="base_classif_config", node=ClassifFineTuneConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


# * Models


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, device: torch.device, conf: ClassifFineTuneConf):
        super().__init__()

        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)
        self.classif_layer = nn.Linear(in_dim, conf.nlabels)
        self.classif_layer = self.classif_layer.to(device=device)

    def forward(self, embeddings: torch.Tensor):
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return self.classif_layer(embeddings)


class MLPClassifierHead(nn.Module):
    def __init__(self, in_dim: int, device: torch.device, conf: ClassifFineTuneConf):
        super().__init__()

        self.dense = nn.Linear(in_dim, in_dim, device=device)
        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)

        self.out_proj = nn.Linear(in_dim, conf.nlabels, device=device)

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


class DocClassifier(nn.Module):
    def __init__(self, encoder: EncodeModule, conf: ClassifFineTuneConf):
        super().__init__()
        self.encoder = encoder

        if conf.classif_head == ClassifHeadType.LINEAR:
            self.cls_head = LinearClassifierHead(encoder.doc_embs_dim(), self.encoder.device, conf)
        else:
            self.cls_head = MLPClassifierHead(encoder.doc_embs_dim(), self.encoder.device, conf)

        self.thresholds: list | None = None

    def forward(self, docs, doc_fragments):
        embeddings = self.encoder(docs, doc_fragments)
        return self.cls_head(embeddings)


def _create_model(conf: ClassifFineTuneConf, eval_mode=False):
    doc_encoder = EncodeModule(conf, eval_mode=eval_mode)

    if (ft_cfg := doc_encoder._state_dict.get('fine_tune_cfg')) is not None:
        # update conf's nlabels
        conf.nlabels = ft_cfg.nlabels
        # uses this loaded config for DocClassifier loading
        conf = ft_cfg

    model = DocClassifier(doc_encoder, conf)
    model.thresholds = doc_encoder._state_dict.get('thresholds')
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
        self, outputs: torch.Tensor, labels: torch.Tensor, multi_label: bool, metrics: dict
    ):
        self._multi_label = multi_label
        with torch.no_grad():
            if multi_label:
                thresholds = metrics.get('thresholds')
                if thresholds is None:
                    thresholds = [0.5] * labels.size(1)
                t_t = torch.tensor(thresholds, dtype=torch.float, device=outputs.device)

                predicted = torch.where(F.sigmoid(outputs) < t_t, 0, 1)
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
    return unique_labels, labels_mapping


def _train_classif(conf: ClassifFineTuneConf):
    torch.manual_seed(2025 * 8)
    random.seed(2025 * 9)

    labels_index, labels_mapping = _create_label_mapping(conf.train_meta_path)
    if OmegaConf.is_missing(conf, 'nlabels'):
        conf.nlabels = len(labels_index)
    logging.info("Found %s unique labels, first 20: %s", len(labels_index), labels_index[:20])

    model = _create_model(conf)
    logging.info("Model:\n%s", model)

    train_iter = ClassifBatchIterator(
        conf.train_meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=False),
        labels_mapping=labels_mapping,
        device=model.encoder.device,
    )
    try:
        _train_loop(
            conf, train_iter, model, labels_index=labels_index, labels_mapping=labels_mapping
        )
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

        model = _create_model(conf, eval_mode=True)
        labels_mapping = model.encoder._state_dict['labels_mapping']
        with torch.inference_mode():
            metrics = eval_on_dataset(
                conf, conf.test_meta_path, model, labels_mapping=labels_mapping
            )
        logging.info("Test metrics:\n%s", metrics)
        print('micro_F1,macro_F1')
        print(f'{metrics["micro_F1"]:.3f},{metrics["macro_F1"]:.3f}')


def _eval_on_dev_and_maybe_save(conf, model, prev_best_metrics, labels_index, labels_mapping):
    with torch.no_grad():
        metrics = eval_on_dataset(
            conf,
            conf.dev_meta_path,
            model,
            labels_mapping=labels_mapping,
            last_eval_results=prev_best_metrics,
        )

    logging.info(
        "\nEval metrics: %s\n",
        '\n'.join(f'{m}: {v}' for m, v in metrics.items() if not m.startswith('_')),
    )

    if (m := re.match(r'F1_(\d+)', conf.validation_metric)) is not None:
        cls_num = int(m.group(1))
        if f1m := metrics.get('F1'):
            m = f1m[cls_num]
        else:
            raise RuntimeError(
                f'Validation metric (F1) not found among metrics: {list(metrics.keys())}'
            )
        prev_best_metric = 0
        if prev_f1m := prev_best_metrics.get('F1'):
            prev_best_metric = prev_f1m[cls_num]
    elif (m := metrics.get(conf.validation_metric)) is not None:
        prev_best_metric = prev_best_metrics.get(conf.validation_metric, 0.0)
    else:
        raise RuntimeError(
            f'Validation metric ({conf.validation_metric}) not found among metrics: {list(metrics.keys())}'
        )

    logging.info(
        "%s on dev %.3f; best %s %.3f; per classes: %s",
        conf.validation_metric,
        m,
        conf.validation_metric,
        prev_best_metric,
        metrics.get('predictions_per_cls', 'N/A'),
    )
    if m >= prev_best_metric:
        _save_model(
            conf,
            model,
            metrics,
            labels_index=labels_index,
            labels_mapping=labels_mapping,
        )
        logging.info("new best model was saved")
        prev_best_metrics = metrics
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
    model: DocClassifier,
    labels_index,
    labels_mapping,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        conf.lr,
        total_steps=conf.max_updates,
        **conf.lr_scheduler_kwargs,
    )

    scaler = GradScaler(enabled=conf.enable_amp)
    best_metrics = {}
    update_nums = 0
    epoch = 0
    running_stat = RunningStat()

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
            running_stat.update_running_metric(
                outputs, labels, train_iter.multi_label(), best_metrics
            )
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
                best_metrics = _eval_on_dev_and_maybe_save(
                    conf, model, best_metrics, labels_index, labels_mapping
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
    def __init__(self, nlbl: int, device: torch.device, thresholds: list):
        self.nlbl = nlbl
        if len(thresholds) != nlbl:
            raise RuntimeError(
                f"Len of thresholds list should be equal to number of labels: {len(thresholds)} != {nlbl}!"
            )
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32, device=device)

        # per class aggregations
        self.cls_total = torch.zeros(nlbl, dtype=torch.int32)
        self.cls_predicted = torch.zeros(nlbl, dtype=torch.int32)
        self.cls_tp = torch.zeros(nlbl, dtype=torch.int32)

        # per examples aggregation
        # Micro
        self.xmpl_tp = 0.0
        self.xmpl_rel = 0.0
        self.xmpl_predicted = 0.0
        # Macro
        self.xmpl_rec = 0.0
        self.xmpl_prec = 0.0
        self.xmpl_ap = 0.0
        self.total_examples = 0

    def _update_per_class(self, predictions: torch.Tensor, labels: torch.Tensor):
        # Predictions are already either 0 or 1 for each class, just compute
        # metrics for binary classification of each class.
        for i in range(self.nlbl):
            labels_i = labels[:, i]
            pred_i = predictions[:, i]
            self.cls_tp[i] += (pred_i * labels_i).int().sum().cpu().item()
            self.cls_total[i] += labels_i.int().sum().cpu().item()
            self.cls_predicted[i] += pred_i.sum().cpu().item()

    def _update_per_example(
        self, outputs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor
    ):
        tp = (predictions * labels).sum(dim=1)
        rel = labels.sum(dim=1)
        predicted = predictions.sum(dim=1)
        self.xmpl_tp += tp.sum().item()
        self.xmpl_rel += rel.sum().item()
        self.xmpl_predicted += predicted.sum().item()

        self.total_examples += labels.size(0)
        self.xmpl_rec += (tp / rel).nan_to_num().sum().item()
        self.xmpl_prec += (tp / predicted).nan_to_num().sum().item()

        sort_idx = torch.argsort(outputs, dim=-1, descending=True)
        # N x L
        sorted_tp = torch.gather(predictions * labels, -1, sort_idx)
        # N x L
        prec_at_i = torch.cumsum(sorted_tp, dim=-1) / torch.arange(
            1, labels.size(1) + 1, device=sorted_tp.device
        )

        self.xmpl_ap += ((prec_at_i * sorted_tp).sum(dim=-1) / rel).sum().item()

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        predictions = F.sigmoid(outputs)
        predictions = torch.where(torch.lt(predictions, self.thresholds), 0, 1)
        self._update_per_class(predictions, labels)

        self._update_per_example(outputs, predictions, labels)

    def _compute_per_class_metrics(self):
        rec = self.cls_tp / self.cls_total
        prec = self.cls_tp / self.cls_predicted
        f1 = 2 * rec * prec / (rec + prec)
        f1 = f1.nan_to_num()
        metrics = {
            'cls_macro_F1': f1.mean().item(),
            'cls_recall': rec.nan_to_num().tolist(),
            'cls_precision': prec.nan_to_num().tolist(),
            'cls_F1': f1.tolist(),
        }
        return metrics

    def _compute_per_example_metrics(self):
        if self.total_examples:
            macro_rec = self.xmpl_rec / self.total_examples
            macro_prec = self.xmpl_prec / self.total_examples
            macro_f1 = 2 * macro_rec * macro_prec / (macro_prec + macro_rec)
            mean_ap = self.xmpl_ap / self.total_examples
        else:
            macro_rec = 0
            macro_prec = 0
            macro_f1 = 0
            mean_ap = 0

        if self.xmpl_predicted:
            micro_prec = self.xmpl_tp / self.xmpl_predicted
        else:
            micro_prec = 0

        if self.xmpl_rel:
            micro_rec = self.xmpl_tp / self.xmpl_rel
        else:
            micro_rec = 0

        if micro_prec > 0 or micro_rec > 0:
            micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        else:
            micro_f1 = 0

        return {
            'micro_F1': micro_f1,
            'micro_recall': micro_rec,
            'micro_precision': micro_prec,
            'macro_F1': macro_f1,
            'macro_recall': macro_rec,
            'macro_precision': macro_prec,
            'MAP': mean_ap,
        }

    def compute(self):
        metrics = self._compute_per_class_metrics()
        metrics.update(self._compute_per_example_metrics())
        return metrics


class MultiLabelDevEvaluator:
    """MultiLabelTestEvaluator is used for evaluation at test time for given thresholds,
    whereas this class is used to compute best thresholds for each label."""

    def __init__(
        self,
        nlbl: int,
        device: torch.device,
        last_eval_results: dict,
        select_threshold_metric: str = 'micro_F1',
    ):
        self.nlbl = nlbl
        self.last_eval_results = last_eval_results

        # per class aggregations
        self.cls_metric_for_threshold_selecting = 'cls_F1'
        self.cls_thresh_bins_cnt = 10
        self.cls_total = torch.zeros(nlbl, dtype=torch.int32)
        self.cls_predicted = [
            torch.zeros(self.cls_thresh_bins_cnt, dtype=torch.int32) for _ in range(nlbl)
        ]
        self.cls_tp = [
            torch.zeros(self.cls_thresh_bins_cnt, dtype=torch.int32) for _ in range(nlbl)
        ]
        self.cls_thresholds = torch.linspace(0.0, 0.9, self.cls_thresh_bins_cnt).to(device=device)

        # per example aggregation
        self.select_threshold_metric = select_threshold_metric
        self.xmpl_max_threshold_vars = 20
        if tvars := self.last_eval_results.get('_threshold_vars'):
            threshold_vars_cnt = len(tvars)
        else:
            threshold_vars_cnt = 10
            tvars = [[v / threshold_vars_cnt] * nlbl for v in range(0, threshold_vars_cnt)]

        self.xmpl_thresholds = torch.tensor(tvars, device=device)

        self.xmpl_tp = torch.zeros(threshold_vars_cnt, dtype=torch.int32)
        self.xmpl_rel = torch.zeros(1, dtype=torch.int32)
        self.xmpl_predicted = torch.zeros(threshold_vars_cnt, dtype=torch.int32)

        self.xmpl_rec = torch.zeros(threshold_vars_cnt, dtype=torch.float32)
        self.xmpl_prec = torch.zeros(threshold_vars_cnt, dtype=torch.float32)
        self.xmpl_ap = torch.zeros(threshold_vars_cnt, dtype=torch.float32)
        self.total_examples = 0

    def _update_per_class(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]

        assert self.nlbl == labels.size(1), "Input data has wrong num of labels!"

        for i in range(self.nlbl):
            labels_i = labels[:, i].squeeze()
            out_i = outputs[:, i].squeeze()

            self.cls_total[i] += labels_i.int().sum().item()

            # out_i      [ 1    x n ]
            # thresholds [ bins x 1 ]
            # pred_i     [ bins x n ]
            pred_i = out_i >= self.cls_thresholds[:, None]
            # labels_i    [ n ]
            self.cls_tp[i] += (pred_i & labels_i.bool()).sum(dim=1).cpu()
            self.cls_predicted[i] += pred_i.sum(dim=1).cpu()

    def _update_per_example(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs [ N x L ]
        # labels [ N x L ]
        # thresholds [ V x L ]

        # outputs     [ N x 1 x L ]
        # thresholds  [ 1 x V x L ]
        # predictions [ N x V x L ]
        predictions = outputs.unsqueeze(1) >= self.xmpl_thresholds

        # N x V
        tp = (predictions * labels.unsqueeze(1)).sum(dim=2)
        # N
        rel = labels.sum(dim=1)
        # [N x V]
        predicted = predictions.sum(dim=2)
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
        sorted_tp = torch.gather(
            predictions * labels.unsqueeze(1),
            2,
            sort_idx.unsqueeze(1).expand(-1, self.xmpl_tp.size(0), -1),
        )

        # N x V x L
        prec_at_i = torch.cumsum(sorted_tp, dim=-1) / torch.arange(
            1, labels.size(1) + 1, device=sorted_tp.device
        )
        self.xmpl_ap += ((prec_at_i * sorted_tp).sum(dim=-1) / rel[:, None]).sum(dim=0).cpu()

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        sig_out = F.sigmoid(outputs)
        self._update_per_class(sig_out, labels)
        self._update_per_example(sig_out, labels)

    def _compute_per_class_metrics(self):
        metrics = {
            'cls_macro_F1': 0.0,
            'cls_recall': [],
            'cls_precision': [],
            'cls_F1': [],
            'cls_thresholds': [],
        }
        for i in range(self.nlbl):
            # cls_total [ nlbl ]
            # tp        [ nbins ]
            rec = self.cls_tp[i] / self.cls_total[i]
            prec = self.cls_tp[i] / self.cls_predicted[i]
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

            max_m_idx = torch.argmax(decider_m)

            for ms, mt in metrics_i.items():
                metrics[ms].append(mt[max_m_idx].item())

            metrics['cls_thresholds'].append(self.cls_thresholds[max_m_idx].cpu().item())

        metrics['cls_macro_F1'] = sum(metrics['cls_F1']) / len(metrics['cls_F1'])
        return metrics

    def _compute_per_example_metrics(self):
        # V
        micro_rec = (self.xmpl_tp / self.xmpl_rel).nan_to_num()
        micro_prec = (self.xmpl_tp / self.xmpl_predicted).nan_to_num()
        micro_f1 = (2 * micro_rec * micro_prec / (micro_rec + micro_prec)).nan_to_num()

        macro_rec = (self.xmpl_rec / self.total_examples).nan_to_num()
        macro_prec = (self.xmpl_prec / self.total_examples).nan_to_num()
        macro_f1 = (2 * macro_rec * macro_prec / (macro_rec + macro_prec)).nan_to_num()
        mean_ap = (self.xmpl_ap / self.total_examples).nan_to_num()

        decider_m_tensor = None
        match self.select_threshold_metric:
            case 'micro_F1':
                decider_m_tensor = micro_f1
            case 'macro_F1':
                decider_m_tensor = macro_f1
            case _:
                raise RuntimeError(f'Unsupported decider metric: {self.select_threshold_metric}')
        assert decider_m_tensor is not None, "Logic error comp 32892"

        max_m_idx = torch.argmax(decider_m_tensor)
        if decider_m_tensor.size(0) > self.xmpl_max_threshold_vars:
            _, ind = decider_m_tensor.topk(self.xmpl_max_threshold_vars, sorted=False)
            thresh_vars = self.xmpl_thresholds[ind].tolist()
        else:
            thresh_vars = self.xmpl_thresholds.tolist()

        fmt_str = list(
            (m.item(), ','.join("%.1f" % v for v in thr[:3].tolist()) + '..')
            for m, thr in zip(decider_m_tensor, self.xmpl_thresholds)
        )
        logging.info("select thresholds from  %s, selected %s", fmt_str, fmt_str[max_m_idx])
        return {
            'micro_F1': micro_f1[max_m_idx].item(),
            'micro_recall': micro_rec[max_m_idx].item(),
            'micro_precision': micro_prec[max_m_idx].item(),
            'macro_F1': macro_f1[max_m_idx].item(),
            'macro_recall': macro_rec[max_m_idx].item(),
            'macro_precision': macro_prec[max_m_idx].item(),
            'MAP': mean_ap[max_m_idx].item(),
            '_threshold_vars': thresh_vars,
            'thresholds': self.xmpl_thresholds[max_m_idx].tolist(),
        }

    def compute(self):
        m = self._compute_per_class_metrics()
        m.update(self._compute_per_example_metrics())
        m['_threshold_vars'].append(m['cls_thresholds'])
        return m


def eval_on_dataset(
    conf: ClassifFineTuneConf,
    meta_path,
    model: DocClassifier,
    labels_mapping: dict,
    last_eval_results: dict | None = None,
):
    model.eval()
    device = model.encoder.device
    test_iter = ClassifBatchIterator(
        meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=True),
        labels_mapping=labels_mapping,
        device=device,
    )

    if test_iter.multi_label():
        if model.thresholds is not None:
            evalor = MultiLabelTestEvaluator(
                conf.nlabels, device=device, thresholds=model.thresholds
            )
        else:
            assert last_eval_results is not None, "Logic error dev 821"
            evalor = MultiLabelDevEvaluator(
                conf.nlabels,
                device=device,
                last_eval_results=last_eval_results,
                select_threshold_metric=conf.validation_metric,
            )
    else:
        evalor = MultiClassEvaluator(conf.nlabels)

    for docs, doc_fragments, labels in test_iter.batches():

        with autocast(device.type, enabled=conf.enable_amp):
            output = model(docs, doc_fragments)

        evalor(output, labels)

    return evalor.compute()


def _save_model(conf: FineTuneConf, model: DocClassifier, metrics, labels_index, labels_mapping):
    d = model.encoder.to_dict()
    d['fine_tune_cfg'] = conf
    d['cls_head'] = model.cls_head.state_dict()
    d['labels_index'] = labels_index
    d['labels_mapping'] = labels_mapping
    if 'thresholds' in metrics:
        d['thresholds'] = metrics['thresholds']

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
