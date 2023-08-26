#!/usr/bin/env python3


import logging
import dataclasses
import csv
import os
from pathlib import Path


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast


from doc_enc.doc_encoder import DocEncoderConf, EncodeModule, BatchIterator, file_path_fetcher

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

    validation_metric: str = 'acc'

    only_eval_test: bool = False


@dataclasses.dataclass
class ClassifFineTuneConf(FineTuneConf):
    nlabels: int = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=FineTuneConf)
cs.store(name="base_classif_config", node=ClassifFineTuneConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


# * Models


class DocClassifier(nn.Module):
    def __init__(self, encoder: EncodeModule, conf: ClassifFineTuneConf):
        super().__init__()
        self.encoder = encoder

        self.dropout = None
        if conf.dropout > 0.0:
            self.dropout = nn.Dropout(conf.dropout)
        self.classif_layer = nn.Linear(encoder.doc_embs_dim(), conf.nlabels)
        self.classif_layer = self.classif_layer.to(device=self.encoder.device)

    def forward(self, docs, doc_fragments):
        embeddings = self.encoder(docs, doc_fragments)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        return self.classif_layer(embeddings)


def _create_model(conf: ClassifFineTuneConf):
    doc_encoder = EncodeModule(conf)
    if (
        OmegaConf.is_missing(conf, 'nlabels')
        and (ft_cfg := doc_encoder._state_dict.get('fine_tune_cfg')) is not None
    ):
        conf.nlabels = ft_cfg.nlabels
    model = DocClassifier(doc_encoder, conf)
    if 'classif_layer' in doc_encoder._state_dict:
        model.classif_layer.load_state_dict(doc_encoder._state_dict['classif_layer'])
    return model


# * Data Iterators


class ClassifBatchIterator:
    def __init__(
        self, meta_path, base_data_dir, docs_iter: BatchIterator, labels_mapping, device=None
    ) -> None:
        self._docs_iter = docs_iter

        self._path_list = []
        self._labels_list = []
        self._device = device

        with open(meta_path, 'r', encoding='utf8') as infp:
            reader = csv.reader(infp)
            for row in reader:
                fp = Path(base_data_dir) / row[0]
                self._path_list.append(fp)
                label = row[1]
                if label not in labels_mapping:
                    raise RuntimeError(f"Unknown label: {label}")
                self._labels_list.append(labels_mapping[label])

    def examples_cnt(self):
        return len(self._labels_list)

    def destroy(self):
        self._docs_iter.destroy()

    def batches(self):
        self._docs_iter.start_workers_for_item_list(self._path_list, file_path_fetcher)

        for docs, doc_fragments, idxs in self._docs_iter.batches():
            labels = torch.as_tensor(
                [self._labels_list[i] for i in idxs], dtype=torch.long, device=self._device
            )
            yield docs, doc_fragments, labels


# * Train loop


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

        model = _create_model(conf)
        labels_mapping = model.encoder._state_dict['labels_mapping']
        with torch.inference_mode():
            metrics = eval_on_dataset(
                conf, conf.test_meta_path, model, labels_mapping=labels_mapping
            )
        print(metrics)
        print('Acc,macro_F1')
        print(f'{metrics["acc"]:.3f},{metrics["macro_F1"]:.3f}')


def _eval_on_dev_and_maybe_save(conf, model, best_metric, labels_index, labels_mapping):
    with torch.no_grad():
        metrics = eval_on_dataset(conf, conf.dev_meta_path, model, labels_mapping=labels_mapping)

    best_metric = _save_model_if_best(
        conf,
        model,
        metrics,
        best_metric,
        labels_index=labels_index,
        labels_mapping=labels_mapping,
    )
    return best_metric


def _train_loop(
    conf: ClassifFineTuneConf,
    train_iter: ClassifBatchIterator,
    model: DocClassifier,
    labels_index,
    labels_mapping,
):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        conf.lr,
        total_steps=conf.max_updates,
        **conf.lr_scheduler_kwargs,
    )

    scaler = GradScaler(enabled=True)
    best_metric = 0.0
    running_loss = 0.0
    update_nums = 0
    running_correct = 0
    running_examples_num = 0
    docs_total = 0
    epoch = 0
    while update_nums < conf.max_updates:
        epoch += 1
        logging.info("Starting %s epoch", epoch)
        for docs, doc_fragments, labels in train_iter.batches():
            # zero the parameter gradients
            model.train(mode=True)
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast(True):
                outputs = model(docs, doc_fragments)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).int().sum().cpu().item()
            running_examples_num += labels.size(0)
            docs_total += len(docs)
            update_nums += 1

            if update_nums % 100 == 0:
                logging.info(
                    "#%d, docs_per_batch: %.1f, avg loss: %.3f, acc: %.3f, lr:%.5e",
                    update_nums,
                    docs_total / 100,
                    running_loss / 100,
                    running_correct / running_examples_num,
                    scheduler.get_last_lr()[0],
                )
                running_loss = 0
                running_correct = 0
                running_examples_num = 0
                docs_total = 0

            if update_nums % conf.eval_every == 0:
                best_metric = _eval_on_dev_and_maybe_save(
                    conf, model, best_metric, labels_index, labels_mapping
                )
            if update_nums >= conf.max_updates:
                break


def eval_on_dataset(conf: ClassifFineTuneConf, meta_path, model: DocClassifier, labels_mapping):
    model.eval()
    test_iter = ClassifBatchIterator(
        meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=True),
        labels_mapping=labels_mapping,
        device=model.encoder.device,
    )

    nlbl = conf.nlabels
    correct = torch.zeros(1)
    total = 0
    cls_total = torch.zeros(nlbl, dtype=torch.int32)
    cls_predicted = torch.zeros(nlbl, dtype=torch.int32)
    tp = torch.zeros(nlbl, dtype=torch.int32)
    for docs, doc_fragments, labels in test_iter.batches():
        with autocast():
            output = model(docs, doc_fragments)
        _, predicted = torch.max(output, 1)

        total += labels.size(0)
        correct += (predicted == labels).int().sum().cpu()
        for i in range(nlbl):
            cls_predicted[i] += (predicted == i).int().sum().cpu()
            cls_total[i] += (labels == i).int().sum().cpu()
            tp[i] += (predicted[(labels == i)] == i).int().sum().cpu()

    if total == 0:
        return {}

    rec = tp / cls_total
    prec = tp / cls_predicted
    f1 = 2 * rec * prec / (rec + prec)

    metrics = {
        'acc': correct.item() / total,
        'macro_F1': f1.mean().item(),
        'recall': rec.tolist(),
        'precision': prec.tolist(),
        'F1': f1.tolist(),
        'predictions_per_cls': [c.item() / total for c in cls_predicted],
    }

    return metrics


def _save_model_if_best(
    conf: FineTuneConf, model: DocClassifier, metrics, best_metric, labels_index, labels_mapping
):
    if conf.validation_metric == 'acc':
        m = metrics.get('acc', 0)
    elif conf.validation_metric.startswith('F1'):
        cls_num = int(conf.validation_metric.split('_')[-1])
        if f1m := metrics.get('F1'):
            m = f1m[cls_num]
        else:
            m = 0
    else:
        raise RuntimeError(f"Unknown validation metric: {conf.validation_metric}")

    logging.info(
        "%s on dev %.3f; best %s %.3f; per classes: %s",
        conf.validation_metric,
        m,
        conf.validation_metric,
        best_metric,
        metrics['predictions_per_cls'],
    )
    if m >= best_metric:
        d = model.encoder.to_dict()
        d['fine_tune_cfg'] = conf
        d['classif_layer'] = model.classif_layer.state_dict()
        d['labels_index'] = labels_index
        d['labels_mapping'] = labels_mapping

        if not conf.save_path:
            conf.save_path = os.path.join(os.getcwd(), 'model.pt')

        if not (p := Path(conf.save_path).parent).exists():
            p.mkdir(parents=True)

        torch.save(d, conf.save_path)
        logging.info("new best model was saved")
        return m
    return best_metric


@hydra.main(config_path=None, config_name="config", version_base=None)
def fine_tune_classif_cli(conf: ClassifFineTuneConf) -> None:
    classif_fine_tune(conf)
