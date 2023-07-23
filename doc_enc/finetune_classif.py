#!/usr/bin/env python3


import logging
import dataclasses
import csv
import os


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

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

    lr: float = 0.0002
    nepoch: int = 5

    validation_metric: str = 'acc'


@dataclasses.dataclass
class ClassifFineTuneConf(FineTuneConf):
    nlabels: int = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=FineTuneConf)
cs.store(name="base_classif_config", node=ClassifFineTuneConf)
cs.store(name="base_doc_encoder", group="doc_encoder", node=DocEncoderConf)


# * Models


class DocClassifier(nn.Module):
    def __init__(self, encoder: EncodeModule, classif_nlabels=0):
        super().__init__()
        self.encoder = encoder

        self.classif_layer = nn.Linear(encoder.doc_embs_dim(), classif_nlabels)
        self.classif_layer = self.classif_layer.to(device=self.encoder.device())

    def forward(self, docs, doc_fragments):
        embeddings = self.encoder(docs, doc_fragments)
        return self.classif_layer(embeddings)


# * Data Iterators


class ClassifBatchIterator:
    def __init__(self, meta_path, base_data_dir, docs_iter: BatchIterator, device=None) -> None:
        self._docs_iter = docs_iter

        self._path_list = []
        self._labels_list = []
        self._device = device

        with open(meta_path, 'r', encoding='utf8') as infp:
            reader = csv.reader(infp)
            for row in reader:
                fp = base_data_dir + row[0]
                self._path_list.append(fp)
                self._labels_list.append(int(row[1]))

    def batches(self):
        self._docs_iter.start_workers_for_item_list(self._path_list, file_path_fetcher)

        for docs, doc_fragments, idxs in self._docs_iter.batches():
            labels = torch.as_tensor(
                [self._labels_list[i] for i in idxs], dtype=torch.long, device=self._device
            )
            yield docs, doc_fragments, labels


# * Train loop


def classif_fine_tune(conf: ClassifFineTuneConf):
    doc_encoder = EncodeModule(conf)
    model = DocClassifier(doc_encoder, conf.nlabels)
    train_iter = ClassifBatchIterator(
        conf.train_meta_path,
        conf.data_dir,
        doc_encoder.create_batch_iterator(eval_mode=False),
        device=doc_encoder.device(),
    )
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    scaler = GradScaler(enabled=True)
    best_metric = 0.0
    for epoch in range(conf.nepoch):
        loss_epoch = 0.0
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
            loss_epoch += loss.item()

        logging.info('Ep %s | loss %s', epoch, loss_epoch)

        with torch.no_grad():
            metrics = eval_on_dataset(conf, model)

        best_metric = _save_model_if_best(conf, model, metrics, best_metric)


def eval_on_dataset(conf: ClassifFineTuneConf, model: DocClassifier):
    model.eval()
    dev_iter = ClassifBatchIterator(
        conf.dev_meta_path,
        conf.data_dir,
        model.encoder.create_batch_iterator(eval_mode=True),
        device=model.encoder.device(),
    )

    nlbl = conf.nlabels
    correct = torch.zeros(1)
    total = 0
    cls_total = torch.zeros(nlbl, dtype=torch.int32)
    cls_predicted = torch.zeros(nlbl, dtype=torch.int32)
    tp = torch.zeros(nlbl, dtype=torch.int32)
    for docs, doc_fragments, labels in dev_iter.batches():
        output = model(docs, doc_fragments)
        _, predicted = torch.max(output, 1)

        total += labels.size(0)
        correct += (predicted == labels).int().sum().cpu()
        for i in range(nlbl):
            cls_predicted[i] += (predicted == i).int().sum().cpu()
            cls_total[i] += (labels == i).int().sum().cpu()
            tp[i] += (predicted[(labels == i)] == i).int().sum().cpu()

    rec = tp / cls_total
    prec = tp / cls_predicted

    metrics = {
        'acc': correct.item() / total,
        'recall': rec.tolist(),
        'precision': prec.tolist(),
        'F1': (2 * rec * prec / (rec + prec)).tolist(),
        'predictions_per_cls': [c.item() / total for c in cls_predicted],
    }

    return metrics


def _save_model_if_best(conf: FineTuneConf, model: DocClassifier, metrics, best_metric):
    if conf.validation_metric == 'acc':
        m = metrics['acc']
    elif conf.validation_metric.startswith('F1'):
        cls_num = int(conf.validation_metric.split('_')[-1])
        m = metrics['F1'][cls_num]
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

        save_path = conf.save_path
        if not save_path:
            save_path = os.path.join(os.getcwd(), 'model.pt')
        torch.save(d, save_path)
        logging.info("new best model was saved")
        return m
    return best_metric


@hydra.main(config_path=None, config_name="config", version_base=None)
def fine_tune_classif_cli(conf: ClassifFineTuneConf) -> None:
    classif_fine_tune(conf)
