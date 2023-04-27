#!/usr/bin/env python3

import logging

import torch
import torch.nn.functional as F

from LibVQ.utils import dist_gather_tensor

from doc_enc.training.index.index_train_conf import IndexTrainConf, IndexLossType
from doc_enc.training.index.ivf_pq_model import TrainableIvfPQ
from doc_enc.training.models.base_model import DualEncModelOutput
from doc_enc.training.models.base_sent_model import BaseSentModel
from doc_enc.training.models.base_doc_model import BaseDocModel


# from LibVQ
def compute_distil_loss(teacher_score: torch.Tensor, student_score: torch.Tensor):
    preds_smax = F.softmax(student_score, dim=1)
    true_smax = F.softmax(teacher_score, dim=1)
    preds_smax = preds_smax + 1e-6
    preds_log = torch.log(preds_smax)
    loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
    return loss


def compute_loss(score_matrix: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(score_matrix, labels)


def combine_dense_and_index_loss(conf: IndexTrainConf, dense_loss, ivf_loss, pq_loss, world_size):
    ivf_weight = conf.ivf.weight
    if conf.ivf.scale_ivf_weight_to_pq:
        weight = pq_loss / (ivf_loss + 1e-6)
        if weight == 0:
            logging.warning("There is no loss for pq; use 'scaled_to_denseloss' ")
            weight = dense_loss / (ivf_loss + 1e-6)
        if world_size > 1:
            weight = dist_gather_tensor(weight.unsqueeze(0), world_size=world_size)
            weight = torch.mean(weight)
        ivf_weight = weight

    batch_loss = conf.dense_weight * dense_loss + conf.pq.weight * pq_loss + ivf_weight * ivf_loss
    return batch_loss


def calculate_ivf_pq_loss(conf: IndexTrainConf, output: DualEncModelOutput, labels):
    if output.ivf_score_matrix is None or output.pq_score_matrix is None:
        return None, None

    if conf.ivf.loss_type == IndexLossType.BASIC:
        ivf_loss = compute_loss(output.ivf_score_matrix, labels)
    elif conf.ivf.loss_type == IndexLossType.DISTIL:
        ivf_loss = compute_distil_loss(output.dense_score_matrix, output.ivf_score_matrix)
    else:
        raise RuntimeError(f"Unknown loss type {conf.ivf.loss_type}")

    if conf.pq.loss_type == IndexLossType.BASIC:
        pq_loss = compute_loss(output.pq_score_matrix, labels)
    elif conf.pq.loss_type == IndexLossType.DISTIL:
        pq_loss = compute_distil_loss(output.dense_score_matrix, output.pq_score_matrix)
    else:
        raise RuntimeError(f"Unknown loss type {conf.pq.loss_type}")

    return ivf_loss, pq_loss


def update_ivf_model(index: TrainableIvfPQ | None, world_size, ivf_lr):
    if index is None:
        return
    index.ivf.grad_accumulate(world_size)
    index.ivf.update_centers(lr=ivf_lr)
    index.ivf.zero_grad()


def update_doc_index_model(mod: BaseDocModel, world_size):
    if mod.index is None:
        return

    if not mod.conf.index.ivf.fixed:
        update_ivf_model(mod.index, world_size, mod.conf.index.ivf.lr)


def update_sent_index_model(mod: BaseSentModel, world_size):
    if mod.index is None:
        return

    if not mod.conf.index.ivf.fixed:
        update_ivf_model(mod.index, world_size, mod.conf.index.ivf.lr)
