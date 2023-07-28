#!/usr/bin/env python3

from enum import Enum
import logging

import faiss
import numpy as np


def gpu_knn(src_embs, tgt_embs, topk, device_num=-1):
    res = faiss.StandardGpuResources()
    d, i = faiss.knn_gpu(
        res, src_embs, tgt_embs, topk, metric=faiss.METRIC_INNER_PRODUCT, device=device_num
    )
    return d, i


def cpu_knn(src_embs, tgt_embs, topk):
    (
        d,
        i,
    ) = faiss.knn(src_embs, tgt_embs, topk, metric=faiss.METRIC_INNER_PRODUCT)
    return d, i


def knn(src_embs, tgt_embs, topk: int, use_gpu=-1):
    if use_gpu != -1:
        return gpu_knn(src_embs, tgt_embs, topk, device_num=use_gpu)
    return cpu_knn(src_embs, tgt_embs, topk)


def score_candidates_by_margin_crit(x, y, candidate_inds, fwd_mean, bwd_mean, margin):
    def score(x, y, fwd_mean, bwd_mean, margin):
        return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)

    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]

            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def margin_criterion(topk: int, src_embs, tgt_embs, use_gpu=-1):
    x2y_sim, x2y_ind = knn(src_embs, tgt_embs, topk, use_gpu)
    x2y_mean = x2y_sim.mean(axis=1)

    y2x_sim, _y2x_ind = knn(
        tgt_embs, src_embs, topk, use_gpu
    )  # pylint: disable=arguments-out-of-order
    y2x_mean = y2x_sim.mean(axis=1)

    margin = lambda a, b: a / b

    fwd_scores = score_candidates_by_margin_crit(
        src_embs, tgt_embs, x2y_ind, x2y_mean, y2x_mean, margin
    )
    idxs = (-fwd_scores).argsort(1)
    fwd_scores = np.take_along_axis(fwd_scores, idxs, 1)
    x2y_ind = np.take_along_axis(x2y_ind, idxs, 1)
    return fwd_scores, x2y_ind

    # fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    # for i, j in enumerate(fwd_best):
    #     print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)


class SimKind(Enum):
    COS = 1
    MARGIN = 2


def calc_sim(sim_kind: SimKind, topk: int, src_embs, tgt_embs, use_gpu=-1):
    src_embs = src_embs.astype(np.float32, copy=False)
    tgt_embs = tgt_embs.astype(np.float32, copy=False)
    faiss.normalize_L2(src_embs)
    faiss.normalize_L2(tgt_embs)

    if sim_kind == SimKind.COS:
        return knn(src_embs, tgt_embs, topk, use_gpu=use_gpu)
    if sim_kind == SimKind.MARGIN:
        logging.info("use margin sim")
        return margin_criterion(topk, src_embs, tgt_embs, use_gpu)

    raise RuntimeError(f"Unsupported sim_kind {sim_kind}")
