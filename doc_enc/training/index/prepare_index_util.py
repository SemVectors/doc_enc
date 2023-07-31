#!/usr/bin/env python3

import os
import logging
import random
from pathlib import Path

import torch
import faiss
import numpy as np

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf
from doc_enc.utils import file_line_cnt

from doc_enc.training.index.index_train_conf import IndexTrainConf
from doc_enc.training.sents_batch_generator import SentsBatchGeneratorConf
from doc_enc.training.models.model_conf import BaseModelConf


def sents_generator(sent_file, limit=0):
    with open(sent_file, 'r', encoding='utf8') as fp:
        cnt = 0
        for l in fp:
            sent_id, sent = l.rstrip().split('\t', 1)
            yield sent, int(sent_id)
            cnt += 1
            if limit and cnt >= limit:
                break


def hn_sents_generator(sent_file, last_id, added_ids):
    with open(sent_file, 'r', encoding='utf8') as fp:
        finished = False
        for l in fp:
            t = l.rstrip().split('\t', 2)
            src_sent_id = int(t[0])
            if last_id == src_sent_id:
                # this is hns for last id,
                # add them and break after that
                finished = True
            elif finished:
                break

            if len(t) < 3:
                continue
            _, target_sent_id, sent = t

            target_sent_id = int(target_sent_id)
            if target_sent_id in added_ids:
                continue
            added_ids.add(target_sent_id)
            yield target_sent_id, sent


def sents_sample_generator(inp_file, sample_ratio, limit=0):
    sents_cnt = file_line_cnt(inp_file, limit)
    sample_size = int(sents_cnt * sample_ratio)
    logging.info("%d sents in input file, sample %d from them", sents_cnt, sample_size)

    sampled_idxs = random.sample(range(sents_cnt), sample_size)
    sampled_idxs.sort()
    if not sampled_idxs:
        return

    cur_sample_idx = 0
    next_sent_idx = sampled_idxs[cur_sample_idx]
    with open(inp_file, 'r', encoding='utf8') as fp:
        for idx, l in enumerate(fp):
            if idx < next_sent_idx:
                continue
            sent_id, sent = l.rstrip().split('\t', 1)
            yield int(sent_id), sent

            cur_sample_idx += 1
            if cur_sample_idx >= len(sampled_idxs):
                break
            next_sent_idx = sampled_idxs[cur_sample_idx]


def _create_doc_enc(model_conf: BaseModelConf):
    if not model_conf.load_params_from:
        raise RuntimeError("have to pass load_params_from when training index from scratch")

    model_path = Path(model_conf.load_params_from)
    if not model_path.exists():
        raise RuntimeError(f"{model_path} does not exist.")

    conf = DocEncoderConf(model_path=str(model_path), use_gpu=0)
    doc_encoder = DocEncoder(conf)
    return doc_encoder


def collect_vectors(vecs_generator, normalize_l2=True):
    processed_vectors = 0
    vectors = []

    processed_vectors = 0
    for ids, embs in vecs_generator:
        if normalize_l2:
            embs = embs.astype(np.float32, copy=False)
            faiss.normalize_L2(embs)
        vectors.append(embs)
        processed_vectors += len(ids)
        if processed_vectors >= 100_000:
            logging.info(
                "Another %dk vectors were collected for training", processed_vectors // 1000
            )
            processed_vectors = 0

    vectors = np.vstack(vectors)
    logging.info("collected vectors shape: %s", vectors.shape)
    return vectors


def _index2gpu(index, gpu_index=0):
    res = faiss.StandardGpuResources()
    res.noTempMemory()
    co = faiss.GpuClonerOptions()
    # co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(res, gpu_index, index, co)
    return index


def create_index(conf: IndexTrainConf, emb_dim, gpu_index=0):
    quantizer = faiss.IndexFlatIP(emb_dim)
    quantizer = _index2gpu(quantizer, gpu_index=gpu_index)

    index = faiss.IndexIVFPQ(
        quantizer,
        emb_dim,
        conf.ivf_centers_num,
        conf.subvector_num,
        conf.subvector_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    return index


def create_and_train_index(conf: IndexTrainConf, vec_iter, emb_dim):
    index = create_index(conf, emb_dim)

    vecs = collect_vectors(vec_iter)
    index.train(vecs)
    return index


def prepare_sent_index(
    model_conf: BaseModelConf,
    sents_conf: SentsBatchGeneratorConf,
):
    index_conf = model_conf.index
    if index_conf.init_index_file:
        # nothing to be done
        return

    logging.info("Starting faiss index training")
    doc_encoder = _create_doc_enc(model_conf)
    train_file = sents_conf.input_dir + '/train.tgt'

    def _create_gen_func():
        def _gen():
            yield from sents_sample_generator(
                train_file, index_conf.train_sample, sents_conf.sents_limit
            )

        return _gen

    vec_gen = doc_encoder.encode_sents_from_generators([_create_gen_func()])
    index = create_and_train_index(index_conf, vec_gen, doc_encoder.enc_module().sent_embs_dim())
    logging.info("Train of index was ended")

    _add_sent_vectors_impl('train', index, doc_encoder, sents_conf)
    _add_sent_vectors_impl('dev', index, doc_encoder, sents_conf)

    save_path = os.path.join(
        os.getcwd(),
        f"IVF{index_conf.ivf_centers_num}_PQ{index_conf.subvector_num}x{index_conf.subvector_bits}.faiss",
    )

    faiss.write_index(faiss.index_gpu_to_cpu(index), save_path)
    index_conf.init_index_file = save_path

    logging.info("Index was saved to %s", save_path)
    del doc_encoder
    del index
    torch.cuda.empty_cache()


def _add_sent_vectors_impl(
    split,
    index,
    doc_encoder,
    sents_conf: SentsBatchGeneratorConf,
):
    def _add(ids, embs):
        ids = np.array(ids, dtype=np.int64)
        embs = embs.astype(np.float32, copy=False)
        faiss.normalize_L2(embs)
        index.add_with_ids(embs, ids)

    train_file = sents_conf.input_dir + f'/{split}.tgt'

    logging.info("Add vectors to Index")
    added_ids = set()
    last_id = 0
    gen = doc_encoder.generate_sent_embs_from_file(
        train_file, lines_limit=sents_conf.sents_limit, first_column_is_id=True
    )
    for ids, embs in gen:
        ids = [int(i) for i in ids]
        _add(ids, embs)
        added_ids.update(ids)
        last_id = ids[-1]

    logging.info("add from %s.tgt: added_cnt=%d", split, len(added_ids))
    if sents_conf.dont_use_hns:
        return

    # add also hard negatives

    def _create_gen_func():
        def _gen():
            yield from hn_sents_generator(sents_conf.input_dir + f'/{split}.hn', last_id, added_ids)

        return _gen

    hn_added = 0
    for ids, embs in doc_encoder.encode_sents_from_generators([_create_gen_func()]):
        hn_added += len(ids)
        _add(ids, embs)
    logging.info("add from %s.hn: added_cnt=%d", split, hn_added)


def re_add_sent_vectors(index_path, model_path, sents_conf: SentsBatchGeneratorConf, gpu_index=0):
    conf = DocEncoderConf(model_path=model_path, use_gpu=gpu_index)
    doc_encoder = DocEncoder(conf)

    logging.info("start readding vectors to index %s", index_path)
    index = faiss.read_index(index_path)
    index.reset()

    index.quantizer = _index2gpu(index.quantizer, gpu_index=gpu_index)

    _add_sent_vectors_impl('train', index, doc_encoder, sents_conf)
    _add_sent_vectors_impl('dev', index, doc_encoder, sents_conf)

    faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
