#!/usr/bin/env python3

import logging
import math
from typing import Optional
import dataclasses
from pathlib import Path
import multiprocessing

import torch

from doc_enc.text_processor import TextProcessor, TextProcessorConf

from doc_enc.encoders.enc_factory import (
    create_sent_encoder,
    create_emb_seq_encoder,
)
from doc_enc.encoders.sent_encoder import split_sents_and_embed


@dataclasses.dataclass
class DocEncoderConf:
    model_path: str
    use_gpu: Optional[int] = None

    async_batch_gen: int = 2

    max_sents: int = 1024
    max_tokens: int = 0


class FromPathsBatchGenerator:
    def __init__(self, tp_conf: TextProcessorConf, conf: DocEncoderConf, tp_state_dict) -> None:
        self._conf = conf
        self._tp = TextProcessor(tp_conf)
        self._tp.load_state_dict(tp_state_dict)

    def _gen_idxs(self, offset, i, doc_len):
        return list(range(offset + i, offset + i + doc_len))

    def batches(self, paths, offset):
        docs = []
        doc_fragments = []
        cur_token_cnt = 0
        cur_sent_cnt = 0
        batch_idx_beg = 0

        for i, p in enumerate(paths):
            sents, fragment_len_list = self._tp.prepare_text_from_file(p)
            token_cnt = sum(len(s) for s in sents)
            if not token_cnt:
                sents = [[self._tp.vocab().pad_idx()]]
                fragment_len_list = [1]

            if (
                docs
                and self._conf.max_sents
                and cur_sent_cnt + len(sents) > self._conf.max_sents
                or self._conf.max_tokens
                and cur_token_cnt + token_cnt > self._conf.max_tokens
            ):
                idxs = self._gen_idxs(offset, batch_idx_beg, len(docs))
                yield docs, doc_fragments, idxs
                docs = []
                doc_fragments = []
                batch_idx_beg = i
                cur_sent_cnt = 0
                cur_token_cnt = 0

            docs.append(sents)
            doc_fragments.append(fragment_len_list)
            cur_sent_cnt += len(sents)
            cur_token_cnt += token_cnt
        if docs:
            yield docs, doc_fragments, self._gen_idxs(offset, batch_idx_beg, len(docs))


def _generator_proc_wrapper(queue: multiprocessing.Queue, GenCls, items, offset, *args, **kwargs):
    try:
        generator = GenCls(*args, **kwargs)
        for b in generator.batches(items, offset):
            queue.put(b)
    except Exception as e:
        logging.error("Failed to process batches: GenCls=%s: %s", GenCls, e)

    queue.put(None)


class BatchIterator:
    def __init__(
        self,
        generator_cls=None,
        generator_args=(),
        async_generators=1,
    ):

        self._generator_cls = generator_cls
        self._generator_args = generator_args
        self._async_generators = async_generators

        self._processes = []
        self._queue = multiprocessing.Queue(4 * async_generators)

    def destroy(self):
        self._terminate_workers()
        self._queue.close()

    def _terminate_workers(self):
        for p in self._processes:
            p.terminate()
            p.join()
        self._processes = []

    def start_workers(self, items):
        per_worker_items = math.ceil(len(items) / self._async_generators)
        for offs in range(0, len(items), per_worker_items):
            p = multiprocessing.Process(
                target=_generator_proc_wrapper,
                args=(
                    self._queue,
                    self._generator_cls,
                    items[offs : offs + per_worker_items],
                    offs,
                )
                + self._generator_args,
                kwargs={},
            )
            p.start()

            self._processes.append(p)

    def batches(self):
        if not self._processes:
            raise RuntimeError("Sent batch Iterator is not initialized!")

        finished_processes = 0
        while finished_processes < self._async_generators:
            logging.debug("queue len: %s", self._queue.qsize())
            batch = self._queue.get()
            if batch is None:
                finished_processes += 1
                continue
            yield batch

        for p in self._processes:
            p.join()
        self._processes = []


class DocEncoder:
    def __init__(self, conf: DocEncoderConf) -> None:
        self._conf = conf

        state_dict = torch.load(conf.model_path)
        self._tp_conf: TextProcessorConf = state_dict['tp_conf']
        self._tp_conf.tokenizer.vocab_path = None
        self._tp_state_dict = state_dict['tp']
        self._tp = TextProcessor(self._tp_conf)
        self._tp.load_state_dict(self._tp_state_dict)

        if conf.use_gpu is not None and torch.cuda.is_available():
            logging.info("Computing on gpu:%s", conf.use_gpu)
            self._device = torch.device(f'cuda:{conf.use_gpu}')
        else:
            logging.info("Computing on cpu")
            self._device = torch.device('cpu')

        mc = state_dict['model_conf']
        self._sent_layer = create_sent_encoder(mc.sent.encoder, self._tp.vocab())
        self._sent_layer.load_state_dict(state_dict['sent_enc'])
        self._sent_layer = self._sent_layer.to(device=self._device).eval()
        logging.debug("sent layer\n%s", self._sent_layer)

        self._fragment_layer = None
        sent_embs_out_size = self._sent_layer.out_embs_dim()
        if 'frag_enc' in state_dict:
            self._fragment_layer = create_emb_seq_encoder(mc.fragment, sent_embs_out_size)
            self._fragment_layer.load_state_dict(state_dict['frag_enc'])
            self._fragment_layer = self._fragment_layer.to(device=self._device).eval()
            doc_input_size = self._fragment_layer.out_embs_dim()
            logging.debug("fragment layer\n:%s", self._fragment_layer)
        else:
            doc_input_size = sent_embs_out_size

        self._doc_layer = create_emb_seq_encoder(mc.doc, doc_input_size)
        self._doc_layer.load_state_dict(state_dict['doc_enc'])
        self._doc_layer = self._doc_layer.to(device=self._device).eval()

        logging.debug("doc layer\n:%s", self._doc_layer)

    def _make_sent_tensor(self, cnt, max_len, sents):
        sent_tensor = torch.full((cnt, max_len), self._tp.vocab().pad_idx(), dtype=torch.int32)
        for i, sent in enumerate(sents):
            sent_tensor[i, 0 : len(sent)] = torch.as_tensor(sent)
        return sent_tensor

    def _encode_sents_impl(self, sents):
        cnt = len(sents)
        sent_lengths = [len(t) for t in sents]
        lengths_tensor = torch.as_tensor(sent_lengths, dtype=torch.int64, device=self._device)
        sent_tensor = self._make_sent_tensor(cnt, max(sent_lengths), sents)
        sent_tensor = sent_tensor.to(device=self._device)

        if cnt > self._conf.max_sents:
            return split_sents_and_embed(
                self._sent_layer,
                sent_tensor,
                lengths_tensor,
                self._conf.max_sents,
            )

        res = self._sent_layer(sent_tensor, lengths_tensor, enforce_sorted=False)
        sent_embs = res.pooled_out
        return sent_embs

    def _encode_docs_impl(self, docs, doc_fragments):
        """Each doc is a list of tokenized sentences."""

        all_sents = [s for d in docs for s in d]
        sent_embs = self._encode_sents_impl(all_sents)

        if self._fragment_layer is not None:
            frag_len = []
            len_list = []
            for fragments in doc_fragments:
                frag_len.extend(fragments)
                len_list.append(len(fragments))

            embs = self._fragment_layer(sent_embs, frag_len, enforce_sorted=False).pooled_out
        else:
            embs = sent_embs
            len_list = [len(d) for d in docs]

        doc_embs = self._doc_layer(embs, len_list, enforce_sorted=False).pooled_out
        return doc_embs

    def _encode_docs(self, docs, doc_fragments):
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                return self._encode_docs_impl(docs, doc_fragments)

    def _encode_sents(self, sents):
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                return self._encode_sents_impl(sents)

    def encode_sents(self, sents):
        sent_ids = self._tp.prepare_sents(sents)
        return self._encode_sents(sent_ids).cpu().numpy()

    def encode_docs_from_path_list(self, path_list):
        embs = []
        embs_idxs = []
        batch_iter = BatchIterator(
            FromPathsBatchGenerator,
            (self._tp_conf, self._conf, self._tp_state_dict),
            self._conf.async_batch_gen,
        )
        batch_iter.start_workers(path_list)
        for docs, doc_fragments, idxs in batch_iter.batches():
            doc_embs = self._encode_docs(docs, doc_fragments)
            embs.append(doc_embs.to(device='cpu', dtype=torch.float32))
            embs_idxs.extend(idxs)

        stacked = torch.vstack(embs)
        assert len(stacked) == len(path_list)

        embs_idxs = torch.tensor(embs_idxs)
        initial_order_idxs = torch.empty_like(embs_idxs)
        initial_order_idxs.scatter_(0, embs_idxs, torch.arange(0, embs_idxs.numel()))
        reordered_embs = stacked.index_select(0, initial_order_idxs)
        return reordered_embs.numpy()

    def encode_docs_from_dir(self, path: Path):
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.encode_docs_from_path_list(paths)
