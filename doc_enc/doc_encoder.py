#!/usr/bin/env python3

import logging
import math
from typing import Optional, Any
import dataclasses
from pathlib import Path
import threading
import multiprocessing
import collections.abc

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast

from doc_enc.text_processor import TextProcessor, TextProcessorConf
from doc_enc.training.base_batch_generator import create_padded_tensor

from doc_enc.encoders.enc_factory import (
    create_encoder,
    create_sent_encoder,
    create_emb_seq_encoder,
)
from doc_enc.encoders.sent_encoder import SentEncoder, split_sents_and_embed, SentForDocEncoder
from doc_enc.training.models.model_conf import DocModelConf


@dataclasses.dataclass
class DocEncoderConf:
    model_path: str
    use_gpu: Optional[int] = None

    async_batch_gen: int = 2

    max_sents: int = 2048
    max_tokens: int = 96_000


class BaseBatchGenerator:
    def __init__(self, tp_conf: TextProcessorConf, conf: DocEncoderConf, tp_state_dict) -> None:
        self._conf = conf
        self._tp = TextProcessor(tp_conf, inference_mode=True)
        self._tp.load_state_dict(tp_state_dict)

    def batches(self, items: list[Any], fetcher):
        docs = []
        doc_fragments = []
        cur_token_cnt = 0
        cur_sent_cnt = 0
        batch_idx_list = []

        for idx, doc in fetcher(items):
            if isinstance(doc, str):
                doc = doc.split('\n')

            sents, fragment_len_list = self._tp.prepare_text(doc)
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
                yield docs, doc_fragments, batch_idx_list
                docs = []
                doc_fragments = []
                batch_idx_list = []
                cur_sent_cnt = 0
                cur_token_cnt = 0

            docs.append(sents)
            doc_fragments.append(fragment_len_list)
            batch_idx_list.append(idx)
            cur_sent_cnt += len(sents)
            cur_token_cnt += token_cnt
        if docs:
            yield docs, doc_fragments, batch_idx_list


def _proc_wrapper_for_item_list(
    queue: multiprocessing.Queue, items: list[Any], fetcher, offset, *args, **kwargs
):

    try:
        generator = BaseBatchGenerator(*args, **kwargs)
        for docs, doc_fragments, batch_idx_list in generator.batches(items, fetcher):
            batch_idx_list = [offset + i for i in batch_idx_list]
            queue.put((docs, doc_fragments, batch_idx_list))
    except Exception as e:
        print(type(e), str(e))
        logging.exception("Failed to process batches: %s", e)

    queue.put(None)


def _proc_wrapper_for_item_generator(
    in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue, fetcher, *args, **kwargs
):
    try:
        generator = BaseBatchGenerator(*args, **kwargs)

        while True:
            items = in_queue.get()
            if items is None:
                break
            for docs, doc_fragments, batch_idx_list in generator.batches(items, fetcher):
                batch_items = [items[i] for i in batch_idx_list]
                out_queue.put((docs, doc_fragments, batch_items))
    except Exception as e:
        print(type(e), str(e))
        logging.exception("Failed to process batches: %s", e)

    out_queue.put(None)


class BatchIterator:
    def __init__(
        self,
        generator_args=(),
        async_generators=1,
    ):

        self._generator_args = generator_args
        self._async_generators = async_generators

        self._processes = []
        self._out_queue = multiprocessing.Queue(4 * async_generators)

        # input stream support
        self._in_queue = multiprocessing.Queue(10 * async_generators)
        self._generator_thread = None

    def destroy(self):
        self._terminate_workers()
        self._in_queue.close()
        self._out_queue.close()

    def _terminate_workers(self):
        for p in self._processes:
            p.terminate()
            p.join()
        self._processes = []

    def start_workers_for_item_list(self, items: list[Any], fetcher):
        per_worker_items = math.ceil(len(items) / self._async_generators)
        for offs in range(0, len(items), per_worker_items):
            p = multiprocessing.Process(
                target=_proc_wrapper_for_item_list,
                args=(
                    self._out_queue,
                    items[offs : offs + per_worker_items],
                    fetcher,
                    offs,
                )
                + self._generator_args,
                kwargs={},
            )
            p.start()

            self._processes.append(p)

    def _input_generator_thread(self, items_generator, batch_size):

        try:
            items_batch = []
            for item in items_generator:
                items_batch.append(item)
                if len(items_batch) >= batch_size:
                    self._in_queue.put(items_batch)
                    items_batch = []

            if items_batch:
                self._in_queue.put(items_batch)
        finally:
            for _ in self._processes:
                self._in_queue.put(None)

    def start_workers_for_stream(self, items_generator, fetcher, batch_size=10):
        for _ in range(self._async_generators):
            p = multiprocessing.Process(
                target=_proc_wrapper_for_item_generator,
                args=(
                    self._in_queue,
                    self._out_queue,
                    fetcher,
                )
                + self._generator_args,
                kwargs={},
            )
            p.start()

            self._processes.append(p)

        self._generator_thread = threading.Thread(
            target=self._input_generator_thread, args=(items_generator, batch_size)
        )
        self._generator_thread.start()

    def batches(self):
        if not self._processes:
            raise RuntimeError("Batch Iterator is not initialized!")

        finished_processes = 0
        while finished_processes < self._async_generators:
            logging.debug("queue len: %s", self._out_queue.qsize())
            batch = self._out_queue.get()
            if batch is None:
                finished_processes += 1
                continue
            yield batch

        for p in self._processes:
            p.join()
        self._processes = []

        if self._generator_thread is not None:
            self._generator_thread.join()


def file_path_fetcher(paths):
    for idx, path in enumerate(paths):
        with open(path, 'r', encoding='utf8') as fp:
            sents = []
            for s in fp:
                sents.append(s.rstrip())
            yield idx, sents


class DocEncoder:
    def __init__(self, conf: DocEncoderConf) -> None:
        self._conf = conf
        if conf.use_gpu is not None and torch.cuda.is_available():
            logging.info("Computing on gpu:%s", conf.use_gpu)
            self._device = torch.device(f'cuda:{conf.use_gpu}')
        else:
            logging.info("Computing on cpu")
            self._device = torch.device('cpu')

        state_dict = torch.load(conf.model_path, map_location=self._device)
        self._tp_conf: TextProcessorConf = state_dict['tp_conf']
        self._tp_conf.tokenizer.vocab_path = None
        self._tp_state_dict = state_dict['tp']
        self._tp = TextProcessor(self._tp_conf, inference_mode=True)
        self._tp.load_state_dict(self._tp_state_dict)

        mc: DocModelConf = state_dict['model_conf']
        base_sent_enc = create_sent_encoder(mc.sent.encoder, self._tp.vocab())
        base_sent_enc.load_state_dict(state_dict['sent_enc'])
        sent_for_doc_layer = None
        if 'sent_for_doc' in state_dict and mc.sent_for_doc is not None:
            sent_for_doc_layer = create_encoder(mc.sent_for_doc)
            sent_for_doc_layer.load_state_dict(state_dict['sent_for_doc'])
        self._sent_layer = SentForDocEncoder.from_base(base_sent_enc, sent_for_doc_layer)
        self._sent_layer = self._sent_layer.to(device=self._device).eval()
        logging.debug("sent layer\n%s", self._sent_layer)

        self._fragment_layer = None
        sent_embs_out_size = self._sent_layer.out_embs_dim()
        if 'frag_enc' in state_dict and mc.fragment is not None:
            self._fragment_layer = create_emb_seq_encoder(mc.fragment, sent_embs_out_size)
            self._fragment_layer = self._load_layer(self._fragment_layer, state_dict['frag_enc'])
            doc_input_size = self._fragment_layer.out_embs_dim()
            logging.debug("fragment layer\n:%s", self._fragment_layer)
        else:
            doc_input_size = sent_embs_out_size

        self._doc_layer = create_emb_seq_encoder(mc.doc, doc_input_size)
        self._doc_layer = self._load_layer(self._doc_layer, state_dict['doc_enc'])

        logging.debug("doc layer\n:%s", self._doc_layer)

    def _load_layer(self, module, state_dict):
        module.load_state_dict(state_dict)
        return module.to(device=self._device).eval()

    def load_params_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        sent_state_dict = {}
        doc_state_dict = {}
        frag_state_dict = {}
        for k, v in state['model'].items():
            if k.startswith('sent_encoder'):
                sent_state_dict[k.removeprefix('sent_encoder.')] = v
            elif k.startswith('doc_encoder'):
                doc_state_dict[k.removeprefix('doc_encoder.')] = v
            elif k.startswith('frag_encoder'):
                frag_state_dict[k.removeprefix('frag_encoder.')] = v

        self._load_layer(self._sent_layer, sent_state_dict)
        self._load_layer(self._doc_layer, doc_state_dict)

        if frag_state_dict:
            self._load_layer(self._fragment_layer, frag_state_dict)

    def _encode_sents_impl(self, sents, encoder: SentEncoder, collect_on_cpu=False):
        max_len = len(max(sents, key=len))
        sent_tensor, lengths_tensor = create_padded_tensor(
            sents,
            max_len,
            pad_idx=self._tp.vocab().pad_idx(),
            device=self._device,
            pad_to_multiple_of=encoder.pad_to_multiple_of,
        )

        return split_sents_and_embed(
            encoder,
            sent_tensor,
            lengths_tensor,
            max_chunk_size=self._conf.max_sents,
            max_tokens_in_chunk=self._conf.max_tokens,
            collect_on_cpu=collect_on_cpu,
        )

    def _encode_docs_impl(self, docs, doc_fragments):
        """Each doc is a list of tokenized sentences."""

        all_sents = [s for d in docs for s in d]
        sent_embs = self._encode_sents_impl(all_sents, self._sent_layer)

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
            with autocast():
                return self._encode_docs_impl(docs, doc_fragments)

    def _encode_sents(self, sents):
        with torch.inference_mode():
            with autocast():
                encoder = self._sent_layer.cast_to_base()
                return self._encode_sents_impl(sents, encoder, collect_on_cpu=True)

    def encode_sents(self, sents: list[str]) -> np.ndarray:
        """Encode bunch of sents to vectors."""
        sent_ids = self._tp.prepare_sents(sents)
        return self._encode_sents(sent_ids).numpy()

    def encode_sents_stream(
        self, sents_generator, sents_batch_size=4096
    ) -> collections.abc.Iterable[tuple[list[str], np.ndarray, Any]]:
        """Encode stream of sents (represented via generator) to vectors.
        `sents_generator` should yield a sent text. It can also yield extra
        fields that will be yielded back with vectors. At first it collects
        sents in a batch of size `sents_batch_size`. Then invokes `encode_sents`
        on each batch.
        This method yields the tuple of sentence tetxt, sent vectors, and extra stuff from the generator.
        """
        sents_batch = []
        sents_misc = []
        for sent, *misc in sents_generator:
            sents_batch.append(sent)
            sents_misc.append(misc)
            if len(sents_batch) > sents_batch_size:
                embs = self.encode_sents(sents_batch)
                yield sents_batch, embs, sents_misc
                sents_batch = []
                sents_misc = []
        if sents_batch:
            embs = self.encode_sents(sents_batch)
            yield sents_batch, embs, sents_misc

    def encode_docs_from_path_list(self, path_list) -> np.ndarray:
        """This method encodes a texts from a path_list into vectors and returns them.
        Texts should be presegmented, i.e. each sentence should be placed on separate line.
        """
        embs = []
        embs_idxs = []
        batch_iter = BatchIterator(
            (self._tp_conf, self._conf, self._tp_state_dict),
            self._conf.async_batch_gen,
        )
        batch_iter.start_workers_for_item_list(path_list, fetcher=file_path_fetcher)
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

    def encode_docs_from_dir(self, path: Path) -> tuple[list[Path], np.ndarray]:
        """This method iterates over files in the directory and returns a tuple of paths
        and vector representation of texts.
        Texts should be presegmented, i.e. each sentence should be placed on separate line.
        """
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.encode_docs_from_path_list(paths)

    def encode_docs(self, docs: list[list[str] | str]) -> np.ndarray:
        """Encode a list of documents into vector representation. A doc is eiher
        a list of sentences or a text where each sentence is placed on a new
        line."""

        def dummy_fetcher(items):
            yield from enumerate(items)

        embs = []
        batch_generator = BaseBatchGenerator(self._tp_conf, self._conf, self._tp_state_dict)
        for batch, doc_fragments, _ in batch_generator.batches(docs, fetcher=dummy_fetcher):
            doc_embs = self._encode_docs(batch, doc_fragments)
            embs.append(doc_embs.to(device='cpu', dtype=torch.float32))
        stacked = torch.vstack(embs)
        assert len(stacked) == len(docs)
        return stacked.numpy()

    def encode_docs_stream(
        self, doc_id_generator, fetcher, batch_size: int = 10
    ) -> collections.abc.Iterable[tuple[list[Any], np.ndarray]]:
        """The most general method that accepts a generator of document ids and a function `fetcher`.
        `fetcher` will be invoked on a batch of ids to get a text of a documents.
        It should return a list of texts along with text index. Example
        ```
        def file_path_fetcher(paths):
            for idx, path in enumerate(paths):
                with open(path, 'r', encoding='utf8') as fp:
                    sents = []
                    for s in fp:
                        sents.append(s.rstrip())
                    yield idx, sents
        ```
        At first this method collects doc ids in a batch of size `batch_size`.
        Then invokes `fetcher` on each batch and then encode docs into vectors.
        This method yields the tuple of doc id (that was returned by `doc_id_generator`) and doc vectors.
        """

        batch_iter = BatchIterator(
            (self._tp_conf, self._conf, self._tp_state_dict),
            self._conf.async_batch_gen,
        )
        batch_iter.start_workers_for_stream(
            doc_id_generator, fetcher=fetcher, batch_size=batch_size
        )
        for docs, doc_fragments, ids in batch_iter.batches():
            doc_embs = self._encode_docs(docs, doc_fragments)
            yield ids, doc_embs.to(device='cpu', dtype=torch.float32).numpy()
