#!/usr/bin/env python3

import logging
import math
import copy
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
    create_seq_encoder,
)
from doc_enc.encoders.sent_encoder import SentEncoder, split_sents_and_embed, SentForDocEncoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoder
from doc_enc.training.models.model_conf import DocModelConf


@dataclasses.dataclass
class DocEncoderConf:
    model_path: str
    use_gpu: Optional[int] = None

    async_batch_gen: int = 2

    max_sents: int = 2048
    max_tokens: int = 96_000


class BaseBatchGenerator:
    def __init__(
        self, tp_conf: TextProcessorConf, conf: DocEncoderConf, tp_state_dict, eval_mode=True
    ) -> None:
        self._conf = conf
        self._tp = TextProcessor(tp_conf, inference_mode=eval_mode)
        self._tp.load_state_dict(tp_state_dict)

    def batches(self, items: list[Any], fetcher):
        docs = []
        doc_lengths = []
        cur_token_cnt = 0
        cur_segments_cnt = 0
        batch_idx_list = []

        for idx, doc in fetcher(items):
            if isinstance(doc, str):
                doc = doc.split('\n')

            segmented_text, doc_segments_length = self._tp.prepare_text(doc)
            token_cnt = sum(len(s) for s in segmented_text)
            if not token_cnt:
                segmented_text = [[self._tp.vocab().pad_idx()]]
                doc_segments_length = [1]

            if (
                docs
                and self._conf.max_sents
                and cur_segments_cnt + len(segmented_text) > self._conf.max_sents
                or self._conf.max_tokens
                and cur_token_cnt + token_cnt > self._conf.max_tokens
            ):
                yield docs, doc_lengths, batch_idx_list
                docs = []
                doc_lengths = []
                batch_idx_list = []
                cur_segments_cnt = 0
                cur_token_cnt = 0

            docs.append(segmented_text)
            doc_lengths.append(doc_segments_length)
            batch_idx_list.append(idx)
            cur_segments_cnt += len(segmented_text)
            cur_token_cnt += token_cnt
        if docs:
            yield docs, doc_lengths, batch_idx_list


def _proc_wrapper_for_item_list(
    queue: multiprocessing.Queue, items: list[Any], fetcher, offset, *args, **kwargs
):
    try:
        generator = BaseBatchGenerator(*args, **kwargs)
        for docs, doc_lengths, batch_idx_list in generator.batches(items, fetcher):
            batch_idx_list = [offset + i for i in batch_idx_list]
            queue.put((docs, doc_lengths, batch_idx_list))
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
            for docs, doc_lengths, batch_idx_list in generator.batches(items, fetcher):
                batch_items = [items[i] for i in batch_idx_list]
                out_queue.put((docs, doc_lengths, batch_items))
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


class BaseEncodeModule(torch.nn.Module):
    def __init__(
        self,
        doc_layer: SeqEncoder,
        pad_idx: int,
        device: torch.device,
        sent_layer: SentForDocEncoder | None = None,
        frag_layer: SeqEncoder | None = None,
    ) -> None:
        super().__init__()

        self.doc_layer = doc_layer
        self.sent_layer = sent_layer
        self.frag_layer = frag_layer
        self.device = device
        self._pad_idx = pad_idx

    def sent_encoding_supported(self):
        return self.sent_layer is not None

    def sent_embs_dim(self):
        if self.sent_layer is not None:
            return self.sent_layer.out_embs_dim()
        return 0

    def doc_embs_dim(self):
        return self.doc_layer.out_embs_dim()

    def _encode_sents_impl(
        self,
        sents,
        encoder: SentEncoder,
        collect_on_cpu=False,
        already_sorted=False,
        split_sents=True,
        max_chunk_size=1024,
        max_tokens_in_chunk=48_000,
    ):
        max_len = len(max(sents, key=len))
        sent_tensor, lengths_tensor = create_padded_tensor(
            sents,
            max_len,
            pad_idx=self._pad_idx,
            device=self.device,
            pad_to_multiple_of=encoder.pad_to_multiple_of,
        )

        if not split_sents:
            res = encoder(sents, lengths_tensor, enforce_sorted=already_sorted)
            return res.pooled_out

        return split_sents_and_embed(
            encoder,
            sent_tensor,
            lengths_tensor,
            max_chunk_size=max_chunk_size,
            max_tokens_in_chunk=max_tokens_in_chunk,
            collect_on_cpu=collect_on_cpu,
        )

    def _encode_docs_impl(
        self,
        doc_segments: list[list[int]],
        doc_lengths: list[list[int]],
        split_sents=True,
        max_chunk_size=1024,
        max_tokens_in_chunk=48_000,
        batch_info: dict | None = None,
    ):
        """Docs is represented as sequence of segments in a flat list doc_segments"""

        if batch_info is None:
            batch_info = {}
        if self.sent_layer is not None:
            # 1. document is a sequence of sentences
            sent_embs = self._encode_sents_impl(
                doc_segments,
                self.sent_layer,
                split_sents=split_sents,
                max_chunk_size=max_chunk_size,
                max_tokens_in_chunk=max_tokens_in_chunk,
            )

            if self.frag_layer is not None:
                frag_len: list[int] = []
                doc_len_list: list[int] = []
                for fragments in doc_lengths:
                    frag_len.extend(fragments)
                    doc_len_list.append(len(fragments))

                embs = self.frag_layer(
                    sent_embs,
                    frag_len,
                    enforce_sorted=False,
                    padded_seq_len=batch_info.get('fragment_len'),
                ).pooled_out
                padded_seq_len = batch_info.get('doc_len_in_frags')
            else:
                embs = sent_embs
                doc_len_list = [l for d in doc_lengths for l in d]
                padded_seq_len = batch_info.get('doc_len_in_sents')

            doc_embs = self.doc_layer(
                embs, doc_len_list, enforce_sorted=False, padded_seq_len=padded_seq_len
            ).pooled_out
            return doc_embs

        if self.frag_layer is not None:
            # 2. document is a sequence of fragments
            # TODO add enforce_sorted=False??
            embs = self.frag_layer(input_token_ids=doc_segments).pooled_out
            doc_len_list = [l for d in doc_lengths for l in d]
            doc_embs = self.doc_layer(embs, doc_len_list).pooled_out
            return doc_embs

        # 3. document is a sequence of tokens
        doc_embs = self.doc_layer(input_token_ids=doc_segments).pooled_out
        return doc_embs


class EncodeModule(BaseEncodeModule):
    def __init__(self, conf: DocEncoderConf) -> None:
        self._conf = conf
        if conf.use_gpu is not None and torch.cuda.is_available():
            logging.info("Computing on gpu:%s", conf.use_gpu)
            device = torch.device(f'cuda:{conf.use_gpu}')
        else:
            logging.info("Computing on cpu")
            device = torch.device('cpu')

        state_dict = torch.load(conf.model_path, map_location=device)
        self._state_dict = state_dict
        self._tp_conf: TextProcessorConf = state_dict['tp_conf']
        self._tp_conf.tokenizer.vocab_path = None
        self._tp_state_dict = state_dict['tp']
        self._tp = TextProcessor(self._tp_conf, inference_mode=True)
        self._tp.load_state_dict(self._tp_state_dict)

        mc: DocModelConf = state_dict['model_conf']
        sent_layer: SentForDocEncoder | None = None
        sent_embs_out_size = 0
        if mc.sent is not None:
            base_sent_enc = create_sent_encoder(mc.sent.encoder, self._tp.vocab())
            base_sent_enc.load_state_dict(state_dict['sent_enc'])
            sent_for_doc_layer = None
            if 'sent_for_doc' in state_dict and mc.sent_for_doc is not None:
                sent_for_doc_layer = create_encoder(mc.sent_for_doc)
                sent_for_doc_layer.load_state_dict(state_dict['sent_for_doc'])
            sent_layer = SentForDocEncoder.from_base(
                base_sent_enc, sent_for_doc_layer, freeze_base_sents_layer=False
            )
            sent_layer = sent_layer.to(device=device)
            logging.debug("sent layer\n%s", sent_layer)
            sent_embs_out_size = sent_layer.out_embs_dim()

        frag_layer = None
        if 'frag_enc' in state_dict and mc.fragment is not None:
            frag_layer = create_seq_encoder(
                mc.fragment,
                pad_idx=self._tp.vocab().pad_idx(),
                device=device,
                prev_output_size=sent_embs_out_size,
            )
            frag_layer = self._load_layer(frag_layer, state_dict['frag_enc'], device)
            doc_input_size = frag_layer.out_embs_dim()
            logging.debug("fragment layer\n:%s", frag_layer)
        else:
            doc_input_size = sent_embs_out_size

        doc_layer = create_seq_encoder(
            mc.doc,
            pad_idx=self._tp.vocab().pad_idx(),
            device=device,
            prev_output_size=doc_input_size,
        )
        doc_layer = self._load_layer(doc_layer, state_dict['doc_enc'], device)

        logging.debug("doc layer\n:%s", doc_layer)

        super().__init__(
            doc_layer=doc_layer,
            pad_idx=self._tp.vocab().pad_idx(),
            device=device,
            sent_layer=sent_layer,
            frag_layer=frag_layer,
        )

    def _load_layer(self, module, state_dict, device):
        if state_dict:
            module.load_state_dict(state_dict)
        return module.to(device=device)

    def to_dict(self):
        # module weights may have changed; update them
        state_dict = copy.copy(self._state_dict)
        state_dict['doc_enc'] = self.doc_layer.state_dict()
        if self.sent_layer is not None:
            state_dict['sent_enc'] = self.sent_layer.cast_to_base().state_dict()
            if 'sent_for_doc' in state_dict and self.sent_layer.doc_mode_encoder is not None:
                state_dict['sent_for_doc'] = self.sent_layer.doc_mode_encoder.state_dict()

        if 'frag_enc' in state_dict and self.frag_layer is not None:
            state_dict['frag_enc'] = self.frag_layer.state_dict()

        return state_dict

    def tp(self):
        return self._tp

    def load_params_from_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        sent_state_dict = {}
        doc_state_dict = {}
        frag_state_dict = {}
        for k, v in state['model'].items():
            if k.startswith('sent_layer'):
                sent_state_dict[k.removeprefix('sent_layer.')] = v
            elif k.startswith('doc_layer'):
                doc_state_dict[k.removeprefix('doc_layer.')] = v
            elif k.startswith('frag_layer'):
                frag_state_dict[k.removeprefix('frag_layer.')] = v

        self._load_layer(self.doc_layer, doc_state_dict, self.device)
        if sent_state_dict:
            self._load_layer(self.sent_layer, sent_state_dict, self.device)

        if frag_state_dict:
            self._load_layer(self.frag_layer, frag_state_dict, self.device)

    def create_batch_generator(self, eval_mode=True):
        return BaseBatchGenerator(self._tp_conf, self._conf, self._tp_state_dict, eval_mode)

    def create_batch_iterator(self, eval_mode=True):
        return BatchIterator(
            (self._tp_conf, self._conf, self._tp_state_dict, eval_mode),
            self._conf.async_batch_gen,
        )

    def encode_sents(self, sents, collect_on_cpu=False):
        if self.sent_layer is None:
            raise RuntimeError("Sent layer is absent in this model")
        encoder = self.sent_layer.cast_to_base()
        return self._encode_sents_impl(sents, encoder, collect_on_cpu=collect_on_cpu)

    def encode_docs(self, docs: list[list[list[int]]], doc_lengths: list[list[int]]):
        """Each doc is a list of tokenized sequences."""
        all_segments = [s for d in docs for s in d]
        return self._encode_docs_impl(all_segments, doc_lengths, split_sents=False)

    def forward(self, docs, doc_lengths):
        return self.encode_docs(docs, doc_lengths)


def file_path_fetcher(paths):
    for idx, path in enumerate(paths):
        with open(path, 'r', encoding='utf8', errors='ignore') as fp:
            sents = []
            for s in fp:
                sents.append(s.rstrip())
            yield idx, sents


class DocEncoder:
    def __init__(self, conf: DocEncoderConf, eval_mode: bool = True) -> None:
        self._enc_module = EncodeModule(conf)
        self._enc_module.train(not eval_mode)
        self._eval_mode = eval_mode

    def enc_module(self):
        return self._enc_module

    def _encode_docs(self, docs: list[list[list[int]]], doc_lengths: list[list[int]]):
        with torch.inference_mode(self._eval_mode):
            with autocast():
                return self._enc_module.encode_docs(docs, doc_lengths)

    def _encode_sents(self, sents):
        with torch.inference_mode(self._eval_mode):
            with autocast():
                return self._enc_module.encode_sents(sents, collect_on_cpu=True)

    def load_params_from_checkpoint(self, checkpoint_path):
        self._enc_module.load_params_from_checkpoint(checkpoint_path)

    def encode_sents(self, sents: list[str]) -> np.ndarray:
        """Encode bunch of sents to vectors."""
        if not self._enc_module.sent_encoding_supported():
            raise RuntimeError("Sent encoding is unsupported by this model!")
        sent_ids = self._enc_module.tp().prepare_sents(sents)
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
        if not self._enc_module.sent_encoding_supported():
            raise RuntimeError("Sent encoding is unsupported by this model!")

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
        batch_iter = self._enc_module.create_batch_iterator()

        batch_iter.start_workers_for_item_list(path_list, fetcher=file_path_fetcher)
        for docs, doc_lengths, idxs in batch_iter.batches():
            doc_embs = self._encode_docs(docs, doc_lengths)
            embs.append(doc_embs.to(device='cpu', dtype=torch.float32))
            embs_idxs.extend(idxs)

        stacked = torch.vstack(embs)
        assert stacked.shape[0] == len(
            path_list
        ), f"Missaligned data: {stacked.shape[0]} != {len(path_list)}"

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

        batch_generator = self._enc_module.create_batch_generator()
        for batch, doc_lengths, _ in batch_generator.batches(docs, fetcher=dummy_fetcher):
            doc_embs = self._encode_docs(batch, doc_lengths)
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

        batch_iter = self._enc_module.create_batch_iterator()
        batch_iter.start_workers_for_stream(
            doc_id_generator, fetcher=fetcher, batch_size=batch_size
        )
        for docs, doc_lengths, ids in batch_iter.batches():
            doc_embs = self._encode_docs(docs, doc_lengths)
            yield ids, doc_embs.to(device='cpu', dtype=torch.float32).numpy()
