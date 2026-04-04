#!/usr/bin/env python3

import contextlib
import logging
import itertools
import math
import copy
from typing import Callable, Dict, Generator, Optional, Any, Sequence
import dataclasses
from pathlib import Path
import multiprocessing
import collections.abc
import threading

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from doc_enc.encoders.enc_in import (
    EncoderInputType,
    EncoderInData,
    SeqEncoderBatchedInput,
    TextsRepr,
)
from doc_enc.inter_proc_utils import deserialize_enc_in_data, serialize_enc_in_data
from doc_enc.shared_tensors import EncInputSharedTensors
from doc_enc.utils import file_line_cnt
from doc_enc.embs.emb_factory import create_emb_layer
from doc_enc.embs.token_embed import TokenEmbedding


from doc_enc.text_processor import TextProcessor, TextProcessorConf

from doc_enc.encoders.enc_factory import (
    create_encoder,
    create_seq_encoder,
)

from doc_enc.encoders.enc_config import SeqEncoderConf
from doc_enc.encoders.pad_utils import PadOpts
from doc_enc.encoders.split_input import split_padded_input_and_encode
from doc_enc.encoders.sent_for_doc_encoder import SentForDocEncoder
from doc_enc.encoders.seq_encoder import SeqEncoder
from doc_enc.training.models.model_conf import DocModelConf


@dataclasses.dataclass
class TextProcOverride:
    auto_tokenizer_max_seq_len: Optional[int] = None


@dataclasses.dataclass
class EncOverride:
    use_adapter: Optional[str] = None
    adapter_kwargs: Optional[Dict[str, Any]] = None
    transformers_torch_fp16: Optional[bool] = None
    transformers_kwargs: Optional[Dict[str, Any]] = None

    input_type: Optional[EncoderInputType] = None


@dataclasses.dataclass
class ConfOverrides:
    text_proc: Optional[TextProcOverride] = None
    sent: Optional[EncOverride] = None
    frag: Optional[EncOverride] = None
    doc: Optional[EncOverride] = None


@dataclasses.dataclass
class DocEncoderConf:
    model_path: str
    use_gpu: Optional[int] = None

    # number of processes for async generation of batches
    async_batch_gen: int = 2

    max_sents: int = 2048
    max_tokens: int = 96_000

    bucket_multiplier: int = 2

    # truncate docs that are excessively long
    # truncate_long_docs: bool = False

    # perform l2 normalization on encoded vectors.
    normalize_vecs: bool = False
    enable_amp: bool = False
    ensure_flash_attn: bool = False

    overrides: Optional[ConfOverrides] = None


# * Batch helpers


SentsGenFuncT = Callable[[], Generator[tuple[str | int, str | list[str]], None, None]]
TextGenFuncT = SentsGenFuncT


# ** Texts batch generators


class BaseBatchGenerator:
    def __init__(
        self,
        enc_input_type: EncoderInputType,
        conf: DocEncoderConf,
        tp_conf: TextProcessorConf,
        tp_state_dict,
        pad_opts: PadOpts = PadOpts(),
        eval_mode=True,
    ) -> None:
        self._enc_input_type = enc_input_type
        self._conf = conf
        self._tp = TextProcessor(tp_conf, inference_mode=eval_mode)
        self._tp.load_state_dict(tp_state_dict)
        self._pad_opts = pad_opts

    def _prepare_sent(
        self, text: str | list[str], truncate_length_in_tokens: int, truncate_length_in_seqs: int
    ) -> tuple[list[list[int]], list[int]]:
        assert isinstance(text, str), "Prepare sent works only with str type."
        return ([self._tp.prepare_sent(text)], [1])

    def _prepare_text(
        self, text: str | list[str], truncate_length_in_tokens: int, truncate_length_in_seqs: int
    ) -> tuple[list[list[int]], list[int]]:
        if isinstance(text, str):
            text = text.split('\n')
        return self._tp.prepare_text(
            text,
            truncate_length_in_tokens=truncate_length_in_tokens,
            truncate_length_in_seqs=truncate_length_in_seqs,
        )

    def _create_in_data(
        self,
        segmented_texts: list[list[int]],
        text_lengths: list[list[int]],
        text_ids: list[str | int],
        input_are_sents: bool = False,
    ):
        return EncoderInData(
            SeqEncoderBatchedInput.from_input_ids(
                self._enc_input_type, segmented_texts, self._tp.vocab().pad_idx(), self._pad_opts
            ),
            text_ids,
            (
                TextsRepr(segmented_texts, text_lengths)
                if not input_are_sents
                else TextsRepr(segmented_texts, [])
            ),
        )

    def batches(self, generator_func: TextGenFuncT, input_are_sents: bool = False):
        texts: list[list[int]] = []
        text_lengths: list[list[int]] = []
        text_ids_list: list[str | int] = []
        cur_tokens_cnt = 0
        cur_seqs_cnt = 0

        if input_are_sents:
            prepare_text_f = self._prepare_sent
        else:
            prepare_text_f = self._prepare_text

        # TODO collect more data sort them by length?
        # Only do it for PADDed and Packed?
        # m = self._conf.bucket_multiplier
        m = 1
        truncate_length_in_tokens = self._conf.max_tokens
        truncate_length_in_seqs = self._conf.max_sents
        ntruncated_by_tokens = 0
        ntruncated_by_seqs = 0

        for text_id, text in generator_func():
            segmented_text, text_segments_length = prepare_text_f(
                text,
                truncate_length_in_tokens=truncate_length_in_tokens,
                truncate_length_in_seqs=truncate_length_in_seqs,
            )

            tokens_cnt = sum(len(s) for s in segmented_text)
            if (by_seqs := len(segmented_text) == truncate_length_in_seqs) or (
                tokens_cnt == truncate_length_in_tokens
            ):
                logging.debug(
                    "text_truncated; text_id=%s, nsegments=%s, ntokens=%s",
                    text_id,
                    len(segmented_text),
                    tokens_cnt,
                )
                if by_seqs:
                    ntruncated_by_seqs += 1
                else:
                    ntruncated_by_tokens += 1

            if not tokens_cnt:
                segmented_text = [[self._tp.vocab().pad_idx()]]
                text_segments_length = [1]
                tokens_cnt = 1

            if texts and (
                cur_seqs_cnt + len(segmented_text) > m * self._conf.max_sents
                or cur_tokens_cnt + tokens_cnt > m * self._conf.max_tokens
            ):
                yield self._create_in_data(
                    texts, text_lengths, text_ids_list, input_are_sents=input_are_sents
                )
                texts = []
                text_lengths = []
                text_ids_list = []
                cur_seqs_cnt = 0
                cur_tokens_cnt = 0

            texts.extend(segmented_text)
            text_lengths.append(text_segments_length)
            text_ids_list.append(text_id)
            cur_seqs_cnt += len(segmented_text)
            cur_tokens_cnt += tokens_cnt
        if texts:
            yield self._create_in_data(
                texts, text_lengths, text_ids_list, input_are_sents=input_are_sents
            )

        if ntruncated_by_tokens or ntruncated_by_seqs:
            logging.debug(
                "text_truncated: by_tokens=%s, by_seqs_cnt=%s",
                ntruncated_by_tokens,
                ntruncated_by_seqs,
            )


def _proc_wrapper_for_texts_generator(
    in_queue: multiprocessing.Queue,
    queue: multiprocessing.Queue,
    shared_tensors_holder: EncInputSharedTensors,
    *args,
    **kwargs,
):
    torch.set_num_threads(1)
    try:
        generator = BaseBatchGenerator(shared_tensors_holder.enc_input_type, *args, **kwargs)

        while True:
            indata = in_queue.get()
            if indata is None:
                break
            generator_func, input_are_sents = indata
            for b in generator.batches(generator_func, input_are_sents):

                d = serialize_enc_in_data(b, shared_tensors_holder)
                queue.put(d)
            queue.put(None)

    except Exception as e:
        print(type(e), str(e))
        logging.exception("Failed to process batches: %s", e)


class BatchAsyncGenerator:
    """The same as BaseBatchGenerator, but uses background workers for
    producing batches.
    """

    def __init__(
        self,
        enc_input_type: EncoderInputType,
        conf: DocEncoderConf,
        other_generator_args=(),
        nworkers=1,
        fix_batch_order: bool = False,
    ):
        """fix_batch_order is used to ensure determinism while
        training/fine-tunining. When it is true, batches will be always
        generated in the same order.

        """
        self._generator_args = (conf,) + other_generator_args
        self._nworkers = nworkers

        self._processes = []
        self._out_queues: list[multiprocessing.Queue] = []
        self._shared_tensors_holders: list[EncInputSharedTensors] = []
        self._fix_batch_order = fix_batch_order

        cap_m = 3
        if fix_batch_order:
            self._out_queues = [multiprocessing.Queue(cap_m) for _ in range(nworkers)]
            self._shared_tensors_holders = [
                EncInputSharedTensors(enc_input_type, conf.max_tokens, conf.max_sents, cap_m)
                for _ in range(nworkers)
            ]

        else:
            self._out_queues = [multiprocessing.Queue(cap_m * nworkers)]
            self._shared_tensors_holders = [
                EncInputSharedTensors(
                    enc_input_type,
                    conf.max_tokens,
                    conf.max_sents,
                    cap_m * nworkers,
                )
            ]

        self._in_queue = multiprocessing.Queue()

    def nworkers(self):
        return self._nworkers

    def is_batch_order_fixed(self):
        return self._fix_batch_order

    def destroy(self):
        self._terminate_workers()
        self._in_queue.close()
        # Generator might be destroyed before batches() is exhausted.
        self._in_queue.cancel_join_thread()

        for out_q in self._out_queues:
            out_q.close()
            out_q.cancel_join_thread()

    def _terminate_workers(self):
        for p in self._processes:
            p.terminate()
            p.join()
        self._processes = []

    def _input_generator_thread(self, gens: Sequence[TextGenFuncT], input_are_sents: bool):
        for func in gens:
            self._in_queue.put((func, input_are_sents))

    def start_workers(self):
        proc_cnt = self._nworkers
        for i in range(proc_cnt):
            if self._fix_batch_order:
                out_q = self._out_queues[i]
                t_holder = self._shared_tensors_holders[i]
            else:
                out_q = self._out_queues[0]
                t_holder = self._shared_tensors_holders[0]

            p = multiprocessing.Process(
                target=_proc_wrapper_for_texts_generator,
                args=(self._in_queue, out_q, t_holder) + self._generator_args,
                kwargs={},
            )
            p.start()

            self._processes.append(p)

    def _print_debug_info_for_batch(self, batch: EncoderInData):
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return

        seqs_cnt = 0
        if batch.texts_repr.second_level_lengths is not None:
            seqs_cnt = batch.texts_repr.second_level_lengths.shape[0]

        logging.debug(
            "docs_cnt=%s, segments_cnt=%s, tokens_cnt=%s; seq_max_len=%s",
            len(batch.text_ids),
            seqs_cnt,
            batch.seq_encoder_input.ntokens(),
            batch.seq_encoder_input.max_len,
        )

    def _fixed_batch_order_queue_gen(self):
        last_q_idx = 0
        finished = [False] * self._nworkers
        nfinished = 0
        while nfinished < self._nworkers:
            is_finished = yield last_q_idx
            if is_finished:
                finished[last_q_idx] = True
                nfinished += 1
                # for send invoker
                yield 0
            # condition to prevent infinite loop when all(f for f in finished)
            while nfinished < self._nworkers:
                last_q_idx = (last_q_idx + 1) % self._nworkers
                if not finished[last_q_idx]:
                    break

    def _simple_queue_gen(self, ngens: int):
        completed_gens = 0
        while completed_gens < ngens:
            is_finished = yield 0
            if is_finished:
                completed_gens += 1
                # for send invoker
                yield 0

    def batches(
        self, generator_funcs: Sequence[TextGenFuncT], input_are_sents: bool = False
    ) -> Generator[EncoderInData, None, None]:
        """When class is created with fix_batch_order == True, length of
        generator_funcs should be <= # of workers (set by nworkers in
        the constructor).

        """
        if not self._processes:
            raise RuntimeError("Batch Iterator is not initialized!")

        if self._in_queue.qsize() != 0 or any(q.qsize() for q in self._out_queues):
            raise RuntimeError(
                "Previous batches are not fully exhausted! Destroy generator and create it anew!"
            )

        ngens = len(generator_funcs)
        if self._fix_batch_order and ngens != self._nworkers:
            raise RuntimeError(
                "AsyncGenerator was create with fix_batch_order==True, len(generator_funcs) should be == # of workers."
            )

        for g in generator_funcs:
            self._in_queue.put((g, input_are_sents))

        if not self._fix_batch_order:
            consume_order_gen = self._simple_queue_gen(ngens)
        else:
            consume_order_gen = self._fixed_batch_order_queue_gen()

        for q_idx in consume_order_gen:
            out_q = self._out_queues[q_idx]
            logging.debug("queue len: %s", out_q.qsize())
            batch = out_q.get()

            if batch is None:
                consume_order_gen.send(True)
                continue

            shared_t = self._shared_tensors_holders[q_idx]
            with deserialize_enc_in_data(batch, shared_t) as b:
                self._print_debug_info_for_batch(b)
                yield b


# * Input helpers


def encode_input_data(
    input_data: SeqEncoderBatchedInput,
    encoder: SeqEncoder,
    embed: TokenEmbedding | None = None,
    collect_on_cpu=False,
    split_data=True,
    max_chunk_size=1024,
    max_tokens_in_chunk=48_000,
):
    def _enc_cb(input_data: SeqEncoderBatchedInput, **kwargs):
        if embed is not None:
            input_data.embed_(embed)

        return encoder(input_data, **kwargs)

    # logging.error("Encode input data: bs %s, max len %s", input_data.batch_size, input_data.max_len)

    if encoder.input_type() == EncoderInputType.PADDED:

        if not split_data:
            # TODO handle collect_on_cpu == True ??
            return _enc_cb(input_data).pooled_out

        return split_padded_input_and_encode(
            _enc_cb,
            input_data,
            max_chunk_size=max_chunk_size,
            max_tokens_in_chunk=max_tokens_in_chunk,
            collect_on_cpu=collect_on_cpu,
            # TODO
            # already_sorted=input_data.input_sorted(),
            pad_opts=PadOpts(encoder.pad_to_multiple_of, encoder.get_padding_side()),
        )
    else:
        if embed is not None:
            input_data.embed_(embed)
        enc_out = encoder(input_data)
        if collect_on_cpu:
            return enc_out.pooled_out.cpu()
        return enc_out.pooled_out


# * Encode modules


class BaseSentEncodeModule(torch.nn.Module):
    def __init__(
        self,
        pad_idx: int,
        device: torch.device,
        embed: TokenEmbedding | None = None,
        sent_layer: SentForDocEncoder | None = None,
    ):
        super().__init__()
        self._pad_idx = pad_idx
        self.device = device

        self.embed = embed
        self.sent_layer = sent_layer

    def first_encode_layer(self) -> SeqEncoder:
        if self.sent_layer is not None:
            return self.sent_layer
        raise RuntimeError("No encode layers in Sent Encode module")

    def _prepare_input_data(self, input_data: EncoderInData):
        input_data.seq_encoder_input.to_(self.device)
        input_data.seq_encoder_input.init_padding_mask_(
            self._pad_idx,
            self.first_encode_layer().get_padding_side(),
        )

        return input_data
        # max_len = len(max(doc_segments, key=len))

        # tokens_tensor, lengths_tensor = create_padded_tensor(
        #     doc_segments,
        #     max_len,
        #     pad_idx=self._pad_idx,
        #     device=self.device,
        #     pad_to_multiple_of=self.first_encode_layer().pad_to_multiple_of,
        #     padding_side=self.first_encode_layer().get_padding_side(),
        # )

        # return _InputData(
        #     tokens_tensor=tokens_tensor,
        #     lengths_tensor=lengths_tensor,
        #     already_sorted=already_sorted,
        # )

    def _encode_sents_impl(
        self,
        sents: list[list[int]],
        already_sorted=False,
        collect_on_cpu=False,
        split_sents=True,
        max_chunk_size=1024,
        max_tokens_in_chunk=48_000,
    ):
        assert self.sent_layer is not None, "Logic error 38389"
        # TODO
        input_data = self._prepare_input_data(sents, already_sorted=already_sorted)
        return encode_input_data(
            input_data,
            self.sent_layer,
            embed=self.embed,
            collect_on_cpu=collect_on_cpu,
            split_data=split_sents,
            max_chunk_size=max_chunk_size,
            max_tokens_in_chunk=max_tokens_in_chunk,
        )


class BaseEncodeModule(BaseSentEncodeModule):
    def __init__(
        self,
        doc_layer: SeqEncoder,
        pad_idx: int,
        device: torch.device,
        embed: TokenEmbedding | None = None,
        sent_layer: SentForDocEncoder | None = None,
        frag_layer: SeqEncoder | None = None,
    ) -> None:
        super().__init__(embed=embed, sent_layer=sent_layer, pad_idx=pad_idx, device=device)

        self.doc_layer = doc_layer
        self.frag_layer = frag_layer

    def sent_encoding_supported(self):
        return self.sent_layer is not None or (
            self.frag_layer is None and self.doc_layer is not None
        )

    def sent_embs_dim(self):
        if self.sent_layer is not None:
            return self.sent_layer.out_embs_dim()
        return 0

    def doc_embs_dim(self):
        return self.doc_layer.out_embs_dim()

    def input_type(self):
        return self.first_encode_layer().input_type()

    def first_encode_layer(self) -> SeqEncoder:
        if self.sent_layer is not None:
            return self.sent_layer
        if self.frag_layer is not None:
            return self.frag_layer
        return self.doc_layer

    def last_encode_layer(self) -> SeqEncoder:
        if self.doc_layer is not None:
            return self.doc_layer
        if self.frag_layer is not None:
            return self.frag_layer
        if self.sent_layer is not None:
            return self.sent_layer
        raise RuntimeError("last_encode_layer: logic error!")

    def _sent_level_ctx_mgr(self):
        return contextlib.nullcontext()

    def _encode_docs_impl(
        self,
        input_data: EncoderInData,
        split_input=True,
        max_chunk_size=1024,
        max_tokens_in_chunk=48_000,
        batch_info: dict | None = None,
    ):

        # """Docs is represented as sequence of segments in a flat list doc_segments"""

        if batch_info is None:
            batch_info = {}

        input_data = self._prepare_input_data(input_data)

        if self.sent_layer is not None:
            # 1. document is a sequence of sentences

            with self._sent_level_ctx_mgr():
                sent_embs = encode_input_data(
                    input_data.seq_encoder_input,
                    self.sent_layer,
                    embed=self.embed,
                    split_data=split_input,
                    max_chunk_size=max_chunk_size,
                    max_tokens_in_chunk=max_tokens_in_chunk,
                )

            if self.frag_layer is not None:
                # frag_len: list[int] = []
                # doc_len_list: list[int] = []
                # for fragments in doc_lengths:
                #     frag_len.extend(fragments)
                #     doc_len_list.append(len(fragments))
                #
                if input_data.texts_repr.second_level_lengths is None:
                    raise RuntimeError(
                        "Fragment layer is not None but second_level_lengths is None"
                    )

                input_for_frag_layer = SeqEncoderBatchedInput.from_embs(
                    self.frag_layer.input_type(),
                    sent_embs,
                    input_data.texts_repr.second_level_lengths.to(sent_embs.device),
                    padded_prepend_with_zero=self.frag_layer.beg_seq_param is not None,
                )

                # len_tensor = torch.as_tensor(frag_len, dtype=torch.int64, device=sent_embs.device)
                embs = self.frag_layer(
                    input_for_frag_layer
                    # padded_seq_len=batch_info.get('fragment_len'),
                ).pooled_out
                # padded_seq_len = batch_info.get('doc_len_in_frags')
                # TODO temp
                doc_len_tens = input_data.texts_repr.third_level_lengths
                # if doc_len_tens is None:
                #     doc_len_tens = torch.ones((embs.shape[0]), dtype=torch.int32)
                assert doc_len_tens is not None, "Logic error 3425181"
            else:
                embs = sent_embs
                # doc_len_list = [l for d in doc_lengths for l in d]
                # padded_seq_len = batch_info.get('doc_len_in_sents')
                doc_len_tens = input_data.texts_repr.second_level_lengths
                assert doc_len_tens is not None, "Logic error 3425182"

            input_for_doc_layer = SeqEncoderBatchedInput.from_embs(
                self.doc_layer.input_type(),
                embs,
                doc_len_tens.to(embs.device),
                padded_prepend_with_zero=self.doc_layer.beg_seq_param is not None,
            )
            # len_tensor = torch.as_tensor(doc_len_list, dtype=torch.int64, device=embs.device)
            doc_embs = self.doc_layer(
                input_for_doc_layer
                # padded_seq_len=padded_seq_len,
            ).pooled_out
            return doc_embs

        if self.frag_layer is not None:
            # 2. document is a sequence of fragments

            embs = encode_input_data(
                input_data.seq_encoder_input,
                self.frag_layer,
                embed=self.embed,
                split_data=split_input,
                max_chunk_size=max_chunk_size,
                max_tokens_in_chunk=max_tokens_in_chunk,
            )

            if input_data.texts_repr.second_level_lengths is None:
                raise RuntimeError("Fragment layer is not None but second_level_lengths is None")

            input_for_frag_layer = SeqEncoderBatchedInput.from_embs(
                self.frag_layer.input_type(),
                embs,
                input_data.texts_repr.second_level_lengths.to(embs.device),
                padded_prepend_with_zero=self.frag_layer.beg_seq_param is not None,
            )

            # len_tensor = torch.as_tensor(
            #     [l for d in doc_lengths for l in d], dtype=torch.int64, device=embs.device
            # )
            doc_embs = self.doc_layer(input_for_frag_layer).pooled_out
            return doc_embs

        # 3. document is a sequence of tokens
        return encode_input_data(
            input_data.seq_encoder_input,
            self.doc_layer,
            self.embed,
            split_data=split_input,
            max_chunk_size=max_chunk_size,
            max_tokens_in_chunk=max_tokens_in_chunk,
        )


def _adjust_enc_config(
    config: SeqEncoderConf, eval_mode: bool, override: EncOverride | None = None
):
    # wipe out cache dir since it is different from machine on which model was trained
    if config.transformers_cache_dir:
        config.transformers_cache_dir = None

    if not eval_mode:
        config.transformers_fix_pretrained_params = False

    # It is dumb merge, even if optional value is not set, it will override value on the left with None from the right.
    # if override is not None:
    #     return OmegaConf.structured(OmegaConf.merge(config, OmegaConf.to_container(override)))
    if override is not None:
        if override.input_type is not None:
            config.input_type = override.input_type

        if override.transformers_torch_fp16 is not None:
            config.transformers_torch_fp16 = override.transformers_torch_fp16
        if override.use_adapter is not None:
            config.use_adapter = override.use_adapter
        if override.adapter_kwargs is not None:
            config.adapter_kwargs = override.adapter_kwargs
        if override.transformers_kwargs is not None:
            if config.transformers_kwargs is None:
                config.transformers_kwargs = override.transformers_kwargs
            else:
                config.transformers_kwargs.update(override.transformers_kwargs)


def _adjust_tp_config(tp: TextProcessor, override: ConfOverrides | None):
    if (
        override is not None
        and override.text_proc is not None
        and (msl := override.text_proc.auto_tokenizer_max_seq_len) is not None
    ):
        tp._tokenizer.set_max_seq_length(msl)
        tp.conf().tokenizer.auto_tokenizer_max_seq_len = msl


class EncodeModule(BaseEncodeModule):
    def __init__(self, conf: DocEncoderConf, eval_mode: bool = True) -> None:
        self._conf = conf
        if conf.use_gpu is not None and conf.use_gpu >= 0 and torch.cuda.is_available():
            logging.info("Computing on gpu:%s", conf.use_gpu)
            device = torch.device(f'cuda:{conf.use_gpu}')
        else:
            logging.info("Computing on cpu")
            device = torch.device('cpu')

        state_dict = torch.load(conf.model_path, map_location=device, weights_only=False)
        self._state_dict = state_dict
        self._tp_conf: TextProcessorConf = state_dict['tp_conf']
        self._tp_conf.tokenizer.vocab_path = None
        self._tp_conf.tokenizer.transformers_cache_dir = None
        self._tp_state_dict = state_dict['tp']
        self._tp = TextProcessor(self._tp_conf, inference_mode=True)
        self._tp.load_state_dict(self._tp_state_dict)
        _adjust_tp_config(self._tp, conf.overrides)

        vocab = self._tp.vocab()
        mc: DocModelConf = state_dict['model_conf']
        embed: TokenEmbedding | None = None
        emb_dim = 0
        if 'embed' in state_dict and mc.embed is not None:
            embed = create_emb_layer(mc.embed, vocab.vocab_size(), vocab.pad_idx())
            emb_dim = mc.embed.emb_dim

        sent_layer: SentForDocEncoder | None = None
        sent_embs_out_size = 0
        if mc.sent is not None:
            _adjust_enc_config(
                mc.sent.encoder, eval_mode, None if not conf.overrides else conf.overrides.sent
            )
            base_sent_enc = create_seq_encoder(
                mc.sent.encoder, prev_output_size=emb_dim, eval_mode=eval_mode
            )
            sent_for_doc_layer = None
            if 'sent_for_doc' in state_dict and mc.sent_for_doc is not None:
                sent_for_doc_layer = create_encoder(mc.sent_for_doc, eval_mode)
            sent_layer = SentForDocEncoder.from_base(
                base_sent_enc, sent_for_doc_layer, freeze_base_sents_layer=False
            )
            logging.debug("sent layer\n%s", sent_layer)
            sent_embs_out_size = sent_layer.out_embs_dim()

        frag_layer = None
        if 'frag_enc' in state_dict and mc.fragment is not None:
            _adjust_enc_config(
                mc.fragment, eval_mode, None if not conf.overrides else conf.overrides.frag
            )
            frag_layer = create_seq_encoder(
                mc.fragment, prev_output_size=sent_embs_out_size, eval_mode=eval_mode
            )
            doc_input_size = frag_layer.out_embs_dim()
            logging.debug("fragment layer\n:%s", frag_layer)
        else:
            doc_input_size = sent_embs_out_size

        _adjust_enc_config(mc.doc, eval_mode, None if not conf.overrides else conf.overrides.doc)
        doc_layer = create_seq_encoder(mc.doc, prev_output_size=doc_input_size, eval_mode=eval_mode)

        logging.debug("doc layer\n:%s", doc_layer)

        super().__init__(
            doc_layer=doc_layer,
            pad_idx=vocab.pad_idx(),
            device=device,
            embed=embed,
            sent_layer=sent_layer,
            frag_layer=frag_layer,
        )
        self.load_params_from_state_dict(state_dict)

    def _load_layer(self, module, state_dict):
        if state_dict:
            module.load_state_dict(state_dict)
        return module.to(device=self.device)

    def to_dict(self):
        # module weights may have changed; update them
        state_dict = copy.copy(self._state_dict)
        state_dict['doc_enc'] = self.doc_layer.state_dict()
        if self.embed is not None:
            state_dict['embed'] = self.embed.state_dict()
        if self.sent_layer is not None:
            state_dict['sent_enc'] = self.sent_layer.cast_to_base().state_dict()
            if self.sent_layer.doc_mode_encoder is not None:
                state_dict['sent_for_doc'] = self.sent_layer.doc_mode_encoder.state_dict()

        if self.frag_layer is not None:
            state_dict['frag_enc'] = self.frag_layer.state_dict()

        return state_dict

    def tp(self):
        return self._tp

    def load_params_from_state_dict(self, state_dict):
        if self.embed is not None and 'embed' in state_dict:
            self.embed = self._load_layer(self.embed, state_dict['embed'])

        if self.sent_layer is not None:
            if (sent_enc_state := state_dict.get('sent_enc')) is not None:
                self.sent_layer.cast_to_base().load_state_dict(sent_enc_state)

            if self.sent_layer.doc_mode_encoder is not None:
                self.sent_layer.doc_mode_encoder.load_state_dict(state_dict['sent_for_doc'])

            self.sent_layer = self.sent_layer.to(device=self.device)

        if self.frag_layer is not None:
            self.frag_layer = self._load_layer(self.frag_layer, state_dict['frag_enc'])

        self.doc_layer = self._load_layer(self.doc_layer, state_dict['doc_enc'])

    def load_params_from_checkpoint(self, checkpoint: str | dict):
        if isinstance(checkpoint, (str, Path)):
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        else:
            state = checkpoint

        embed_state_dict = {}
        sent_state_dict = {}
        doc_state_dict = {}
        frag_state_dict = {}
        for k, v in state['model'].items():
            if k.startswith('embed'):
                embed_state_dict[k.removeprefix('embed.')] = v
            if k.startswith('sent_layer'):
                sent_state_dict[k.removeprefix('sent_layer.')] = v
            elif k.startswith('doc_layer'):
                doc_state_dict[k.removeprefix('doc_layer.')] = v
            elif k.startswith('frag_layer'):
                frag_state_dict[k.removeprefix('frag_layer.')] = v

        self._load_layer(self.doc_layer, doc_state_dict)
        if embed_state_dict:
            self._load_layer(self.embed, embed_state_dict)

        if sent_state_dict:
            self._load_layer(self.sent_layer, sent_state_dict)

        if frag_state_dict:
            self._load_layer(self.frag_layer, frag_state_dict)

    def _create_pad_opts(self):
        return PadOpts(
            self.first_encode_layer().pad_to_multiple_of,
            self.first_encode_layer().get_padding_side(),
        )

    def create_batch_generator(self, eval_mode=True):
        po = self._create_pad_opts()
        return BaseBatchGenerator(
            self.input_type(), self._conf, self._tp_conf, self._tp_state_dict, po, eval_mode
        )

    def create_batch_async_generator(self, eval_mode=True, fix_batch_order: bool = False):
        po = self._create_pad_opts()
        gen = BatchAsyncGenerator(
            self.input_type(),
            self._conf,
            (self._tp_conf, self._tp_state_dict, po, eval_mode),
            self._conf.async_batch_gen,
            fix_batch_order=fix_batch_order,
        )
        gen.start_workers()
        return gen

    def encode_sents(self, input_data: EncoderInData, collect_on_cpu=False, already_sorted=False):
        if self.sent_layer is not None:
            encoder = self.sent_layer.cast_to_base()
        elif self.frag_layer is None and self.doc_layer is not None:
            encoder = self.doc_layer
        else:
            raise RuntimeError("Sentence encoding is not supported!")

        input_data = self._prepare_input_data(input_data)
        return encode_input_data(
            input_data.seq_encoder_input,
            encoder,
            embed=self.embed,
            collect_on_cpu=collect_on_cpu,
            split_data=True,
            max_chunk_size=self._conf.max_sents,
            max_tokens_in_chunk=self._conf.max_tokens,
        )

    def encode_docs(self, input_data: EncoderInData):
        return self._encode_docs_impl(
            input_data,
            split_input=True,
            max_chunk_size=self._conf.max_sents,
            max_tokens_in_chunk=self._conf.max_tokens,
        )

    def forward(self, input_data: EncoderInData):
        return self.encode_docs(input_data)


# * Main classes


class SentEncodeStat:
    def __init__(self) -> None:
        self.total_tokens_cnt = 0
        self.sents_cnt = 0


class DocEncodeStat:
    def __init__(self) -> None:
        self.total_tokens_cnt = 0
        self.total_sents_cnt = 0

        self.docs_cnt = 0


# def file_path_fetcher(paths):
#     for idx, path in enumerate(paths):
#         with open(path, 'r', encoding='utf8', errors='ignore') as fp:
#             sents = []
#             for s in fp:
#                 sents.append(s.rstrip())
#             yield idx, sents


class TextsFromPathListGen:
    def __init__(self, paths: list[str] | list[Path], offs: int = 0):
        self.paths = paths
        self.offs = offs

    def __call__(self):
        for idx, path in enumerate(self.paths):
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                sents = []
                for s in fp:
                    sents.append(s.rstrip())
                yield self.offs + idx, sents


def create_text_gens_from_ids_list(
    text_id_list: list[Any], nsplits: int, gen_cls: type = TextsFromPathListGen
):
    per_worker_items = math.ceil(len(text_id_list) / nsplits)
    gens = []
    for offs in range(0, len(text_id_list), per_worker_items):
        gens.append(gen_cls(text_id_list[offs : offs + per_worker_items], offs))
    return gens


class SentsFromFileGen:
    def __init__(self, fn: str, offset: int, limit: int, first_column_is_id: bool, sep: str):
        self.fn = fn
        self.offset = offset
        self.limit = limit
        self.first_column_is_id = first_column_is_id
        self.sep = sep

    def __call__(self):
        with open(self.fn, 'r') as inpf:
            if self.first_column_is_id:
                for line in itertools.islice(inpf, self.offset, self.offset + self.limit):
                    sent_id, sent = line.rstrip().split(self.sep, 1)
                    yield sent_id, sent
            else:
                for i, line in enumerate(
                    itertools.islice(inpf, self.offset, self.offset + self.limit)
                ):
                    yield i + self.offset, line.rstrip()


# def _create_sents_gen_func(fn: str, offset: int, limit: int, first_column_is_id: bool, sep: str):
#     def _gen():
#         with open(fn, 'r') as inpf:
#             if first_column_is_id:
#                 for line in itertools.islice(inpf, offset, offset + limit):
#                     sent_id, sent = line.rstrip().split(sep, 1)
#                     yield sent_id, sent
#             else:
#                 for i, line in enumerate(itertools.islice(inpf, offset, offset + limit)):
#                     yield i + offset, line.rstrip()

#     return _gen


class DocEncoder:
    """This class provides efficient methods for offline encoding batch of
    texts. It could be used for encoding stream of texts, but it will not be
    efficient at this.
    """

    def __init__(self, conf: DocEncoderConf, eval_mode: bool = True) -> None:
        self._enc_module = EncodeModule(conf, eval_mode)
        self._enc_module.train(not eval_mode)
        self._eval_mode = eval_mode

        self._batch_gen = self._enc_module.create_batch_async_generator()

        self._destroyed = False

    def __del__(self):
        self.destroy()

    def destroy(self):
        if not self._destroyed:
            self._batch_gen.destroy()
            self._destroyed = True

    def sent_encoding_supported(self):
        return self._enc_module.sent_encoding_supported()

    def enc_module(self):
        return self._enc_module

    def conf(self):
        return self._enc_module._conf

    def _sdpa_mods(self):
        if self.conf().ensure_flash_attn:
            sdpa_mods = SDPBackend.FLASH_ATTENTION
        else:
            sdpa_mods = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

        return sdpa_mods

    def _encode_docs(self, input_data: EncoderInData):
        with torch.inference_mode(self._eval_mode):
            with autocast(self._enc_module.device.type, enabled=self.conf().enable_amp):
                with sdpa_kernel(self._sdpa_mods()):
                    return self._enc_module.encode_docs(input_data)

    def _encode_sents(self, input_data: EncoderInData):
        with torch.inference_mode(self._eval_mode):
            with autocast(self._enc_module.device.type, enabled=self.conf().enable_amp):
                with sdpa_kernel(self._sdpa_mods()):
                    return self._enc_module.encode_sents(input_data, collect_on_cpu=True)

    def load_params_from_checkpoint(self, checkpoint_path):
        self._enc_module.load_params_from_checkpoint(checkpoint_path)

    def encode_sents(self, sents: list[str]) -> np.ndarray:
        """Encode bunch of sents to vectors.
        async_batch_gen option is ignored."""
        if not self._enc_module.sent_encoding_supported():
            raise RuntimeError("Sent encoding is unsupported by this model!")

        batch_gen = self._enc_module.create_batch_generator()

        all_embs = []
        all_ids = []

        def _gen():
            yield from enumerate(sents)

        for input_data in batch_gen.batches(_gen):
            sent_embs = self._encode_sents(input_data)
            all_embs.append(sent_embs.to(device='cpu'))
            all_ids.extend(input_data.text_ids)

        stacked = torch.vstack(all_embs)
        assert all_ids == list(range(len(sents))), "Misaligned data 3812"
        assert len(stacked) == len(sents), "Misaligned data 3813"
        if self.conf().normalize_vecs:
            stacked = F.normalize(stacked, p=2, dim=1)

        return stacked.numpy()

    def encode_sents_generator(
        self, sents_generator: Generator[tuple[str, ...], None, None], sents_batch_size=512
    ) -> collections.abc.Iterable[tuple[list[str], np.ndarray, Any]]:
        """Encode batch of sents (produced by generator) to vectors.
        `sents_generator` should yield a sentence text. It can also yield extra
        fields that will be yielded back with vectors. At first it collects
        sents in a batch of size `sents_batch_size`. Then invokes `encode_sents`
        on each batch.
        This method yields the tuple of sentence tetxt, sent vectors, and extra stuff from the generator.
        async_batch_gen option from DocEncoderConf is ignored.
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

    def encode_sents_from_generators(
        self, generator_funcs: Sequence[SentsGenFuncT], stat: SentEncodeStat | None = None
    ):
        """Low level function that utilizes async batch preparation to speed up encoding for some cases.
        You can pass up to async_batch_gen generator functions.
        Each function should produce a unique set of tuples: (sent_id, sent_text).
        Example:

        def create_gen_func(fn, offset, limit):
            def _gen():
                with open(fn, 'r') as inpf:
                    for i, line in enumerate(itertools.islice(inpf, offset, offset + limit)):
                        yield offset + i, line.strip()
            return _gen
        encode_sents_from_generators([create_gen_func('/f/p', 0, 100), create_gen_func('/f/p', 100, 100)])


        or

        class SentsGen:
            def __init__(self, fn, offset, limit):
                self.fn = fn
                self.offset = offset
                self.limit = limit

            def __call__(self):
                with open(self.fn, 'r') as inpf:
                    for i, line in enumerate(itertools.islice(inpf, self.offset, self.offset + self.limit)):
                        yield self.offset + i, line.strip()

        encode_sents_from_generators([SentsGen('/f/p', 0, 100), SentsGen('/f/p', 100, 100)])
        """

        if not self._enc_module.sent_encoding_supported():
            raise RuntimeError("Sent encoding is unsupported by this model!")
        # batch_generator = self._enc_module.create_batch_async_generator()
        # batch_generator.start_workers()

        for batch in self._batch_gen.batches(generator_funcs, input_are_sents=True):
            if stat is not None:
                stat.total_tokens_cnt += batch.seq_encoder_input.ntokens()
                stat.sents_cnt += batch.seq_encoder_input.batch_size

            sent_embs = self._encode_sents(batch)
            if self.conf().normalize_vecs:
                sent_embs = F.normalize(sent_embs, p=2, dim=1)
            sent_embs = sent_embs.to(device='cpu').numpy()
            yield batch.text_ids, sent_embs

    def generate_sent_embs_from_file(
        self,
        file_path,
        lines_limit=0,
        first_column_is_id=False,
        sep='\t',
        stat: SentEncodeStat | None = None,
    ):
        """Generator of embeddings from the given file.
        File is split on async_batch_gen parts.
        Sentences from each part are prepared in its own process.
        """
        line_cnt = file_line_cnt(file_path, limit=lines_limit)
        proc_num = self.conf().async_batch_gen
        per_proc = math.ceil(line_cnt / proc_num)
        gens = [
            SentsFromFileGen(
                file_path, offs, per_proc, first_column_is_id=first_column_is_id, sep=sep
            )
            for offs in range(0, line_cnt, per_proc)
        ]

        for ids, embs in self.encode_sents_from_generators(gens, stat=stat):
            yield ids, embs

    def encode_sents_from_file(
        self,
        file_path,
        lines_limit=0,
        first_column_is_id=False,
        sep='\t',
        stat: SentEncodeStat | None = None,
    ):
        sent_ids = []
        sent_embs = []
        for ids, embs in self.generate_sent_embs_from_file(
            file_path, lines_limit, first_column_is_id, sep, stat=stat
        ):
            sent_ids.extend(ids)
            sent_embs.append(embs)

        stacked = np.vstack(sent_embs)
        assert len(sent_ids) == stacked.shape[0], "Missaligned data 83292"
        return sent_ids, stacked

    def _update_doc_stat(self, input_data: EncoderInData, stat: DocEncodeStat):

        stat.docs_cnt += len(input_data.text_ids)
        stat.total_sents_cnt += input_data.seq_encoder_input.batch_size
        stat.total_tokens_cnt += input_data.seq_encoder_input.ntokens()

    def _reorder_collected_arrays(self, stacked_array, idxs):
        assert stacked_array.shape[0] == len(
            idxs
        ), f"Missaligned data: {stacked_array.shape[0]} != {len(idxs)}"

        idxs = torch.tensor(idxs)
        initial_order_idxs = torch.empty_like(idxs)
        initial_order_idxs.scatter_(0, idxs, torch.arange(0, idxs.numel()))
        reordered_embs = stacked_array.index_select(0, initial_order_idxs)
        return reordered_embs

    def encode_docs_from_path_list(
        self, path_list: list[str] | list[Path], stat: DocEncodeStat | None = None
    ) -> np.ndarray:
        """This method encodes texts from a path_list into vectors and returns them.
        Texts should be presegmented, i.e. each sentence should be placed on a separate line.
        """
        embs = []
        embs_idxs = []
        # batch_iter = self._enc_module.create_batch_async_generator()

        # per_worker_items = math.ceil(len(path_list) / self.conf().async_batch_gen)
        # gens = []
        # for offs in range(0, len(path_list), per_worker_items):
        #     gens.append(TextsFromPathListGen(path_list[offs : offs + per_worker_items], offs))
        gens = create_text_gens_from_ids_list(
            path_list, 10 * self.conf().async_batch_gen, TextsFromPathListGen
        )

        # batch_iter.start_workers_for_item_list(path_list, fetcher=file_path_fetcher)
        # batch_iter.start_workers(gens)
        for input_data in self._batch_gen.batches(gens):
            if stat is not None:
                self._update_doc_stat(input_data, stat)
            doc_embs = self._encode_docs(input_data)
            embs.append(doc_embs.to(device='cpu'))
            embs_idxs.extend(input_data.text_ids)

        stacked = torch.vstack(embs)
        assert stacked.shape[0] == len(
            path_list
        ), f"Missaligned data with paths: {stacked.shape[0]} != {len(path_list)}"

        if self.conf().normalize_vecs:
            stacked = F.normalize(stacked, p=2, dim=1)

        reordered_embs = self._reorder_collected_arrays(stacked, embs_idxs)
        return reordered_embs.numpy()

    def encode_docs_from_dir(
        self, path: Path, stat: DocEncodeStat | None = None
    ) -> tuple[list[Path], np.ndarray]:
        """This method iterates over files in the directory and returns a tuple of paths
        and vector representation of texts.
        Texts should be presegmented, i.e. each sentence should be placed on separate line.
        """
        paths = list(path.iterdir())
        paths.sort()
        return paths, self.encode_docs_from_path_list(paths, stat=stat)

    def encode_docs(
        self, docs: list[list[str] | str], stat: DocEncodeStat | None = None
    ) -> np.ndarray:
        """Encode a list of documents into vector representation. A doc is eiher
        a list of sentences or a text where each sentence is placed on a new
        line."""

        def dummy_fetcher():
            yield from enumerate(docs)

        embs = []

        batch_generator = self._enc_module.create_batch_generator()
        for input_data in batch_generator.batches(dummy_fetcher):
            if stat is not None:
                self._update_doc_stat(input_data, stat)
            doc_embs = self._encode_docs(input_data)
            embs.append(doc_embs.to(device='cpu'))
        stacked = torch.vstack(embs)
        assert len(stacked) == len(docs)

        if self.conf().normalize_vecs:
            stacked = F.normalize(stacked, p=2, dim=1)

        return stacked.numpy()

    # TODO doc string
    def encode_docs_from_generators(
        self, generator_funcs: list[TextGenFuncT], stat: DocEncodeStat | None = None
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

        # batch_async_gen = self._enc_module.create_batch_async_generator()
        # batch_async_gen.start_workers()
        for input_data in self._batch_gen.batches(generator_funcs):
            if stat is not None:
                self._update_doc_stat(input_data, stat)
            doc_embs = self._encode_docs(input_data)
            if self.conf().normalize_vecs:
                doc_embs = F.normalize(doc_embs, p=2, dim=1)
            yield input_data.text_ids, doc_embs.to(device='cpu').numpy()
