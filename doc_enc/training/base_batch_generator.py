#!/usr/bin/env python3


from ast import TypeVar
from dataclasses import dataclass
import multiprocessing
import math
import random
import logging
from typing import Generator

import torch
from hydra.core.utils import configure_log

from doc_enc.encoders.enc_in import EncoderInData, EncoderInputType
from doc_enc.inter_proc_utils import deserialize_training_data, serialize_training_data
from doc_enc.shared_tensors import EncInputSharedTensors
from doc_enc.utils import file_line_cnt


def _split_between_nproc(n, start_offs, line_cnt):
    per_rank = math.ceil(line_cnt / n)
    return range(start_offs, start_offs + line_cnt, per_rank)


def _generator_proc_wrapper(
    queue: multiprocessing.Queue,
    shared_tensors_holder: EncInputSharedTensors,
    logging_conf,
    is_master,
    GenCls,
    *args,
    seed=None,
    **kwargs,
):
    if seed is None:
        seed = 42 * 42 + 51
    random.seed(seed)
    torch.set_num_threads(1)

    if logging_conf:
        configure_log(*logging_conf)
        if not is_master:
            logging.getLogger().setLevel(logging.WARNING)

    gen_stat = None
    try:
        generator = GenCls(*args, **kwargs)
        for b in generator.batches():
            d = serialize_training_data(b.src_data, b.tgt_data, b.labels, shared_tensors_holder)
            queue.put(d)
        gen_stat = generator.get_stat()
    except Exception as e:
        logging.exception(
            "Failed to process batches: GenCls=%s; Args=%s; kwargs=%s : %s", GenCls, args, kwargs, e
        )

    queue.put((None, gen_stat))


def skip_to_line(fp, line_offset):
    i = 0
    line = ''
    while i < line_offset:
        line = fp.readline()
        i += 1
    if line_offset and not line:
        raise RuntimeError("Unexpected end of file!")


@dataclass
class BaseBatchAsyncGeneratorConf:
    async_generators: int = 1


BatchT = TypeVar('BatchT')


class BaseBatchAsyncGenerator[BatchT]:
    _gen_cls = None
    _name = ''

    def __init__(
        self,
        enc_input_type: EncoderInputType,
        opts: BaseBatchAsyncGeneratorConf,
        max_tokens: int,
        max_seqs: int,
        logging_conf,
        other_generator_args=(),
        rank=0,
        world_size=-1,
    ):
        self._opts = opts
        self._logging_conf = logging_conf
        self._generator_args = (enc_input_type,) + other_generator_args

        self._rank = rank
        self._world_size = world_size

        self._nworkers = opts.async_generators
        self._cap_m = 3
        self._out_queues = [multiprocessing.Queue(self._cap_m) for _ in range(self._nworkers)]
        self._shared_tensors_holders = [
            EncInputSharedTensors(
                enc_input_type, max_tokens, max_seqs, self._cap_m, is_training=True
            )
            for _ in range(self._nworkers)
        ]

        self._processes = []

    def destroy(self):
        self._terminate_workers()
        for out_q in self._out_queues:
            out_q.close()
            out_q.cancel_join_thread()

    def init_epoch(self, epoch):
        raise NotImplementedError("Impl init_epoch")

    def end_epoch(self):
        self.destroy()
        for shared_t in self._shared_tensors_holders:
            shared_t.reset()

        self._out_queues = [multiprocessing.Queue(self._cap_m) for _ in range(self._nworkers)]
        logging.info("%s: end epoch done", self._name)

    def _get_line_offs_for_rank(self, filepath, limit):
        line_cnt = file_line_cnt(filepath, limit)

        if not line_cnt or self._world_size == -1:
            return 0, line_cnt

        r = _split_between_nproc(self._world_size, start_offs=0, line_cnt=line_cnt)
        return r[self._rank], r.step

    def _terminate_workers(self):
        for p in self._processes:
            p.terminate()
            p.join()
        self._processes = []

    def _start_workers(self, filepath, seed=None, limit=0):
        rank_offs, per_rank_lines = self._get_line_offs_for_rank(filepath, limit)
        if not per_rank_lines:
            return False
        logging.info(
            "%s split for rank=%d: offs=%d; lines=%d",
            self._name,
            self._rank,
            rank_offs,
            per_rank_lines,
        )
        r = _split_between_nproc(
            self._opts.async_generators, start_offs=rank_offs, line_cnt=per_rank_lines
        )
        per_proc_lines = r.step
        for i, offs in enumerate(r):

            logging.error(
                "create proc with offs %s, line cnt %s, limit %s", offs, per_proc_lines, limit
            )

            p = multiprocessing.Process(
                target=_generator_proc_wrapper,
                args=(
                    self._out_queues[i],
                    self._shared_tensors_holders[i],
                    self._logging_conf,
                    self._rank == 0,
                    self._gen_cls,
                )
                + self._generator_args,
                kwargs={
                    'line_offset': offs,
                    'line_cnt': per_proc_lines,
                    'limit': limit,
                    'seed': seed,
                },
            )
            p.start()

            self._processes.append(p)
        return bool(self._processes)

    def batches(self) -> Generator[BatchT, None, None]:
        if not self._processes:
            raise RuntimeError("Base batch Iterator is not initialized!")

        nworkers = len(self._processes)
        last_q_idx = 0
        finished = [False] * nworkers
        nfinished = 0
        stats = []
        while nfinished < nworkers:
            out_q = self._out_queues[last_q_idx]
            shared_t = self._shared_tensors_holders[last_q_idx]
            batch = out_q.get()

            match batch:
                case (None, _stat):
                    finished[last_q_idx] = True
                    nfinished += 1
                    if _stat is not None:
                        stats.append(_stat)
                case tuple():
                    with deserialize_training_data(batch, shared_t) as b:
                        yield self._prepare_batch(*b)

        if stats:
            if len(stats) > 1:
                total_stat = sum(stats[1:], start=stats[0])
            else:
                total_stat = stats[0]
            logging.info("%s: stat:\n%s", self._name, total_stat)

        for p in self._processes:
            p.join()
        self._processes = []

    def _prepare_batch(self, src_in: EncoderInData, tgt_in: EncoderInData, labels: torch.Tensor):
        raise NotImplementedError("Implement _prepare_batch")
