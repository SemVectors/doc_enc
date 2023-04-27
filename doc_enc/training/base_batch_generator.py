#!/usr/bin/env python3


from dataclasses import dataclass
import multiprocessing
import math
import random
import logging

import torch
from hydra.core.utils import configure_log

from doc_enc.utils import calc_line_cnt


def _split_between_nproc(n, start_offs, line_cnt):
    per_rank = math.ceil(line_cnt / n)
    return range(start_offs, start_offs + line_cnt, per_rank)


def _generator_proc_wrapper(
    queue: multiprocessing.Queue, logging_conf, is_master, GenCls, *args, seed=None, **kwargs
):
    if seed is None:
        seed = 42 * 42 + 51
    random.seed(seed)

    if logging_conf:
        configure_log(*logging_conf)
        if not is_master:
            logging.getLogger().setLevel(logging.WARNING)
    try:
        generator = GenCls(*args, **kwargs)
        for b in generator.batches():
            queue.put(b)
    except Exception as e:
        logging.error(
            "Failed to process batches: GenCls=%s; Args=%s; kwargs=%s : %s", GenCls, args, kwargs, e
        )

    queue.put(None)


def skip_to_line(fp, line_offset):
    i = 0
    l = ''
    while i < line_offset:
        l = fp.readline()
        i += 1
    if line_offset and not l:
        raise RuntimeError("Unexpected end of file!")


def create_padded_tensor(tokens, max_len, pad_idx, device, pad_to_multiple_of=0):
    bs = len(tokens)

    if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    batch = torch.full((bs, max_len), pad_idx, dtype=torch.int32)
    for i in range(bs):
        batch[i, 0 : len(tokens[i])] = torch.as_tensor(tokens[i])

    batch = batch.to(device=device)
    lengths = torch.as_tensor([len(t) for t in tokens], dtype=torch.int64, device=device)
    return batch, lengths


@dataclass
class BaseBatchIteratorConf:
    async_generators: int = 1


class BaseBatchIterator:
    def __init__(
        self,
        name: str,
        opts: BaseBatchIteratorConf,
        logging_conf,
        generator_cls=None,
        generator_args=(),
        queue_size=0,
        rank=0,
        world_size=-1,
    ):
        self._name = name
        self._opts = opts
        self._logging_conf = logging_conf
        self._generator_cls = generator_cls
        self._generator_args = generator_args

        self._rank = rank
        self._world_size = world_size

        self._processes = []
        if queue_size == 0:
            queue_size = 10 * opts.async_generators
        self._queue = multiprocessing.Queue(queue_size)

    def destroy(self):
        self._terminate_workers()
        self._queue.close()

    def init_epoch(self, epoch):
        raise NotImplementedError("Impl init_epoch")

    def end_epoch(self):
        self.destroy()
        self._queue = multiprocessing.Queue(4 * self._opts.async_generators)

    def _get_line_offs_for_rank(self, filepath, limit):
        line_cnt = calc_line_cnt(filepath, limit)

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
        for offs in r:
            p = multiprocessing.Process(
                target=_generator_proc_wrapper,
                args=(
                    self._queue,
                    self._logging_conf,
                    self._rank == 0,
                    self._generator_cls,
                )
                + self._generator_args,
                kwargs={'line_offset': offs, 'line_cnt': per_proc_lines, 'seed': seed},
            )
            p.start()

            self._processes.append(p)
        return bool(self._processes)

    def batches(self):
        if not self._processes:
            raise RuntimeError("Base batch Iterator is not initialized!")

        finished_processes = 0
        while finished_processes < self._opts.async_generators:
            logging.debug("%s queue len: %s", self._name, self._queue.qsize())
            batch = self._queue.get()
            if batch is None:
                finished_processes += 1
                continue
            yield self._prepare_batch(batch)

        for p in self._processes:
            p.join()
        self._processes = []

    def _prepare_batch(self, batch):
        raise NotImplementedError("Implement _prepare_batch")
