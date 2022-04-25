#!/usr/bin/env python3

from dataclasses import dataclass
import multiprocessing
import math
import logging


def _calc_line_cnt(fp):
    with open(fp, 'rb') as f:
        i = -1
        for i, _ in enumerate(f):
            pass
    return i + 1


def _split_between_nproc(n, start_offs, line_cnt):
    per_rank = math.ceil(line_cnt / n)
    return range(start_offs, start_offs + line_cnt, per_rank)


def _generator_proc_wrapper(queue: multiprocessing.Queue, GenCls, *args, **kwargs):
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


@dataclass
class BaseBatchIteratorConf:
    async_generators: int = 1


class BaseBatchIterator:
    def __init__(
        self,
        opts: BaseBatchIteratorConf,
        generator_cls=None,
        generator_args=(),
        rank=0,
        world_size=-1,
    ):
        self._opts = opts
        self._generator_cls = generator_cls
        self._generator_args = generator_args

        self._rank = rank
        self._world_size = world_size

        self._processes = []
        self._queue = multiprocessing.Queue(4 * self._opts.async_generators)

    def destroy(self):
        assert not self._processes
        self._queue.close()

    def _get_line_offs_for_rank(self, filepath):
        line_cnt = _calc_line_cnt(filepath)

        if self._world_size == -1:
            return 0, line_cnt

        r = _split_between_nproc(self._world_size, start_offs=0, line_cnt=line_cnt)
        return r[self._rank], r.step

    def _start_workers(self, filepath):
        rank_offs, per_rank_lines = self._get_line_offs_for_rank(filepath)
        r = _split_between_nproc(
            self._opts.async_generators, start_offs=rank_offs, line_cnt=per_rank_lines
        )
        per_proc_lines = r.step
        for offs in r:
            p = multiprocessing.Process(
                target=_generator_proc_wrapper,
                args=(
                    self._queue,
                    self._generator_cls,
                )
                + self._generator_args,
                kwargs={'line_offset': offs, 'line_cnt': per_proc_lines},
            )
            p.start()

            self._processes.append(p)

    def batches(self):
        if not self._processes:
            raise RuntimeError("Sent batch Iterator is not initialized!")

        finished_processes = 0
        while finished_processes < self._opts.async_generators:
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
