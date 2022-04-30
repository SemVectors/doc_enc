#!/usr/bin/env python3


from typing import Union, Optional
from dataclasses import dataclass
import multiprocessing
import math
import logging
from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path

from hydra.core.utils import configure_log


def _is_gzipped(fp):
    if isinstance(fp, Path):
        n = fp.name
    elif isinstance(fp, str):
        n = fp
    else:
        raise RuntimeError("logic error 82093")

    return n.endswith('.gz')


def open_bin_file(fp: Union[Path, str]):
    if _is_gzipped(fp):
        return GzipFile(fp, mode='rb')
    return open(fp, mode='rb')


def open_file(fp: Union[Path, str]):
    if _is_gzipped(fp):
        return TextIOWrapper(GzipFile(fp, 'rb'), encoding='utf8')
    return open(fp, 'rt', encoding='utf8')


def find_file(fp: Union[Path, str], throw_if_not_exist=True):
    if isinstance(fp, Path):
        sp = str(fp)
    elif isinstance(fp, str):
        sp = fp
        fp = Path(fp)
    else:
        raise RuntimeError("logic error 82094")

    np = Path(f"{sp}.gz")
    if np.exists():
        return np
    if fp.exists():
        return fp

    if throw_if_not_exist:
        raise RuntimeError(f"Failed to find {fp}[.gz]")
    return fp


def _calc_line_cnt(fp):
    with open_bin_file(fp) as f:
        i = -1
        for i, _ in enumerate(f):
            pass
    return i + 1


def _split_between_nproc(n, start_offs, line_cnt):
    per_rank = math.ceil(line_cnt / n)
    return range(start_offs, start_offs + line_cnt, per_rank)


def _generator_proc_wrapper(
    queue: multiprocessing.Queue, logging_conf, rank, GenCls, *args, **kwargs
):
    if logging_conf:
        configure_log(logging_conf, False)
        if rank != 0:
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


@dataclass
class BaseBatchIteratorConf:
    async_generators: int = 1


class BaseBatchIterator:
    def __init__(
        self,
        opts: BaseBatchIteratorConf,
        logging_conf,
        generator_cls=None,
        generator_args=(),
        rank=0,
        world_size=-1,
    ):
        self._opts = opts
        self._logging_conf = logging_conf
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
                    self._logging_conf,
                    self._rank,
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
