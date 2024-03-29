#!/usr/bin/env python3

import dataclasses
import logging
import time
from pathlib import Path

import torch

from doc_enc.doc_encoder import DocEncoder, SentEncodeStat, DocEncodeStat
from doc_enc.eval.caching_doc_encoder import CachingDocEncoder


@dataclasses.dataclass
class DocDatasetConf:
    name: str
    texts: str | None = None
    paths_file: str | None = None
    repeat_times: int = 3


@dataclasses.dataclass
class SentDatasetConf:
    name: str
    sents_file: str
    repeat_times: int = 3
    first_column_is_id: bool = True


@dataclasses.dataclass
class BenchConf:
    doc_ds_base_dir: str = ''
    doc_datasets: list[DocDatasetConf] = dataclasses.field(default_factory=list)
    sent_ds_base_dir: str = ''
    sent_datasets: list[SentDatasetConf] = dataclasses.field(default_factory=list)
    bench_sents_encoding: bool = False

    keep_full_stats: bool = False


class StatsRecorder:
    def __init__(self, device) -> None:
        self.device = device
        self.stats = {
            'runs': 0,
            'min_peak_alloc_mem': float('inf'),
            'max_peak_alloc_mem': 0,
            'mean_peak_alloc_mem': 0,
            'min_peak_cached_mem': float('inf'),
            'max_peak_cached_mem': 0,
            'mean_peak_cached_mem': 0,
            'min_runtime': float('inf'),
            'max_runtime': 0,
            'mean_runtime': 0,
        }
        self._start_time = 0
        self._texts_cnt = 0

    def set_texts_cnt(self, cnt):
        self._texts_cnt = cnt

    def set_extra_stat(self, **kwargs):
        self.stats.update(kwargs)

    def __enter__(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time = time.time()

    def __exit__(self, *args, **kwargs):
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_cached = torch.cuda.max_memory_reserved(self.device)
        runtime = time.time() - self._start_time

        self.stats['runs'] += 1

        self.stats['min_peak_alloc_mem'] = min(peak_alloc, self.stats['min_peak_alloc_mem'])
        self.stats['max_peak_alloc_mem'] = max(peak_alloc, self.stats['max_peak_alloc_mem'])
        self.stats['mean_peak_alloc_mem'] += peak_alloc

        self.stats['min_peak_cached_mem'] = min(peak_cached, self.stats['min_peak_cached_mem'])
        self.stats['max_peak_cached_mem'] = max(peak_cached, self.stats['max_peak_cached_mem'])
        self.stats['mean_peak_cached_mem'] += peak_cached

        self.stats['min_runtime'] = min(runtime, self.stats['min_runtime'])
        self.stats['max_runtime'] = max(runtime, self.stats['max_runtime'])
        self.stats['mean_runtime'] += runtime

    def finalize_stats(self, keep_full_stats: bool = False):
        if not keep_full_stats:
            keys = list(self.stats.keys())
            for k in keys:
                if k[:3] in ('min', 'max'):
                    del self.stats[k]
        for k in self.stats:
            if 'peak' in k:
                self.stats[k] /= 1024 * 1024

        if not self.stats['runs']:
            return self.stats

        n = self.stats['runs']
        self.stats['mean_peak_alloc_mem'] /= n
        self.stats['mean_peak_cached_mem'] /= n
        self.stats['mean_runtime'] /= n

        if self._texts_cnt:
            self.stats['texts_per_second'] = self._texts_cnt / self.stats['mean_runtime']

        return self.stats


def _run_bench_on_sents_file(
    base_dir, conf: SentDatasetConf, doc_encoder: DocEncoder, stats_recorder: StatsRecorder
):
    path = base_dir + '/' + conf.sents_file

    for _ in range(conf.repeat_times):
        stat = SentEncodeStat()
        gen = doc_encoder.generate_sent_embs_from_file(
            path, first_column_is_id=conf.first_column_is_id, stat=stat
        )
        with stats_recorder:
            for _ in gen:
                pass
            stats_recorder.set_texts_cnt(stat.sents_cnt)
            stats_recorder.set_extra_stat(avg_sent_len=stat.total_tokens_cnt / stat.sents_cnt)


def _check_sent_ds(conf: BenchConf, dsconf: SentDatasetConf):
    base_dir = Path(conf.sent_ds_base_dir)
    if not (base_dir / dsconf.sents_file).exists():
        logging.warning("%s does not exist. Skip this dataset", base_dir / dsconf.sents_file)
        return False
    return True


def bench_sents_encoding(conf: BenchConf, doc_encoder: DocEncoder):
    if not doc_encoder.sent_encoding_supported():
        logging.warning("Sent encoding is not supported by this model! Skip benching sent encoding")
        return []

    results = []
    for dsconf in conf.sent_datasets:
        if not _check_sent_ds(conf, dsconf):
            continue
        stats_recorder = StatsRecorder(doc_encoder.enc_module().device)
        _run_bench_on_sents_file(conf.sent_ds_base_dir, dsconf, doc_encoder, stats_recorder)
        results.append((dsconf.name, stats_recorder.finalize_stats(conf.keep_full_stats)))

    return results


def _run_bench_on_docs(
    base_dir, conf: DocDatasetConf, doc_encoder: DocEncoder, stats_recorder: StatsRecorder
):
    base_path = Path(base_dir)
    paths = []
    if conf.paths_file is not None:
        pathfile = base_path / conf.paths_file
        with open(pathfile, 'r', encoding='utf8') as inpf:
            for p in inpf:
                p = Path(p.strip())
                if not p.is_absolute():
                    p = pathfile.parent / p
                paths.append(p)
    else:
        text_dir = base_dir / conf.texts
        paths = list(text_dir.iterdir())
        paths.sort()

    kwargs = {}
    if isinstance(doc_encoder, CachingDocEncoder):
        kwargs['dont_cache'] = True

    for _ in range(conf.repeat_times):
        stat = DocEncodeStat()
        with stats_recorder:
            doc_encoder.encode_docs_from_path_list(paths, stat=stat, **kwargs)

            stats_recorder.set_texts_cnt(stat.docs_cnt)
            if stat.docs_cnt:
                stats_recorder.set_extra_stat(
                    avg_doc_len_tokens=stat.total_tokens_cnt / stat.docs_cnt,
                    avg_doc_len_sents=stat.total_sents_cnt / stat.docs_cnt,
                )


def _check_doc_ds(conf: BenchConf, dsconf: DocDatasetConf):
    base_dir = Path(conf.doc_ds_base_dir)

    if dsconf.texts is not None:
        path = dsconf.texts
    elif dsconf.paths_file is not None:
        path = dsconf.paths_file
    else:
        raise RuntimeError("Pass either texts or paths_file")

    if not (base_dir / path).exists():
        logging.warning("%s does not exist. Skip this dataset", base_dir / path)
        return False
    return True


def bench_docs_encoding(conf: BenchConf, doc_encoder: DocEncoder):
    results = []
    for dsconf in conf.doc_datasets:
        if not _check_doc_ds(conf, dsconf):
            continue
        stats_recorder = StatsRecorder(doc_encoder.enc_module().device)
        _run_bench_on_docs(conf.doc_ds_base_dir, dsconf, doc_encoder, stats_recorder)
        results.append((dsconf.name, stats_recorder.finalize_stats(conf.keep_full_stats)))

    return results
