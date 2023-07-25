#!/usr/bin/env python3

from typing import Optional, MutableMapping, List
import logging
import collections
from pathlib import Path

import numpy as np

from doc_enc.doc_encoder import DocEncoderConf, DocEncoder
from doc_enc.eval.eval_utils import paths_to_ids


class _Cache:
    def __init__(self, name, ids=None, embs=None) -> None:
        self._name = name
        self._ids: List = ids if ids is not None else []
        self._inv_idx: MutableMapping[str, int] = {}
        self._embs: Optional[np.ndarray] = embs

        if self._ids:
            self._make_inv_idx()

    def _make_inv_idx(self):
        self._inv_idx = {i: n for n, i in enumerate(self._ids)}

    def key(self):
        return self._name

    def empty(self):
        return not self._ids

    def cached_idx(self, path: Path) -> int:
        if not self._ids:
            return -1
        while path.suffix:
            path = path.with_suffix('')
        doc_id = path.name
        return self._inv_idx.get(doc_id, -1)

    def get_ids(self):
        return self._ids

    def embs_dim(self):
        if self._embs is None:
            return 0
        return self._embs.shape[1]

    def get_embs(self):
        return self._embs

    def add_embs(self, paths, embs: np.ndarray):
        if self._embs is None:
            self._embs = embs
            self._ids = paths_to_ids(paths)
            self._make_inv_idx()
        else:
            new_ids = paths_to_ids(paths)
            offs = len(self._ids)
            for n, i in enumerate(new_ids):
                self._inv_idx[i] = offs + n
            self._ids.extend(new_ids)
            self._embs = np.vstack((self._embs, embs))

    def __len__(self):
        return len(self._ids)


class CachingDocEncoder(DocEncoder):
    def __init__(self, conf: DocEncoderConf, model_name: str) -> None:
        super().__init__(conf)

        self._caches = {}

        self._cache_dir = f"__cached_embs__/{model_name}"

    def _load_cache_for_path(self, path: Path) -> _Cache:
        parent_dir = path.parent
        cache = self._caches.get(parent_dir)
        if cache is not None:
            return cache

        p = parent_dir / self._cache_dir
        if not p.exists():
            empty_cache = _Cache(parent_dir)
            self._caches[parent_dir] = empty_cache
            return empty_cache

        data = np.load(p / "embs.npz", allow_pickle=True)
        cache = _Cache(parent_dir, ids=data['ids'].tolist(), embs=data['embs'])
        self._caches[parent_dir] = cache
        return cache

    def _save_cache_to_disk(self, cache_key: Path):
        p = cache_key / self._cache_dir
        p.mkdir(exist_ok=True, parents=True)

        # can be used with tar: tar  --exclude-caches-all
        tag_file = cache_key / "__cached_embs__/CACHEDIR.TAG"
        if not tag_file.exists():
            with open(tag_file, 'w', encoding='ascii') as fp:
                fp.write("Signature: 8a477f597d28d172789f06886806bc55\n")

        cache = self._caches[cache_key]
        np.savez(p / 'embs.npz', embs=cache.get_embs(), ids=cache.get_ids())

    def _assign_embs(self, cached_idxs_dict, total_embs_cnt):
        if len(cached_idxs_dict) == 1:
            # try to optimize
            required_idxs, cached_idxs = next(iter(cached_idxs_dict.values()))
            cache_key = next(iter(cached_idxs_dict.keys()))
            cache = self._caches[cache_key]
            if (
                len(required_idxs) == len(cache)
                and len(required_idxs) == total_embs_cnt
                and required_idxs == cached_idxs
            ):
                logging.info("simple case: return all cached embs")
                return cache.get_embs()

        embs_arr = np.empty(
            (total_embs_cnt, self._enc_module.doc_layer.out_embs_dim()), dtype=np.float16
        )
        for cache_key, (required_idxs, cached_idxs) in cached_idxs_dict.items():
            logging.info("assign %d embs from %s", len(cached_idxs), cache_key)
            cache = self._caches[cache_key]
            embs_arr[required_idxs] = cache.get_embs()[cached_idxs]
        return embs_arr

    def _compute_embs(self, compute_idxs_dict, path_list, out_embs):
        for cache_key, idxs_to_compute in compute_idxs_dict.items():
            cache: _Cache = self._caches[cache_key]
            if len(path_list) == len(idxs_to_compute):
                logging.info("computing all paths from initial path list for %s cache", cache_key)
                this_cache_paths = path_list
            else:
                this_cache_paths = [path_list[i] for i in idxs_to_compute]
                logging.info("computing %s paths for %s cache", len(this_cache_paths), cache_key)

            embs = super().encode_docs_from_path_list(this_cache_paths)
            cache.add_embs(this_cache_paths, embs)
            self._save_cache_to_disk(cache_key)
            if id(this_cache_paths) == id(path_list):
                return embs

            out_embs[idxs_to_compute] = embs
        return out_embs

    def encode_docs_from_path_list(self, path_list):
        cached_idxs_dict = collections.defaultdict(lambda: tuple([[], []]))
        compute_idxs_dict = collections.defaultdict(list)
        for i, path in enumerate(path_list):
            cache = self._load_cache_for_path(path)
            cached_idx = cache.cached_idx(path)
            if cached_idx != -1:
                ti = cached_idxs_dict[cache.key()]
                ti[0].append(i)
                ti[1].append(cached_idx)
            else:
                compute_idxs_dict[cache.key()].append(i)

        for cache_key, info in cached_idxs_dict.items():
            logging.info("Found in %s: %d embs", cache_key, len(info[0]))
        for cache_key, info in compute_idxs_dict.items():
            logging.info("To compute for %s: %d embs", cache_key, len(info))

        if not compute_idxs_dict:
            logging.info("nothing to compute")
            return self._assign_embs(cached_idxs_dict, len(path_list))

        out_embs = self._assign_embs(cached_idxs_dict, len(path_list))
        return self._compute_embs(compute_idxs_dict, path_list, out_embs)
