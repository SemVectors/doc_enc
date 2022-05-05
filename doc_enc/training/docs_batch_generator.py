#!/usr/bin/env python3

import itertools
import logging
import os.path
from pathlib import Path
from enum import Enum

import random
from typing import List, Optional
import dataclasses

from omegaconf import MISSING
import torch

from doc_enc.tokenizer import TokenizerConf, create_tokenizer
from doc_enc.training.base_batch_generator import (
    BaseBatchIterator,
    BaseBatchIteratorConf,
    skip_to_line,
    find_file,
    open_file,
)
from doc_enc.training.types import DocsBatch


@dataclasses.dataclass
class DocsBatchGeneratorConf:
    input_dir: str
    meta_prefix: str = "combined"

    batch_src_sents_cnt: int = 512
    batch_total_sents_cnt: int = 1296
    max_sents_cnt_delta: int = 64
    batch_size: int = 96

    positives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 2])
    negatives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [2, 4])

    fragment_size: int = 24

    max_sent_size: int = 128
    min_sent_size: int = 4
    max_sents_per_doc: int = 1024
    min_sents_per_doc: int = 5
    min_tgt_docs_per_src_doc: int = 1
    allow_docs_without_positives: bool = False


EXMPL_DATASET = 0
EXMPL_SRC_ID = 1
EXMPL_TGT_ID = 2
EXMPL_LABEL = 3
EXMPL_SRC_LEN = 4
EXMPL_TGT_LEN = 5
EXMPL_SRC_HASH = 6
EXMPL_TGT_HASH = 7


class _ProcSrcStatus(Enum):
    ADDED = 1
    CANT_FIT_IN_BATCH = 2
    NON_VALID_SRC = 3


class DocsBatchGenerator:
    def __init__(
        self,
        opts: DocsBatchGeneratorConf,
        tok_conf: TokenizerConf,
        split,
        line_offset=0,
        line_cnt=-1,
    ):
        self._opts = opts

        self._tokenizer = create_tokenizer(tok_conf)
        self._doc_pair_metas = self._load_metas(split, line_offset, line_cnt)

        self._text_dirs_dict = {}
        for p in Path(self._opts.input_dir).iterdir():
            if not p.is_dir():
                continue
            if (p / "texts").is_dir():
                self._text_dirs_dict[p.name] = ("texts", "texts")
            elif (p / "texts_1").is_dir() and (p / "texts_2").is_dir():
                self._text_dirs_dict[p.name] = ("texts_1", "texts_2")

    def _load_metas(self, split, line_offset, line_cnt):
        all_metas = []
        fp = f"{self._opts.input_dir}/{self._opts.meta_prefix}_{split}.csv"

        with open(fp, 'r', encoding='utf8') as fp:
            skip_to_line(fp, line_offset)
            for i, l in enumerate(fp):
                if i == line_cnt:
                    break
                row = l.rstrip().split(',')
                ds, src, tgt, label, slen, tlen, shash, thash = row
                t = (ds, int(src), int(tgt), int(label), int(slen), int(tlen), shash, thash)
                all_metas.append(t)

        all_metas.sort(
            key=lambda t: (-t[EXMPL_SRC_LEN], t[EXMPL_SRC_HASH], -t[EXMPL_LABEL], t[EXMPL_TGT_LEN])
        )
        return all_metas

    def _select_targets(self, targets, min_max_list):
        if not targets:
            return []

        a, b = min_max_list
        n = random.randrange(a, b + 1)

        if n >= len(targets):
            return targets
        return random.sample(targets, n)

    def _tokenize_doc(self, path):
        with open_file(path) as f:
            sents = []
            for l in f:
                if not l.strip():
                    continue

                tokens = self._tokenizer(l.rstrip())
                if len(tokens) >= self._opts.min_sent_size:
                    tokens = tokens[: self._opts.max_sent_size]
                    sents.append(tokens)
            return sents

    def _split_on_fragments(self, sents: List, fragment_len_list: List):
        l = len(sents)

        fragments_cnt = 0
        for offs in range(0, l, self._opts.fragment_size):
            cnt = min(l - offs, self._opts.fragment_size)
            fragment_len_list.append(cnt)
            fragments_cnt += 1
        return fragments_cnt

    def _populate_doc_len(
        self, sents: List, fragments_cnt, doc_len_in_sents_list, doc_len_in_frags_list: List
    ):
        doc_no = len(doc_len_in_sents_list)
        doc_len_in_sents_list.append(len(sents))
        doc_len_in_frags_list.append(fragments_cnt)
        return doc_no

    def _prepare_all_targets(
        self, positive_targets, negative_targets, tgt_hashes, batch_dups, batch
    ):
        positive_idxs = []
        for tgt_path, tgt_id, _, tgt_hash, lbl in itertools.chain(
            (t + (1,) for t in positive_targets),
            (t + (0,) for t in negative_targets),
        ):

            if tgt_hash in tgt_hashes:
                if lbl == 1:
                    positive_idxs.append(tgt_hashes[tgt_hash])
                continue

            if lbl == 0 and not self._opts.allow_docs_without_positives and not positive_idxs:
                return positive_idxs

            tgt_sents = self._tokenize_doc(tgt_path)
            if (
                not tgt_sents
                or len(tgt_sents) < self._opts.min_sents_per_doc
                or len(tgt_sents) >= self._opts.max_sents_per_doc
            ):
                continue

            batch.tgt_ids.append(tgt_id)
            batch.tgt_sents.extend(tgt_sents)
            fragments_cnt = self._split_on_fragments(tgt_sents, batch.tgt_fragment_len)
            tgt_no = self._populate_doc_len(
                tgt_sents, fragments_cnt, batch.tgt_doc_len_in_sents, batch.tgt_doc_len_in_frags
            )
            tgt_hashes[tgt_hash] = tgt_no
            if lbl == 1:
                positive_idxs.append(tgt_no)

            if tgt_hash in batch_dups:
                # this tgt is a positive example for some document that is already in the batch
                # we need to adjust positive_idxs of this document accordingly
                batch.positive_idxs[batch_dups[tgt_hash]].append(tgt_no)
        return positive_idxs

    def _process_src_doc(
        self,
        src_info,
        all_positive_targets,
        all_negative_targets,
        batch: DocsBatch,
        tgt_hashes: dict,
        batch_dups: dict,
    ) -> _ProcSrcStatus:

        if not all_positive_targets:
            if not self._opts.allow_docs_without_positives:
                return _ProcSrcStatus.NON_VALID_SRC
            if not all_negative_targets:
                return _ProcSrcStatus.NON_VALID_SRC

        src_path, src_id, src_len = src_info

        if not os.path.exists(src_path):
            logging.warning("src text is missing: %s", src_path)
            return _ProcSrcStatus.NON_VALID_SRC

        positive_targets = self._select_targets(all_positive_targets, self._opts.positives_per_doc)
        negative_targets = self._select_targets(all_negative_targets, self._opts.negatives_per_doc)

        tgt_extra_len = sum(t[2] for t in itertools.chain(positive_targets, negative_targets))
        if self._is_batch_ready(batch, src_extra=src_len, tgt_extra=tgt_extra_len):
            return _ProcSrcStatus.CANT_FIT_IN_BATCH

        src_sents = self._tokenize_doc(src_path)
        if (
            not src_sents
            or len(src_sents) < self._opts.min_sents_per_doc
            or len(src_sents) >= self._opts.max_sents_per_doc
        ):
            return _ProcSrcStatus.NON_VALID_SRC

        positive_idxs = self._prepare_all_targets(
            positive_targets, negative_targets, tgt_hashes, batch_dups, batch
        )
        if not positive_idxs and not self._opts.allow_docs_without_positives:
            return _ProcSrcStatus.NON_VALID_SRC

        batch.positive_idxs.append(positive_idxs)
        batch.src_ids.append(src_id)
        batch.src_sents.extend(src_sents)
        fragments_cnt = self._split_on_fragments(src_sents, batch.src_fragment_len)
        self._populate_doc_len(
            src_sents, fragments_cnt, batch.src_doc_len_in_sents, batch.src_doc_len_in_frags
        )

        src_no = len(batch.src_ids) - 1
        selected_hashes = [t[-1] for t in positive_targets]
        for _, _, _, h in all_positive_targets:
            if h not in selected_hashes:
                batch_dups[h] = src_no

        return _ProcSrcStatus.ADDED

    def _empty_batch(self):
        iterable = [[] for _ in range(13)]
        iterable.append(
            {
                'src_docs_cnt': 0,
                'tgt_docs_cnt': 0,
                'src_frags_cnt': 0,
                'tgt_frags_cnt': 0,
                'max_positives_per_doc': 0,
            }
        )
        return DocsBatch._make(iterable), {}, {}

    def _finalize_batch(self, batch: DocsBatch):
        for l in batch.positive_idxs:
            l.sort()
        src_sz = len(batch.src_doc_len_in_sents)
        batch.info['bs'] = src_sz
        batch.info['src_docs_cnt'] = src_sz
        batch.info['tgt_docs_cnt'] = len(batch.tgt_doc_len_in_sents)
        batch.info['src_frags_cnt'] = len(batch.src_fragment_len)
        batch.info['tgt_frags_cnt'] = len(batch.tgt_fragment_len)

        batch.info['max_positives_per_doc'] = len(max(batch.positive_idxs, key=len))

    def _is_defect_batch(self, batch):
        return (
            batch.info['bs'] == 1
            and batch.info['tgt_docs_cnt'] < self._opts.min_tgt_docs_per_src_doc
        )

    def _is_batch_ready(self, batch: DocsBatch, src_extra=0, tgt_extra=0):
        src_sz = len(batch.src_sents)
        d = self._opts.max_sents_cnt_delta if src_extra or tgt_extra else 0
        return (
            src_sz + src_extra > self._opts.batch_src_sents_cnt + d
            or src_sz + src_extra + len(batch.tgt_sents) + tgt_extra
            > self._opts.batch_total_sents_cnt + d
            or len(batch.src_ids) > self._opts.batch_size
        )

    def batches(self):
        positive_targets = []
        negative_targets = []
        cur_hash = ''
        src_info = tuple()

        batch, tgt_hashes, batch_dups = self._empty_batch()

        for metas in self._doc_pair_metas:
            src_texts, tgt_texts = self._text_dirs_dict[metas[EXMPL_DATASET]]
            if cur_hash != metas[EXMPL_SRC_HASH]:
                status = self._process_src_doc(
                    src_info,
                    positive_targets,
                    negative_targets,
                    batch,
                    tgt_hashes,
                    batch_dups,
                )

                if (
                    status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_sents
                ) or self._is_batch_ready(batch):
                    self._finalize_batch(batch)
                    if not self._is_defect_batch(batch):
                        yield batch
                    batch, tgt_hashes, batch_dups = self._empty_batch()

                if status == _ProcSrcStatus.CANT_FIT_IN_BATCH:
                    self._process_src_doc(
                        src_info,
                        positive_targets,
                        negative_targets,
                        batch,
                        tgt_hashes,
                        batch_dups,
                    )

                # preparations for new src doc
                positive_targets = []
                negative_targets = []
                cur_hash = metas[EXMPL_SRC_HASH]

                src_id = metas[EXMPL_SRC_ID]
                src_len = metas[EXMPL_SRC_LEN]
                src_path = find_file(
                    f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/{src_texts}/{metas[EXMPL_SRC_ID]}.txt",
                    throw_if_not_exist=False,
                )
                src_info = (src_path, src_id, src_len)

            tgt_id = metas[EXMPL_TGT_ID]
            tgt_path = find_file(
                f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/{tgt_texts}/{metas[EXMPL_TGT_ID]}.txt",
                throw_if_not_exist=False,
            )
            if not os.path.exists(tgt_path):
                logging.warning("tgt text is missing: %s", tgt_path)
                continue

            tgt_len = metas[EXMPL_TGT_LEN]
            tgt_info = (tgt_path, tgt_id, tgt_len, metas[EXMPL_TGT_HASH])

            label = metas[EXMPL_LABEL]
            if label == 1:
                positive_targets.append(tgt_info)
            else:
                negative_targets.append(tgt_info)

        status = self._process_src_doc(
            src_info, positive_targets, negative_targets, batch, tgt_hashes, batch_dups
        )
        if status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_sents:
            self._finalize_batch(batch)
            if not self._is_defect_batch(batch):
                yield batch
            batch, tgt_hashes, batch_dups = self._empty_batch()
            self._process_src_doc(
                src_info, positive_targets, negative_targets, batch, tgt_hashes, batch_dups
            )

        if batch.src_sents:
            self._finalize_batch(batch)
            if not self._is_defect_batch(batch):
                yield batch
        return


@dataclasses.dataclass
class DocsBatchIteratorConf(BaseBatchIteratorConf):
    batch_generator_conf: DocsBatchGeneratorConf = MISSING

    use_existing_combined_meta: bool = False
    include_datasets: Optional[List[str]] = None
    exclude_datasets: Optional[List[str]] = None


class DocsBatchIterator(BaseBatchIterator):
    def __init__(
        self,
        opts: DocsBatchIteratorConf,
        tok_conf: TokenizerConf,
        logging_conf,
        split,
        rank=0,
        world_size=-1,
        pad_idx=0,
    ):

        super().__init__(
            opts,
            logging_conf,
            DocsBatchGenerator,
            (opts.batch_generator_conf, tok_conf, split),
            rank=rank,
            world_size=world_size,
        )

        self._opts = opts
        self._split = split

        if torch.cuda.is_available():
            self._device = torch.device(f'cuda:{rank}')
        else:
            self._device = torch.device('cpu')

        self._pad_idx = pad_idx
        self._epoch = 0

    def init_epoch(self, epoch):
        self._epoch = epoch - 1
        opts = self._opts.batch_generator_conf
        fp = f"{opts.input_dir}/{opts.meta_prefix}_{self._split}.csv"
        self._start_workers(fp)

    def _make_batch_for_retr_task(self, batch: DocsBatch):

        src_lengths = torch.as_tensor(
            [len(t) for t in batch.src_sents], dtype=torch.int64, device=self._device
        )

        tgt_lengths = torch.as_tensor(
            [len(t) for t in batch.tgt_sents], dtype=torch.int64, device=self._device
        )

        src_cnt = batch.info['src_docs_cnt']
        labels = torch.full(
            (src_cnt, batch.info['tgt_docs_cnt']), 0.0, dtype=torch.float32, device=self._device
        )
        for i in range(src_cnt):
            positive_tgts = batch.positive_idxs[i]
            if positive_tgts:
                labels[i][positive_tgts] = 1.0

        b = batch._replace(src_sent_len=src_lengths, tgt_sent_len=tgt_lengths)
        return b, labels

    def _prepare_batch(self, batch):
        return self._make_batch_for_retr_task(batch)
