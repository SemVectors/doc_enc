#!/usr/bin/env python3

import itertools
import logging
import os.path
from pathlib import Path
from enum import Enum

import random
from typing import List, Optional, Union, Mapping
import dataclasses

from omegaconf import MISSING
import torch

from doc_enc.text_processor import (
    TextProcessorConf,
    TextProcessor,
    pad_sent_sequences,
    pad_fragment_sequences,
)
from doc_enc.training.base_batch_generator import (
    BaseBatchIterator,
    BaseBatchIteratorConf,
    skip_to_line,
)
from doc_enc.utils import find_file
from doc_enc.training.types import DocsBatch


@dataclasses.dataclass
class DocsBatchGeneratorConf:
    input_dir: str
    meta_prefix: str = "combined"

    batch_docs_cnt: int = 512
    batch_total_tokens_cnt: int = 0
    batch_total_sents_cnt: int = 8192
    max_sents_cnt_delta: int = 64

    positives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 3])
    negatives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [0, 3])

    max_sents_per_doc: int = 1280
    min_sents_per_doc: int = 5
    min_tgt_docs_per_src_doc: int = 2
    allow_docs_without_positives: bool = False

    pad_src_sentences: bool = False
    pad_tgt_sentences: bool = False
    pad_fragments_level: bool = False


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
        tp_conf: TextProcessorConf,
        split,
        line_offset=0,
        line_cnt=-1,
        limit=0,
    ):
        self._opts = opts

        self._text_proc = TextProcessor(tp_conf)
        self._doc_pair_metas = self._load_metas(split, line_offset, line_cnt)
        if not self._doc_pair_metas:
            raise RuntimeError("Empty docs dataset")

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

        return all_metas

    def _select_targets(self, targets, min_max_list):
        if not targets:
            return []

        a, b = min_max_list
        n = random.randrange(a, b + 1)

        if n >= len(targets):
            return targets
        return random.sample(targets, n)

    def _populate_doc_len(
        self,
        segmented_text: list[list[int]],
        doc_segments_length: list[int],
        sent_len_list: list[int],
        frag_len_in_sents: list[int],
        doc_len_in_sents_list: list[int],
        doc_len_in_frags_list: list[int],
    ):
        if self._text_proc.conf().split_into_sents and self._text_proc.conf().split_into_fragments:
            frag_len_in_sents.extend(doc_segments_length)
            doc_len_in_frags_list.append(len(doc_segments_length))
        elif self._text_proc.conf().split_into_fragments:
            doc_len_in_frags_list.append(doc_segments_length[0])

        if self._text_proc.conf().split_into_sents:
            sent_len_list.extend(len(t) for t in segmented_text)
            doc_len_in_sents_list.append(len(segmented_text))

    def _prepare_all_targets(
        self,
        positive_targets,
        negative_targets,
        tgt_hashes,
        batch_dups,
        batch,
        tokenized_texts_cache,
    ):
        positive_idxs = []
        for _, tgt_hash, tgt_id, lbl in itertools.chain(
            (t + (1,) for t in positive_targets),
            (t + (0,) for t in negative_targets),
        ):
            if tgt_hash in tgt_hashes:
                if lbl == 1:
                    positive_idxs.append(tgt_hashes[tgt_hash])
                continue

            if tgt_hash not in tokenized_texts_cache:
                continue
            segmented_text, doc_segments_length = tokenized_texts_cache[tgt_hash]

            tgt_no = len(batch.tgt_ids)
            batch.tgt_ids.append(tgt_id)
            batch.tgt_texts.extend(segmented_text)
            batch.tgt_doc_segments_length.append(doc_segments_length)

            self._populate_doc_len(
                segmented_text,
                doc_segments_length,
                batch.tgt_sent_len,
                batch.tgt_fragment_len,
                batch.tgt_doc_len_in_sents,
                batch.tgt_doc_len_in_frags,
            )
            tgt_hashes[tgt_hash] = tgt_no
            if lbl == 1:
                positive_idxs.append(tgt_no)

            if tgt_hash in batch_dups:
                # this tgt is a positive example for some document that is already in the batch
                # we need to adjust positive_idxs of this document accordingly
                batch.positive_idxs[batch_dups[tgt_hash]].append(tgt_no)
        return positive_idxs

    def _prepare_doc_texts(self, path_gen, tokenized_texts_cache):
        for path, doc_hash, *_ in path_gen:
            if doc_hash in tokenized_texts_cache:
                continue
            if not os.path.exists(path):
                continue

            segmented_text, doc_segments_length = self._text_proc.prepare_text_from_file(path)
            if not segmented_text or (
                self._text_proc.conf().split_into_sents
                and (
                    len(segmented_text) < self._opts.min_sents_per_doc
                    or len(segmented_text) >= self._opts.max_sents_per_doc
                )
            ):
                continue

            tokenized_texts_cache[doc_hash] = (segmented_text, doc_segments_length)

    def _check_src_doc(self, src_info, positive_targets, tokenized_texts_cache):
        src_hash = src_info[1]
        if src_hash not in tokenized_texts_cache:
            return False

        if not self._opts.allow_docs_without_positives:
            for _, tgt_hash, *_ in positive_targets:
                if tgt_hash in tokenized_texts_cache:
                    break
            else:
                return False

        return True

    def _process_src_doc(
        self,
        src_info,
        all_positive_targets,
        all_negative_targets,
        batch: DocsBatch,
        tgt_hashes: dict,
        batch_dups: dict,
        tokenized_texts_cache: dict,
    ) -> _ProcSrcStatus:
        if not all_positive_targets:
            if not self._opts.allow_docs_without_positives:
                return _ProcSrcStatus.NON_VALID_SRC
            if not all_negative_targets:
                return _ProcSrcStatus.NON_VALID_SRC

        positive_targets = self._select_targets(all_positive_targets, self._opts.positives_per_doc)
        negative_targets = self._select_targets(all_negative_targets, self._opts.negatives_per_doc)

        def _gen():
            yield src_info
            yield from positive_targets
            yield from negative_targets

        self._prepare_doc_texts(_gen(), tokenized_texts_cache)
        if not self._check_src_doc(src_info, positive_targets, tokenized_texts_cache):
            return _ProcSrcStatus.NON_VALID_SRC

        def _texts_gen():
            yield tokenized_texts_cache[src_info[1]][0]
            for _, tgt_hash, *_ in itertools.chain(positive_targets, negative_targets):
                if tgt_hash not in tgt_hashes and tgt_hash in tokenized_texts_cache:
                    yield tokenized_texts_cache[tgt_hash][0]

        if self._is_batch_ready(batch, new_docs=list(_texts_gen())):
            return _ProcSrcStatus.CANT_FIT_IN_BATCH

        positive_idxs = self._prepare_all_targets(
            positive_targets, negative_targets, tgt_hashes, batch_dups, batch, tokenized_texts_cache
        )

        src_hash = src_info[1]
        segmented_text, doc_segments_length = tokenized_texts_cache[src_hash]

        batch.positive_idxs.append(positive_idxs)
        batch.src_ids.append(src_info[2])
        batch.src_texts.extend(segmented_text)
        batch.src_doc_segments_length.append(doc_segments_length)

        self._populate_doc_len(
            segmented_text,
            doc_segments_length,
            batch.src_sent_len,
            batch.src_fragment_len,
            batch.src_doc_len_in_sents,
            batch.src_doc_len_in_frags,
        )

        src_no = len(batch.src_ids) - 1
        selected_hashes = [t[-1] for t in positive_targets]
        for _, h, _ in all_positive_targets:
            if h not in selected_hashes:
                batch_dups[h] = src_no

        return _ProcSrcStatus.ADDED

    def _empty_batch(self):
        iterable: List[Union[List, Mapping[str, int]]] = [[] for _ in range(15)]
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

    def _pad_batch_with_fragments(
        self, sents, fragment_lengths, doc_length_in_fragments, prefix, batch
    ):
        padded_sents, fragment_len = pad_sent_sequences(
            sents, fragment_lengths, self._text_proc.vocab()
        )
        padded_sents, frag_lens_with_padding, doc_len_in_frags = pad_fragment_sequences(
            padded_sents,
            doc_length_in_fragments,
            fragment_len=fragment_len,
            fragment_len_list=fragment_lengths,
            vocab=self._text_proc.vocab(),
        )
        fragment_lengths[:] = frag_lens_with_padding
        batch.info[f'{prefix}_fragment_len'] = fragment_len
        batch.info[f'{prefix}_doc_len_in_frags'] = doc_len_in_frags
        return padded_sents

    def _finalize_batch(self, batch: DocsBatch):
        for l in batch.positive_idxs:
            l.sort()
        src_sz = len(batch.src_doc_segments_length)
        batch.info['bs'] = src_sz
        batch.info['src_docs_cnt'] = src_sz
        batch.info['tgt_docs_cnt'] = len(batch.tgt_doc_segments_length)
        batch.info['src_frags_cnt'] = len(batch.src_fragment_len)
        batch.info['tgt_frags_cnt'] = len(batch.tgt_fragment_len)

        batch.info['max_positives_per_doc'] = len(max(batch.positive_idxs, key=len))
        if not self._opts.pad_src_sentences and not self._opts.pad_tgt_sentences:
            return batch

        if (
            self._opts.pad_fragments_level
            and self._text_proc.conf().split_into_sents
            and self._text_proc.conf().split_into_fragments
        ):
            src_padded_sents = batch.src_texts
            if self._opts.pad_src_sentences:
                src_padded_sents = self._pad_batch_with_fragments(
                    batch.src_texts,
                    batch.src_fragment_len,
                    batch.src_doc_len_in_frags,
                    'src',
                    batch,
                )
                batch.info['src_frags_cnt'] = len(batch.src_fragment_len)

            tgt_padded_sents = batch.tgt_texts
            if self._opts.pad_tgt_sentences:
                tgt_padded_sents = self._pad_batch_with_fragments(
                    batch.tgt_texts,
                    batch.tgt_fragment_len,
                    batch.tgt_doc_len_in_frags,
                    'tgt',
                    batch,
                )
                batch.info['tgt_frags_cnt'] = len(batch.tgt_fragment_len)

            return batch._replace(src_texts=src_padded_sents, tgt_texts=tgt_padded_sents)

        if self._opts.pad_src_sentences and self._text_proc.conf().split_into_sents:
            src_padded_sents, src_doc_len = pad_sent_sequences(
                batch.src_texts, batch.src_doc_len_in_sents, self._text_proc.vocab()
            )
            batch.info['src_doc_len_in_sents'] = src_doc_len

            batch = batch._replace(
                src_texts=src_padded_sents,
                src_fragment_len=None,
                src_doc_len_in_frags=None,
            )
        if self._opts.pad_tgt_sentences and self._text_proc.conf().split_into_sents:
            tgt_padded_sents, tgt_doc_len = pad_sent_sequences(
                batch.tgt_texts, batch.tgt_doc_len_in_sents, self._text_proc.vocab()
            )
            batch.info['tgt_doc_len_in_sents'] = tgt_doc_len

            batch = batch._replace(
                tgt_texts=tgt_padded_sents,
                tgt_fragment_len=None,
                tgt_doc_len_in_frags=None,
            )
        return batch

    def _is_defect_batch(self, batch):
        return (
            batch.info['bs'] == 1
            and batch.info['tgt_docs_cnt'] < self._opts.min_tgt_docs_per_src_doc
        )

    def _is_batch_ready(self, batch: DocsBatch, new_docs: list | None = None):
        batch_doc_size = len(batch.src_ids) + len(batch.tgt_ids)

        new_docs_cnt = 0
        if new_docs is not None:
            new_docs_cnt = len(new_docs)

        # check number of documents
        if self._opts.batch_docs_cnt and batch_doc_size + new_docs_cnt > self._opts.batch_docs_cnt:
            return True

        if self._opts.batch_total_tokens_cnt:
            # check number of tokens
            tokens_cnt = sum(len(t) for t in batch.src_texts)
            tokens_cnt += sum(len(t) for t in batch.tgt_texts)
            new_tokens_cnt = 0
            if new_docs is not None:
                for d in new_docs:
                    new_tokens_cnt += sum(len(s) for s in d)
            if tokens_cnt + new_tokens_cnt > self._opts.batch_total_tokens_cnt:
                return True

        if self._text_proc.conf().split_into_sents and self._opts.batch_total_sents_cnt:
            # check number of sentences
            d = self._opts.max_sents_cnt_delta if new_docs is not None else 0
            new_sents_cnt = 0
            if new_docs is not None:
                new_sents_cnt += sum(len(d) for d in new_docs)

            sents_cnt = len(batch.src_texts) + len(batch.tgt_texts)
            if sents_cnt + new_sents_cnt > self._opts.batch_total_sents_cnt + d:
                return True

        return False

    def _buckets_iter(self):
        bucket = []
        unique_docs_cnt = 0
        cur_hash = self._doc_pair_metas[0][EXMPL_SRC_HASH]

        def _fin_bucket():
            bucket.sort(
                key=lambda t: (
                    -t[EXMPL_SRC_LEN],
                    t[EXMPL_SRC_HASH],
                    -t[EXMPL_LABEL],
                    t[EXMPL_TGT_LEN],
                )
            )

        for metas in self._doc_pair_metas:
            if cur_hash != metas[EXMPL_SRC_HASH]:
                unique_docs_cnt += 1

                if unique_docs_cnt >= 1000:
                    _fin_bucket()
                    yield from bucket
                    bucket = []
                    unique_docs_cnt = 0
                cur_hash = metas[EXMPL_SRC_HASH]

            bucket.append(metas)

        if bucket:
            _fin_bucket()
            yield from bucket

    def batches(self):
        positive_targets = []
        negative_targets = []
        cur_hash = ''
        src_info = tuple()
        tokenized_texts_cache = {}

        batch, tgt_hashes, batch_dups = self._empty_batch()

        for metas in self._buckets_iter():
            if len(tokenized_texts_cache) > 4000:
                tokenized_texts_cache.clear()

            src_texts, tgt_texts = self._text_dirs_dict[metas[EXMPL_DATASET]]
            if cur_hash != metas[EXMPL_SRC_HASH]:
                status = self._process_src_doc(
                    src_info,
                    positive_targets,
                    negative_targets,
                    batch,
                    tgt_hashes,
                    batch_dups,
                    tokenized_texts_cache,
                )

                if (
                    status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_texts
                ) or self._is_batch_ready(batch):
                    batch = self._finalize_batch(batch)
                    if not self._is_defect_batch(batch):
                        yield batch
                    batch, tgt_hashes, batch_dups = self._empty_batch()

                if status == _ProcSrcStatus.CANT_FIT_IN_BATCH:
                    # add to new batch current document
                    self._process_src_doc(
                        src_info,
                        positive_targets,
                        negative_targets,
                        batch,
                        tgt_hashes,
                        batch_dups,
                        tokenized_texts_cache,
                    )

                # preparations for new src doc
                positive_targets = []
                negative_targets = []
                cur_hash = metas[EXMPL_SRC_HASH]

                src_path = find_file(
                    f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/{src_texts}/{metas[EXMPL_SRC_ID]}.txt",
                    throw_if_not_exist=False,
                )
                src_info = (src_path, cur_hash, metas[EXMPL_SRC_ID])

            tgt_id = metas[EXMPL_TGT_ID]
            tgt_path = find_file(
                f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/{tgt_texts}/{metas[EXMPL_TGT_ID]}.txt",
                throw_if_not_exist=False,
            )
            if not os.path.exists(tgt_path):
                logging.warning("tgt text is missing: %s", tgt_path)
                continue

            tgt_info = (tgt_path, metas[EXMPL_TGT_HASH], tgt_id)

            label = metas[EXMPL_LABEL]
            if label == 1:
                positive_targets.append(tgt_info)
            else:
                negative_targets.append(tgt_info)

        status = self._process_src_doc(
            src_info,
            positive_targets,
            negative_targets,
            batch,
            tgt_hashes,
            batch_dups,
            tokenized_texts_cache,
        )
        if status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_texts:
            batch = self._finalize_batch(batch)
            if not self._is_defect_batch(batch):
                yield batch
            batch, tgt_hashes, batch_dups = self._empty_batch()
            self._process_src_doc(
                src_info,
                positive_targets,
                negative_targets,
                batch,
                tgt_hashes,
                batch_dups,
                tokenized_texts_cache,
            )

        if batch.src_texts:
            batch = self._finalize_batch(batch)
            if not self._is_defect_batch(batch):
                yield batch


@dataclasses.dataclass
class DocsBatchIteratorConf(BaseBatchIteratorConf):
    batch_generator_conf: DocsBatchGeneratorConf = MISSING

    use_existing_combined_meta: bool = False
    combine_procs_cnt: int = 4
    include_datasets: Optional[List[str]] = None
    exclude_datasets: Optional[List[str]] = None


class DocsBatchIterator(BaseBatchIterator):
    def __init__(
        self,
        opts: DocsBatchIteratorConf,
        tp_conf: TextProcessorConf,
        logging_conf,
        split,
        rank=0,
        world_size=-1,
        device=None,
        pad_to_multiple_of=0,
    ):
        super().__init__(
            "DocsIter",
            opts,
            logging_conf,
            DocsBatchGenerator,
            (opts.batch_generator_conf, tp_conf, split),
            rank=rank,
            world_size=world_size,
        )

        self._opts = opts
        self._split = split

        if device is None:
            device = torch.device('cpu')
        self._device = device

        self._pad_to_multiple_of = pad_to_multiple_of
        self._epoch = 0

    def init_epoch(self, epoch, iter_no=1):
        self._epoch = epoch - 1
        opts = self._opts.batch_generator_conf
        fp = f"{opts.input_dir}/{opts.meta_prefix}_{self._split}.csv"
        if not self._start_workers(fp, seed=10_000 * epoch + iter_no):
            raise RuntimeError("Failed to init docs batch generator, empty folder or config error")

    def _make_batch_for_retr_task(self, batch: DocsBatch):
        src_cnt = batch.info['src_docs_cnt']
        labels = torch.full(
            (src_cnt, batch.info['tgt_docs_cnt']), 0.0, dtype=torch.float32, device=self._device
        )
        for i in range(src_cnt):
            positive_tgts = batch.positive_idxs[i]
            if positive_tgts:
                labels[i][positive_tgts] = 1.0

        return batch, labels

    def _prepare_batch(self, batch):
        return self._make_batch_for_retr_task(batch)
