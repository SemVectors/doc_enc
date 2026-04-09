#!/usr/bin/env python3

import itertools
import logging
import os.path
from pathlib import Path
from enum import Enum

import random
from typing import Generator, List, Optional, Union, Mapping
import dataclasses

from omegaconf import MISSING
import torch

from doc_enc.encoders.enc_in import (
    EncoderInData,
    EncoderInputType,
    SeqEncoderBatchedInput,
    TextReprType,
    TextsRepr,
)
from doc_enc.encoders.pad_utils import PadOpts
from doc_enc.text_processor import (
    TextProcessorConf,
    TextProcessor,
    pad_sent_sequences,
    pad_fragment_sequences,
)
from doc_enc.training.base_batch_generator import (
    BaseBatchAsyncGenerator,
    BaseBatchAsyncGeneratorConf,
    skip_to_line,
)
from doc_enc.utils import find_file
from doc_enc.training.types import DocRetrPairs, DocsBatch


@dataclasses.dataclass
class DocsBatchGeneratorConf:
    input_dir: str
    meta_prefix: str = "combined"

    batch_docs_cnt: int = 512
    batch_total_tokens_cnt: int = 96_000
    batch_total_sents_cnt: int = 8192

    positives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 3])
    negatives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [0, 3])

    max_sents_per_doc: int = 1280
    min_sents_per_doc: int = 5
    min_tgt_docs_per_src_doc: int = 2
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
        enc_input_type: EncoderInputType,
        opts: DocsBatchGeneratorConf,
        tp_conf: TextProcessorConf,
        split,
        pad_opts: PadOpts = PadOpts(),
        line_offset=0,
        line_cnt=-1,
        limit=0,
    ):
        self._enc_input_type = enc_input_type

        self._opts = opts
        self._pad_opts = pad_opts

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

        if tp_conf.split_into_fragments and tp_conf.split_into_sents:
            self._text_repr_type = TextReprType.SEQ_OF_FRAGMENTS_OF_SENTS
        elif tp_conf.split_into_sents:
            self._text_repr_type = TextReprType.SEQ_OF_SENTS
        elif tp_conf.split_into_fragments:
            self._text_repr_type = TextReprType.SEQ_OF_FRAGMENTS
        else:
            self._text_repr_type = TextReprType.SEQ_OF_TOKENS

    def _load_metas(self, split, line_offset, line_cnt):
        all_metas = []
        fp = f"{self._opts.input_dir}/{self._opts.meta_prefix}_{split}.csv"

        with open(fp, 'r', encoding='utf8') as fp:
            skip_to_line(fp, line_offset)
            for i, line in enumerate(fp):
                if i == line_cnt:
                    break
                row = line.rstrip().split(',')
                ds, src, tgt, label, slen, tlen, shash, thash = row
                t = (ds, int(src), int(tgt), int(label), int(slen), int(tlen), shash, thash)
                all_metas.append(t)

        return all_metas

    def _select_targets(self, targets: list[tuple], min_max_list) -> list[tuple]:
        if not targets:
            return []

        a, b = min_max_list
        n = random.randrange(a, b + 1)

        if n >= len(targets):
            return targets
        return random.sample(targets, n)

    # def _populate_doc_len(
    #     self,
    #     segmented_text: list[list[int]],
    #     doc_segments_length: list[int],
    #     sent_len_list: list[int],
    #     frag_len_in_sents: list[int],
    #     doc_len_in_sents_list: list[int],
    #     doc_len_in_frags_list: list[int],
    # ):
    #     if self._text_proc.conf().split_into_sents and self._text_proc.conf().split_into_fragments:
    #         frag_len_in_sents.extend(doc_segments_length)
    #         doc_len_in_frags_list.append(len(doc_segments_length))
    #     elif self._text_proc.conf().split_into_fragments:
    #         doc_len_in_frags_list.append(doc_segments_length[0])

    #     if self._text_proc.conf().split_into_sents:
    #         sent_len_list.extend(len(t) for t in segmented_text)
    #         doc_len_in_sents_list.append(len(segmented_text))

    def _prepare_all_targets(
        self,
        positive_targets: list[tuple],
        negative_targets: list[tuple],
        tgt_hashes,
        batch_dups,
        batch: DocRetrPairs,
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
            batch.tgt_text_lengths.append(doc_segments_length)

            # self._populate_doc_len(
            #     segmented_text,
            #     doc_segments_length,
            #     batch.tgt_sent_len,
            #     batch.tgt_fragment_len,
            #     batch.tgt_doc_len_in_sents,
            #     batch.tgt_doc_len_in_frags,
            # )
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
        batch: DocRetrPairs,
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
        batch.src_text_lengths.append(doc_segments_length)

        # self._populate_doc_len(
        #     segmented_text,
        #     doc_segments_length,
        #     batch.src_sent_len,
        #     batch.src_fragment_len,
        #     batch.src_doc_len_in_sents,
        #     batch.src_doc_len_in_frags,
        # )

        src_no = len(batch.src_ids) - 1
        selected_hashes = [t[-1] for t in positive_targets]
        for _, h, _ in all_positive_targets:
            if h not in selected_hashes:
                batch_dups[h] = src_no

        return _ProcSrcStatus.ADDED

    def _empty_batch(self):
        iterable: List[Union[List, Mapping[str, int]]] = [[] for _ in range(7)]
        empty_info = {
            'src_docs_cnt': 0,
            'tgt_docs_cnt': 0,
            'src_frags_cnt': 0,
            'tgt_frags_cnt': 0,
            'max_positives_per_doc': 0,
        }
        iterable.append(empty_info)
        return DocRetrPairs._make(iterable), {}, {}

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

    def _finalize_batch(self, batch: DocRetrPairs) -> DocsBatch:
        src_cnt = len(batch.src_text_lengths)
        tgt_cnt = len(batch.tgt_text_lengths)

        for pis in batch.positive_idxs:
            pis.sort()

        labels = torch.full((src_cnt, tgt_cnt), 0.0, dtype=torch.float32)
        for i in range(src_cnt):
            positive_tgts = batch.positive_idxs[i]
            if positive_tgts:
                labels[i][positive_tgts] = 1.0

        pad_idx = self._text_proc.vocab().pad_idx()
        prep_batch = DocsBatch(
            EncoderInData(
                SeqEncoderBatchedInput.from_input_ids(
                    self._enc_input_type, batch.src_texts, pad_idx, self._pad_opts
                ),
                batch.src_ids,
                TextsRepr(self._text_repr_type, batch.src_texts, batch.src_text_lengths),
            ),
            EncoderInData(
                SeqEncoderBatchedInput.from_input_ids(
                    self._enc_input_type, batch.tgt_texts, pad_idx, self._pad_opts
                ),
                batch.tgt_ids,
                TextsRepr(self._text_repr_type, batch.tgt_texts, batch.tgt_text_lengths),
            ),
            labels=labels,
        )

        # prep_batch.info['bs'] = src_cnt
        # prep_batch.info['src_docs_cnt'] = src_cnt
        # prep_batch.info['tgt_docs_cnt'] = tgt_cnt
        # if self._text_proc.conf().split_into_sents and self._text_proc.conf().split_into_fragments:
        #     prep_batch.info['src_frags_cnt'] = sum(len(tl) for tl in batch.src_text_lengths)
        #     prep_batch.info['tgt_frags_cnt'] = sum(len(tl) for tl in batch.tgt_text_lengths)

        # prep_batch.info['max_positives_per_doc'] = len(max(batch.positive_idxs, key=len))

        return prep_batch

    def _is_defect_batch(self, batch: DocsBatch):
        return (
            batch.batch_size() == 1
            and batch.get_tgt_docs_cnt() < self._opts.min_tgt_docs_per_src_doc
        )

    def _is_batch_ready(self, batch: DocRetrPairs, new_docs: list | None = None):
        batch_doc_size = len(batch.src_ids) + len(batch.tgt_ids)

        new_docs_cnt = 0
        if new_docs is not None:
            new_docs_cnt = len(new_docs)

        # check number of documents
        if self._opts.batch_docs_cnt and batch_doc_size + new_docs_cnt > self._opts.batch_docs_cnt:
            return True

        # check number of tokens
        tokens_cnt = sum(len(t) for t in batch.src_texts)
        tokens_cnt += sum(len(t) for t in batch.tgt_texts)
        new_tokens_cnt = 0
        if new_docs is not None:
            for d in new_docs:
                new_tokens_cnt += sum(len(s) for s in d)
        if tokens_cnt + new_tokens_cnt > self._opts.batch_total_tokens_cnt:
            return True

        if self._text_proc.conf().split_into_sents:
            # check number of sentences
            new_sents_cnt = 0
            if new_docs is not None:
                new_sents_cnt += sum(len(d) for d in new_docs)

            sents_cnt = len(batch.src_texts) + len(batch.tgt_texts)
            if sents_cnt + new_sents_cnt > self._opts.batch_total_sents_cnt:
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

    def batches(self) -> Generator[DocsBatch, None, None]:
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
                    prep_batch = self._finalize_batch(batch)
                    if not self._is_defect_batch(prep_batch):
                        yield prep_batch
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
class DocsBatchAsyncGeneratorConf(BaseBatchAsyncGeneratorConf):
    batch_generator_conf: DocsBatchGeneratorConf = MISSING

    use_existing_combined_meta: bool = False
    combine_procs_cnt: int = 4
    include_datasets: Optional[List[str]] = None
    exclude_datasets: Optional[List[str]] = None


class DocsBatchAsyncGenerator(BaseBatchAsyncGenerator[DocsBatch]):
    _gen_cls = DocsBatchGenerator
    _name = "DocsBatchGenerator"

    def __init__(
        self,
        enc_input_type: EncoderInputType,
        opts: DocsBatchAsyncGeneratorConf,
        tp_conf: TextProcessorConf,
        logging_conf,
        split,
        pad_opts: PadOpts = PadOpts(),
        rank=0,
        world_size=-1,
    ):
        super().__init__(
            enc_input_type,
            opts,
            opts.batch_generator_conf.batch_total_tokens_cnt,
            opts.batch_generator_conf.batch_total_sents_cnt,
            logging_conf,
            (opts.batch_generator_conf, tp_conf, split, pad_opts),
            rank=rank,
            world_size=world_size,
        )

        self._opts = opts
        self._split = split

        self._epoch = 0

    def init_epoch(self, epoch, iter_no=1):
        self._epoch = epoch - 1
        opts = self._opts.batch_generator_conf
        fp = f"{opts.input_dir}/{opts.meta_prefix}_{self._split}.csv"
        if not self._start_workers(fp, seed=10_000 * epoch + iter_no):
            raise RuntimeError("Failed to init docs batch generator, empty folder or config error")

    # def _make_batch_for_retr_task(self, batch: DocsBatch):
    #     src_cnt = batch.info['src_docs_cnt']
    #     labels = torch.full(
    #         (src_cnt, batch.info['tgt_docs_cnt']), 0.0, dtype=torch.float32, device=self._device
    #     )
    #     for i in range(src_cnt):
    #         positive_tgts = batch.positive_idxs[i]
    #         if positive_tgts:
    #             labels[i][positive_tgts] = 1.0

    #     return batch, labels

    def _prepare_batch(self, src_in: EncoderInData, tgt_in: EncoderInData, labels: torch.Tensor):
        return DocsBatch(src_in, tgt_in, labels)
