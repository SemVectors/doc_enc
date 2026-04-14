#!/usr/bin/env python3

import itertools
import logging
import math
import os.path
from pathlib import Path
from enum import Enum

import random
from typing import Generator, List, NamedTuple, Optional, Union, Mapping
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
    SegmentedText,
    TextProcessorConf,
    TextProcessor,
    pad_sent_sequences,
    pad_fragment_sequences,
)
from doc_enc.training.base_batch_generator import (
    BaseBatchAsyncGenerator,
    BaseBatchAsyncGeneratorConf,
)
from doc_enc.utils import find_file, skip_to_line
from doc_enc.training.types import DocRetrPairs, DocRetrPairsStat, DocsBatch


@dataclasses.dataclass
class DocsBatchGeneratorConf:
    input_dir: str
    meta_prefix: str = "combined"

    batch_docs_cnt: int = 512
    batch_total_tokens_cnt: int = 96_000
    batch_total_sents_cnt: int = 8192

    # It is guaranteed that each src text will have one positive and one
    # negative (if there are any). Additionally, some more examples will be
    # added. Those options specify a range of number of positive/negative
    # examples. The actual value is randomly selected from this range for each
    # arc text. Note, that if there will be no room for an extra examples in the
    # mini-batch, then these actions will have no effect.
    positives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 3])
    negatives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 3])

    # Drop text if number of sequences is larger than max_sents_per_doc.
    max_sents_per_doc: int | None = None
    # Drop text if number of sequences is less than min_sents_per_doc.
    min_sents_per_doc: int = 5

    # Truncation settings
    # Do not truncate if  1 - len(truncated)/len(orig) > <acceptable_*>
    acceptable_seqs_trunc_ratio: float = 0.5
    acceptable_toks_trunc_ratio: float = 0.5


#            ds,  src, tgt, label, slen, tlen, shash, thash
_MetaT = tuple[str, int, int, int, int, int, str, str]


#                 path, hash, id
# _TextInfoT = tuple[Path, str, int]
class _TextInfoT(NamedTuple):
    path: Path
    hash: str
    text_id: int


EXMPL_DATASET = 0
EXMPL_SRC_ID = 1
EXMPL_TGT_ID = 2
EXMPL_LABEL = 3
EXMPL_SRC_LEN = 4
EXMPL_TGT_LEN = 5
EXMPL_SRC_HASH = 6
EXMPL_TGT_HASH = 7

# _TokTextT = list[list[int]]
_TokTextCacheT = dict[str, SegmentedText]


class _ProcSrcStatus(Enum):
    ADDED = 1
    CANT_FIT_IN_BATCH = 2
    NON_VALID_SRC = 3
    HUGE_SRC_NOT_TRUNCATABLE = 4


class DocsBatchGeneratorStat:
    def __init__(self):
        self.src_texts = 0
        self.tgt_texts = 0
        self.positives = 0
        self.negatives = 0
        self.src_segments = 0
        self.tgt_segments = 0
        self.src_tokens = 0
        self.tgt_tokens = 0
        self.filtered_by_min_sents_per_doc = 0
        self.filtered_by_max_sents_per_doc = 0
        self.path_does_not_exist = 0
        self.empty_text = 0
        self.huge_src = 0
        self.bad_src = 0
        self.no_positives = 0
        self.cant_fit = 0
        self.defective_batch = 0

        self.total_seqs_trunc_ratio = 0.0
        self.seqs_truncated = 0
        self.total_toks_trunc_ratio = 0.0
        self.toks_truncated = 0

    def __str__(self):
        return (
            f'{{src/tgt}} texts,segments,tokens: {self.src_texts}/{self.tgt_texts},{self.src_segments}/{self.tgt_segments},{self.src_tokens}/{self.tgt_tokens}, '
            f'positives: {self.positives}, negatives: {self.negatives}\n'
            f'filtered_by_{{min/max}}_sents_per_doc: {self.filtered_by_min_sents_per_doc}/{self.filtered_by_max_sents_per_doc}, '
            f'path_does_not_exist: {self.path_does_not_exist}, empty_texts: {self.empty_text}, huge_pair: {self.huge_src}, '
            f'bad_src: {self.bad_src}, no_positives: {self.no_positives}, cant_fit: {self.cant_fit}, defective_batch: {self.defective_batch}, '
            f'{{toks/seqs}}_truncated (trunc ratio): {self.toks_truncated} ({self.total_toks_trunc_ratio/self.toks_truncated if self.toks_truncated else 0:.2f})'
            f'/{self.seqs_truncated} ({self.total_seqs_trunc_ratio/self.seqs_truncated if self.seqs_truncated else 0:.2f})'
        )

    def __add__(self, other: 'DocsBatchGeneratorStat'):
        new = DocsBatchGeneratorStat()
        new.src_texts = self.src_texts + other.src_texts
        new.src_segments = self.src_segments + other.src_segments
        new.src_tokens = self.src_tokens + other.src_tokens
        new.tgt_texts = self.tgt_texts + other.tgt_texts
        new.tgt_segments = self.tgt_segments + other.tgt_segments
        new.tgt_tokens = self.tgt_tokens + other.tgt_tokens

        new.positives = self.positives + other.positives
        new.negatives = self.negatives + other.negatives

        new.filtered_by_min_sents_per_doc = (
            self.filtered_by_min_sents_per_doc + other.filtered_by_min_sents_per_doc
        )
        new.filtered_by_max_sents_per_doc = (
            self.filtered_by_max_sents_per_doc + other.filtered_by_max_sents_per_doc
        )
        new.path_does_not_exist = self.path_does_not_exist + other.path_does_not_exist
        new.empty_text = self.empty_text + other.empty_text
        new.huge_src = self.huge_src + other.huge_src
        new.bad_src = self.bad_src + other.bad_src
        new.no_positives = self.no_positives + other.no_positives
        new.cant_fit = self.cant_fit + other.cant_fit
        new.defective_batch = self.defective_batch + other.defective_batch
        new.total_seqs_trunc_ratio = self.total_seqs_trunc_ratio + other.total_seqs_trunc_ratio
        new.seqs_truncated = self.seqs_truncated + other.seqs_truncated
        new.total_toks_trunc_ratio = self.total_toks_trunc_ratio + other.total_toks_trunc_ratio
        new.toks_truncated = self.toks_truncated + other.toks_truncated
        return new


# From itertools recipes
def _roundrobin(*iterables):
    "Visit input iterables in a cycle until each is exhausted."
    # roundrobin('ABC', 'D', 'EF') → A D E B F C
    # Algorithm credited to George Sakkis
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = itertools.cycle(itertools.islice(iterators, num_active))
        yield from map(next, iterators)


def _interleave_examples(
    positives: list[_TextInfoT], pos_offs, negatives: list[_TextInfoT], neg_offs
) -> Generator[tuple[_TextInfoT, int], None, None]:
    pos_slice = itertools.islice(positives, pos_offs, None)
    neg_slice = itertools.islice(negatives, neg_offs, None)
    return _roundrobin(zip(pos_slice, itertools.repeat(1)), zip(neg_slice, itertools.repeat(0)))


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

        self._conf = opts
        self._pad_opts = pad_opts

        self._text_proc = TextProcessor(tp_conf)
        self._doc_pair_metas = self._load_metas(split, line_offset, line_cnt)
        if not self._doc_pair_metas:
            raise RuntimeError("Empty docs dataset")

        self._text_dirs_dict = {}
        for p in Path(self._conf.input_dir).iterdir():
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

        self._stat = DocsBatchGeneratorStat()

    def get_stat(self):
        return self._stat

    def _load_metas(self, split: str, line_offset: int, line_cnt: int) -> list[_MetaT]:
        all_metas = []
        fp = f"{self._conf.input_dir}/{self._conf.meta_prefix}_{split}.csv"

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

    def _select_targets(self, targets: list[_TextInfoT], min_max_list) -> list[_TextInfoT]:
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
        selected_texts: list[tuple[_TextInfoT, int]],
        prepped_texts: list[SegmentedText],
        batch: DocRetrPairs,
    ):
        positive_idxs = []

        for ((_, tgt_hash, tgt_id), label), prepped_text in zip(
            selected_texts, prepped_texts, strict=True
        ):
            if (idx := batch.tgt_hashes.get(tgt_hash)) is not None:
                if label == 1:
                    positive_idxs.append(idx)
                continue

            tgt_no = len(batch.tgt_ids)
            batch.tgt_ids.append(tgt_id)
            batch.tgt_texts.extend(prepped_text.token_seqs)
            batch.tgt_text_lengths.append(prepped_text.segment_lengths)

            self._stat.tgt_texts += 1
            self._stat.tgt_segments += prepped_text.segments_cnt()
            self._stat.tgt_tokens += prepped_text.ntokens()
            if label == 0:
                self._stat.negatives += 1
            elif label == 1:
                self._stat.positives += 1

            batch.tgt_hashes[tgt_hash] = tgt_no
            if label == 1:
                positive_idxs.append(tgt_no)

            if tgt_hash in batch.dups:
                # this tgt is a positive example for some document that is already in the batch
                # we need to adjust positive_idxs of this document accordingly
                batch.positive_idxs[batch.dups[tgt_hash]].append(tgt_no)
        return positive_idxs

    def _encode_text(
        self, path: Path, doc_hash: str, tokenized_texts_cache: _TokTextCacheT
    ) -> SegmentedText | None:
        if cached := tokenized_texts_cache.get(doc_hash):
            return cached
        if not os.path.exists(path):
            self._stat.path_does_not_exist += 1
            return None

        segmented_text = self._text_proc.prepare_text_from_file(path, add_special_tokens=False)
        if not segmented_text.token_seqs:
            self._stat.empty_text += 1
            return None

        if self._text_proc.conf().split_into_sents:
            if segmented_text.segments_cnt() < self._conf.min_sents_per_doc:
                self._stat.filtered_by_min_sents_per_doc += 1
                return None
            if (
                self._conf.max_sents_per_doc is not None
                and self._conf.max_sents_per_doc < segmented_text.segments_cnt()
            ):
                self._stat.filtered_by_max_sents_per_doc += 1
                return None

        # Put in cache only full (not truncated) text without special tokens.
        # So we can truncate it later when stitching batch together.
        tokenized_texts_cache[doc_hash] = segmented_text
        return segmented_text

    def _add_pair_to_batch(
        self,
        selected_texts: list[tuple[_TextInfoT, int]],
        ntokens: int,
        prepped_texts: list[SegmentedText],
        all_positive_targets: list[_TextInfoT],
        batch: DocRetrPairs,
    ):

        assert selected_texts[0][1] == -1, "First text is src with label == -1"
        src_info = selected_texts[0][0]
        src_seg_text = prepped_texts[0]
        batch.src_ids.append(src_info.text_id)
        batch.src_texts.extend(src_seg_text.token_seqs)
        batch.src_text_lengths.append(src_seg_text.segment_lengths)
        batch.stat.ntokens += ntokens
        self._stat.src_texts += 1
        self._stat.src_segments += src_seg_text.segments_cnt()
        self._stat.src_tokens += src_seg_text.ntokens()

        positive_idxs = self._prepare_all_targets(selected_texts[1:], prepped_texts[1:], batch)

        batch.positive_idxs.append(positive_idxs)

        # update batch_dups
        src_no = len(batch.src_ids) - 1
        selected_hashes = [t.hash for t, label in selected_texts if label == 1]
        for _, h, _ in all_positive_targets:
            if h not in selected_hashes:
                batch.dups[h] = src_no

    def _process_src_doc(
        self,
        src_info: _TextInfoT,
        all_positive_targets: list[_TextInfoT],
        all_negative_targets: list[_TextInfoT],
        batch: DocRetrPairs,
        tokenized_texts_cache: _TokTextCacheT,
    ) -> _ProcSrcStatus:
        if not src_info.hash:
            return _ProcSrcStatus.NON_VALID_SRC

        if not all_positive_targets:
            self._stat.no_positives += 1
            return _ProcSrcStatus.NON_VALID_SRC

        positive_targets = self._select_targets(all_positive_targets, self._conf.positives_per_doc)
        negative_targets = self._select_targets(all_negative_targets, self._conf.negatives_per_doc)

        # Minimal mini-batch would contain: src text, positive example, negative
        # example (optional). If there is no negative examples for this pair
        # then all other texts from mini-batch would be negative examples. But
        # we need to ensure that this mini-batch will contain other examples. In
        # the end it checked in _is_defect_batch.

        selected_texts: list[tuple[_TextInfoT, int]] = [(src_info, -1)]

        if not self._encode_text(src_info.path, src_info.hash, tokenized_texts_cache):
            self._stat.bad_src += 1
            return _ProcSrcStatus.NON_VALID_SRC

        pos_offs = 0
        for e in positive_targets:
            pos_offs += 1
            if self._encode_text(e.path, e.hash, tokenized_texts_cache):
                selected_texts.append((e, 1))
                break
        else:
            self._stat.no_positives += 1
            return _ProcSrcStatus.NON_VALID_SRC

        neg_offs = 0
        for e in negative_targets:
            neg_offs += 1
            if self._encode_text(e.path, e.hash, tokenized_texts_cache):
                selected_texts.append((e, 0))
                break

        cur_src_seqs_cnt = 0
        cur_src_ntokens = 0
        for cand, _ in selected_texts:
            if cand.hash not in batch.tgt_hashes and (
                cached := tokenized_texts_cache.get(cand.hash)
            ):
                cur_src_seqs_cnt += cached.segments_cnt()
                cur_src_ntokens += cached.ntokens()

        # Add more positive/negative examples.
        max_seqs_cnt = self._conf.batch_total_sents_cnt - batch.total_seqs_cnt()
        max_tokens_cnt = self._conf.batch_total_tokens_cnt - batch.stat.ntokens

        next_cand_gen = _interleave_examples(positive_targets, pos_offs, negative_targets, neg_offs)
        while cur_src_seqs_cnt < max_seqs_cnt and cur_src_ntokens < max_tokens_cnt:
            try:
                cand, label = next(next_cand_gen)
                if (
                    segmented_text := self._encode_text(cand.path, cand.hash, tokenized_texts_cache)
                ) is None:
                    continue
                if cand.hash in batch.tgt_hashes:
                    # This text is already in batch so we have not need to increase current total values.
                    selected_texts.append((cand, label))
                    continue
                nsc = segmented_text.segments_cnt()
                ntkn = segmented_text.ntokens()
                if cur_src_seqs_cnt + nsc > max_seqs_cnt or cur_src_ntokens + ntkn > max_tokens_cnt:
                    break

                cur_src_seqs_cnt += nsc
                cur_src_ntokens += ntkn
                selected_texts.append((cand, label))
            except StopIteration:
                break

        if cur_src_seqs_cnt < max_seqs_cnt and cur_src_ntokens < max_tokens_cnt:
            # Add this example to the batch.

            prepped_texts: list[SegmentedText] = []
            for (_, hash, _), label in selected_texts:
                seg_text = tokenized_texts_cache[hash]
                prepped = self._text_proc.prepare_for_model(seg_text)
                prepped_texts.append(prepped)

            self._add_pair_to_batch(
                selected_texts, cur_src_ntokens, prepped_texts, all_positive_targets, batch
            )
            return _ProcSrcStatus.ADDED

        # Current pair does not fit into the existing batch.
        # Try to truncate it if this is the only pair in the batch.
        if not batch.src_ids:
            truncated_texts = self._truncate_examples(selected_texts, tokenized_texts_cache)
            if truncated_texts is not None:
                self._add_pair_to_batch(
                    selected_texts, max_tokens_cnt, truncated_texts, all_positive_targets, batch
                )
                return _ProcSrcStatus.ADDED
            else:
                self._stat.huge_src += 1
                return _ProcSrcStatus.HUGE_SRC_NOT_TRUNCATABLE

        self._stat.cant_fit += 1
        return _ProcSrcStatus.CANT_FIT_IN_BATCH

    def _calc_truncated_lengths(self, orig_lengths: list[int], max_length: int):
        cur_total_length = sum(orig_lengths)
        truncated_lengths = [
            math.floor(max_length * (orig / cur_total_length)) for orig in orig_lengths
        ]
        trunc_ratio_list = [1 - new / orig for new, orig in zip(truncated_lengths, orig_lengths)]
        avg_ratio = sum(trunc_ratio_list) / len(trunc_ratio_list)
        return truncated_lengths, avg_ratio

    def _truncate_examples(
        self, selected_texts: list[tuple[_TextInfoT, int]], tokenized_texts_cache: _TokTextCacheT
    ) -> list[SegmentedText] | None:
        seqs_cnt_list = []
        ntokens_list = []
        seg_texts: list[SegmentedText] = []
        for cand, _ in selected_texts:
            if cached := tokenized_texts_cache.get(cand.hash):
                seg_texts.append(cached)
                seqs_cnt_list.append(cached.segments_cnt())
                ntokens_list.append(cached.ntokens())
        assert len(seg_texts) == len(
            selected_texts
        ), "_truncate_examples: All texts should be in the cache."

        cur_total_seqs_cnt = sum(seqs_cnt_list)
        max_seqs_cnt = self._conf.batch_total_sents_cnt
        if max_seqs_cnt and cur_total_seqs_cnt > max_seqs_cnt:
            trunc_seqs_lengths, avg_ratio = self._calc_truncated_lengths(
                seqs_cnt_list, max_seqs_cnt
            )
            if avg_ratio > self._conf.acceptable_seqs_trunc_ratio:
                return None
            self._stat.seqs_truncated += 1
            self._stat.total_seqs_trunc_ratio += avg_ratio
        else:
            trunc_seqs_lengths = seqs_cnt_list

        cur_total_tokens_cnt = sum(ntokens_list)
        max_tokens_cnt = self._conf.batch_total_tokens_cnt
        if max_tokens_cnt and cur_total_tokens_cnt > max_tokens_cnt:
            trunc_tok_lengths, avg_ratio = self._calc_truncated_lengths(
                ntokens_list, max_tokens_cnt
            )
            if avg_ratio > self._conf.acceptable_toks_trunc_ratio:
                return None
            self._stat.toks_truncated += 1
            self._stat.total_toks_trunc_ratio += avg_ratio
        else:
            trunc_tok_lengths = ntokens_list

        prepped_texts = []
        for seg_text, max_seqs, max_toks in zip(
            seg_texts, trunc_seqs_lengths, trunc_tok_lengths, strict=True
        ):
            prepped = self._text_proc.prepare_for_model(
                seg_text, truncate_length_in_tokens=max_toks, truncate_length_in_seqs=max_seqs
            )

            prepped_texts.append(prepped)

        return prepped_texts

    def _empty_batch(self):
        iterable: List[Union[List, Mapping[str, int]]] = [[] for _ in range(7)]
        empty_info = {
            'src_docs_cnt': 0,
            'tgt_docs_cnt': 0,
            'src_frags_cnt': 0,
            'tgt_frags_cnt': 0,
            'max_positives_per_doc': 0,
        }
        iterable += [empty_info, {}, {}, DocRetrPairsStat()]

        return DocRetrPairs._make(iterable)

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

        return prep_batch

    def _is_defect_batch(self, batch: DocsBatch):
        # Check that there is at least one negative example.
        is_defective = (
            batch.batch_size() == 1 and batch.max_positives_per_doc() == batch.get_tgt_docs_cnt()
        )
        if is_defective:
            self._stat.defective_batch += 1
        return is_defective

    def _is_batch_ready(self, batch: DocRetrPairs):
        batch_doc_size = len(batch.src_ids) + len(batch.tgt_ids)

        # check number of documents
        if self._conf.batch_docs_cnt and batch_doc_size > self._conf.batch_docs_cnt:
            return True

        if batch.stat.ntokens > self._conf.batch_total_tokens_cnt:
            return True

        if self._text_proc.conf().split_into_sents:
            if batch.total_seqs_cnt() > self._conf.batch_total_sents_cnt:
                return True

        return False

    def _buckets_iter(self):
        bucket: list[_MetaT] = []
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
        positive_targets: list[_TextInfoT] = []
        negative_targets: list[_TextInfoT] = []
        cur_hash = ''
        src_info = _TextInfoT(Path(), '', 0)
        tokenized_texts_cache: _TokTextCacheT = {}

        batch = self._empty_batch()

        for metas in self._buckets_iter():
            if len(tokenized_texts_cache) > 4000:
                tokenized_texts_cache.clear()

            src_texts, tgt_texts = self._text_dirs_dict[metas[EXMPL_DATASET]]
            if cur_hash != metas[EXMPL_SRC_HASH]:
                status = self._process_src_doc(
                    src_info, positive_targets, negative_targets, batch, tokenized_texts_cache
                )

                if (
                    status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_texts
                ) or self._is_batch_ready(batch):
                    prep_batch = self._finalize_batch(batch)
                    if not self._is_defect_batch(prep_batch):
                        yield prep_batch
                    batch = self._empty_batch()

                if status == _ProcSrcStatus.CANT_FIT_IN_BATCH:
                    # add to new batch current document
                    self._process_src_doc(
                        src_info, positive_targets, negative_targets, batch, tokenized_texts_cache
                    )

                # preparations for new src doc
                positive_targets = []
                negative_targets = []
                cur_hash = metas[EXMPL_SRC_HASH]

                src_path = find_file(
                    f"{self._conf.input_dir}/{metas[EXMPL_DATASET]}/{src_texts}/{metas[EXMPL_SRC_ID]}.txt",
                    throw_if_not_exist=False,
                )
                src_info = _TextInfoT(src_path, cur_hash, metas[EXMPL_SRC_ID])

            tgt_id = metas[EXMPL_TGT_ID]
            tgt_path = find_file(
                f"{self._conf.input_dir}/{metas[EXMPL_DATASET]}/{tgt_texts}/{metas[EXMPL_TGT_ID]}.txt",
                throw_if_not_exist=False,
            )
            if not os.path.exists(tgt_path):
                logging.warning("tgt text is missing: %s", tgt_path)
                continue

            tgt_info = _TextInfoT(tgt_path, metas[EXMPL_TGT_HASH], tgt_id)

            label = metas[EXMPL_LABEL]
            if label == 1:
                positive_targets.append(tgt_info)
            else:
                negative_targets.append(tgt_info)

        status = self._process_src_doc(
            src_info, positive_targets, negative_targets, batch, tokenized_texts_cache
        )
        if status == _ProcSrcStatus.CANT_FIT_IN_BATCH and batch.src_texts:
            batch = self._finalize_batch(batch)
            if not self._is_defect_batch(batch):
                yield batch
            batch = self._empty_batch()
            self._process_src_doc(
                src_info, positive_targets, negative_targets, batch, tokenized_texts_cache
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
        max_seq_length: int | None = None,
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
            max_seq_length=max_seq_length,
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
