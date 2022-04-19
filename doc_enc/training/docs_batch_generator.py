#!/usr/bin/env python3

import itertools

# import math
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
)
from doc_enc.training.types import DocsBatch


@dataclasses.dataclass
class DocsBatchGeneratorConf:
    input_dir: str
    meta_prefix: str = "combined"

    include_datasets: Optional[List[str]] = None
    exclude_datasets: Optional[List[str]] = None

    batch_sent_size: int = 512
    batch_size: int = 128

    positives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [1, 2])
    negatives_per_doc: List[int] = dataclasses.field(default_factory=lambda: [2, 4])

    fragment_size: int = 24


EXMPL_DATASET = 0
EXMPL_SRC_ID = 1
EXMPL_TGT_ID = 2
EXMPL_LABEL = 3
EXMPL_SRC_HASH = 6
EXMPL_TGT_HASH = 7


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

        self._line_offset = line_offset
        self._line_cnt = line_cnt

        self._tokenizer = create_tokenizer(tok_conf)
        self._meta_file = None

        fp = f"{self._opts.input_dir}/{self._opts.meta_prefix}_{split}.csv"
        self._meta_file = open(fp, 'r', encoding='utf8')
        skip_to_line(self._meta_file, self._line_offset)

    def __del__(self):
        if self._meta_file is not None:
            self._meta_file.close()

    def _select_targets(self, targets, min_max_list):
        if not targets:
            return []

        a, b = min_max_list
        n = random.randrange(a, b + 1)

        if n >= len(targets):
            return targets
        return random.sample(targets, n)

    def _tokenize_doc(self, path):
        with open(path, 'r', encoding='utf8') as f:
            sents = []
            for l in f:
                tokens = self._tokenizer(l.rstrip())
                sents.append(tokens)
            # TODO truncate large sents
            return sents

    def _split_on_fragments(self, sents: List, fragment_len_list: List):
        l = len(sents)

        # start_fragment_id = 0
        # if fragment_id_list:
        #     start_fragment_id = fragment_id_list[-1] + 1
        fragments_cnt = 0
        for offs in range(0, l, self._opts.fragment_size):
            cnt = min(l - offs, self._opts.fragment_size)
            fragment_len_list.append(cnt)
            fragments_cnt += 1
            # i = offs // self._opts.fragment_size
            # i += start_fragment_id
            # fragment_id_list.extend(itertools.repeat(i, cnt))
        return fragments_cnt

    def _populate_doc_len(
        self, sents: List, fragments_cnt, doc_len_in_sents_list, doc_len_in_frags_list: List
    ):
        # doc_no = 0
        # if doc_len_list:
        #     doc_no = doc_len_list[-1] + 1
        # doc_len_list.extend(itertools.repeat(doc_no, len(sents)))
        doc_no = len(doc_len_in_sents_list)
        doc_len_in_sents_list.append(len(sents))
        doc_len_in_frags_list.append(fragments_cnt)
        return doc_no

    def _process_src_doc(
        self, src_path, positive_targets, negative_targets, batch: DocsBatch, tgt_hashes: dict
    ):
        if not positive_targets and not negative_targets:
            return 0
        positive_targets = self._select_targets(positive_targets, self._opts.positives_per_doc)
        negative_targets = self._select_targets(negative_targets, self._opts.negatives_per_doc)

        src_sents = self._tokenize_doc(src_path)
        batch.src_sents.extend(src_sents)
        fragments_cnt = self._split_on_fragments(src_sents, batch.src_fragment_len)
        self._populate_doc_len(
            src_sents, fragments_cnt, batch.src_doc_len_in_sents, batch.src_doc_len_in_frags
        )

        batch.positive_idxs.append([])
        for tgt_path, tgt_hash, lbl in itertools.chain(
            ((p, h, 1) for p, h in positive_targets), ((p, h, 0) for p, h in negative_targets)
        ):

            if tgt_hash in tgt_hashes:
                if lbl == 1:
                    batch.positive_idxs[-1].append(tgt_hashes[tgt_hash])
                continue

            tgt_sents = self._tokenize_doc(tgt_path)
            batch.tgt_sents.extend(tgt_sents)
            fragments_cnt = self._split_on_fragments(tgt_sents, batch.tgt_fragment_len)
            tgt_no = self._populate_doc_len(
                tgt_sents, fragments_cnt, batch.tgt_doc_len_in_sents, batch.tgt_doc_len_in_frags
            )
            tgt_hashes[tgt_hash] = tgt_no
            if lbl == 1:
                batch.positive_idxs[-1].append(tgt_no)

    def _empty_batch(self):
        iterable = [[] for _ in range(11)]
        iterable.append(
            {'src_docs_cnt': 0, 'tgt_docs_cnt': 0, 'src_frags_cnt': 0, 'tgt_frags_cnt': 0}
        )
        return DocsBatch._make(iterable), {}

    def _finalize_batch(self, batch: DocsBatch):
        batch.info['src_docs_cnt'] = len(batch.src_doc_len_in_sents)
        batch.info['tgt_docs_cnt'] = len(batch.tgt_doc_len_in_sents)
        batch.info['src_frags_cnt'] = len(batch.src_fragment_len)
        batch.info['tgt_frags_cnt'] = len(batch.tgt_fragment_len)

    def batches(self):
        if self._meta_file is None:
            raise RuntimeError("Files are not initialized")

        positive_targets = []
        negative_targets = []
        cur_hash = ''
        src_path = ''

        batch, tgt_hashes = self._empty_batch()

        for i, l in enumerate(self._meta_file):
            if i == self._line_cnt:
                break
            metas = l.split(',')

            if cur_hash != metas[EXMPL_SRC_HASH]:
                self._process_src_doc(
                    src_path, positive_targets, negative_targets, batch, tgt_hashes
                )
                if len(batch.src_sents) > self._opts.batch_sent_size:
                    self._finalize_batch(batch)
                    yield batch
                    batch, tgt_hashes = self._empty_batch()

                positive_targets = []
                negative_targets = []
                cur_hash = metas[EXMPL_SRC_HASH]

                src_path = (
                    f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/texts/{metas[EXMPL_SRC_ID]}.txt"
                )

            tgt_path = (
                f"{self._opts.input_dir}/{metas[EXMPL_DATASET]}/texts/{metas[EXMPL_TGT_ID]}.txt"
            )
            tgt_info = (tgt_path, metas[EXMPL_TGT_HASH])

            label = int(metas[EXMPL_LABEL])
            if label == 1:
                positive_targets.append(tgt_info)
            else:
                negative_targets.append(tgt_info)

        self._process_src_doc(src_path, positive_targets, negative_targets, batch, tgt_hashes)
        self._finalize_batch(batch)
        yield batch


@dataclasses.dataclass
class DocsBatchIteratorConf(BaseBatchIteratorConf):
    batch_generator_conf: DocsBatchGeneratorConf = MISSING
    pad_to_multiple_of: int = 0


class DocsBatchIterator(BaseBatchIterator):
    def __init__(
        self,
        opts: DocsBatchIteratorConf,
        tok_conf: TokenizerConf,
        split,
        rank=0,
        world_size=-1,
        pad_idx=0,
    ):

        super().__init__(
            opts,
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

    def _create_padded_tensor(self, tokens, max_len):
        # logging.debug('make batch with max len %s for %s', str(max_len), str(tokens))
        bs = len(tokens)

        if self._opts.pad_to_multiple_of and max_len % self._opts.pad_to_multiple_of != 0:
            max_len = (
                (max_len // self._opts.pad_to_multiple_of) + 1
            ) * self._opts.pad_to_multiple_of

        batch = torch.full((bs, max_len), self._pad_idx, dtype=torch.int32)
        for i in range(bs):
            batch[i, 0 : len(tokens[i])] = torch.as_tensor(tokens[i])

        batch = batch.to(device=self._device)
        lengths = torch.as_tensor([len(t) for t in tokens], dtype=torch.int64, device=self._device)
        return batch, lengths

    def _make_batch_for_retr_task(self, batch: DocsBatch):

        src_max_len = len(max(batch.src_sents, key=len))
        src, src_len = self._create_padded_tensor(batch.src_sents, src_max_len)

        tgt_max_len = len(max(batch.tgt_sents, key=len))
        tgt, tgt_len = self._create_padded_tensor(batch.tgt_sents, tgt_max_len)

        src_cnt = batch.info['src_docs_cnt']
        labels = torch.full((src_cnt, batch.info['tgt_docs_cnt']), 0, dtype=torch.int16)
        for i in range(src_cnt):
            positive_tgts = batch.positive_idxs[i]
            if positive_tgts:
                labels[i][positive_tgts] = 1

        b = batch._replace(src_sents=src, src_sent_len=src_len, tgt_sents=tgt, tgt_sent_len=tgt_len)
        return b, labels

    def _prepare_batch(self, batch):
        return self._make_batch_for_retr_task(batch)
