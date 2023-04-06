#!/usr/bin/env python3
#!/usr/bin/env python3

from typing import NamedTuple, Tuple, List
import logging
import collections
import itertools
import random
from dataclasses import dataclass


from omegaconf import MISSING
import torch

from doc_enc.training.base_batch_generator import (
    BaseBatchIterator,
    BaseBatchIteratorConf,
    skip_to_line,
    create_padded_tensor,
)
from doc_enc.training.types import SentsBatch
from doc_enc.tokenizer import TokenizerConf, create_tokenizer

from doc_enc.utils import find_file, open_file


class Example(NamedTuple):
    src_id: int
    src: List[int]
    tgt: List[int]
    dups: List[int]
    hns: Tuple[List[List[int]], List[int]]


def _src_filepath(input_dir, split):
    return find_file(f"{input_dir}/{split}.src")


def _tgt_filepath(input_dir, split):
    return find_file(f"{input_dir}/{split}.tgt")


@dataclass
class SentsBatchGeneratorConf:
    input_dir: str
    batch_size: int = 128
    batch_per_bucket: int = 100
    max_tokens: int = 0
    max_sent_size: int = 256

    adjust_batch_size: bool = True
    dont_use_dups: bool = False
    dont_use_hns: bool = False
    skip_large_hn: bool = False


class SentsBatchGenerator:
    def __init__(
        self,
        opts: SentsBatchGeneratorConf,
        tok_conf: TokenizerConf,
        split,
        line_offset=0,
        line_cnt=-1,
    ):
        self._opts = opts

        self._line_num = line_offset
        self._line_cnt = line_cnt

        self._tokenizer = create_tokenizer(tok_conf)
        self._src_file = None
        self._tgt_file = None
        self._hn_file = None
        self._dup_file = None

        src_fp = _src_filepath(opts.input_dir, split)
        tgt_fp = _tgt_filepath(opts.input_dir, split)
        self._src_file = open_file(src_fp)
        self._tgt_file = open_file(tgt_fp)

        if not opts.dont_use_dups:
            dups_fp = find_file(f"{opts.input_dir}/{split}.dups")
            self._dup_file = open_file(dups_fp)

        if not opts.dont_use_hns:
            hn_fp = find_file(f"{opts.input_dir}/{split}.hn")
            if str(hn_fp).endswith('.gz'):
                raise RuntimeError("gzipped hard negatives are not supported")
            self._hn_file = open(hn_fp, 'rb')
            self._hn_last_pos = 0

        self._init_files()

    def __del__(self):
        if self._src_file is not None:
            self._src_file.close()
        if self._tgt_file is not None:
            self._tgt_file.close()
        if self._dup_file is not None and not isinstance(self._dup_file, itertools.repeat):
            self._dup_file.close()
        if self._hn_file is not None:
            self._hn_file.close()

    def _init_files(self):
        if self._src_file is not None:
            skip_to_line(self._src_file, self._line_num)
            logging.info("initialized sents src file")
        if self._tgt_file is not None:
            skip_to_line(self._tgt_file, self._line_num)
            logging.info("initialized sents tgt file")
        if self._dup_file is not None:
            skip_to_line(self._dup_file, self._line_num)
            logging.info("initialized sents dups file")
        else:
            self._dup_file = itertools.repeat(None)

        if self._src_file is None:
            return
        # get current line id
        last_pos = self._src_file.tell()
        line = self._src_file.readline()
        self._src_file.seek(last_pos)
        if not line:
            raise RuntimeError("Unexpected end of src file!")
        line_id = int(line.split('\t', 1)[0])

        # skip hns file to the line_id
        if self._hn_file is not None:
            line = self._hn_file.readline()
            while line != '':
                read_line_id = int(line.split(b'\t', 1)[0])
                if read_line_id == line_id:
                    self._hn_file.seek(self._hn_last_pos)
                    break
                self._hn_last_pos += len(line)
                line = self._hn_file.readline()

            logging.info("initialized sents hn file")

    def _sort_within_bucket(self, bucket):
        # sort by length of src in a decreasing order
        # if lengths are equal, sort by the number of duplicates: the least the better
        bucket.sort(key=lambda e: (-len(e.src), len(e.dups)))

    def _adjust_batch_size(self, tgt_tokens, tgt_ids, excluded_ids, bucket: List[Example]):
        # its better to keep batch size multiple of 8 for performance reasons
        # see https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
        if not self._opts.adjust_batch_size or len(tgt_ids) < 8:
            return

        tries_cnt = 0
        while len(tgt_ids) % 8 != 0 and tries_cnt < len(bucket):
            tries_cnt += 1
            i = random.randrange(0, len(bucket))
            e = bucket[i]
            if e.src_id not in excluded_ids:
                excluded_ids.add(e.src_id)
                tgt_tokens.append(e.tgt)
                tgt_ids.append(e.src_id)

            for toks, hn_id in zip(*e.hns):
                if len(tgt_ids) % 8 == 0:
                    break
                if hn_id not in excluded_ids:
                    excluded_ids.add(hn_id)
                    tgt_tokens.append(toks)
                    tgt_ids.append(hn_id)

    def _make_batch(self, examples, excluded_ids, bucket: List[Example]):
        src_tokens = []
        src_ids = []
        tgt_tokens = []
        hn_tokens = []
        hn_ids = []
        hn_indices = []
        tgt_max_len = len(max((e.tgt for e in examples), key=len))
        for e in examples:
            src_tokens.append(e.src)
            src_ids.append(e.src_id)
            tgt_tokens.append(e.tgt)

            hns, ids = e.hns
            idxs = []
            for hn_id, toks in zip(ids, hns):
                if hn_id in excluded_ids:
                    continue
                if self._opts.skip_large_hn and len(toks) > 100 and len(toks) > 2 * tgt_max_len:
                    logging.info(
                        "hn %s has len %d that is greater than max tgt len %d. "
                        "It will be skipped ",
                        hn_id,
                        len(toks),
                        tgt_max_len,
                    )
                    continue

                excluded_ids.add(hn_id)

                idxs.append(len(examples) + len(hn_ids))
                hn_tokens.append(toks)
                hn_ids.append(hn_id)
            hn_indices.append(idxs)

        tgt_tokens = tgt_tokens + hn_tokens
        tgt_ids = src_ids + hn_ids
        self._adjust_batch_size(tgt_tokens, tgt_ids, excluded_ids, bucket)

        assert len(hn_indices) == len(src_tokens) == len(src_ids)

        info = {'bs': len(src_ids)}
        b = SentsBatch(src_ids, src_tokens, [], tgt_ids, tgt_tokens, [], hn_indices, info=info)
        return b

    def _maybe_prepend_not_fitted(self, not_fitted, bucket, to_next_bucket):
        if not not_fitted:
            return
        logging.debug("not_fitted: size %s", len(not_fitted))

        if not bucket:
            to_next_bucket.extend(not_fitted)
            return

        if len(not_fitted) > self._opts.batch_size // 2:
            logging.debug("not_fitted: move to the begining of the bucket")
            bucket.extendleft(reversed(not_fitted))
            return

        for e in reversed(not_fitted):
            if len(e.src) - 2 <= len(bucket[0].src):
                logging.debug("not_fitted: appendleft")
                bucket.appendleft(e)
            else:
                logging.debug("not_fitted: to next bucket")
                to_next_bucket.append(e)

    def _is_batch_ready(self, examples, max_len):
        bs = len(examples)
        if bs >= self._opts.batch_size:
            return True

        # we have to strive to keep batch size to be divisible by 8
        # for better gpu performance (see _adjust_batch_size)
        if self._opts.max_tokens and bs % 8 == 0:
            # since target contains many more examples, we will estimate tokens cnt in target batch
            hn_per_example = len(examples[0].hns[1])
            # num tokens with padding in target batch
            tokens_cnt = bs * (hn_per_example + 1) * max_len
            if tokens_cnt >= self._opts.max_tokens:
                logging.info(
                    "batch is ready by tokens: bs=%s, ml=%s, hns=%s ", bs, max_len, hn_per_example
                )

                return True

        return False

    def _create_batches_from_bucket(self, initial_bucket: List[Example]):
        # there should be no fuzzy duplicates inside mini-examples
        # so we should track id of dups of all examples added in minibatch so far
        batch_dups = set()
        examples = []
        not_fitted = []
        to_next_bucket = []
        batches = []
        bucket = collections.deque(initial_bucket)
        cur_max_len = 0
        while bucket:
            e = bucket.popleft()
            if e.src_id in batch_dups or any(i in batch_dups for i in e.dups):
                not_fitted.append(e)
                continue
            batch_dups.add(e.src_id)
            batch_dups.update(e.dups)
            examples.append(e)
            cur_max_len = max(len(e.src), len(e.tgt), cur_max_len, *[len(t) for t in e.hns[0]])

            if self._is_batch_ready(examples, cur_max_len):
                b = self._make_batch(examples, batch_dups, initial_bucket)
                batches.append(b)

                self._maybe_prepend_not_fitted(not_fitted, bucket, to_next_bucket)
                batch_dups = set()
                examples = []
                not_fitted = []
                cur_max_len = 0

        if examples:
            b = self._make_batch(examples, batch_dups, initial_bucket)
            batches.append(b)
            self._maybe_prepend_not_fitted(not_fitted, bucket, to_next_bucket)

        return batches, to_next_bucket

    def _parse_line(self, line):
        line_id, text = line.split('\t', 1)
        sent = self._tokenizer(text)
        return int(line_id), sent[: self._opts.max_sent_size]

    def _parse_dups(self, line):
        line_id, dups_str = line.split('\t', 1)
        return int(line_id), [int(i) for i in dups_str.split()]

    def _read_hard_negatives(self, src_id):
        if self._hn_file is None:
            return [], []

        line = self._hn_file.readline()
        if not line:
            raise RuntimeError("Unexpected end of hard negatives file!")

        hn_sents = []
        hn_ids = []
        while line:
            t = line.rstrip().split(b'\t', 2)

            read_src_id = int(t[0])
            if read_src_id != src_id:
                self._hn_file.seek(self._hn_last_pos)
                return hn_sents, hn_ids

            if len(t) == 1:
                # there is no hard negatives for src_id
                self._hn_last_pos += len(line)
                return [], []

            if len(t) == 2:
                # no tokens were found for this hn
                # nothing to do
                pass
            else:
                _, hn_id, hns_str = t
                hn_id = int(hn_id)
                hn_ids.append(hn_id)
                hn_sents.append(self._tokenizer(hns_str.decode('utf8')))

            self._hn_last_pos += len(line)
            line = self._hn_file.readline()
        return hn_sents, hn_ids

    def _log_bucket_info(self, batches):
        n = len(batches)

        logging.debug('batches cnt %d, avg tgt cnt %d', n, sum(len(b.tgt_id) for b in batches) / n)

    def batches(self):
        if self._src_file is None or self._tgt_file is None or self._dup_file is None:
            raise RuntimeError("Files are not initialized")
        bucket_size = self._opts.batch_per_bucket * self._opts.batch_size

        bucket = []
        cnt = 0
        for s, t, dups in zip(self._src_file, self._tgt_file, self._dup_file):
            if cnt == self._line_cnt:
                break
            cnt += 1
            src_id, st = self._parse_line(s)
            tgt_id, tt = self._parse_line(t)
            assert src_id == tgt_id, f"Data misaligned {src_id} != {tgt_id}"
            if dups is not None:
                dups_id, dups = self._parse_dups(dups)
                assert src_id == dups_id, f"Data misaligned with dups data {src_id} != {dups_id}"
            else:
                dups = []

            hns = self._read_hard_negatives(src_id)

            bucket.append(Example(src_id, src=st, tgt=tt, dups=dups, hns=hns))

            if len(bucket) >= bucket_size:
                self._sort_within_bucket(bucket)
                batches, to_next_bucket = self._create_batches_from_bucket(bucket)
                self._log_bucket_info(batches)
                logging.debug("%d examples left for next bucket", len(to_next_bucket))
                yield from batches
                bucket = to_next_bucket

        if bucket:
            self._sort_within_bucket(bucket)
            batches, to_next_bucket = self._create_batches_from_bucket(bucket)
            self._log_bucket_info(batches)
            yield from batches

            if to_next_bucket:
                logging.info("Failed to find a batch for %d examples", len(to_next_bucket))


@dataclass
class SentsBatchIteratorConf(BaseBatchIteratorConf):
    batch_generator_conf: SentsBatchGeneratorConf = MISSING


class SentsBatchIterator(BaseBatchIterator):
    def __init__(
        self,
        opts: SentsBatchIteratorConf,
        tok_conf: TokenizerConf,
        logging_conf,
        split,
        rank=0,
        world_size=-1,
        device=None,
        pad_idx=0,
        pad_to_multiple_of=0,
    ):
        super().__init__(
            "SentsIter",
            opts,
            logging_conf,
            SentsBatchGenerator,
            (opts.batch_generator_conf, tok_conf, split),
            rank=rank,
            world_size=world_size,
        )

        self._opts = opts

        self._split = split

        if device is None:
            device = torch.device('cpu')
        self._device = device

        self._pad_idx = pad_idx
        self._pad_to_multiple_of = pad_to_multiple_of

        self._epoch = 0

    def init_epoch(self, epoch, iter_no=1):
        self._epoch = epoch - 1
        src_fp = _tgt_filepath(self._opts.batch_generator_conf.input_dir, self._split)
        if not self._start_workers(src_fp, seed=10_000 * epoch + iter_no):
            raise RuntimeError("Failed to init sents batch generator, empty folder or config error")

    def _make_batch_for_retr_task(self, batch):
        src, src_len = create_padded_tensor(
            batch.src, len(batch.src[0]), self._pad_idx, self._device, self._pad_to_multiple_of
        )

        tgt_max_len = len(max(batch.tgt, key=len))
        tgt, tgt_len = create_padded_tensor(
            batch.tgt, tgt_max_len, self._pad_idx, self._device, self._pad_to_multiple_of
        )

        labels = torch.arange(0, batch.info['bs'], device=self._device)
        b = batch._replace(src=src, src_len=src_len, tgt=tgt, tgt_len=tgt_len, hn_idxs=[])
        return b, labels

    def _prepare_batch(self, batch):
        batch, labels = self._make_batch_for_retr_task(batch)
        return batch, labels
