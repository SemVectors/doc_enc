#!/usr/bin/env python3
#!/usr/bin/env python3

from typing import Generator, NamedTuple, Tuple, List
import logging
import collections
import itertools
import random
from dataclasses import dataclass


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
from doc_enc.training.base_batch_generator import (
    BaseBatchAsyncGenerator,
    BaseBatchAsyncGeneratorConf,
    skip_to_line,
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
    sents_limit: int = 0
    batches_per_bucket: int = 100
    batch_size: int = 128
    max_sents: int = 128 * 4
    max_tokens: int = 96_000
    # Deprecated: use max_seq_length in TokenizerConf
    max_sent_size: int | None = None

    adjust_batch_size: bool = False
    dont_use_dups: bool = False
    dont_use_hns: bool = False
    # skip_large_hn: bool = False
    min_hn_cnt: int = 0


class SentsBatchGenerator:
    def __init__(
        self,
        enc_input_type: EncoderInputType,
        conf: SentsBatchGeneratorConf,
        tok_conf: TokenizerConf,
        split,
        pad_opts: PadOpts = PadOpts(),
        line_offset=0,
        line_cnt=-1,
        limit=0,
    ):

        self._enc_input_type = enc_input_type
        self._conf = conf

        self._pad_opts = pad_opts

        self._line_num = line_offset
        self._line_cnt = line_cnt
        self._limit = limit

        if conf.max_sent_size is not None and tok_conf.max_seq_length is None:
            logging.warning("max_sent_size is deprecated use TokenizerConf.max_seq_length instead!")
            tok_conf.max_seq_length = conf.max_sent_size

        self._tokenizer = create_tokenizer(tok_conf)
        self._src_file = None
        self._tgt_file = None
        self._hn_file = None
        self._dup_file = None

        src_fp = _src_filepath(conf.input_dir, split)
        tgt_fp = _tgt_filepath(conf.input_dir, split)
        self._src_file = open_file(src_fp)
        self._tgt_file = open_file(tgt_fp)

        if not conf.dont_use_dups:
            dups_fp = find_file(f"{conf.input_dir}/{split}.dups")
            self._dup_file = open_file(dups_fp)

        if not conf.dont_use_hns:
            hn_fp = find_file(f"{conf.input_dir}/{split}.hn")
            if str(hn_fp).endswith('.gz'):
                raise RuntimeError("gzipped hard negatives are not supported")
            self._hn_file = open(hn_fp, 'rb')
            self._hn_last_line = b''

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
                    self._hn_last_line = line
                    break
                line = self._hn_file.readline()

            logging.info("initialized sents hn file")

    def get_stat(self):
        return None

    def _sort_within_bucket(self, bucket):
        # sort by length of src in a decreasing order
        # if lengths are equal, sort by the number of duplicates: the least the better
        bucket.sort(key=lambda e: (-len(e.src), len(e.dups)))

    def _adjust_batch_size(self, tgt_tokens, tgt_ids, excluded_ids, bucket: List[Example]):
        # its better to keep batch size multiple of 8 for performance reasons
        # see https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
        if not self._conf.adjust_batch_size or len(tgt_ids) < 8:
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
        # logging.debug("batch was adjusted with %d tries", tries_cnt)

    def _make_batch(self, examples: list[Example], excluded_ids: set[int], bucket: list[Example]):
        # logging.error("MAKE BATCH %s", [len(e.src) for e in examples])

        src_tokens: list[list[int]] = []
        src_ids = []
        tgt_tokens: list[list[int]] = []
        hn_tokens = []
        hn_ids = []
        hn_indices = []
        # tgt_max_len = len(max((e.tgt for e in examples), key=len))
        for e in examples:
            src_tokens.append(e.src)
            src_ids.append(e.src_id)
            tgt_tokens.append(e.tgt)

            ex_hns, ex_hn_ids = e.hns
            hn_idxs = []
            for hn_id, hn_toks in zip(ex_hn_ids, ex_hns):
                if hn_id in excluded_ids:
                    continue
                # if self._conf.skip_large_hn and len(toks) > 100 and len(toks) > 2 * tgt_max_len:
                #     logging.info(
                #         "hn %s has len %d that is greater than max tgt len %d. "
                #         "It will be skipped ",
                #         hn_id,
                #         len(toks),
                #         tgt_max_len,
                #     )
                #     continue

                excluded_ids.add(hn_id)

                hn_idxs.append(len(examples) + len(hn_ids))
                hn_tokens.append(hn_toks)
                hn_ids.append(hn_id)
            hn_indices.append(hn_idxs)

        tgt_tokens = tgt_tokens + hn_tokens
        tgt_ids = src_ids + hn_ids
        self._adjust_batch_size(tgt_tokens, tgt_ids, excluded_ids, bucket)

        assert len(hn_indices) == len(src_tokens) == len(src_ids)

        # info = {'bs': len(src_ids)}
        # b = SentsBatch(src_ids, src_tokens, [], tgt_ids, tgt_tokens, [], hn_indices)
        labels = torch.arange(0, len(src_ids), dtype=torch.float32)

        pad_idx = self._tokenizer.pad_idx()
        b = SentsBatch(
            EncoderInData(
                SeqEncoderBatchedInput.from_input_ids(
                    self._enc_input_type, src_tokens, pad_idx, self._pad_opts, sorted_by_length=True
                ),
                src_ids,
                TextsRepr(TextReprType.SEQ_OF_TOKENS, src_tokens, []),
            ),
            EncoderInData(
                SeqEncoderBatchedInput.from_input_ids(
                    self._enc_input_type, tgt_tokens, pad_idx, self._pad_opts
                ),
                tgt_ids,
                TextsRepr(TextReprType.SEQ_OF_TOKENS, tgt_tokens, []),
            ),
            labels,
            hn_indices,
        )
        return b

    def _maybe_prepend_not_fitted(self, not_fitted, bucket: collections.deque, to_next_bucket):
        if not not_fitted:
            return
        # logging.debug("not_fitted: size %s", len(not_fitted))

        if not bucket:
            to_next_bucket.extend(not_fitted)
            return

        # if len(not_fitted) > self._conf.batch_size // 2:
        #     # logging.debug("not_fitted: move to the begining of the bucket")
        #     bucket.extendleft(reversed(not_fitted))
        #     self._sort_within_bucket(bucket)
        #     return
        # logging.error("NOT FITTED %s", [len(e.src) for e in not_fitted])
        # logging.error("BEFORE maybe prepend first %s", [len(e.src) for e in list(bucket)[:50]])

        if len(not_fitted[-1].src) >= len(bucket[0].src):
            bucket.extendleft(reversed(not_fitted))
            # logging.error(
            #     "AFTER SHORTCUT maybe prepend first %s", [len(e.src) for e in list(bucket)[:50]]
            # )
            return

        for e in reversed(not_fitted):
            if len(e.src) >= len(bucket[0].src):
                # logging.debug("not_fitted: appendleft")
                bucket.appendleft(e)
            else:
                # logging.debug("not_fitted: to next bucket")
                to_next_bucket.append(e)
        # logging.error("AFTER maybe prepend first %s", len(bucket[0].src))

    # def _is_batch_ready(self, examples: list[Example], cur_tokens_cnt:int):
    #     bs = len(examples)
    #     if bs >= self._conf.batch_size:
    #         return True

    #     if cur_tokens_cnt >=
    #     # we have to strive to keep batch size to be divisible by 8
    #     # for better gpu performance (see _adjust_batch_size)
    #     if bs % 8 == 0:
    #         # since target contains many more examples, we will estimate tokens cnt in target batch
    #         hn_per_example = len(examples[0].hns[1])
    #         # num tokens with padding in target batch
    #         tokens_cnt = bs * (hn_per_example + 1) * max_len
    #         if tokens_cnt >= self._conf.max_tokens:
    #             logging.info(
    #                 "batch is ready by tokens: bs=%s, ml=%s, hns=%s ", bs, max_len, hn_per_example
    #             )

    #             return True

    #     return False

    def _create_batches_from_bucket(self, initial_bucket: list[Example]):
        # There should be no fuzzy duplicates inside mini-batch.
        # So we should track id of dups of all examples added in minibatch so far.
        batch_dups: set[int] = set()
        examples: list[Example] = []
        not_fitted = []
        to_next_bucket = []
        batches = []
        bucket = collections.deque(initial_bucket)
        cur_sents_cnt = 0
        cur_tokens_cnt = 0
        while bucket:
            e = bucket.popleft()
            if e.src_id in batch_dups or any(i in batch_dups for i in e.dups):
                not_fitted.append(e)
                continue

            example_sents_cnt = 2 + len(e.hns[1])
            example_tokens_cnt = (
                len(e.src)
                + len(e.tgt)
                + sum(len(t) for i, t in enumerate(e.hns[0]) if e.hns[1][i] not in batch_dups)
            )

            if (
                len(examples) + 1 > self._conf.batch_size
                or cur_sents_cnt + example_sents_cnt > self._conf.max_sents
                or cur_tokens_cnt + example_tokens_cnt > self._conf.max_tokens
            ):
                b = self._make_batch(examples, batch_dups, initial_bucket)
                batches.append(b)

                not_fitted.append(e)
                self._maybe_prepend_not_fitted(not_fitted, bucket, to_next_bucket)
                batch_dups = set()
                examples = []
                not_fitted = []
                cur_sents_cnt = 0
                cur_tokens_cnt = 0
                continue

            batch_dups.add(e.src_id)
            batch_dups.update(e.dups)

            examples.append(e)
            cur_sents_cnt += example_sents_cnt
            cur_tokens_cnt += example_tokens_cnt

        if examples:
            b = self._make_batch(examples, batch_dups, initial_bucket)
            batches.append(b)
            self._maybe_prepend_not_fitted(not_fitted, bucket, to_next_bucket)

        return batches, to_next_bucket

    def _tokenize(self, text):
        sent = self._tokenizer(text, max_length=self._tokenizer.get_max_seq_length())
        return sent

    def _parse_line(self, line):
        line_id, text = line.split('\t', 1)
        return int(line_id), self._tokenize(text)

    def _parse_dups(self, line):
        line_id, dups_str = line.split('\t', 1)
        return int(line_id), [int(i) for i in dups_str.split()]

    def _read_hard_negatives(self, src_id):
        if self._hn_file is None:
            return [], []

        line = self._hn_last_line
        if not line:
            raise RuntimeError("Unexpected end of hard negatives file!")

        raw_hn_sents = []
        hn_ids = []

        def _fin_hns():
            if not raw_hn_sents:
                return [], []
            min_hn = min(self._conf.min_hn_cnt, len(raw_hn_sents))
            if min_hn != len(raw_hn_sents):
                hn_cnt = random.randint(min_hn, len(raw_hn_sents))
                hn_indexes = random.sample(range(len(raw_hn_sents)), hn_cnt)
            else:
                hn_indexes = range(min_hn)

            fin_hn_ids = []
            fin_hn_sents = []
            for hn_idx in hn_indexes:
                fin_hn_ids.append(hn_ids[hn_idx])
                fin_hn_sents.append(self._tokenize(raw_hn_sents[hn_idx].decode('utf8')))

            return fin_hn_sents, fin_hn_ids

        while line:
            t = line.rstrip().split(b'\t', 2)

            read_src_id = int(t[0])
            if read_src_id != src_id:
                self._hn_last_line = line
                return _fin_hns()

            if len(t) < 3:
                # there is no hard negatives for src_id
                # or
                # no tokens were found for this hn
                # nothing to do
                pass
            else:
                _, hn_id, hns_str = t
                hn_id = int(hn_id)
                hn_ids.append(hn_id)
                raw_hn_sents.append(hns_str)

            line = self._hn_file.readline()
        return _fin_hns()

    def _log_bucket_info(self, batches: list[SentsBatch]):
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            n = len(batches)
            logging.debug(
                'batches cnt %d, avg tgt cnt %d',
                n,
                sum(len(b.tgt_data.text_ids) for b in batches) / n,
            )

    def batches(self) -> Generator[SentsBatch, None, None]:
        if self._src_file is None or self._tgt_file is None or self._dup_file is None:
            raise RuntimeError("Files are not initialized")
        bucket_size = self._conf.batches_per_bucket * self._conf.batch_size

        bucket = []
        cnt = 0
        for s, t, dups in zip(self._src_file, self._tgt_file, self._dup_file):
            if cnt == self._line_cnt or (self._limit and self._line_num + cnt >= self._limit):
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
                logging.debug("bucket is full, prepare and create batches")
                self._sort_within_bucket(bucket)
                batches, to_next_bucket = self._create_batches_from_bucket(bucket)
                self._log_bucket_info(batches)
                logging.debug("%d examples left for next bucket", len(to_next_bucket))
                yield from batches
                logging.debug(
                    "start creating new bucket, with %d initial items", len(to_next_bucket)
                )
                bucket = to_next_bucket

        if bucket:
            self._sort_within_bucket(bucket)
            batches, to_next_bucket = self._create_batches_from_bucket(bucket)
            self._log_bucket_info(batches)
            yield from batches

            if to_next_bucket:
                logging.info("Failed to find a batch for %d examples", len(to_next_bucket))


@dataclass
class SentsBatchAsyncGeneratorConf(BaseBatchAsyncGeneratorConf):
    batch_generator_conf: SentsBatchGeneratorConf = MISSING


class SentsBatchAsyncGenerator(BaseBatchAsyncGenerator[SentsBatch]):
    _gen_cls = SentsBatchGenerator
    _name = "SentsBatchGenerator"

    def __init__(
        self,
        enc_input_type: EncoderInputType,
        conf: SentsBatchAsyncGeneratorConf,
        tok_conf: TokenizerConf,
        logging_conf,
        split,
        pad_opts: PadOpts = PadOpts(),
        rank=0,
        world_size=-1,
        max_seq_length: int | None = None,
    ):
        super().__init__(
            enc_input_type,
            conf,
            conf.batch_generator_conf.max_tokens,
            conf.batch_generator_conf.max_sents,
            logging_conf,
            (conf.batch_generator_conf, tok_conf, split, pad_opts),
            rank=rank,
            world_size=world_size,
            max_seq_length=max_seq_length,
        )

        self._opts = conf

        self._split = split

        self._epoch = 0

    def init_epoch(self, epoch, iter_no=1):
        self._epoch = epoch - 1
        src_fp = _tgt_filepath(self._opts.batch_generator_conf.input_dir, self._split)
        if not self._start_workers(
            src_fp,
            seed=10_000 * epoch + iter_no,
            limit=self._opts.batch_generator_conf.sents_limit,
        ):
            raise RuntimeError("Failed to init sents batch generator, empty folder or config error")

    def _prepare_batch(self, src_in: EncoderInData, tgt_in: EncoderInData, labels: torch.Tensor):
        return SentsBatch(src_in, tgt_in, labels, [])
