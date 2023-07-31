#!/usr/bin/env python3

import argparse
import logging

import numpy as np
import faiss

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf


def _create_mapping(opts):
    mapping = {}
    with open(opts.mapping_file, encoding='utf8') as f:
        prev_src = None
        gold_tgt = []
        for l in f:
            src, tgt = l.split(',')
            src, tgt = int(src), int(tgt)

            if src != prev_src:
                if gold_tgt:
                    mapping[prev_src] = gold_tgt
                    gold_tgt = []
                prev_src = src
            gold_tgt.append(tgt)
    return mapping


def _create_found_list(dists, idxs):
    found_list = []
    for sim, found_id in zip(dists, idxs):
        found_list.append((found_id, sim))
    found_list.sort(key=lambda t: -t[1])
    return found_list


def run_eval_index_cli(args):
    conf = DocEncoderConf(
        model_path=str(args.model_path), use_gpu=args.use_gpu, max_sents=args.max_sents
    )
    doc_encoder = DocEncoder(conf)

    index = faiss.read_index(args.index)
    # https://github.com/facebookresearch/faiss/blob/main/faiss/AutoTune.cpp
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)

    mapping = None
    if args.mapping_file:
        mapping = _create_mapping(args)

    def _get_gold_set(src_id):
        if mapping is None:
            return [src_id]
        if src_id not in mapping:
            logging.error("No gold data for %s", src_id)
            return []
        return mapping[src_id]

    levels = [int(l) for l in args.levels]
    scores = [0] * len(levels)
    total = 0
    gen = doc_encoder.generate_sent_embs_from_file(
        args.input_file, lines_limit=args.lines_limit, first_column_is_id=True
    )
    for ids, embs in gen:
        ids = [int(i) for i in ids]
        embs = embs.astype(np.float32, copy=False)
        faiss.normalize_L2(embs)
        D, I = index.search(embs, args.topk)

        for query_id, dists, idxs in zip(ids, D, I):
            query_id = int(query_id)
            found_list = _create_found_list(dists, idxs)

            gold = _get_gold_set(query_id)
            total += len(gold)

            for i, l in enumerate(levels):
                found_top = [f for f, _ in found_list[:l]]
                cnt = sum(1 for f in found_top if f in gold)
                scores[i] += cnt

    for i, l in enumerate(levels):
        s = scores[i] / total
        print(f"Rec@{l}: {s*100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True, help="")
    parser.add_argument("--index", "-idx", required=True)
    parser.add_argument("--input_file", "-i", required=True, help="")
    parser.add_argument("--lines_limit", "-l", default=0, type=int)
    parser.add_argument("--nprobe", default=16, type=int)
    parser.add_argument("--topk", "-k", default=30, type=int)

    parser.add_argument("--use_gpu", "-g", default=0)
    parser.add_argument("--max_sents", default=8192, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")

    parser.add_argument("--levels", "-L", default=[1, 5, 30], nargs="*")
    parser.add_argument(
        "--mapping_file",
        "-mapf",
        default='',
        help="if mapping file is not provided than target id should be equal to source id",
    )

    parser.set_defaults(func=run_eval_index_cli)

    args = parser.parse_args()

    FORMAT = "%(asctime)s %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=FORMAT)

    try:
        args.func(args)
    except Exception as e:
        logging.exception("failed to index: %s ", e)


if __name__ == '__main__':
    main()
