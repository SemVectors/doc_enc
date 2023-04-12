#!/usr/bin/env python3

import argparse

import logging
import os
import numpy as np

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf, file_path_fetcher


def _save(args, output_dir, ids, embs, batch_i):
    embs = np.vstack(embs)
    logging.info("Batch #%s; shape of vecs: %s", batch_i, embs.shape)
    dtype = np.float32
    if args.save_as_fp16:
        dtype = np.float16
    np.savez(
        os.path.join(output_dir, str(batch_i).zfill(4) + ".npz"),
        ids=ids,
        embs=embs.astype(dtype, copy=False),
    )


def _paths_gen(sent_file):
    with open(sent_file, 'r', encoding='utf8') as fp:
        for l in fp:
            yield l.rstrip()


def _compute_embs(args, doc_encoder: DocEncoder):
    output_dir_path = args.output_dir
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    batch_i = 0
    embs = []
    ids = []
    for batch_ids, batch_embs in doc_encoder.encode_docs_stream(
        _paths_gen(args.input_file), fetcher=file_path_fetcher, batch_size=args.batch_size
    ):
        embs.append(batch_embs)
        ids.extend(batch_ids)
        if len(ids) >= args.batch_size:
            _save(args, output_dir_path, ids, embs, batch_i)
            batch_i += 1
            embs = []
            ids = []
    if ids:
        _save(args, output_dir_path, ids, embs, batch_i)


def run_compute(args):
    conf = DocEncoderConf(
        model_path=args.model_path, use_gpu=args.gpu, max_sents=args.max_sents_per_batch
    )
    doc_encoder = DocEncoder(conf)
    _compute_embs(args, doc_encoder)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    doc_parser = subparsers.add_parser('docs', help='help of segment')

    doc_parser.add_argument(
        "--input_file", "-i", required=True, help="file with paths to segmented texts"
    )
    doc_parser.add_argument("--output_dir", "-o", required=True, help="")
    doc_parser.add_argument("--model_path", "-m", required=True, help="")
    doc_parser.add_argument("--gpu", "-g", default=0, help='GPU device number ')
    doc_parser.add_argument("--max_sents_per_batch", "-l", default=2048)
    doc_parser.add_argument("--batch_size", "-b", default=1000)
    doc_parser.add_argument("--save_as_fp16", default=False, action='store_true')
    doc_parser.add_argument("--verbose", "-v", action="store_true", help="")
    doc_parser.set_defaults(func=run_compute)

    args = parser.parse_args()

    FORMAT = "%(asctime)s %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=FORMAT)

    try:
        args.func(args)
    except Exception as e:
        logging.exception("failed to compute: %s ", e)


if __name__ == '__main__':
    main()
