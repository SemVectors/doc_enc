#!/usr/bin/env python3

import argparse

import logging
import os
import numpy as np
import torch

from doc_enc.doc_encoder import DocEncoder, DocEncoderConf, file_path_fetcher, TextProcOverride, ConfOverrides
from doc_enc.doc_classifier import DocClassifier

from doc_enc.utils import global_init


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
        for ll in fp:
            yield ll.rstrip()


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

def _create_doc_enc_conf(args):
    conf = DocEncoderConf(
        model_path=args.model_path,
        use_gpu=args.gpu,
        max_sents=args.max_sents_per_batch,
        max_tokens=args.max_tokens_per_batch,
        enable_amp=args.enable_amp,
        bucket_multiplier=args.bucket_multiplier,
        async_batch_gen=args.async_batch_gen,
        ensure_flash_attn = args.ensure_flash_attn
    )
    if args.max_seq_len != -1:
        conf.overrides = ConfOverrides(text_proc=TextProcOverride(args.max_seq_len))
    
    return conf

def run_compute_embs(args):
    conf = _create_doc_enc_conf(args)
    doc_encoder = DocEncoder(conf)
    _compute_embs(args, doc_encoder)


def run_compute_sent_embs(args):
    conf = _create_doc_enc_conf(args)
    doc_encoder = DocEncoder(conf)

    output_dir_path = args.output_dir
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    batch_i = 0
    embs = []
    ids = []
    for batch_ids, batch_embs in doc_encoder.generate_sent_embs_from_file(
        args.input_file, first_column_is_id=args.first_column_is_id, sep=args.column_sep
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


def run_classif(args):
    conf = _create_doc_enc_conf(args)
    doc_encoder = DocClassifier(conf, topk=args.topk, threshold=args.threshold)
    with open(args.output_file, 'w', encoding='utf8') as outf:
        for batch_paths, predictions in doc_encoder.clsf_docs_stream(
            _paths_gen(args.input_file), fetcher=file_path_fetcher, batch_size=args.batch_size
        ):
            for path, preds in zip(batch_paths, predictions):

                for lbl, weight in sorted(preds, key=lambda t: -t[1]):
                    outf.write(f'{path}\t{lbl}\t{weight}\n')


def merge_checkpoints(args):
    def _zero_out(prms):
        for v in prms.values():
            if isinstance(v, dict):
                _zero_out(v)
            else:
                v.fill_(0)

    def _merge(prms, ckp_w, out_prms, seen=None):
        if seen is None:
            seen = set()
        for k in out_prms.keys():
            if (v := prms.get(k)) is None:
                raise RuntimeError(f'Checkpoint {ckp_path} has no {k} parameter.')
            if isinstance(v, dict):
                _merge(v, ckp_w, out_prms[k], seen)
            else:
                # There might be shared tensor. Update them only once.
                if v.data_ptr() not in seen:
                    out_prms[k] += ckp_w * v
                    seen.add(v.data_ptr())

    ckp_file_paths: list[str] = args.ckp_files
    weights: list[float] = args.ckp_weights
    if not weights:
        weights = [1 / len(ckp_file_paths)] * len(ckp_file_paths)
    elif len(weights) != len(ckp_file_paths):
        raise RuntimeError('Number of specified weights != number of checkpoints!')

    device = torch.device('cpu')
    merged_state = torch.load(ckp_file_paths[0], map_location=device, weights_only=False)
    mod_params = merged_state['model']
    _zero_out(mod_params)

    for ckp_path, ckp_w in zip(ckp_file_paths, weights):
        ckp_state = torch.load(ckp_path, map_location=device, weights_only=False)
        cw = ckp_state['model']
        _merge(cw, ckp_w, mod_params)

    conf = DocEncoderConf(args.base_model)
    doc_enc = DocEncoder(conf)
    doc_enc.load_params_from_checkpoint(merged_state)

    d = doc_enc.enc_module().to_dict()
    torch.save(d, args.output_file)


def _add_common_opts(parser):
    parser.add_argument("--model_path", "-m", required=True, help="")
    parser.add_argument(
        "--gpu", "-g", default=0, help='GPU device number, pass -1 to use CPU.', type=int
    )
    parser.add_argument("--max_sents_per_batch", "-ms", default=2048, type=int)
    parser.add_argument("--max_tokens_per_batch", "-mt", default=128_000, type=int)
    parser.add_argument("--batch_size", "-b", default=1000, type=int)
    parser.add_argument("--enable_amp", "-amp", default=False, action='store_true')
    parser.add_argument("--async_batch_gen", default=2, type=int)
    parser.add_argument("--bucket_multiplier", default=2, type=int)
    parser.add_argument("--max_seq_len", default=-1, type=int)
    parser.add_argument("--ensure_flash_attn", default=False, action='store_true')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="")

    subparsers = parser.add_subparsers(help='sub-command help')

    doc_parser = subparsers.add_parser('docs', help='compute embeddings of documents')
    doc_parser.add_argument(
        "--input_file", "-i", required=True, help="file with paths to segmented texts"
    )
    doc_parser.add_argument("--output_dir", "-o", required=True, help="")
    _add_common_opts(doc_parser)
    doc_parser.add_argument("--save_as_fp16", "-fp16", default=False, action='store_true')
    doc_parser.set_defaults(func=run_compute_embs)

    sent_parser = subparsers.add_parser('sents', help='compute embeddings of sentences')

    sent_parser.add_argument(
        "--input_file", "-i", required=True, help="File with each sentencen on the new line"
    )
    sent_parser.add_argument("--output_dir", "-o", required=True, help="")
    _add_common_opts(sent_parser)
    sent_parser.add_argument("--first_column_is_id", default=False, action='store_true')
    sent_parser.add_argument("--column_sep", default='\t')
    sent_parser.add_argument("--save_as_fp16", "-fp16", default=False, action='store_true')
    sent_parser.set_defaults(func=run_compute_sent_embs)

    classif_parser = subparsers.add_parser('classif', help='Do classification of documents')
    classif_parser.add_argument(
        "--input_file", "-i", required=True, help="file with paths to texts."
    )
    classif_parser.add_argument("--output_file", "-o", required=True, help="")
    _add_common_opts(classif_parser)
    classif_parser.add_argument("--topk", "-k", default=None, type=int)
    classif_parser.add_argument("--threshold", "-t", default=None, type=float)
    classif_parser.set_defaults(func=run_classif)

    merge_ckp_parser = subparsers.add_parser('merge_checkpoints', help='Merge various checkpoints')

    merge_ckp_parser.add_argument(
        "--base_model",
        "-b",
        required=True,
        help=(
            "Base model path. This model is used only for copying metadata/configs to the new model."
            " Weight only loaded from checkpoints, weight from this model are not used."
        ),
    )
    merge_ckp_parser.add_argument(
        "--ckp_files", "-i", required=True, nargs='+', help="Checkpoint paths."
    )
    merge_ckp_parser.add_argument(
        "--ckp_weights",
        "-w",
        nargs='*',
        type=float,
        help="Checkpoints weights. By default 1/n where n is the number of checkpoints.",
    )

    merge_ckp_parser.add_argument(
        "--output_file", "-o", required=True, help="Path to the new model with merged weights."
    )
    merge_ckp_parser.set_defaults(func=merge_checkpoints)

    args = parser.parse_args()

    FORMAT = "%(asctime)s %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=FORMAT)

    try:
        args.func(args)
    except Exception as e:
        logging.exception("failed to compute: %s ", e)


if __name__ == '__main__':
    global_init()
    main()
