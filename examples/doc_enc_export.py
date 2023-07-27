#!/usr/bin/env python3

import argparse
import logging

import torch


from doc_enc.encoders.pad_utils import create_padded_tensor

from doc_enc.training.models.model_conf import DocModelConf
from doc_enc.encoders.enc_factory import create_sent_encoder, create_seq_encoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoderConf
from doc_enc.text_processor import TextProcessor, TextProcessorConf


class EmbSeqPrepare(torch.nn.Module):
    def __init__(
        self,
        conf: SeqEncoderConf,
    ):
        super().__init__()

        input_size = conf.input_size if conf.input_size is not None else conf.hidden_size
        self._beg_seq_param = torch.nn.parameter.Parameter(torch.zeros(input_size))
        self._extra_len: int = 1

    def pad_embs_seq(self, embs: torch.Tensor, lengths: list[int]):
        emb_sz: int = embs.size(1)
        # pad sequence of embs
        max_len: int = max(lengths) + self._extra_len

        padded_seq = torch.zeros(
            (len(lengths) * max_len, emb_sz),
            device=embs.device,
            dtype=embs.dtype,
        )
        idx: list[list[int]] = []
        offs: int = 0 + self._extra_len
        for l in lengths:
            idx.extend([i] for i in range(offs, offs + l))
            offs += max_len
        idx_tensor = torch.tensor(idx, dtype=torch.int64, device=embs.device).expand(-1, emb_sz)
        padded_seq.scatter_(0, idx_tensor, embs)
        padded_seq[0 : padded_seq.size(0) : max_len] = self._beg_seq_param
        return padded_seq, max_len

    def create_key_padding_mask(self, max_len: int, src_lengths: list[int], device: torch.device):
        bs = len(src_lengths)
        int_tensor = torch.full([bs, max_len], 1, dtype=torch.int32, device=device)
        for i, l in enumerate(src_lengths):
            int_tensor[i, 0:l] = 0

        mask = int_tensor == 1
        return mask

    def create_padded_tensor(self, embs: torch.Tensor, lengths: list[int]):
        padded_seq, max_len = self.pad_embs_seq(embs, lengths)
        len_tensor = torch.as_tensor(lengths, dtype=torch.int64, device=embs.device)
        if self._extra_len:
            len_tensor += self._extra_len

        emb_sz = embs.size(1)
        seqs_tensor = padded_seq.reshape(len(lengths), max_len, emb_sz)
        key_padding_mask = self.create_key_padding_mask(max_len, lengths, device=embs.device)
        return seqs_tensor, len_tensor, key_padding_mask


class DocEncoderTS(torch.nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()

        state_dict = torch.load(model_path)
        tp_conf: TextProcessorConf = state_dict['tp_conf']
        tp_conf.tokenizer.vocab_path = None
        tp_state_dict = state_dict['tp']
        tp = TextProcessor(tp_conf, inference_mode=True)
        tp.load_state_dict(tp_state_dict)

        mc: DocModelConf = state_dict['model_conf']
        sent_enc = create_sent_encoder(mc.sent.encoder, tp.vocab())
        sent_enc.load_state_dict(state_dict['sent_enc'])
        sent_enc.eval()

        # tracing sent layer
        sents = ['I sing my song', 'It is simple and free'] * 32
        tokens, doc_fragments = tp.prepare_text(sents)
        sent_tensors, sent_lengths = self._create_input_tensor(
            [tokens], tp.vocab().pad_idx(), 'cpu'
        )
        self.sent_layer = torch.jit.trace(sent_enc, (sent_tensors, sent_lengths))

        sent_embs = sent_enc(sent_tensors, sent_lengths).pooled_out

        # init fragment layer

        self.fragment_prepare = None
        self.fragment_encoder = None
        sent_embs_out_size = sent_enc.out_embs_dim()
        if 'frag_enc' in state_dict and mc.fragment is not None:
            self.fragment_prepare = EmbSeqPrepare(mc.fragment)
            self.fragment_prepare.load_state_dict(
                {'_beg_seq_param': state_dict['frag_enc']['_beg_seq_param']}
            )
            frag_layer = create_seq_encoder(mc.fragment, sent_embs_out_size)
            frag_layer.load_state_dict(state_dict['frag_enc'])
            frag_layer.eval()

            # tracing fragment layer
            (
                padded_sents_embs,
                frag_len_tensor,
                key_padding_mask,
            ) = self.fragment_prepare.create_padded_tensor(sent_embs, doc_fragments)
            self.fragment_encoder = torch.jit.trace(
                frag_layer.encoder, (padded_sents_embs, frag_len_tensor, key_padding_mask)
            )

            doc_input_size = frag_layer.out_embs_dim()
            input_doc_layer_embs = frag_layer.encoder(
                padded_sents_embs, frag_len_tensor, key_padding_mask
            ).pooled_out
            input_doc_layer_lens = [len(doc_fragments)]
        else:
            doc_input_size = sent_embs_out_size
            input_doc_layer_embs = sent_embs
            input_doc_layer_lens = [len(tokens)]

        # init doc layer
        self.doc_prepare = EmbSeqPrepare(mc.doc)
        self.doc_prepare.load_state_dict(
            {'_beg_seq_param': state_dict['doc_enc']['_beg_seq_param']}
        )
        doc_layer = create_seq_encoder(mc.doc, doc_input_size)
        doc_layer.load_state_dict(state_dict['doc_enc'])
        doc_layer.eval()

        (
            padded_embs,
            len_tensor,
            key_padding_mask,
        ) = self.doc_prepare.create_padded_tensor(input_doc_layer_embs, input_doc_layer_lens)
        self.doc_encoder = torch.jit.trace(
            doc_layer.encoder, (padded_embs, len_tensor, key_padding_mask)
        )

    def _create_input_tensor(self, docs: list[list[list[int]]], pad_idx: int, device: str):
        # all_sents = [s for d in docs for s in d]
        all_sents: list[list[int]] = []
        for d in docs:
            for s in d:
                all_sents.append(s)

        # max_len = len(max(all_sents, key=lambda l: len(l)))
        max_len: int = 0
        for s in all_sents:
            max_len = max(max_len, len(s))

        sent_tensor, lengths_tensor = create_padded_tensor(
            all_sents,
            max_len,
            pad_idx=pad_idx,
            device=device,
            pad_to_multiple_of=0,
        )
        return sent_tensor, lengths_tensor

    def forward(
        self, docs: list[list[list[int]]], doc_fragments: list[list[int]], pad_idx: int, device: str
    ):
        sent_tensor, lengths_tensor = self._create_input_tensor(docs, pad_idx, device)
        sent_embs = self.sent_layer(sent_tensor, lengths_tensor)[0]

        if self.fragment_prepare is not None:
            frag_len: list[int] = []
            len_list: list[int] = []
            for fragments in doc_fragments:
                frag_len.extend(fragments)
                len_list.append(len(fragments))

            (
                padded_sents_embs,
                len_tensor,
                key_padding_mask,
            ) = self.fragment_prepare.create_padded_tensor(sent_embs, frag_len)
            embs = self.fragment_encoder(padded_sents_embs, len_tensor, key_padding_mask)[0]
        else:
            embs = sent_embs
            len_list: list[int] = [len(d) for d in docs]

        padded_embs, len_tensor, key_padding_mask = self.doc_prepare.create_padded_tensor(
            embs, len_list
        )

        doc_embs = self.doc_encoder(padded_embs, len_tensor, key_padding_mask)[0]
        return doc_embs


def export_ts_cli(opts):
    doc_enc_ts = DocEncoderTS(opts.input_model)
    sm = torch.jit.script(doc_enc_ts)
    print("exported model")
    print(sm)
    print(sm.code)

    sm.save(opts.output_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    subparsers = parser.add_subparsers(help='sub-command help')

    export_ts_parser = subparsers.add_parser('ts', help='help of export')

    export_ts_parser.add_argument("--input_model", "-i", required=True)
    export_ts_parser.add_argument("--output_model", "-o", required=True)
    export_ts_parser.set_defaults(func=export_ts_cli)

    args = parser.parse_args()

    FORMAT = "%(asctime)s %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=FORMAT)
    try:
        args.func(args)
    except Exception as e:
        logging.exception("failed to export_cli: %s ", e)


if __name__ == '__main__':
    main()
