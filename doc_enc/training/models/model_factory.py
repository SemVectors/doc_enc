#!/usr/bin/env python3


from doc_enc.tokenizer import AbcTokenizer
from doc_enc.training.models.model_conf import SentModelConf, DocModelConf, ModelKind

from doc_enc.encoders.enc_factory import (
    create_sent_encoder,
    create_emb_seq_encoder,
)
from doc_enc.training.models.sent_dual_enc import SentDualEncoder
from doc_enc.training.models.doc_dual_enc import DocDualEncoder


def _create_sent_model(conf: SentModelConf, vocab: AbcTokenizer):
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_sent_encoder(conf.encoder, vocab)
        model = SentDualEncoder(conf, encoder, pad_idx=vocab.pad_idx())
        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def create_model(conf: DocModelConf, vocab: AbcTokenizer):
    sent_model = _create_sent_model(conf.sent, vocab)

    frag_encoder = None
    sent_embs_out_size = sent_model.encoder.out_embs_dim()
    if conf.fragment is not None:
        frag_encoder = create_emb_seq_encoder(conf.fragment, sent_embs_out_size)
        doc_input_size = frag_encoder.out_embs_dim()
    else:
        doc_input_size = sent_embs_out_size

    doc_encoder = create_emb_seq_encoder(conf.doc, doc_input_size)

    if conf.kind == ModelKind.DUAL_ENC:
        model = DocDualEncoder(
            conf,
            sent_model,
            frag_encoder=frag_encoder,
            doc_encoder=doc_encoder,
            pad_idx=vocab.pad_idx(),
        )
        return model
    raise RuntimeError(f"Unknown doc model kind {conf.kind}")
