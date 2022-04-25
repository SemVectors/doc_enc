#!/usr/bin/env python3


from doc_enc.training.models.model_conf import SentModelConf, DocModelConf, ModelKind

from doc_enc.encoders.sent_enc_factory import (
    create_sent_encoder,
    create_frag_encoder,
    create_doc_encoder,
)
from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.training.models.sent_dual_enc import SentDualEncoder
from doc_enc.training.models.doc_dual_enc import DocDualEncoder


def _create_sent_model(conf: SentModelConf, vocab_size, pad_idx):
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_sent_encoder(conf.encoder, vocab_size, pad_idx)
        split_target = isinstance(encoder, SentTransformerEncoder)
        model = SentDualEncoder(conf, encoder, split_target=split_target)
        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def create_model(conf: DocModelConf, vocab_size, pad_idx):
    sent_model = _create_sent_model(conf.sent, vocab_size, pad_idx)

    frag_encoder = None
    if conf.fragment is not None:
        frag_encoder = create_frag_encoder(conf.fragment)
    doc_encoder = create_doc_encoder(conf.doc)

    if conf.kind == ModelKind.DUAL_ENC:
        model = DocDualEncoder(
            conf, sent_model, frag_encoder=frag_encoder, doc_encoder=doc_encoder, pad_idx=pad_idx
        )
        return model
    raise RuntimeError(f"Unknown doc model kind {conf.kind}")
