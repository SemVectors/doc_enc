#!/usr/bin/env python3


from doc_enc.tokenizer import AbcTokenizer
from doc_enc.training.models.model_conf import SentModelConf, DocModelConf, ModelKind

from doc_enc.encoders.enc_factory import create_sent_encoder, create_emb_seq_encoder, create_encoder
from doc_enc.encoders.sent_encoder import SentForDocEncoder

from doc_enc.training.models.sent_dual_enc import SentDualEncoder
from doc_enc.training.models.doc_dual_enc import DocDualEncoder


def _create_sent_model(conf: SentModelConf, vocab: AbcTokenizer):
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_sent_encoder(conf.encoder, vocab)
        model = SentDualEncoder(conf, encoder)
        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def create_models(conf: DocModelConf, vocab: AbcTokenizer):
    sent_model = _create_sent_model(conf.sent, vocab)
    sent_enc_for_doc = None
    if conf.sent_for_doc is not None:
        sent_enc_for_doc = create_encoder(conf.sent_for_doc)

    e = sent_model.encoder
    sent_encoder = SentForDocEncoder(
        e.conf,
        e.embed,
        e.encoder,
        emb_to_hidden_mapping=e.emb_to_hidden_mapping,
        doc_mode_encoder=sent_enc_for_doc,
        freeze_base_sents_layer=conf.freeze_base_sents_layer,
    )

    sent_embs_out_size = sent_model.encoder.out_embs_dim()

    frag_encoder = None
    if conf.fragment is not None:
        frag_encoder = create_emb_seq_encoder(conf.fragment, sent_embs_out_size)
        doc_input_size = frag_encoder.out_embs_dim()
    else:
        doc_input_size = sent_embs_out_size

    doc_encoder = create_emb_seq_encoder(conf.doc, doc_input_size)

    if conf.kind == ModelKind.DUAL_ENC:
        model = DocDualEncoder(
            conf,
            sent_encoder=sent_encoder,
            doc_encoder=doc_encoder,
            frag_encoder=frag_encoder,
        )
    else:
        raise RuntimeError(f"Unknown doc model kind {conf.kind}")

    return sent_model, model
