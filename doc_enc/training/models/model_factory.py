#!/usr/bin/env python3

import torch

from doc_enc.tokenizer import AbcTokenizer
from doc_enc.training.models.model_conf import SentModelConf, DocModelConf, ModelKind

from doc_enc.encoders.enc_factory import create_sent_encoder, create_seq_encoder, create_encoder
from doc_enc.encoders.sent_encoder import SentForDocEncoder

from doc_enc.training.models.sent_dual_enc import SentDualEncoder
from doc_enc.training.models.doc_dual_enc import DocDualEncoder


def _create_sent_model(conf: SentModelConf, vocab: AbcTokenizer, device, state_dict=None):
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_sent_encoder(conf.encoder, vocab)
        if conf.load_params_from:
            state_dict = torch.load(conf.load_params_from, map_location=device)
        if state_dict is not None:
            encoder.load_state_dict(state_dict['sent_enc'])

        model = SentDualEncoder(conf, encoder)

        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def create_models(conf: DocModelConf, vocab: AbcTokenizer, device):
    state_dict = None
    if conf.load_params_from:
        state_dict = torch.load(conf.load_params_from, map_location=device)

    sent_model = None
    sent_layer = None
    sent_embs_out_size = 0
    if conf.sent is not None:
        sent_model = _create_sent_model(conf.sent, vocab, device, state_dict)

        sent_enc_for_doc = None
        if conf.sent_for_doc is not None:
            sent_enc_for_doc = create_encoder(conf.sent_for_doc)
            if state_dict is not None:
                sent_enc_for_doc.load_state_dict(state_dict['sent_for_doc'])

        e = sent_model.encoder
        sent_layer = SentForDocEncoder(
            e.conf,
            e.embed,
            e.encoder,
            emb_to_hidden_mapping=e.emb_to_hidden_mapping,
            pad_to_multiple_of=e.pad_to_multiple_of,
            doc_mode_encoder=sent_enc_for_doc,
            freeze_base_sents_layer=conf.freeze_base_sents_layer,
        )

        sent_embs_out_size = sent_model.encoder.out_embs_dim()

    frag_layer = None
    if conf.fragment is not None:
        frag_layer = create_seq_encoder(
            conf.fragment,
            pad_idx=vocab.pad_idx(),
            device=device,
            prev_output_size=sent_embs_out_size,
        )
        if state_dict is not None:
            frag_layer.load_state_dict(state_dict['frag_enc'])
        doc_input_size = frag_layer.out_embs_dim()
    else:
        doc_input_size = sent_embs_out_size

    doc_layer = create_seq_encoder(
        conf.doc, pad_idx=vocab.pad_idx(), device=device, prev_output_size=doc_input_size
    )
    if state_dict is not None:
        doc_layer.load_state_dict(state_dict['doc_enc'])

    if conf.kind == ModelKind.DUAL_ENC:
        model = DocDualEncoder(
            conf,
            sent_layer=sent_layer,
            doc_layer=doc_layer,
            frag_layer=frag_layer,
            pad_idx=vocab.pad_idx(),
            device=device,
        )
    else:
        raise RuntimeError(f"Unknown doc model kind {conf.kind}")

    return sent_model, model
