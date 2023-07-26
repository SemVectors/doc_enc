#!/usr/bin/env python3

from omegaconf import OmegaConf
import torch

from doc_enc.common_types import EncoderKind
from doc_enc.embs.emb_factory import create_emb_layer
from doc_enc.encoders.enc_config import BaseEncoderConf

from doc_enc.tokenizer import AbcTokenizer
from doc_enc.training.models.model_conf import SentModelConf, DocModelConf, ModelKind

from doc_enc.encoders.enc_factory import create_seq_encoder, create_encoder
from doc_enc.encoders.sent_encoder import SentForDocEncoder

from doc_enc.training.models.sent_dual_enc import SentDualEncoder
from doc_enc.training.models.doc_dual_enc import DocDualEncoder


def _create_sent_model(
    conf: SentModelConf, vocab: AbcTokenizer, device, embed=None, state_dict=None
):
    emb_dim = 0 if embed is None else embed.conf.emb_dim
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_seq_encoder(
            conf.encoder,
            prev_output_size=emb_dim,
        )
        if conf.load_params_from:
            state_dict = torch.load(conf.load_params_from, map_location=device)
        if state_dict is not None:
            encoder.load_state_dict(state_dict['sent_enc'])
            if embed is not None and 'embed' in state_dict:
                embed.load_state_dict(state_dict['embed'])

        model = SentDualEncoder(conf, encoder, embed=embed, pad_idx=vocab.pad_idx(), device=device)

        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def _check_enc_layer_correctness(enc_config: BaseEncoderConf, prev_output_size):
    if enc_config.encoder_kind == EncoderKind.AVERAGING:
        if not prev_output_size:
            raise RuntimeError("Averaging encoder can't be the first one!")

        if OmegaConf.is_missing(enc_config, 'hidden_size'):
            enc_config.hidden_size = prev_output_size


def create_models(conf: DocModelConf, vocab: AbcTokenizer, device):
    state_dict = None
    if conf.load_params_from:
        state_dict = torch.load(conf.load_params_from, map_location=device)

    embed = None
    if conf.embed is not None:
        embed = create_emb_layer(conf.embed, vocab.vocab_size(), vocab.pad_idx())

    sent_model = None
    sent_layer = None
    sent_embs_out_size = 0
    if conf.sent is not None:
        sent_model = _create_sent_model(
            conf.sent, vocab, device, embed=embed, state_dict=state_dict
        )

        sent_enc_for_doc = None
        if conf.sent_for_doc is not None:
            sent_enc_for_doc = create_encoder(conf.sent_for_doc)
            if state_dict is not None:
                sent_enc_for_doc.load_state_dict(state_dict['sent_for_doc'])

        e = sent_model.sent_layer
        assert e is not None, "Logic error 5899"
        sent_layer = SentForDocEncoder(
            e.conf,
            e.encoder,
            pad_to_multiple_of=e.pad_to_multiple_of,
            doc_mode_encoder=sent_enc_for_doc,
            freeze_base_sents_layer=conf.freeze_base_sents_layer,
        )
        sent_layer.emb_to_hidden_mapping = e.emb_to_hidden_mapping
        sent_embs_out_size = e.out_embs_dim()

    frag_layer = None
    if conf.fragment is not None:
        _check_enc_layer_correctness(conf.fragment, sent_embs_out_size)

        frag_layer = create_seq_encoder(
            conf.fragment,
            prev_output_size=sent_embs_out_size,
        )
        if state_dict is not None:
            frag_layer.load_state_dict(state_dict['frag_enc'])
        doc_input_size = frag_layer.out_embs_dim()
    else:
        doc_input_size = sent_embs_out_size

    _check_enc_layer_correctness(conf.doc, doc_input_size)
    doc_layer = create_seq_encoder(conf.doc, prev_output_size=doc_input_size)
    if state_dict is not None:
        doc_layer.load_state_dict(state_dict['doc_enc'])

    if conf.kind == ModelKind.DUAL_ENC:
        model = DocDualEncoder(
            conf,
            embed=embed,
            sent_layer=sent_layer,
            doc_layer=doc_layer,
            frag_layer=frag_layer,
            pad_idx=vocab.pad_idx(),
            device=device,
        )
    else:
        raise RuntimeError(f"Unknown doc model kind {conf.kind}")

    return sent_model, model
