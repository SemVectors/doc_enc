#!/usr/bin/env python3


from doc_enc.tokenizer import AbcTokenizer
from doc_enc.common_types import EncoderKind
from doc_enc.encoders.enc_config import (
    SentEncoderConf,
    FragmentEncoderConf,
    DocEncoderConf,
)

from doc_enc.embs.emb_factory import create_emb_layer
from doc_enc.encoders.sent_encoder import SentEncoder
from doc_enc.encoders.frag_encoder import FragEncoder
from doc_enc.encoders.doc_encoder import DocEncoder
from doc_enc.encoders.base_lstm import LSTMEncoder
from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.encoders.sent_fnet import SentFNetEncoder


def create_sent_encoder(conf: SentEncoderConf, vocab: AbcTokenizer):
    if conf.emb_conf is None:
        raise RuntimeError("Specify emb configuration for sent encoder!")

    embed = create_emb_layer(conf.emb_conf, vocab.vocab_size(), vocab.pad_idx())

    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = LSTMEncoder(conf)

    elif conf.encoder_kind == EncoderKind.TRANSFORMER:
        encoder = SentTransformerEncoder(conf_dict)

    elif conf.encoder_kind == EncoderKind.FNET:
        encoder = SentFNetEncoder(conf_dict)

    else:
        raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")

    return SentEncoder(embed, encoder)


def create_frag_encoder(conf: FragmentEncoderConf, sent_layer_output_size):
    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = LSTMEncoder(conf)
    else:
        raise RuntimeError(f"Unsupported fragment encoder kind: {conf.encoder_kind}")

    return FragEncoder(conf, encoder, sent_layer_output_size)


def create_doc_encoder(conf: DocEncoderConf, prev_layer_output_size):
    # conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = LSTMEncoder(conf)
    else:
        raise RuntimeError(f"Unsupported doc encoder kind: {conf.encoder_kind}")

    return DocEncoder(conf, encoder, prev_layer_output_size)