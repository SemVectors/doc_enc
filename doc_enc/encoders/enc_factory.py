#!/usr/bin/env python3


import logging


from doc_enc.tokenizer import AbcTokenizer
from doc_enc.common_types import EncoderKind
from doc_enc.encoders.enc_config import (
    BaseEncoderConf,
    SentEncoderConf,
    EmbSeqEncoderConf,
)

from doc_enc.embs.emb_factory import create_emb_layer
from doc_enc.encoders.sent_encoder import SentEncoder
from doc_enc.encoders.emb_seq_encoder import EmbSeqEncoder
from doc_enc.encoders.base_lstm import LSTMEncoder, GRUEncoder
from doc_enc.encoders.base_transformer import TransformerEncoder
from doc_enc.encoders.base_fnet import FNetEncoder
from doc_enc.encoders.base_longformer import LongformerEncoder


def _get_extra_padding(conf: BaseEncoderConf):
    pad_to_multiple_of = 0
    if conf.attention_window:
        pad_to_multiple_of = max(conf.attention_window)
    return pad_to_multiple_of


def create_encoder(conf: BaseEncoderConf):
    if conf.encoder_kind == EncoderKind.LSTM:
        return LSTMEncoder(conf)

    if conf.encoder_kind == EncoderKind.TRANSFORMER:
        return TransformerEncoder(conf)

    if conf.encoder_kind == EncoderKind.FNET:
        return FNetEncoder(conf)

    if conf.encoder_kind == EncoderKind.LONGFORMER:
        return LongformerEncoder(conf)

    if conf.encoder_kind == EncoderKind.GRU:
        return GRUEncoder(conf)

    raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")


def create_sent_encoder(conf: SentEncoderConf, vocab: AbcTokenizer) -> SentEncoder:
    if conf.emb_conf is None:
        raise RuntimeError("Specify emb configuration for sent encoder!")

    encoder = create_encoder(conf)
    embed = create_emb_layer(conf.emb_conf, vocab.vocab_size(), vocab.pad_idx())

    pad_to_multiple_of = _get_extra_padding(conf)
    return SentEncoder(conf, embed, encoder, pad_to_multiple_of=pad_to_multiple_of)


def create_emb_seq_encoder(conf: EmbSeqEncoderConf, sent_layer_output_size) -> EmbSeqEncoder:
    # conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    encoder = create_encoder(conf)

    pad_to_multiple_of = _get_extra_padding(conf)

    return EmbSeqEncoder(
        conf, encoder, sent_layer_output_size, pad_to_multiple_of=pad_to_multiple_of
    )
