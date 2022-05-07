#!/usr/bin/env python3

from typing import Dict, Any

from omegaconf import OmegaConf

from doc_enc.encoders.enc_config import (
    SentEncoderConf,
    FragmentEncoderConf,
    DocEncoderConf,
    EncoderKind,
)

from doc_enc.encoders.base_lstm import LSTMEncoder
from doc_enc.encoders.sent_lstm import SentLSTMEncoder
from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.encoders.sent_fnet import SentFNetEncoder


def create_sent_encoder(conf: SentEncoderConf, vocab_size, pad_idx):
    conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = SentLSTMEncoder(vocab_size, pad_idx, **conf_dict)

    elif conf.encoder_kind == EncoderKind.TRANSFORMER:
        encoder = SentTransformerEncoder(vocab_size, pad_idx, **conf_dict)

    elif conf.encoder_kind == EncoderKind.FNET:
        encoder = SentFNetEncoder(vocab_size, pad_idx, **conf_dict)

    else:
        raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")

    return encoder


def create_frag_encoder(conf: FragmentEncoderConf):
    conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = LSTMEncoder(**conf_dict)
    else:
        raise RuntimeError(f"Unsupported fragment encoder kind: {conf.encoder_kind}")

    return encoder


def create_doc_encoder(conf: DocEncoderConf):
    conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    if conf.encoder_kind == EncoderKind.LSTM:
        encoder = LSTMEncoder(**conf_dict)
    else:
        raise RuntimeError(f"Unsupported doc encoder kind: {conf.encoder_kind}")

    return encoder
