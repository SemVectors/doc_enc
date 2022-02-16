#!/usr/bin/env python3

from typing import Dict, Any
from doc_enc.encoders.enc_config import SentEncoderConf, SentEncoderKind
from doc_enc.encoders.sent_lstm import SentLSTMEncoder
from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.encoders.sent_fnet import SentFNetEncoder

from omegaconf import OmegaConf


def create_sent_encoder(conf: SentEncoderConf, vocab_size, pad_idx):
    conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    if conf.encoder_kind == SentEncoderKind.LSTM:
        encoder = SentLSTMEncoder(vocab_size, pad_idx, **conf_dict)

    elif conf.encoder_kind == SentEncoderKind.TRANSFORMER:
        encoder = SentTransformerEncoder(vocab_size, pad_idx, **conf_dict)

    elif conf.encoder_kind == SentEncoderKind.FNET:
        encoder = SentFNetEncoder(vocab_size, pad_idx, **conf_dict)

    else:
        raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")

    return encoder
