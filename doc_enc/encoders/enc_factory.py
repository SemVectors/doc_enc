#!/usr/bin/env python3


import logging


from doc_enc.common_types import EncoderKind
from doc_enc.encoders.enc_config import (
    BaseEncoderConf,
    SeqEncoderConf,
)

from doc_enc.encoders.seq_encoder import SeqEncoder
from doc_enc.encoders.base_lstm import LSTMEncoder, GRUEncoder
from doc_enc.encoders.base_transformer import TransformerEncoder
from doc_enc.encoders.base_fnet import FNetEncoder
from doc_enc.encoders.base_longformer import LongformerEncoder
from doc_enc.encoders.base_local_transformer import LocalAttnEncoder
from doc_enc.encoders.transformers_encoder import (
    TransformersAutoModel,
    TransformersLongformer,
    SbertAutoModel,
)
from doc_enc.encoders.averaging_encoder import AveragingEncoder


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

    if conf.encoder_kind == EncoderKind.LOCAL_ATTN_TRANSFORMER:
        return LocalAttnEncoder(conf)

    if conf.encoder_kind == EncoderKind.TRANSFORMERS_AUTO:
        if 'longformer' in conf.transformers_auto_name:
            return TransformersLongformer(conf)
        return TransformersAutoModel(conf)
    if conf.encoder_kind == EncoderKind.AVERAGING:
        return AveragingEncoder(conf)
    if conf.encoder_kind == EncoderKind.SBERT_AUTO:
        return SbertAutoModel(conf)

    raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")


def create_seq_encoder(conf: SeqEncoderConf, prev_output_size) -> SeqEncoder:
    # conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    encoder = create_encoder(conf)
    pad_to_multiple_of = _get_extra_padding(conf)

    return SeqEncoder(
        conf,
        encoder,
        prev_output_size=prev_output_size,
        pad_to_multiple_of=pad_to_multiple_of,
    )
