#!/usr/bin/env python3


import logging


from doc_enc.tokenizer import AbcTokenizer
from doc_enc.common_types import EncoderKind
from doc_enc.encoders.enc_config import (
    BaseEncoderConf,
    SentEncoderConf,
    SeqEncoderConf,
)

from doc_enc.embs.emb_factory import create_emb_layer
from doc_enc.encoders.sent_encoder import SentEncoder
from doc_enc.encoders.emb_seq_encoder import SeqEncoder
from doc_enc.encoders.base_lstm import LSTMEncoder, GRUEncoder
from doc_enc.encoders.base_transformer import TransformerEncoder
from doc_enc.encoders.base_fnet import FNetEncoder
from doc_enc.encoders.base_longformer import LongformerEncoder
from doc_enc.encoders.base_local_transformer import LocalAttnEncoder
from doc_enc.encoders.transformers_encoder import TransformersAutoModel, TransformersLongformer
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

    raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")


def create_sent_encoder(conf: SentEncoderConf, vocab: AbcTokenizer) -> SentEncoder:
    if conf.emb_conf is None:
        raise RuntimeError("Specify emb configuration for sent encoder!")

    encoder = create_encoder(conf)
    embed = create_emb_layer(conf.emb_conf, vocab.vocab_size(), vocab.pad_idx())

    pad_to_multiple_of = _get_extra_padding(conf)
    return SentEncoder(conf, embed, encoder, pad_to_multiple_of=pad_to_multiple_of)


def create_seq_encoder(conf: SeqEncoderConf, prev_output_size, pad_idx: int, device) -> SeqEncoder:
    # conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    encoder = create_encoder(conf)
    pad_to_multiple_of = _get_extra_padding(conf)

    return SeqEncoder(
        conf,
        encoder,
        pad_idx=pad_idx,
        device=device,
        prev_output_size=prev_output_size,
        pad_to_multiple_of=pad_to_multiple_of,
    )
