#!/usr/bin/env python3


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
from doc_enc.encoders.base_lstm import LSTMEncoder
from doc_enc.encoders.base_transformer import BaseTransformerEncoder
from doc_enc.encoders.base_fnet import BaseFNetEncoder


def create_encoder(conf: BaseEncoderConf):
    if conf.encoder_kind == EncoderKind.LSTM:
        return LSTMEncoder(conf)

    if conf.encoder_kind == EncoderKind.TRANSFORMER:
        return BaseTransformerEncoder(conf)

    if conf.encoder_kind == EncoderKind.FNET:
        return BaseFNetEncoder(conf)

    raise RuntimeError(f"Unsupported encoder kind: {conf.encoder_kind}")


def create_sent_encoder(conf: SentEncoderConf, vocab: AbcTokenizer):
    if conf.emb_conf is None:
        raise RuntimeError("Specify emb configuration for sent encoder!")

    encoder = create_encoder(conf)
    embed = create_emb_layer(conf.emb_conf, vocab.vocab_size(), vocab.pad_idx())

    return SentEncoder(conf, embed, encoder)


def create_emb_seq_encoder(conf: EmbSeqEncoderConf, sent_layer_output_size):
    # conf_dict: Dict[str, Any] = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)
    encoder = create_encoder(conf)

    return EmbSeqEncoder(conf, encoder, sent_layer_output_size)
