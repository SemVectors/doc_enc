#!/usr/bin/env python3


from doc_enc.training.models.model_conf import SentModelConf, ModelConf, ModelKind

from doc_enc.encoders.sent_enc_factory import create_sent_encoder
from doc_enc.encoders.sent_transformer import SentTransformerEncoder
from doc_enc.training.models.sent_dual_enc import SentDualEncoder


def _create_sent_model(conf: SentModelConf, vocab_size, pad_idx):
    if conf.kind == ModelKind.DUAL_ENC:
        encoder = create_sent_encoder(conf.encoder, vocab_size, pad_idx)
        split_target = isinstance(encoder, SentTransformerEncoder)
        model = SentDualEncoder(conf, encoder, split_target=split_target)
        return model
    raise RuntimeError(f"Unknown model kind {conf.kind}")


def create_model(conf: ModelConf, vocab_size, pad_idx):
    return _create_sent_model(conf.sent, vocab_size, pad_idx)
