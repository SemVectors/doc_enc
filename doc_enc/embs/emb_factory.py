#!/usr/bin/env python3

from doc_enc.common_types import EmbKind
from doc_enc.embs.emb_config import BaseEmbConf
from doc_enc.embs.token_embed import TokenEmbedding
from doc_enc.embs.token_with_positional import TokenWithPositionalEmbedding


def create_emb_layer(conf: BaseEmbConf, vocab_size, pad_idx):

    if conf.emb_kind == EmbKind.TOKEN:
        emb = TokenEmbedding(conf.emb_dim, vocab_size, pad_idx)
    elif conf.emb_kind == EmbKind.TOKEN_WITH_POSITIONAL:
        emb = TokenWithPositionalEmbedding(conf.emb_dim, vocab_size, pad_idx)

    else:
        raise RuntimeError(f"Unsupported emb kind: {conf.emb_kind}")
    return emb
