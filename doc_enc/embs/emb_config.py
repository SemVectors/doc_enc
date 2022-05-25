#!/usr/bin/env python3


import dataclasses

from doc_enc.common_types import EmbKind


@dataclasses.dataclass
class BaseEmbConf:
    emb_kind: EmbKind
    emb_dim: int

    scale_by_dim: bool = False

    normalize_emb: bool = False
    dropout: float = 0.0
