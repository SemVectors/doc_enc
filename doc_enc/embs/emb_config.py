#!/usr/bin/env python3


import dataclasses

from doc_enc.common_types import EmbKind


@dataclasses.dataclass
class BaseEmbConf:
    emb_kind: EmbKind
    emb_dim: int
