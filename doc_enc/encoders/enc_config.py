#!/usr/bin/env python3

from enum import Enum
from typing import Any, Optional, List, Dict
import dataclasses


from doc_enc.common_types import EncoderKind
from doc_enc.encoders.base_pooler import BasePoolerConf
from doc_enc.encoders.enc_in import EncoderInputType


class LookAroundMode(Enum):
    NONE = 0
    BACK = 1
    FORWARD = 2
    BACK_AND_FORWARD = 3


@dataclasses.dataclass
class BaseEncoderConf:
    encoder_kind: EncoderKind
    hidden_size: int
    num_layers: int
    pooler: BasePoolerConf
    dropout: float = 0.0

    # If not set will be selected default one that is specific for an encoder.
    input_type: Optional[EncoderInputType] = None

    # lstm opts
    input_size: Optional[int] = None
    bidirectional: Optional[bool] = None
    proj_size: Optional[int] = None
    # transformer opts
    num_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    intermediate_activation: Optional[str] = None
    full_intermediate: bool = True
    share_attn: bool = True
    # longformer and local transformer
    attention_window: List[int] = dataclasses.field(default_factory=list)
    window_look_around_mode: LookAroundMode = LookAroundMode.BACK

    # transformers
    transformers_auto_name: str = ''
    transformers_cache_dir: str | None = None
    transformers_fix_pretrained_params: bool = False
    transformers_torch_fp16: bool = False
    transformers_kwargs: Optional[Dict[str, Any]] = None
    # might be used to override default pool strategy: mean/first/last/max.
    transformers_pooler: str = 'auto'
    # pad options
    left_padding: bool = False

    # supported values: lora
    use_adapter: str = ''
    adapter_kwargs: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class SeqEncoderConf(BaseEncoderConf):
    add_beg_seq_token: bool = False
    input_dropout: float = 0.0
    add_pos_emb: bool = False
