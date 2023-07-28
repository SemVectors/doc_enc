#!/usr/bin/env python3

import torch

from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutput
from sentence_transformers import SentenceTransformer

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_pooler import BasePoolerConf

from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_out import BaseEncoderOut


class BaseTransformersAutoModel(BaseEncoder):
    def __init__(self, config: BaseEncoderConf, auto_model) -> None:
        super().__init__()
        self.config = config

        self.auto_model = auto_model

        auto_config = self._get_auto_config()
        # set missing params for config
        # It is required to be able to save this config later
        config.hidden_size = auto_config.hidden_size
        config.num_layers = auto_config.num_hidden_layers
        config.num_heads = auto_config.num_attention_heads
        config.pooler = BasePoolerConf(pooling_strategy=PoolingStrategy.UNDEFINED)

    def _get_auto_config(self):
        raise NotImplementedError("Impl in subclass")

    def out_embs_dim(self) -> int:
        return self._get_auto_config().hidden_size

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = src_lengths.shape[0]
        mask = torch.full((bs, max_len), 0, dtype=torch.float, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = 1

        return mask

    def state_dict(self, *args, **kwargs):
        # do not save state when params are fixed
        # if finetuning save state_dict
        if self.config.transformers_fix_pretrained_params:
            return {}
        prefix = kwargs.get('prefix', '')
        kwargs['prefix'] = prefix + 'auto_model.'
        return self.auto_model.state_dict(*args, **kwargs)


class TransformersAutoModel(BaseTransformersAutoModel):
    def __init__(self, config: BaseEncoderConf) -> None:
        auto_model = AutoModel.from_pretrained(
            config.transformers_auto_name, cache_dir=config.transformers_cache_dir
        )

        super().__init__(config, auto_model)

    def _get_auto_config(self):
        return self.auto_model.config

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        transformers_kwargs: dict | None = None,
        **kwargs,
    ):
        if input_token_ids is None and input_embs is None:
            raise RuntimeError("pass either input_embs or input_token_ids")

        if lengths is None:
            raise RuntimeError("pass lengths as input")

        # input shape: batch_sz, seq_len, hidden_dim?

        t = input_token_ids if input_token_ids is not None else input_embs
        assert t is not None, "stupid pyright"
        max_len = t.shape[1]

        attention_mask = self._create_key_padding_mask(max_len, lengths, t.device)

        if transformers_kwargs is None:
            transformers_kwargs = {}

        result = self.auto_model(
            input_ids=input_token_ids,
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            **transformers_kwargs,
        )
        if (pooled_out := getattr(result, 'pooler_output', None)) is None:
            if isinstance(result, BaseModelOutput):
                hidden_states = result[0]
                pooled_out = hidden_states[:, 0]
            else:
                raise RuntimeError(f"Unsupported result type from transformers lib {type(result)} ")

        return BaseEncoderOut(pooled_out, result.hidden_states, lengths)


class TransformersLongformer(TransformersAutoModel):
    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        transformers_kwargs: dict | None = None,
        **kwargs,
    ):
        t = None
        if input_token_ids is not None:
            t = input_token_ids
            shape = t.shape
        elif input_embs is not None:
            t = input_embs
            shape = t.shape[:-1]
        else:
            raise RuntimeError("pass either input_embs or input_token_ids")

        global_attention_mask = torch.zeros(*shape, dtype=torch.float, device=t.device)
        # global attention on cls token
        global_attention_mask[:, 0] = 1

        if transformers_kwargs is None:
            transformers_kwargs = {}
        transformers_kwargs['global_attention_mask'] = global_attention_mask
        return super().forward(
            input_embs=input_embs,
            input_token_ids=input_token_ids,
            lengths=lengths,
            transformers_kwargs=transformers_kwargs,
            **kwargs,
        )


class SbertAutoModel(BaseTransformersAutoModel):
    def __init__(self, config: BaseEncoderConf) -> None:
        auto_sbert = SentenceTransformer(
            config.transformers_auto_name, cache_folder=config.transformers_cache_dir
        )
        super().__init__(config, auto_sbert)

    def _get_auto_config(self):
        trans = self.auto_model[0]
        if not hasattr(trans, 'auto_model'):
            raise RuntimeError("Unsupported Sbert model without auto_model attribute")
        return trans.auto_model.config

    def out_embs_dim(self) -> int:
        return self.auto_model.get_sentence_embedding_dimension()

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        transformers_kwargs: dict | None = None,
        **kwargs,
    ):
        if input_embs is not None:
            raise RuntimeError("Sbert does not support input_embs")

        if input_token_ids is None or lengths is None:
            raise RuntimeError("pass input_token_ids and lengths")

        # input shape: batch_sz, seq_len

        max_len = input_token_ids.shape[1]

        attention_mask = self._create_key_padding_mask(max_len, lengths, input_token_ids.device)

        if transformers_kwargs is None:
            transformers_kwargs = {}

        result = self.auto_model.forward(
            {'input_ids': input_token_ids, 'attention_mask': attention_mask}
        )

        return BaseEncoderOut(result['sentence_embedding'], result['token_embeddings'], lengths)
