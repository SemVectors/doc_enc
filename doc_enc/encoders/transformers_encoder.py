#!/usr/bin/env python3

import torch

from transformers import AutoModel

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_pooler import BasePoolerConf

from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_out import BaseEncoderOut


class TransformersAutoModel(BaseEncoder):
    def __init__(self, config: BaseEncoderConf) -> None:
        super().__init__()
        self.config = config

        self.auto_model = AutoModel.from_pretrained(
            config.transformers_auto_name, cache_dir=config.transformers_cache_dir
        )
        # set missing params for config
        # It is required to be able to save this config later
        c = self.auto_model.config
        config.hidden_size = c.hidden_size
        config.num_layers = c.num_hidden_layers
        config.num_heads = c.num_attention_heads
        config.dropout = c.hidden_dropout_prob
        config.pooler = BasePoolerConf(pooling_strategy=PoolingStrategy.UNDEFINED)

    def out_embs_dim(self) -> int:
        return self.auto_model.config.hidden_size

    def _create_key_padding_mask(self, max_len, src_lengths, device):
        bs = src_lengths.shape[0]
        mask = torch.full((bs, max_len), 0, dtype=torch.float, device=device)
        for i, l in enumerate(src_lengths):
            mask[i, 0:l] = 1

        return mask

    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        transformers_kwargs: dict | None = None,
    ):
        # input shape: batch_sz, seq_len, hidden_dim
        if input_token_ids is None and input_embs is None:
            raise RuntimeError("pass either input_embs or input_token_ids")

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
        return BaseEncoderOut(result.pooler_output, result.hidden_states, lengths)

    def state_dict(self, *args, **kwargs):
        # do not save state when params are fixed
        # if finetuning save state_dict
        if self.config.transformers_fix_pretrained_params:
            return {}
        prefix = kwargs.get('prefix', '')
        kwargs['prefix'] = prefix + 'auto_model.'
        return self.auto_model.state_dict(*args, **kwargs)


class TransformersLongformer(TransformersAutoModel):
    def forward(
        self,
        input_embs: torch.Tensor | None = None,
        input_token_ids: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        transformers_kwargs: dict | None = None,
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
        )
