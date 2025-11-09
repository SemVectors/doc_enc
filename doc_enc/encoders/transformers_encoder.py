#!/usr/bin/env python3

import logging
import torch

from transformers import AutoModel
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sentence_transformers import SentenceTransformer

from doc_enc.common_types import PoolingStrategy
from doc_enc.encoders.base_pooler import BasePoolerConf

from doc_enc.encoders.enc_config import BaseEncoderConf
from doc_enc.encoders.base_encoder import BaseEncoder
from doc_enc.encoders.enc_out import BaseEncoderOut
from doc_enc.encoders.pad_utils import create_key_padding_mask


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

    def _get_padding_side(self):
        padding_side = 'right'
        if self.config.left_padding:
            padding_side = 'left'
        return padding_side

    def state_dict(self, *args, **kwargs):
        # do not save state when params are fixed
        # if finetuning save state_dict
        if self.config.transformers_fix_pretrained_params:
            return {}
        if isinstance(self.auto_model, PeftModel):
            all_peft_config = self.auto_model.peft_config
            selected_adapters = list(all_peft_config.keys())
            if len(selected_adapters) > 1:
                raise RuntimeError("More than one adapter is not supported!")
            adapter_name = selected_adapters[0]
            output_state_dict = get_peft_model_state_dict(
                self.auto_model,
                adapter_name=adapter_name,
                # TODO
                # save_embedding_layers=save_embedding_layers,
            )
            if 'destination' in kwargs:
                d = kwargs['destination']
            else:
                d = {}
            d['adapter_weights'] = output_state_dict
            return d

        prefix = kwargs.get('prefix', '')
        kwargs['prefix'] = prefix + 'auto_model.'
        return self.auto_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if isinstance(self.auto_model, PeftModel):
            set_peft_model_state_dict(self.auto_model, state_dict['adapter_weights'])
            self.auto_model = self.auto_model.merge_and_unload()
        else:
            super().load_state_dict(state_dict, *args, **kwargs)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class TransformersAutoModel(BaseTransformersAutoModel):
    def __init__(self, config: BaseEncoderConf, eval_mode: bool) -> None:
        kwargs = {}
        if config.transformers_torch_fp16:
            kwargs['torch_dtype'] = torch.float16
        auto_model = AutoModel.from_pretrained(
            config.transformers_auto_name, cache_dir=config.transformers_cache_dir, **kwargs
        )
        if config.use_adapter:
            logging.info(
                "Create an adapter %s, adapter kwargs: %s",
                config.use_adapter,
                config.adapter_kwargs,
            )
            if not eval_mode:
                print_trainable_parameters(auto_model)
            if config.use_adapter == 'lora':
                kwargs = {}
                if config.adapter_kwargs is not None:
                    kwargs = config.adapter_kwargs
                adapter_config = LoraConfig(inference_mode=eval_mode, **kwargs)
            else:
                raise RuntimeError("Unsupported adapter %s" % config.use_adapter)

            model = get_peft_model(auto_model, adapter_config)
            if not eval_mode:
                print_trainable_parameters(model)
        else:
            model = auto_model

        # TODO optionally enable checkpointing
        # auto_model.gradient_checkpointing_enable()

        super().__init__(config, model)

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

        # B X L
        attention_mask = create_key_padding_mask(
            max_len, lengths, t.device, padding_side=self._get_padding_side()
        )

        if transformers_kwargs is None:
            transformers_kwargs = {}

        result = self.auto_model(
            input_ids=input_token_ids,
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            **transformers_kwargs,
        )
        if (
            self.config.transformers_pooler != 'auto'
            or (pooled_out := getattr(result, 'pooler_output', None)) is None
        ):

            # B x L X D
            if (last_hidden_state := getattr(result, 'last_hidden_state', None)) is not None:

                if self.config.transformers_pooler in ('first', 'auto'):
                    pooled_out = last_hidden_state[:, 0]
                elif self.config.transformers_pooler == 'last':
                    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
                    if left_padding:
                        pooled_out = last_hidden_state[:, -1]
                    else:
                        batch_size = last_hidden_state.shape[0]
                        pooled_out = last_hidden_state[
                            torch.arange(batch_size, device=last_hidden_state.device),
                            lengths - 1,
                        ]

                elif self.config.transformers_pooler == 'mean':
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    )
                    pooled_out = torch.sum(
                        last_hidden_state * input_mask_expanded, 1
                    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif self.config.transformers_pooler == 'max':
                    masked = torch.masked_fill(
                        last_hidden_state, attention_mask.unsqueeze(-1).logical_not(), float('-inf')
                    )
                    pooled_out = torch.max(masked, dim=1)[0]
                else:
                    raise RuntimeError(
                        f"Unsupported transformer_pooler == {self.config.transformers_pooler}"
                    )
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

        attention_mask = create_key_padding_mask(
            max_len, lengths, input_token_ids.device, padding_side=self._get_padding_side()
        )

        if transformers_kwargs is None:
            transformers_kwargs = {}

        result = self.auto_model.forward(
            {'input_ids': input_token_ids, 'attention_mask': attention_mask}
        )

        return BaseEncoderOut(result['sentence_embedding'], result['token_embeddings'], lengths)
