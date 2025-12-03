# coding=utf-8
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original file was released under Apache-2.0 License.
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
# This modified file is released under the same license.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""PyTorch LLaMA model modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers.cache_utils import Cache, SinkCache, StaticCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    # LlamaFlashAttention2,
    # LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from confrover._ext.ligo_af3.models.pairformer import PairformerStack
from confrover.utils import get_pylogger
from confrover.utils.torch.tensor import rearrange

from ..utils.checkpoint_activations import checkpoint_wrapper
from .kv_cache import OffloadedCache

logger = get_pylogger(__name__)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
}


class FusedLlamaPairformerModule(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(
        self, llama_config: LlamaConfig, pairformer_config: DictConfig, **kwargs
    ):
        super().__init__(llama_config)
        self.pairformer_config = pairformer_config
        self.llama_config = llama_config
        self.layers = nn.ModuleList(
            [
                checkpoint_wrapper(
                    LlamaDecoderLayer(llama_config, layer_idx), offload_to_cpu=True
                )
                for layer_idx in range(llama_config.num_hidden_layers)
            ]
        )
        self.pairformers = nn.ModuleList(
            [
                PairformerStack(
                    **(OmegaConf.to_container(pairformer_config, resolve=True))
                )
                for layer_idx in range(llama_config.num_hidden_layers // 2)
            ]
        )
        self.norm = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )
        self.gradient_checkpointing = True
        # self.gradient_checkpointing_enable() # NOTE: enabled using checkpoint_wrapper function

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        rigids_mask: torch.Tensor = None,
        batch_size: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_identity_attention: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            raise ValueError("You Must input embeds")

        # return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            assert hasattr(self.llama_config, "cache_type"), (
                "KVCache: cache_type must be specified in llama_config"
            )
            # Initialize default KVcache
            cache_type = self.llama_config.cache_type
            if cache_type == "offloaded":
                # Default KVcache offloaded to CPU
                past_key_values = OffloadedCache()
                logger.debug("Setting OffloadedCache")
            elif cache_type.startswith("sink"):
                # Sink cache defined wth format sink{num_sink}:{sliding_window_length}
                match = re.fullmatch(r"sink(\d+):(\d+)", cache_type)
                if match:
                    num_sink = int(match.group(1))
                    sliding_window_length = int(match.group(2))
                else:
                    raise ValueError(
                        f"String '{cache_type}' is not in the expected format: sink{{sink_num}}:{{sliding_window_length}}"
                    )
                past_key_values = SinkCache(
                    window_length=sliding_window_length,
                    num_sink_tokens=num_sink,
                )
                logger.debug(
                    f"Setting SinkCache(num_sink={num_sink}, sliding_window_length={sliding_window_length})"
                )
            else:
                raise ValueError(
                    "cache_type should be 'offloaded' or 'sink{sink_num}:{sliding_window_length}"
                )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if use_identity_attention:
            causal_mask = self._update_idenity_mask(
                attention_mask, inputs_embeds, past_key_values
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Spatio attention
            if layer_index % 2 == 0:
                hidden_states = rearrange(
                    hidden_states, "(B L) F C -> (B F) L C", B=batch_size
                )
                L = int(
                    math.sqrt(hidden_states.shape[1] + 0.25) - 0.5
                )  # L(L+1) = shape[2]
                s = hidden_states[:, :L, :]
                z = rearrange(hidden_states[:, L:, :], "N (L1 L2) C -> N L1 L2 C", L1=L)
                single_mask = rigids_mask.to(s.dtype)
                pair_mask = single_mask[:, :, None] * single_mask[:, None, :]

                s, z = self.pairformers[layer_index // 2](
                    s=s,  # (bs, n_tokens, c_s)
                    z=z,  # (bs, n_tokens, c_z)
                    single_mask=single_mask,  # (bs, n_tokens)
                    pair_mask=pair_mask,  # (bs, n_tokens, n_tokens)
                    chunk_size=self.pairformer_config.chunk_size,
                    use_deepspeed_evo_attention=self.pairformer_config.use_deepspeed_evo_attention,
                )
                z = rearrange(z, "N L1 L2 C ->  N (L1 L2)  C")

                hidden_states = rearrange(
                    torch.cat([s, z], dim=1), "(B F) M C ->  (B M) F C", B=batch_size
                )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    def _update_idenity_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: Cache,
    ):
        """Get identity attention mask. Only used for training"""

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        identity_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            identity_mask *= torch.arange(target_length, device=device) != torch.arange(
                sequence_length, device=device
            ).reshape(-1, 1)
        identity_mask = identity_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        return identity_mask

    def prepare_inputs_for_generation(
        self,
        inputs_embeds,
        past_key_values=None,
        attention_mask=None,
        input_ids=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = (
                    cache_position[0]
                    if cache_position is not None
                    else past_key_values.get_seq_length()
                )
                max_cache_length = (
                    torch.tensor(
                        past_key_values.get_max_length(), device=inputs_embeds.device
                    )
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = (
                    past_length
                    if max_cache_length is None
                    else torch.min(max_cache_length, past_length)
                )
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing inputs_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > inputs_embeds.shape[1]
            ):
                inputs_embeds = inputs_embeds[
                    :, -(attention_mask.shape[1] - past_length) :
                ]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < inputs_embeds.shape[1]:
                inputs_embeds = inputs_embeds[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + inputs_embeds.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None and cache_position is not None:
            position_ids = position_ids.index_select(index=cache_position, dim=1)
        model_inputs = {"inputs_embeds": inputs_embeds}

        input_length = (
            position_ids.shape[-1]
            if position_ids is not None
            else inputs_embeds.shape[1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=inputs_embeds.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,  # default None
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        model_kwargs=None,
        **kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        # keep track of which sequences are already finished
        batch_size = inputs_embeds.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=inputs_embeds.device
        )
        model_kwargs = self._get_initial_cache_position(
            inputs_embeds[..., 0], model_kwargs
        )

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=inputs_embeds.device
        ):
            # prepare model inputs_embeds
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            inputs_embeds = torch.cat(
                [inputs_embeds, outputs.last_hidden_state[:, -1, :][:, None]], dim=1
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                inputs_embeds[..., 0], None
            )
            this_peer_finished = unfinished_sequences.max() == 0

        return inputs_embeds

    def prepare_configs_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,  # default None
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_cache=False,
        **kwargs,
    ) -> Dict:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop(
            "tokenizer", None
        )  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        model_kwargs["use_cache"] = use_cache
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        if not kwargs_has_attention_mask:
            model_kwargs["attention_mask"] = (
                self._prepare_attention_mask_for_generation(inputs_tensor, None, None)
            )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = inputs_tensor.shape[1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )

        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        generation_config.eos_token_id = None
        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=[],
            tokenizer=tokenizer,
            **kwargs,
        )

        return {
            "model_kwargs": model_kwargs,
            "synced_gpus": synced_gpus,
            "stopping_criteria": stopping_criteria,
        }
