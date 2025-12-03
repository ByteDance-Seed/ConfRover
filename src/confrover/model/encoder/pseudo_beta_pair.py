# Copyright 2025 Bytedance Ltd. and/or its affiliates.
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

"""Structure Encoder using pseudo-beta pair distances"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import Any, Mapping, Optional

import ml_collections as mlc
import torch
from openfold.model.primitives import Linear
from torch import nn

from confrover.utils.torch.tensor import rearrange

from ._embeddings import sinusoidal_embedding
from ._input_pair_stacks import InputPairStack

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


InputPairStackConfig = mlc.ConfigDict(
    {
        "input_pair_stack": {
            "c_t": mlc.FieldReference(128, field_type=int),
            "c_hidden_tri_att": 64,
            "c_hidden_tri_mul": 64,
            "no_blocks": 4,
            "no_heads": 4,
            "pair_transition_n": 2,
            "dropout_rate": 0.25,
            "blocks_per_ckpt": 1,
        },
        "input_pair_embedder": {
            "min_bin": 3.25,
            "max_bin": 50.75,
            "no_bins": 39,
            "time_emb_dim": 256,
            "inf": 1e8,
        },
        "evoformer_stack": {
            "c_m": mlc.FieldReference(256, field_type=int),
            "c_z": mlc.FieldReference(128, field_type=int),
            # "c_hidden_msa_att": 32,
            # "c_hidden_opm": 32,
            # "c_hidden_mul": 128,
            # "c_hidden_pair_att": 32,
            # "c_s": mlc.FieldReference(384, field_type=int),
            # "no_heads_msa": 8,
            # "no_heads_pair": 4,
            # "no_blocks": 48,
            # "transition_n": 4,
            # "msa_dropout": 0.15,
            # "pair_dropout": 0.25,
            # "clear_cache_between_blocks": False,
            # "inf": 1e9,
            # "eps": 1e-10,
        },
    }
)


class PseudoBetaPairEncoder(nn.Module):
    def __init__(
        self,
        res_idx_emb_size: int,
        pretrained_single_dim: int,
        pretrained_pair_dim: int,
        output_size: int,
        use_trainable_mask_embedding: bool = False,
        use_deepspeed_evo_attention=False,
        chunk_size: int = None,
        input_pair_stack_cfg: mlc.ConfigDict = InputPairStackConfig,
        **kwargs,
    ):
        super().__init__()
        self.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        self.chunk_size = chunk_size
        self.config = input_pair_stack_cfg

        self.res_idx_emb_func = lambda x: sinusoidal_embedding(
            pos=x, emb_size=res_idx_emb_size
        )
        self.aatype_embedding = nn.Embedding(25, res_idx_emb_size)

        if pretrained_single_dim > 0:
            self.pretrained_single_layernorm = nn.LayerNorm(pretrained_single_dim)
        if pretrained_pair_dim > 0:
            self.pretrained_pair_layernorm = nn.LayerNorm(pretrained_pair_dim)

        single_inp_size = res_idx_emb_size + pretrained_single_dim
        self.single_mlp = nn.Sequential(
            nn.Linear(single_inp_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
        )

        pair_inp_size = res_idx_emb_size + pretrained_pair_dim

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_inp_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
        )
        self.input_pair_embedding = Linear(
            self.config.input_pair_embedder.no_bins,
            self.config.evoformer_stack.c_z,
            init="final",
        )
        self.input_pair_stack = InputPairStack(**self.config.input_pair_stack)

        self.use_trainable_mask_embedding = use_trainable_mask_embedding
        if use_trainable_mask_embedding:
            self._pair_mask_embedding = torch.nn.Parameter(
                torch.zeros(1, 1, self.config.evoformer_stack.c_z)
            )

    @property
    def device(self):
        return self.aatype_embedding.device

    def _get_input_pair_embeddings(self, dists, mask):
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        lower = torch.linspace(
            self.config.input_pair_embedder.min_bin,
            self.config.input_pair_embedder.max_bin,
            self.config.input_pair_embedder.no_bins,
            device=dists.device,
        )  # unit: A
        dists = dists.unsqueeze(-1)
        inf = self.config.input_pair_embedder.inf
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)

        inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z = self.input_pair_stack(
            inp_z,
            mask,
            chunk_size=self.chunk_size,
            use_deepspeed_evo_attention=self.use_deepspeed_evo_attention,
        )
        return inp_z

    def forward(
        self,
        aatype,
        padding_mask,
        struct_mask,  # (B * L)
        pseudo_beta,
        pseudo_beta_mask,
        pretrained_single=None,
        pretrained_pair=None,
        **kwargs,
    ):
        res_idx = torch.arange(
            padding_mask.shape[1], device=padding_mask.device
        ).expand_as(padding_mask)  # (B*F, L)
        res_idx = res_idx * padding_mask

        res_idx = res_idx.to(pseudo_beta.dtype)
        """ Single features. """
        res_idx_emb = self.res_idx_emb_func(res_idx)  # (B, L, res_idx_emb_size)
        aatype_emb = self.aatype_embedding(aatype)
        single_feat = res_idx_emb + aatype_emb

        if (pretrained_single is not None) and (
            self.pretrained_single_layernorm is not None
        ):
            single_repr = self.pretrained_single_layernorm(pretrained_single)
            single_repr = single_repr.expand(*single_feat.shape[:-1], -1)
            single_feat = torch.cat([single_feat, single_repr], dim=-1)
        single_feat = self.single_mlp(single_feat)

        """ Pair features. """
        # 1. relative residue index embedding
        rel_res_idx_emb = self.res_idx_emb_func(
            res_idx[:, :, None] - res_idx[:, None, :]
        )  # (B, L, L, res_idx_emb_size)

        # 2. residue distance embedding based on pseudo-beta
        pseudo_beta_dists = (
            torch.sum(
                (pseudo_beta.unsqueeze(-2) - pseudo_beta.unsqueeze(-3)) ** 2, dim=-1
            )
            ** 0.5
        )
        inp_z = self._get_input_pair_embeddings(
            pseudo_beta_dists,
            pseudo_beta_mask,
        )
        if self.use_trainable_mask_embedding:
            pair_mask_embedding = self._pair_mask_embedding.expand(
                *inp_z.shape[:-1], -1
            )  # (B * F, L, L, c_z)
            struct_mask = (
                struct_mask.unsqueeze(-1).unsqueeze(-2).unsqueeze(-3)
            )  # (B * F) -> (B * F, L, L, 1)
            inp_z = inp_z * struct_mask + pair_mask_embedding * (1 - struct_mask)

        pair_feat = rel_res_idx_emb + inp_z
        if (pretrained_pair is not None) and (
            self.pretrained_pair_layernorm is not None
        ):
            pair_repr: torch.Tensor = self.pretrained_pair_layernorm(pretrained_pair)
            pair_repr = pair_repr.expand(*pair_feat.shape[:-1], -1)
            pair_feat = torch.cat([pair_feat, pair_repr], dim=-1)

        pair_feat = self.pair_mlp(pair_feat)

        return single_feat, pair_feat
