# Copyright 2024 Ligo Biosciences Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construct an initial 1D embedding."""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from .components.primitives import Linear, LinearNoBias
from .template import TemplatePairStack
from .utils.checkpointing import get_checkpoint_fn
from .utils.tensor_utils import add

checkpoint = get_checkpoint_fn()


# Template Embedder #


def dgram_from_positions(
    pos: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    no_bins: int = 39,
    inf: float = 1e8,
):
    """Computes a distogram given a position tensor."""
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        no_blocks: int = 2,
        c_template: int = 64,
        c_z: int = 128,
        clear_cache_between_blocks: bool = False,
    ):
        super(TemplateEmbedder, self).__init__()

        self.proj_pair = nn.Sequential(LayerNorm(c_z), LinearNoBias(c_z, c_template))
        no_template_features = 84  # 108
        self.linear_templ_feat = LinearNoBias(no_template_features, c_template)
        self.pair_stack = TemplatePairStack(
            no_blocks=no_blocks,
            c_template=c_template,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        self.v_to_u_ln = LayerNorm(c_template)
        self.output_proj = nn.Sequential(nn.ReLU(), LinearNoBias(c_template, c_z))
        self.clear_cache_between_blocks = clear_cache_between_blocks

    def forward(
        self,
        features: Dict[str, Tensor],
        z_trunk: Tensor,
        pair_mask: Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        inplace_safe: bool = False,
    ) -> Tensor:
        """
        OLDTODO: modify this function to take the same features as the OpenFold template embedder. That will allow
         minimal changes in the data pipeline.
        Args:
            features:
                Dictionary containing the template features:
                    "template_aatype":
                        [*, N_templ, N_token, 32] One-hot encoding of the template sequence.
                    "template_pseudo_beta":
                        [*, N_templ, N_token, 3] coordinates of the representative atoms
                    "template_pseudo_beta_mask":
                        [*, N_templ, N_token] Mask indicating if the Cβ (Cα for glycine)
                        has coordinates for the template at this residue.
                    "asym_id":
                        [*, N_token] Unique integer for each distinct chain.
            z_trunk:
                [*, N_token, N_token, c_z] pair representation from the trunk.
            pair_mask:
                [*, N_token, N_token] mask indicating which pairs are valid (non-padding).
            chunk_size:
                Chunk size for the pair stack.
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo attention within the pair stack.
            inplace_safe:
                Whether to use inplace operations.
        """
        # Grab data about the inputs
        bs, n_templ, n_token = features["template_aatype"].shape

        # Compute template distogram
        template_distogram = dgram_from_positions(features["template_pseudo_beta"])

        # Compute the unit vector
        # pos = Vec3Array.from_array(features["template_pseudo_beta"])
        # template_unit_vector = (pos / pos.norm()).to_tensor().to(template_distogram.dtype)
        # print(f"template_unit_vector shape: {template_unit_vector.shape}")

        # One-hot encode template restype
        template_restype = F.one_hot(  # [*, N_templ, N_token, 22]
            features["template_aatype"],
            num_classes=22,  # 20 amino acids + UNK + gap
        ).to(template_distogram.dtype)

        # OLDTODO: add template backbone frame feature

        # Compute masks
        # b_frame_mask = features["template_backbone_frame_mask"]
        # b_frame_mask = b_frame_mask[..., None] * b_frame_mask[..., None, :]  # [*, n_templ, n_token, n_token]
        b_pseudo_beta_mask = features["template_pseudo_beta_mask"]
        b_pseudo_beta_mask = (
            b_pseudo_beta_mask[..., None] * b_pseudo_beta_mask[..., None, :]
        )

        template_feat = torch.cat(
            [
                template_distogram,
                # b_frame_mask[..., None],  # [*, n_templ, n_token, n_token, 1]
                # template_unit_vector,
                b_pseudo_beta_mask[..., None],
            ],
            dim=-1,
        )

        # Mask out features that are not in the same chain
        asym_id_i = features["asym_id"][..., None, :].expand((bs, n_token, n_token))
        asym_id_j = features["asym_id"][..., None].expand((bs, n_token, n_token))
        same_asym_id = torch.isclose(asym_id_i, asym_id_j).to(
            template_feat.dtype
        )  # [*, n_token, n_token]
        same_asym_id = same_asym_id.unsqueeze(-3)  # for N_templ broadcasting
        template_feat = template_feat * same_asym_id.unsqueeze(-1)

        # Add residue type information
        temp_restype_i = template_restype[..., None, :].expand(
            (bs, n_templ, n_token, n_token, -1)
        )
        temp_restype_j = template_restype[..., None, :, :].expand(
            (bs, n_templ, n_token, n_token, -1)
        )
        template_feat = torch.cat(
            [template_feat, temp_restype_i, temp_restype_j], dim=-1
        )

        # Mask the invalid features
        template_feat = template_feat * b_pseudo_beta_mask[..., None]

        # Run the pair stack per template
        single_templates = torch.unbind(
            template_feat, dim=-4
        )  # each element shape [*, n_token, n_token, no_feat]
        z_proj = self.proj_pair(z_trunk)
        u = torch.zeros_like(z_proj)
        for t in range(len(single_templates)):
            # Project and add the template features
            v = z_proj + self.linear_templ_feat(single_templates[t])
            # Run the pair stack
            v = add(
                v,
                self.pair_stack(
                    v,
                    pair_mask=pair_mask,
                    chunk_size=chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
            )
            # Normalize and add to u
            u = add(u, self.v_to_u_ln(v), inplace=inplace_safe)
            del v
        u = torch.div(u, n_templ)  # average
        u = self.output_proj(u)
        return u
