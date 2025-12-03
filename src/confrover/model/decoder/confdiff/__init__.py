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

"""ConfDiffDecoder derived from ConfDiff (https://github.com/bytedance/ConfDiff)"""

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from openfold.np import residue_constants as rc
from openfold.utils import rigid_utils as ru
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from torch import nn

from confrover.model.decoder import Decoder
from confrover.model.utils.checkpoint_activations import checkpoint_wrapper

from .structure_module import Linear


class ConfDiffDecoder(Decoder):
    def __init__(
        self,
        model_nn,
        diffuser,
        loss: Optional[nn.Module] = None,
        sampler: Optional[Any] = None,
        pretrain_model_ckpt=None,
        freeze_model_nn: bool = True,
    ):
        super().__init__()

        self.model_nn = model_nn
        self.diffuser = diffuser
        self.sampler = sampler
        self.loss = loss
        self.pretrain_model_ckpt = pretrain_model_ckpt
        self.freeze_model_nn = freeze_model_nn

        # Load pretrained model weights and freeze
        if pretrain_model_ckpt is not None:
            cond_ckpt = torch.load(pretrain_model_ckpt, map_location="cpu")
            cond_state_dict = {}
            for key in cond_ckpt.keys():
                if key.startswith("score_network.model_nn."):
                    cond_state_dict[key[len("score_network.model_nn.") :]] = cond_ckpt[
                        key
                    ]
            self.model_nn.load_state_dict(cond_state_dict, strict=False)

            if freeze_model_nn:
                self.freeze_para_name = cond_state_dict.keys()
                for name, param in self.model_nn.named_parameters():
                    if name in self.freeze_para_name:
                        param.requires_grad = False
            del cond_ckpt

    @property
    def device(self):
        return self.model_nn.device

    def forward(
        self,
        aatype,  # (B * F, L)
        s,
        z,
        t,  # (B * F,)
        rigids_t,  # (B * F, L, 7)
        rigids_mask,  # (B * F, L)
        padding_mask,  # (B * F, L)
        gt_feat,  # dict
        torsion_angles_mask,  # (B * F, L, 7)
        pretrained_single=None,  # (B * F, L, single_dim)
        pretrained_pair=None,  # (B * F, L, L, pair_dim)
        **kwargs,
    ):
        assert self.loss is not None, "Loss must be specified for training"
        rigids_t = ru.Rigid.from_tensor_7(rigids_t)
        rigids_mask = rigids_mask * padding_mask

        #### Denoising Model ####
        output = self.model_nn(
            aatype=aatype,
            padding_mask=padding_mask,
            s=s,
            z=z,
            t=t,
            rigids_t=rigids_t,
            rigids_mask=rigids_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
        pred_rigids_0 = output["pred_rigids_0"]
        pred_torsion_sin_cos = output["pred_torsion_sin_cos"]
        pred_atom14 = output["pred_atom14"]
        pred_sidechain_frame = output["pred_sidechain_frames"]

        #### Compute scores ####
        pred_rot_score = self.diffuser.calc_rot_score(
            rigids_t.get_rots(),
            pred_rigids_0.get_rots(),
            t,
            use_cached_score=False,
        )
        pred_rot_score = pred_rot_score * rigids_mask[..., None]

        pred_trans_score = self.diffuser.calc_trans_score(
            rigids_t.get_trans(),
            pred_rigids_0.get_trans(),
            t[:, None, None],
            use_torch=True,
        )
        pred_trans_score = pred_trans_score * rigids_mask[..., None]

        #### Loss #####
        loss, aux_info = self.loss(
            t=t,
            rigids_mask=rigids_mask,
            torsion_angles_mask=torsion_angles_mask,
            pred_rigids_0=pred_rigids_0,
            pred_torsion_sin_cos=pred_torsion_sin_cos,
            pred_atom14=pred_atom14,
            pred_rot_score=pred_rot_score,
            pred_trans_score=pred_trans_score,
            pred_sidechain_frame=pred_sidechain_frame,
            gt_feat=gt_feat,
        )

        return loss, aux_info, output

    @torch.inference_mode()
    def sample(
        self,
        aatype,
        s,
        z,
        padding_mask,
        num_frames,
        pretrained_single=None,
        pretrained_pair=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse sample from the model."""
        assert not self.model_nn.training

        """ Reverse sampling. """
        rigids_mask = padding_mask.to(pretrained_single.dtype)
        batch_size, seq_len = aatype.shape[:2]

        #### Sample from prior distribution ####
        rigids_t = self.diffuser.sample_ref(
            n_samples=batch_size,
            num_frames=num_frames,
            seq_len=seq_len,
            device=aatype.device,
        )  # (B*F, L, 7)

        #### Reverse sampling ####
        pred_atom14, pred_rigids_0 = self.sampler.reverse_sample(
            model_nn=self.model_nn,
            se3_diffuser=self.diffuser,
            rigids_t=rigids_t,
            s=s,
            z=z,
            # model_nn feats
            aatype=aatype,
            padding_mask=padding_mask,
            rigids_mask=rigids_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
        return pred_atom14, pred_rigids_0


class ConfDiffNetwork(nn.Module):
    def __init__(
        self,
        embedder,
        structure_module,
        single_hidden_dim,
        pair_hidden_dim,
        **kwargs,
    ):
        super().__init__()
        self.embedder = checkpoint_wrapper(embedder, offload_to_cpu=True)
        self.structure_module = structure_module
        self.node_layer_norm = nn.LayerNorm(single_hidden_dim)
        self.node_hidden_mlp = Linear(
            single_hidden_dim, embedder.node_emb_size, init="final"
        )
        self.edge_layer_norm = nn.LayerNorm(pair_hidden_dim)
        self.edge_hidden_mlp = Linear(
            pair_hidden_dim, embedder.edge_emb_size, init="final"
        )

    def forward(
        self,
        aatype,
        padding_mask,
        s,
        z,
        t,
        rigids_t,  # unscaled
        rigids_mask,
        pretrained_single=None,
        pretrained_pair=None,
        **kwargs,
    ):
        #### folding model embedding ####
        node_feat, edge_feat = self.embedder(
            # res_idx = res_idx,
            padding_mask=padding_mask,
            t=t,
            rigids_t=rigids_t,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )

        #### residual add temporal signals ####
        node_feat = node_feat + self.node_hidden_mlp(self.node_layer_norm(s))
        edge_feat = edge_feat + self.edge_hidden_mlp(self.edge_layer_norm(z))

        #### Structure Module denoising ####
        sm_output = self.structure_module(
            rigids_t=rigids_t,  # (B, L, 7) # unscaled
            node_feat=node_feat,  # (B, L, node_emb_size)
            edge_feat=edge_feat,  # (B, L, L, edge_emb_size)
            node_mask=rigids_mask,  # (B, L)
            padding_mask=padding_mask,  # (B, L)
            t=t,
        )
        pred_rigids_0 = sm_output["pred_rigids_0"]
        pred_torsion_sin_cos = sm_output[
            "pred_torsion_sin_cos"
        ]  # NOTE: this torsion prediction is only based on the base model

        # ************************************************************************
        #   Post-process: rigids to atom14 mapping
        # ************************************************************************
        all_frames_to_global = self.torsion_angles_to_frames(
            pred_rigids_0,
            pred_torsion_sin_cos,
            aatype,
        )

        pred_atom14 = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )

        pred_sidechain_frames = all_frames_to_global.to_tensor_4x4()

        # pred_torsion_sin_cos = torch.arctan2(pred_torsion_sin_cos[...,0],
        #                         pred_torsion_sin_cos[...,1].masked_fill(~(torsion_angles_mask).bool(),1)) # avoid nan
        sm_output.update(
            {
                "pred_atom14": pred_atom14,
                "pred_sidechain_frames": pred_sidechain_frames,
            }
        )
        return sm_output

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self,
        r,
        f,  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
