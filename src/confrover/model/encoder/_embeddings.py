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

"""Embedding methods"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import math

import torch

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def sinusoidal_embedding(
    pos: torch.Tensor,
    emb_size: int,
    max_pos: int = 10000,
) -> torch.Tensor:
    assert -max_pos <= pos.min().item() <= max_pos
    assert emb_size % 2 == 0, "Please use an even embedding size."
    half_emb_size = emb_size // 2
    idx = torch.arange(half_emb_size, dtype=pos.dtype, device=pos.device)
    exponent = -1 * idx * math.log(max_pos) / (half_emb_size - 1)
    emb = pos[..., None] * torch.exp(exponent)  # (..., half_emb_size)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (..., emb_size)
    assert emb.size() == pos.size() + torch.Size([emb_size]), "Embedding size mismatch."
    return emb


# =============================================================================
# Classes
# =============================================================================
