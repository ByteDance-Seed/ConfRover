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

"""Tensor utilities"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import List, TypeVar, Union

from einops import rearrange as _rearrange

Tensor = TypeVar("Tensor")

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def rearrange(
    tensor: Union[Tensor, List[Tensor]],
    pattern: str,
    check_inplace: bool = True,
    **axes_lengths,
) -> Tensor:
    """"""
    tensor_rearranged = _rearrange(tensor, pattern, **axes_lengths)
    if check_inplace:
        assert tensor_rearranged.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr(), (
            "Check! It was not an inpalce operation."
        )
    return tensor_rearranged


# =============================================================================
# Classes
# ============================================================================
