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

"""Utilities for calculating all atom representations."""

from __future__ import annotations

import torch
from openfold.np import residue_constants as rc
from openfold.utils.tensor_utils import batched_gather

"""Construct denser atom positions (14 dimensions instead of 37)."""
restype_atom37_to_atom14 = []

for rt in rc.restypes:
    atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append(
        [
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in rc.atom_types
        ]
    )

# Add dummy mapping for restype 'UNK'
restype_atom37_to_atom14.append([0] * 37)
restype_atom37_to_atom14 = torch.tensor(
    restype_atom37_to_atom14,
    dtype=torch.int32,
)

restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32)

for restype, restype_letter in enumerate(rc.restypes):
    restype_name = rc.restype_1to3[restype_letter]
    atom_names = rc.residue_atoms[restype_name]
    for atom_name in atom_names:
        atom_type = rc.atom_order[atom_name]
        restype_atom37_mask[restype, atom_type] = 1


def atom14_to_atom37(atom14, aatype):
    residx_atom37_mask = restype_atom37_mask.to(atom14.device)[aatype.long()]
    residx_atom37_to_atom14 = restype_atom37_to_atom14.to(atom14.device)[aatype.long()]
    # protein["atom37_atom_exists"] = residx_atom37_mask

    atom37 = batched_gather(
        atom14,
        residx_atom37_to_atom14.long(),
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37 = atom37 * residx_atom37_mask[..., None]
    return atom37, residx_atom37_mask


def aatype_to_torsion_angles_mask(aatype):
    torch.cat(
        [torch.ones(*aatype.shape, 3), torch.tensor(rc.chi_angles_mask)[aatype]], dim=-1
    )
