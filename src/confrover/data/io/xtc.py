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

"""IO for XTC format"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
from copy import deepcopy

import mdtraj
import numpy as np
from openfold.np import residue_constants as rc

# =============================================================================
# Constants
# =============================================================================

mdtraj.formats.PDBTrajectoryFile._loadNameReplacementTables()
residue_replace = deepcopy(mdtraj.formats.PDBTrajectoryFile._residueNameReplacements)
atom_replace = deepcopy(mdtraj.formats.PDBTrajectoryFile._atomNameReplacements)


# =============================================================================
# Components
# =============================================================================


def xtc_to_atom37(xtc_path, pdb_path, seqlen, frame_idx, unit="A"):
    """
    Load XTC coordinate file as atom37 coordinate array.

    Args:
        xtc_path (str): Path to the XTC file.
        pdb_path (str): Path to the PDB file.
        seqlen (int): Sequence length.
        frame_idx (int): Index of the frame to read from the XTC file.
        unit (str, optional): Unit of the output coordinates, either 'nm' or 'A'. Defaults to 'A'.

    Returns:
        np.ndarray: Atom37 coordinate data with shape (seqlen, 37, 3).
    """

    assert os.path.exists(xtc_path), f"Cannot find xtc file at {xtc_path}."
    assert os.path.exists(pdb_path), f"Cannot find pdb file at {pdb_path}."

    assert unit in ["nm", "A"], "Unit must be either 'nm' or 'A'"

    with mdtraj.formats.XTCTrajectoryFile(xtc_path, "r") as xtc_file:
        xtc_file.seek(frame_idx)
        xyz, _, _, _ = xtc_file.read(n_frames=1, stride=None, atom_indices=None)
        xyz = xyz[0]
        atom_coords = np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        idx = 0
        with open(pdb_path, "r") as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    resName = residue_replace[line[17:20].strip()]
                    if atom_name in atom_replace[resName]:
                        atom_name = atom_replace[resName][atom_name]

                    if atom_name in rc.atom_order.keys():
                        seq_idx = int(line[22:26].strip()) - 1
                        atom_coords[seq_idx, rc.atom_order[atom_name]] = xyz[idx]
                    idx += 1

    if unit == "A":
        atom_coords *= 10

    return atom_coords
