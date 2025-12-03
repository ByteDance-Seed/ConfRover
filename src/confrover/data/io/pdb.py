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

"""IO for PDB files"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
import warnings

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from openfold.np import residue_constants as rc

# =============================================================================
# Constants
# =============================================================================
pdb_parser = PDBParser(QUIET=True)

# =============================================================================
# Functions
# =============================================================================


def pdb_to_atom37(pdb_path, seqlen, unit="A"):
    """
    Load a PDB file as atom37 coordinate array

    Args:
        pdb_path (str): The path to the PDB file.
        seqlen (int): The sequence length of the protein.
        unit (str, optional): The unit of the output coordinates, either 'nm' or 'A'. Defaults to 'A'.

    Returns:
        np.ndarray: Atom37 coordinate data with shape (seqlen, atom_type_num, 3).
    """

    assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
    assert unit in ["nm", "A"], "Unit must be either 'nm' or 'A'"

    struct: Structure = pdb_parser.get_structure("", pdb_path)

    # This function is to load the single model single chain PDB file.
    if len(struct) != 1:
        warnings.warn(f"PDB file {pdb_path} contains {len(struct)} models.")
    if len(struct[0]) != 1:
        warnings.warn(f"PDB file {pdb_path} contains {len(struct[0])} chains.")

    chain = struct[0].child_list[
        0
    ]  # each PDB file contains a single conformation, i.e., model 0

    # load atomic coordinates
    atom_coords = np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan  # (seqlen, 37, 3)
    for residue in chain:
        seq_idx = residue.id[1] - 1  # zero-based numpy indexing
        for atom in residue:
            if atom.name in rc.atom_order.keys():
                atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord

    if unit == "nm":
        atom_coords /= 10

    return atom_coords
