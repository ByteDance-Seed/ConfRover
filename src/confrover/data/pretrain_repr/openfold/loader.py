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

"""Load pretrained single and pair repr from OpenFold"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from confrover.data.msa.msa_loader import _load_seqres_index_pairs
from confrover.env import CachePaths
from confrover.utils import get_pylogger, log_header

from ...msa.msa_loader import MSALoader
from .make_openfold_repr import dump_repr

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
from confrover.utils import PathLike

default_cache = CachePaths()

# =============================================================================
# Components
# =============================================================================


def _get_seqres_index(repr_dir: PathLike) -> tuple[str, str]:
    """Get seqres from cached repr record"""

    repr_dir = Path(repr_dir)
    index = repr_dir.name

    metadata_fpath = repr_dir / f"{index}_meta.json"
    if not metadata_fpath.exists():
        logger.warning(f"Meta data file not found: {metadata_fpath}")
        return "", ""
    with open(metadata_fpath, "r") as handle:
        metadata = json.load(handle)
    seqres = metadata["seqres"]
    return seqres, index


class OpenFoldReprLoader:
    """Handles OpenFold representations generation and loading"""

    def __init__(
        self,
        repr_root: PathLike,
        num_recycles: int = 3,
        load_single: bool = True,
        load_pair: bool = True,
        v1: bool = False,
    ):
        self.repr_root = Path(repr_root).resolve()
        self.index_file = self.repr_root / "seqres_to_index.csv"
        self.num_recycles = num_recycles
        self.load_single = load_single
        self.load_pair = load_pair
        self.v1 = v1

        if self.repr_root.exists():
            logger.info(f"OpenFold repr root exists. Set to {self.repr_root}")
            if self.index_file.exists():
                self.seqres_to_index = (
                    _load_seqres_index_pairs(self.index_file)
                    .set_index("seqres")["index"]
                    .to_dict()
                )
            else:
                logger.warning(f"Index file `seqres_to_index.csv` not found.")
                self.seqres_to_index = self.build_index_file()
                self.save_index_file()
        else:
            logger.info(f"OpenFoldRepr root not found, created at: {self.repr_root}.")
            self.repr_root.mkdir(parents=True)
            self.seqres_to_index = {}

    def index_to_dir(self, index: str) -> str:
        repr_dir = self.repr_root / index[:2] / index
        if not repr_dir.exists():
            # fall-back to v1 dir
            repr_dir = self.repr_root / index
        return str(repr_dir)

        # if self.v1:
        #     return str(self.repr_root / index)
        # else:
        #     return str()

    def seqres_to_dir(self, seqres: str):
        if seqres not in self.seqres_to_index:
            raise FileNotFoundError(
                f"Repr not found: {seqres}. Run `OpenFoldReprLoader.generate_repr()` or `confrover openfold-repr` to generate repr first."
            )
        index = str(self.seqres_to_index[seqres])
        return self.index_to_dir(index), index

    def check_cache(self, seqres_list: List[str]) -> tuple[List[str], List[str]]:
        """Check cache for existed and missing repr

        Returns:
            Tuple of list of sequences (has_cache, not_found)
        """

        num_input_seqres = len(seqres_list)
        seqres_set = set(seqres_list)
        num_unique_seqres = len(seqres_set)

        has_cache = []
        not_found = []
        for seqres in seqres_set:
            if seqres in self.seqres_to_index:
                has_cache.append(seqres)
            else:
                not_found.append(seqres)
        logger.info(
            f"Input seqres: {num_input_seqres:,} (unique {num_unique_seqres:,}), cached: {len(has_cache):,}, missing: {len(not_found):,}"
        )
        return has_cache, not_found

    def generate_repr(
        self,
        seqres_index_pairs: List[tuple[str, str]],
        msa_root: PathLike = default_cache.msa,
        openfold_params: PathLike = default_cache.openfold_params,
        save_struct: bool = True,
        num_gpus: int = 1,
        overwrite: bool = False,
        msa_max_query_size: int = 32,
    ):
        """Generate OpenFold repr for seqres-index pairs.

        This function will:
            1. check folding cache for mising repr; remove existing repr if overwrite is True
            2. check the msa_root, query if corresponding msa does not exist
            3. generate repr for the remaining seqres-index pairs
            4. update and save the index file


        Args:
            seqres_index_pairs (List[tuple[str, str]]): seqres-index pairs to generate repr
            msa_root (PathLike, optional): MSA root directory. Defaults to default_cache.msa.
            openfold_params (PathLike, optional): OpenFold params directory. Defaults to default_cache.openfold_params.
            save_struct (bool, optional): Save structure. Defaults to True.
            num_gpus (int, optional): Number of GPUs to use. Defaults to 1.
            overwrite (bool, optional): Overwrite existing repr. Defaults to False.
            msa_max_query_size (int, optional): Maximum number of MSA to query, pass to MSALoader. Defaults to 32.
        """

        # 1. check cache and remove existing repr if overwrite is True
        logger.info(log_header(logger, "Check OpenFold repr cache"))
        has_cache, not_found = self.check_cache(
            seqres_list=[seqres for seqres, _ in seqres_index_pairs]
        )
        if overwrite and len(has_cache) > 0:
            logger.warning(
                f"Overwrite set to True, deleting {len(has_cache):,} cached repr records ..."
            )
            self.delete_repr(has_cache, enforce=True)
            to_query = seqres_index_pairs
        else:
            to_query = [
                (seqres, index)
                for seqres, index in seqres_index_pairs
                if seqres in not_found
            ]
        if len(to_query) == 0:
            logger.info("Repr found for all seqres.")
            return None
        # deduplicate to_query by seqres
        to_query = list({k: v for k, v in to_query}.items())

        # 2. check if MSA exists and query if not
        msa_loader = MSALoader(msa_root=msa_root)
        msa_loader.query_msa(
            seqres_index_pairs=to_query,
            max_query_size=msa_max_query_size,
            clean_tmp_dir=True,
            overwrite=overwrite,
        )

        # 3. Generate repr
        logger.info(log_header(logger, "Generate OpenFold repr"))
        logger.info(
            f"Generating repr for {len(to_query):,}/{len(seqres_index_pairs):,} seqres ..."
        )
        seqres_to_index, failed = dump_repr(
            seqres_index_pairs=to_query,
            output_root=self.repr_root,
            openfold_params=openfold_params,
            num_recycles=self.num_recycles,
            msa_root=msa_root,
            save_struct=save_struct,
            num_gpus=num_gpus,
            v1=self.v1,
        )  # return with unique index

        # 4. Update index
        self.seqres_to_index.update(seqres_to_index)
        logger.info(
            f"✅ Generated new representations for {len(seqres_index_pairs):,} proteins ({len(seqres_to_index):,} succeeded, {len(failed):,} failed)."
        )
        self.save_index_file()

    def load(self, seqres: str) -> Dict[str, torch.Tensor]:
        """Load node and/or edge representations from pretrained model
        Returns:
            {
                pretrained_single: Tensor[seqlen, single_dim], float
                pretrained_pair: Tensor[seqlen, seqlen, pair_dim], float
            }
        """

        repr_dict = {}
        repr_dir, index = self.seqres_to_dir(seqres)

        if self.load_single:
            single_repr_path = (
                f"{repr_dir}/{index}_recycle{self.num_recycles:d}_single_repr.npy"
            )
            if os.path.exists(single_repr_path):
                singel_repr = np.load(single_repr_path)
                repr_dict["pretrained_single"] = torch.from_numpy(singel_repr).float()
            else:
                raise FileNotFoundError(
                    f"{index}: single_repr not found: {str(single_repr_path)}"
                )
        # pair repr
        if self.load_pair:
            pair_repr_path = (
                f"{repr_dir}/{index}_recycle{self.num_recycles:d}_pair_repr.npy"
            )
            if os.path.exists(pair_repr_path):
                pair_repr = np.load(pair_repr_path)
                repr_dict["pretrained_pair"] = torch.from_numpy(pair_repr).float()
            else:
                raise FileNotFoundError(
                    f"{index}: pair_repr not found: {str(pair_repr_path)}"
                )
        return repr_dict

    def build_index_file(self):
        """Scan through self.repr_root and build index file"""
        logger.info(f"Building index file from {self.repr_root} ...")
        if self.v1:
            subdir_list = [
                subdir
                for subdir in self.repr_root.glob("*")
                if subdir.is_dir() and subdir.stem != ".tmp"
            ]
        else:
            subdir_list = [
                subdir
                for subdir in self.repr_root.glob("*/*")
                if subdir.is_dir() and subdir.stem != ".tmp"
            ]
        logger.info(f"{len(subdir_list):,} records found {self.repr_root}.")
        repr_info = map(_get_seqres_index, subdir_list)
        seqres_to_index = {
            seqres: idx
            for seqres, idx in repr_info
            if seqres is not None and idx is not None
        }

        seqres_to_index_df = pd.DataFrame(
            seqres_to_index.items(), columns=["seqres", "index"]
        )
        seqres_to_index_df.to_csv(self.index_file, index=False)
        logger.info(f"Index file contains {len(seqres_to_index):,} records.")
        return seqres_to_index

    def save_index_file(self):
        """Save updated index file"""
        logger.info(
            f"✅ Index file updated with {len(self.seqres_to_index):,} records: {self.index_file}"
        )
        seqres_to_index_df = pd.DataFrame(
            self.seqres_to_index.items(), columns=["seqres", "index"]
        )
        seqres_to_index_df.to_csv(self.index_file, index=False)

    def delete_repr(self, seqres_list: List[str], enforce: bool = False):
        """Delete repr for given seqres"""
        if isinstance(seqres_list, str):
            seqres_list = [seqres_list]
        seqres_set = set(seqres_list)
        if not enforce:
            logger.warning(f"Deletion not enforced. Set enforce=True to confirm.")
            return
        for seqres in seqres_set:
            index = self.seqres_to_index.pop(seqres, None)
            if index is not None:
                shutil.rmtree(self.index_to_dir(index))
        logger.warning(f"Deleted: {len(seqres_set)} representations ...")
        self.save_index_file()
