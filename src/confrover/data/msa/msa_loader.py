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

"""MSALoader class to handle the MSA query, caching and retrieval."""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from confrover.utils import PathLike, get_pylogger, log_header
from confrover.utils.misc.process import mp_imap_unordered

from .mmseq2_colab import batch_query as _batch_query

log = get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================


SEQRES_COL_NAME = ["seqres", "sequence", "seq"]
INDEX_COL_NAME = ["index", "case_id", "chain_name", "name"]


# =============================================================================
# Components
# =============================================================================


def _get_query_seqres(msa_dir) -> Tuple[str, str] | Tuple[None, None]:
    """Get MSA query seqres from MSA file"""

    msa_dir = Path(msa_dir)
    msa_fpath = msa_dir / "a3m" / f"{msa_dir.stem}.a3m"
    if not msa_fpath.exists():
        log.warning(f"No MSA file found under: {msa_dir}/a3m/*.a3m")
        return None, None
    seqres = ""
    with open(msa_fpath, "r") as handle:
        line1 = handle.readline()
        assert line1.startswith(">")
        line = handle.readline().strip("\n").strip()

        # FASTA-style can have multiple lines
        while not line.startswith(">"):
            seqres += line
            line = handle.readline().strip("\n").strip()

    # check no gap and insertion in the seqres
    assert "-" not in seqres, f"Query seqres should not have gap: {seqres}"
    assert seqres.isupper(), f"Query seqres should not have insertion: {seqres}"
    return str(msa_dir), seqres


def _load_seqres_index_pairs(
    csv_fpath: PathLike,
    seqres_col_names: list = SEQRES_COL_NAME,
    index_col_names: list = INDEX_COL_NAME,
) -> pd.DataFrame:
    """Load index or metadata from csv file and parse seqres-index pairs. Standardize column names."""

    df = pd.read_csv(csv_fpath)

    seqres_col_name = None
    for col_name in seqres_col_names:
        if col_name in df.columns:
            seqres_col_name = col_name
            break
    if seqres_col_name is None:
        raise IndexError(
            f"seqres column not found in {csv_fpath}. Allowed: {seqres_col_names}"
        )

    index_col_name = None
    for col_name in index_col_names:
        if col_name in df.columns:
            index_col_name = col_name
            break
    if index_col_name is None:
        raise IndexError(
            f"index column not found in {csv_fpath}. Allowed: {index_col_names}"
        )

    return df[[seqres_col_name, index_col_name]].rename(
        columns={seqres_col_name: "seqres", index_col_name: "index"}
    )


class MSALoader:
    """MSA cache loading and querying"""

    def __init__(self, msa_root: PathLike):
        """Connect or initialize an MSA loader at msa_root

        Args:
            msa_root (PathLike): MSA root directory
        """
        self.msa_root = Path(msa_root).resolve()
        self.index_file = self.msa_root / "seqres_to_index.csv"

        if self.msa_root.exists():
            log.info(f"MSA root exists. Set to {self.msa_root}")
            if self.index_file.exists():
                self.seqres_to_index = (
                    _load_seqres_index_pairs(self.index_file)
                    .set_index("seqres")["index"]
                    .to_dict()
                )
            else:
                log.warning(f"Index file `seqres_to_index.csv` not found.")
                self.seqres_to_index = self.build_index_file()
                self.save_index_file()
        else:
            log.info(f"MSA root not found, created at {self.msa_root}")
            self.msa_root.mkdir(parents=True)
            self.seqres_to_index = {}

    def index_to_dir(self, index: str) -> str:
        """Return the MSA directory path for a given index"""
        return str(self.msa_root / index[:2] / index)

    def seqres_to_dir(self, seqres: str):
        """Return the MSA directory path and index for a given seqres"""
        if seqres not in self.seqres_to_index:
            raise FileNotFoundError(f"MSA not found: {seqres}. Query MSA first.")
        index = str(self.seqres_to_index[seqres])
        return self.index_to_dir(index), index

    def check_cache(self, seqres_list: List[str]) -> tuple[List[str], List[str]]:
        """Check whether cache exists for a list of seqres

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
        log.info(
            f"Input seqres: {num_input_seqres:,} (unique {num_unique_seqres:,}), cached: {len(has_cache):,}, missing: {len(not_found):,}"
        )
        return has_cache, not_found

    def query_msa(
        self,
        seqres_index_pairs: List[Tuple[str, str]],
        max_query_size=32,
        clean_tmp_dir=True,
        overwrite: bool = False,
    ):
        """Batch query MMseqs2 server.

        Args:
            seqres_index_pairs (pd.DataFrame | List[Tuple[str, str]]): a DataFrame contains 'seqres' and 'index' columns, or a List of seqres-index pairs.
            tmp_dir (str, optional): Temporary directory to save query output. Defaults to {output_dir}/.tmp/.
            deduplicate (bool, optional): Deduplicate indentical seqres. Defaults to True.
            max_query_size (int, optional): Maximum batch size. Defaults to 64.
            clean_tmp_dir (bool, optional): Clean temporary directory after query. Defaults to True.
            overwrite (bool, optional): Overwrite existing cache. Defaults to False.
        """
        log.info(log_header(log, "Check MSA"))

        # 1. Check cache
        has_cache, not_found = self.check_cache(
            seqres_list=[seqres for seqres, _ in seqres_index_pairs]
        )
        if overwrite and len(has_cache) > 0:
            log.info(
                f"Overwrite set to True, deleting {len(has_cache):,} cached MSA records ..."
            )
            self.delete_msa(has_cache, enforce=True)
            to_query = seqres_index_pairs
        else:
            to_query = [
                (seqres, index)
                for seqres, index in seqres_index_pairs
                if seqres in not_found
            ]
        if len(to_query) == 0:
            log.info("MSA found for all seqres.")
            return None

        # 2. Query MSA
        log.info(
            f"Running queries for {len(to_query):,}/{len(seqres_index_pairs):,} seqres ..."
        )
        seqres_index_pairs = _batch_query(
            to_query,
            output_dir=self.msa_root,
            max_query_size=max_query_size,
            clean_tmp_dir=clean_tmp_dir,
        )

        # 3. Update index
        self.seqres_to_index.update(
            {seqres: index for seqres, index in seqres_index_pairs}
        )
        self.save_index_file()

    def build_index_file(self, n_proc=1) -> dict[str, str]:
        """Scan through self.msa_root and build an index file"""
        log.info(f"Building index file at {self.msa_root} ...")
        subdir_list = [
            subdir
            for subdir in self.msa_root.glob("*/*")
            if subdir.is_dir() and subdir.stem != ".tmp"
        ]
        log.info(f"{len(subdir_list):,} records found.")

        msa_info = mp_imap_unordered(
            iter=subdir_list, func=_get_query_seqres, n_proc=n_proc
        )
        self.seqres_to_index = seqres_to_index = {
            seqres: Path(idx).stem
            for idx, seqres in msa_info
            if seqres is not None and idx is not None
        }
        log.info(f"Successfully built index file for {len(seqres_to_index):,} records.")
        return seqres_to_index

    def save_index_file(self):
        """Save updated index file"""
        log.info(
            f"✅ Index file updated with {len(self.seqres_to_index):,} records: {self.index_file}"
        )
        seqres_to_index_df = pd.DataFrame(
            self.seqres_to_index.items(), columns=["seqres", "index"]
        )
        seqres_to_index_df.to_csv(self.index_file, index=False)

    def delete_msa(self, seqres_list: list[str] | str, enforce: bool = False):
        """Delete MSA records for a list of seqres. Only run if enforce is set to True.

        Args:
            seqres_list (list[str] | str): List of seqres to delete
            enforce (bool, optional): Whether to enforce deletion. Defaults to False.
        """
        if isinstance(seqres_list, str):
            seqres_list = [seqres_list]
        seqres_set = set(seqres_list)
        log.warning(f"Deleting {len(seqres_set)} MSA records ...")
        if not enforce:
            log.warning(f"Deletion not enforced. Set enforce=True to confirm.")
            return
        for seqres in seqres_set:
            index = self.seqres_to_index.pop(seqres, None)
            if index is not None:
                shutil.rmtree(self.index_to_dir(index))
        self.save_index_file()
