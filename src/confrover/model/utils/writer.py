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

"""Structure and trajectory writer"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import mdtraj
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from openfold.np.protein import Protein, to_pdb
from openfold.np.residue_constants import atom_order

from confrover.utils import PathLike, get_pylogger
from confrover.utils.misc.process import mp_imap_unordered

_logger = get_pylogger(__name__)
# =============================================================================
# Constants
# =============================================================================


# =============================================================================
# Components
# =============================================================================
def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        assert isinstance(tensor, np.ndarray)
        return tensor


def pack_pdb_to_xtc(subdir: Path, n_preview_frames: int, cleanup: bool):
    """Pack PDB output as a xtc output. For iid outputs."""
    pdb_files = list(subdir.glob("*.pdb"))
    json_files = list(subdir.glob("*.json"))
    if len(pdb_files) == 0:
        raise FileNotFoundError(f"No pdb files found in {subdir}")
    output_prefix = subdir.name

    # Pack pdb files
    start_time = perf_counter()
    traj = mdtraj.load([str(pdb) for pdb in pdb_files])
    traj = traj.superpose(traj, frame=0)
    traj.save_xtc(subdir / f"{output_prefix}.xtc")
    traj[0].save_pdb(subdir / f"{output_prefix}.pdb")
    if n_preview_frames:
        traj[:n_preview_frames].save_pdb(subdir / f"{output_prefix}_preview.pdb")

    # Merge json files
    merged_json = {}
    for json_file in json_files:
        with open(json_file, "r") as handle:
            data = json.load(handle)
            merged_json[json_file.stem] = data
    with open(subdir / f"{output_prefix}.json", "w") as handle:
        json.dump(merged_json, handle, indent=4)

    # clean up the original files
    if cleanup:
        for pdb_file in pdb_files:
            pdb_file.unlink()
        for json_file in json_files:
            json_file.unlink()
    else:
        # move under the /raw folder
        raw_dir = subdir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for pdb_file in pdb_files:
            pdb_file.rename(raw_dir / pdb_file.name)
        for json_file in json_files:
            json_file.rename(raw_dir / json_file.name)

    _logger.info(
        f"[{subdir.stem}] PDB -> XTC for {len(traj)} files in {perf_counter() - start_time:.2f} s"
    )


class Writer:
    def __init__(
        self,
        output_dir: PathLike,
        output_format: Literal["pdb", "xtc", "auto"],
        preview_frames: int = 20,
    ) -> None:
        self.output_dir = Path(output_dir)
        assert output_format in ["pdb", "xtc", "auto"], (
            f"Invalid output format: {output_format}"
        )
        self.output_format = output_format
        self.preview_frames = preview_frames
        self.task_mode = "iid"

    def _set_auto_output_format(self, job_info):
        if self.output_format == "auto":
            self.output_format = "pdb" if job_info["task_mode"] == "iid" else "xtc"

    def get_output_prefix(self, job_info) -> Path:
        self._set_auto_output_format(job_info)
        task_mode = job_info["task_mode"]
        if task_mode == "iid":
            case_id = job_info["case_id"]
            rep_id = job_info["rep_id"]
            prefix = f"{case_id}/{case_id}_sample{rep_id}.pdb"
        else:
            case_id = job_info["case_id"]
            rep_id = job_info["rep_id"]
            prefix = f"{case_id}/{case_id}_sample{rep_id}"
        return self.output_dir / prefix

    def write(
        self,
        aatype: torch.Tensor | np.ndarray,
        atom37: torch.Tensor | np.ndarray,
        atom37_mask: torch.Tensor | np.ndarray,
        padding_mask: torch.Tensor | np.ndarray,
        job_info,
    ):
        self._set_auto_output_format(job_info)
        self.task_mode = job_info["task_mode"]
        if self.output_format == "pdb":
            self.write_pdb(aatype, atom37, atom37_mask, padding_mask, job_info)
        elif self.output_format == "xtc":
            self.write_xtc(aatype, atom37, atom37_mask, padding_mask, job_info)
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")

    def write_pdb(
        self,
        aatype: torch.Tensor | np.ndarray,
        atom37: torch.Tensor | np.ndarray,
        atom37_mask: torch.Tensor | np.ndarray,
        padding_mask: torch.Tensor | np.ndarray,
        job_info: dict,
    ):
        """Write trajectory output as a folder of frame pdb.

        NOTE: writing/loading a large number of PDB files are slow
        """

        num_frame = atom37.shape[0]
        aatype = to_numpy(aatype[padding_mask])
        atom37_mask = to_numpy(atom37_mask[padding_mask])
        atom37 = to_numpy(atom37[:, padding_mask, ...])
        res_idx = np.arange(aatype.shape[0]) + 1
        if num_frame >= 50:
            _logger.warning(
                "Writing/loading a large number of PDB files can be slow. Suggest to use `xtc` format for trajectory/ensembles."
            )

        output_prefix = self.get_output_prefix(job_info)
        if job_info["task_mode"] == "iid":
            assert num_frame == 1, "iid task should only have 1 frame"
            atom37_slice: np.ndarray = atom37[0]

            gen_protein = Protein(
                aatype=aatype,
                atom_positions=atom37_slice,
                atom_mask=atom37_mask,
                residue_index=res_idx,
                chain_index=np.zeros_like(aatype),
                b_factors=np.zeros_like(atom37_mask),
            )
            pdb_fpath = output_prefix
            pdb_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(pdb_fpath, "w") as fp:
                fp.write(to_pdb(gen_protein))
        else:
            output_prefix.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(num_frame):
                # padding_mask = padding_mask[j]
                atom37_slice: np.ndarray = atom37[frame_idx]

                gen_protein = Protein(
                    aatype=aatype,
                    atom_positions=atom37_slice,
                    atom_mask=atom37_mask,
                    residue_index=res_idx,
                    chain_index=np.zeros_like(aatype),
                    b_factors=np.zeros_like(atom37_mask),
                )
                pdb_fpath = output_prefix / (
                    output_prefix.stem + f"_frame{frame_idx}.pdb"
                )
                pdb_fpath.parent.mkdir(parents=True, exist_ok=True)
                with open(pdb_fpath, "w") as fp:
                    fp.write(to_pdb(gen_protein))

        job_json_path = output_prefix.with_suffix(".json")
        with open(job_json_path, "w") as fp:
            json.dump(job_info, fp, indent=4)

    def write_xtc(
        self,
        aatype: torch.Tensor | np.ndarray,
        atom37: torch.Tensor | np.ndarray,
        atom37_mask: torch.Tensor | np.ndarray,
        padding_mask: torch.Tensor | np.ndarray,
        job_info: dict,
    ):
        """Write trajectory output as a folder of frame pdb.

        NOTE: writing/loading a large number of PDB files are slow
        """

        num_frame = atom37.shape[0]
        aatype = to_numpy(aatype[padding_mask])
        atom37_mask = to_numpy(atom37_mask[padding_mask])
        atom37 = to_numpy(atom37[:, padding_mask, ...])
        res_idx = np.arange(aatype.shape[0]) + 1

        # Prepare output
        output_prefix = self.get_output_prefix(job_info)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        pdb_fpath = output_prefix.with_suffix(".pdb")
        xtc_fpath = output_prefix.with_suffix(".xtc")

        # Get top file
        atom37_slice: np.ndarray = atom37[0]
        gen_protein = Protein(
            aatype=aatype,
            atom_positions=atom37_slice,
            atom_mask=atom37_mask,
            residue_index=res_idx,
            chain_index=np.zeros_like(aatype),
            b_factors=np.zeros_like(atom37_mask),
        )
        pdb_fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(pdb_fpath, "w") as fp:
            fp.write(to_pdb(gen_protein))

        traj: mdtraj.Trajectory = mdtraj.load(str(pdb_fpath))
        top: mdtraj.Topology = traj.topology

        new_xyz = np.zeros((num_frame, top.n_atoms, 3))
        for atom_idx, atom in enumerate(top.atoms):
            res_idx = atom.residue.index
            atom37_idx = atom_order[atom.name]
            new_xyz[:, atom_idx, :] = atom37[:, res_idx, atom37_idx, :]

        new_traj = mdtraj.Trajectory(new_xyz / 10.0, top)  # A -> nm
        new_traj = new_traj.center_coordinates()
        new_traj = new_traj.superpose(new_traj, frame=0)
        new_traj.save_xtc(str(xtc_fpath))

        job_json_path = output_prefix.with_suffix(".json")
        with open(job_json_path, "w") as fp:
            json.dump(job_info, fp, indent=4)

        if self.preview_frames:
            self.save_preview(
                new_traj, pdb_fpath.with_stem(pdb_fpath.stem + "_preview")
            )

    def save_preview(self, traj, preview_fpath):
        """Save a multi-model PDB file for previewing the result"""
        if traj.n_frames < self.preview_frames:
            # save all as pdb
            traj.save_pdb(str(preview_fpath))
        else:
            frame_idx = np.linspace(
                0, traj.n_frames - 1, self.preview_frames, dtype=int
            )
            traj[frame_idx].save_pdb(str(preview_fpath))

    def check_output_exists(self, batch) -> bool:
        """Check if all the output of the batch already exists."""
        num_frames = batch["num_frames"]
        bsz = batch["aatype"].shape[0] // num_frames
        for i in range(bsz):
            job_info = batch["job_info"][i]
            task_mode = job_info["task_mode"]
            self._set_auto_output_format(job_info)
            if task_mode == "iid":
                assert num_frames == 1, "iid task should only have 1 frame"
                pdb_fpath = self.get_output_prefix(job_info)
                if not pdb_fpath.exists():
                    return False
            else:
                output_prefix = self.get_output_prefix(job_info)
                if self.output_format == "xtc":
                    xtc_fpath = output_prefix.with_suffix(".xtc")
                    if not xtc_fpath.exists():
                        return False
                else:
                    assert self.output_format == "pdb"
                    last_frame_pdb = output_prefix / (
                        output_prefix.stem + f"_frame{num_frames - 1}.pdb"
                    )
                    if not last_frame_pdb.exists():
                        return False
            # exists: advance random states
            for j in range(num_frames):
                _ = np.random.rand()
                _ = torch.rand(1)

        return True

    @rank_zero_only
    def cleanup(self, remove_pdb: bool = False, n_workers: int = 1):
        if self.task_mode == "iid":
            # merge pdb files into a single xtc file
            _logger.info(f"Merging pdb files into a xtc file: {self.output_dir}")
            case_dir_list = [
                subdir
                for subdir in self.output_dir.glob("*")
                if subdir.is_dir()
                and not any(
                    [res_folder in subdir.stem for res_folder in ["result", "openmm"]]
                )
            ]
            mp_imap_unordered(
                iter=case_dir_list,
                func=pack_pdb_to_xtc,
                n_proc=n_workers,
                # kwargs
                n_preview_frames=self.preview_frames,
                cleanup=remove_pdb,
            )

        else:
            _logger.info(f"No cleanup needed for {self.task_mode} task")
