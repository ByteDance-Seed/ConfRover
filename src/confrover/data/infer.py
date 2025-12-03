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

"""torch.Dataset for inference"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from openfold.data import data_transforms
from openfold.np import residue_constants as rc
from openfold.utils import rigid_utils as ru
from torch.nn.utils.rnn import pad_sequence

from confrover.utils import PathLike, get_pylogger
from confrover.utils.torch.tensor import rearrange

from .io.pdb import pdb_to_atom37
from .io.xtc import xtc_to_atom37


@dataclass
class LoaderConfig:
    """Additional configuration for torch.DataLoader"""

    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    shuffle: bool | None = None

    def to_dict(self, drop_none: bool = True):
        obj_dict = asdict(self)
        if drop_none:
            return {k: v for k, v in obj_dict.items() if v is not None}
        else:
            return obj_dict


logger = get_pylogger(__name__)


# =============================================================================
# Constants
# =============================================================================

TASK_MODE_TYPE = Literal["forward", "iid", "interp"]


# Torsion angle mask for each type of residues
all_angles_mask = torch.cat(
    [torch.ones(20, 3), torch.tensor(rc.chi_angles_mask)], dim=-1
)
all_angles_mask_with_x = torch.cat([all_angles_mask, torch.zeros(1, 7)], dim=0)


# =============================================================================
# Manifests
# =============================================================================

__FORMAT_MSG = """
Input JSON file format:

```
{
    "name": <str>,
    "task_mode": <forward|interp|iid>,
    "n_replicates": <int>,
    "n_frames": <int>,
    "stride_in_10ps": <int>,
    "cases": [
        {
            "case_id": <str>, 
            "seqres": <str>,
            "conditions": <info for conditioning frames>,
        },
        ...
    ]
}
```

"conditions" specifies the conditioning frames and accepts:
    1) a list of PDB files:
        "conditions": ["pdb_fpath_1.pdb", "pdb_fpath_2.pdb"],
    2) a dict of XTC file info:
        "conditions": {
            "xtc_fpath": "xtc_fpath.xtc",
            "pdb_fpath": "pdb_fpath.pdb",
            "frame_idxs": [<int>, ...]
        }

Different tasks requires different number of conditioning frames:
    - forward simulation: one initial frame
    - interpolation: one initial frame and one target frame
    - iid: no conditioning frames

"""


@dataclass
class PDBConditions:
    """Context conditions defined by a list of PDB files

    Supported formats:
        - forward simulation: a single pdb file as the initial structure
        - interpolation: a pair of pdb files corresponding to the start and end conformation
    """

    pdb_fpath_list: list[str]

    @classmethod
    def from_list(
        cls,
        pdb_fpath_list: Any,
        task_mode: TASK_MODE_TYPE,
        relpath_to: str | None = None,
    ):
        """Create and verify PDBConditions from a list of PDB files"""

        try:
            # cast single pdb file to a list
            if isinstance(pdb_fpath_list, (str, Path)):
                pdb_fpath_list = [pdb_fpath_list]
            assert isinstance(pdb_fpath_list, list), (
                f"PDBConditions must be single PDB file or a list of PDB files, but got {type(pdb_fpath_list)}."
            )

            # verify task_mode and input format
            if task_mode == "forward":
                assert len(pdb_fpath_list) == 1, (
                    f"Task mode `forward` requires one PDB file, but got {len(pdb_fpath_list)}."
                )
            else:
                assert task_mode == "interp", (
                    f"Expected task_mode to be `forward` or `interp`, but got {task_mode}"
                )
                assert len(pdb_fpath_list) == 2, (
                    f"Task mode `interp` requires two PDB files, but got {len(pdb_fpath_list)}."
                )
        except AssertionError as e:
            logger.error(__FORMAT_MSG)
            raise e

        # pdb_fpath_list are relative paths if relpath_to is provided
        if relpath_to is not None:
            pdb_fpath_list = [
                str(Path(relpath_to) / pdb_fpath) for pdb_fpath in pdb_fpath_list
            ]

        # check pdb files exist
        for pdb_fpath in pdb_fpath_list:
            assert Path(pdb_fpath).exists(), f"PDB file {pdb_fpath} does not exist."
            assert Path(pdb_fpath).suffix == ".pdb", (
                f"PDB file does not have suffix `.pdb`: {pdb_fpath}"
            )
        return cls(pdb_fpath_list)

    def __len__(self):
        return len(self.pdb_fpath_list)

    def load_coords(self, seqlen: int):
        """Load coordinates from PDB files"""
        return [
            pdb_to_atom37(pdb_path=pdb_fpath, seqlen=seqlen, unit="A")
            for pdb_fpath in self.pdb_fpath_list
        ]


@dataclass
class XTCConditions:
    """Frame conditions from XTC files"""

    xtc_fpath: str
    pdb_fpath: str
    frame_idxs: list[int]  # List of frame indices (in ps) to load

    @classmethod
    def from_dict(
        cls, cfg_dict: dict, task_mode: TASK_MODE_TYPE, relpath_to: str | None = None
    ):
        """Create and verify XTCConditions from a dict"""

        try:
            xtc_fpath = cfg_dict["xtc_fpath"]
            pdb_fpath = cfg_dict["pdb_fpath"]
            frame_idxs = cfg_dict["frame_idxs"]
            if isinstance(frame_idxs, int):
                frame_idxs = [frame_idxs]

            # verify input format
            assert isinstance(frame_idxs, list), (
                f"Frame indices must be a list of integers, but got {type(frame_idxs)}."
            )
            for frame_idx in frame_idxs:
                assert isinstance(frame_idx, int), (
                    f"Frame index must be an integer, but got {type(frame_idx)}."
                )
            # verify the task_mode and frame_idxs matches
            if task_mode == "forward":
                assert len(frame_idxs) == 1, (
                    f"Task mode `forward` requires one frame index, but got {len(frame_idxs)}."
                )
            else:
                assert task_mode == "interp", (
                    f"Expected task_mode to be `forward` or `interp`, but got {task_mode}"
                )
                assert len(frame_idxs) == 2, (
                    f"Task mode `interp` requires two frame indices, but got {len(frame_idxs)}."
                )
        except Exception as e:
            logger.error(__FORMAT_MSG)
            raise e

        # xtc_fpath and pdb_fpath are relative paths if relpath_to is provided
        if relpath_to is not None:
            xtc_fpath = str(Path(relpath_to) / xtc_fpath)
            pdb_fpath = str(Path(relpath_to) / pdb_fpath)

        # verify xtc and pdb file exist
        assert Path(xtc_fpath).exists(), f"XTC file {xtc_fpath} does not exist."
        assert Path(xtc_fpath).suffix == ".xtc", (
            f"XTC file does not have suffix `.xtc`: {xtc_fpath}"
        )
        assert Path(pdb_fpath).exists(), f"PDB file {pdb_fpath} does not exist."
        assert Path(pdb_fpath).suffix == ".pdb", (
            f"PDB file does not have suffix `.pdb`: {pdb_fpath}"
        )
        return cls(xtc_fpath=xtc_fpath, pdb_fpath=pdb_fpath, frame_idxs=frame_idxs)

    def __len__(self):
        return len(self.frame_idxs)

    def load_coords(self, seqlen: int):
        """Load coordinates from XTC files"""
        all_frame_atom37 = []
        for frame_idx in self.frame_idxs:
            frame_atom37 = xtc_to_atom37(
                xtc_path=self.xtc_fpath,
                pdb_path=self.pdb_fpath,
                frame_idx=frame_idx,
                seqlen=seqlen,
                unit="A",
            )
            all_frame_atom37.append(frame_atom37)
        return all_frame_atom37


@dataclass
class GenCaseConfig:
    """Input config for one generation case"""

    case_id: str
    seqres: str
    seqlen: int
    task_mode: TASK_MODE_TYPE

    n_replicates: int  # total number of repeats
    rep_id: int  # this repeat
    n_frames: int | None
    stride_in_10ps: int | None
    conditions: PDBConditions | XTCConditions | None

    @classmethod
    def from_dict(
        cls,
        case_dict: dict,
        rep_id: int,
        dataset_cfg: GenDatasetConfig,
        relpath_to: str | None = None,
    ):
        """Construct from a dictionary"""

        case_dict = dict(case_dict)

        # parse task_mode and conditions
        task_mode = case_dict.get("task_mode", dataset_cfg.task_mode)
        conditions = case_dict.get("conditions", None)
        if isinstance(conditions, (str, list)):
            # Guess is PDB format
            conditions = PDBConditions.from_list(conditions, task_mode, relpath_to)
        elif isinstance(conditions, dict):
            conditions = XTCConditions.from_dict(conditions, task_mode, relpath_to)
        else:
            assert conditions is None and task_mode == "iid", (
                f"Task mode `iid` requires no conditions, but got {conditions}."
            )

        return cls(
            # required key in the dict
            case_id=case_dict["case_id"],
            seqres=case_dict["seqres"],
            seqlen=len(case_dict["seqres"]),
            task_mode=task_mode,
            conditions=conditions,
            rep_id=rep_id,
            # optionally to populate from GenDatasetConfig
            n_replicates=case_dict.get("n_replicates", dataset_cfg.n_replicates),
            n_frames=case_dict.get("n_frames", dataset_cfg.n_frames),
            stride_in_10ps=case_dict.get("stride_in_10ps", dataset_cfg.stride_in_10ps),
        )

    def __post_init__(self):
        """Post-initialize the class"""
        if self.task_mode == "iid":
            if self.n_frames is not None and self.n_frames > 1:
                raise ValueError(
                    f"Task mode `iid` requires `n_frames` to be 1, but got {self.n_frames}."
                )
            self.n_frames = 1
        else:
            assert self.n_frames is not None, (
                f"Task mode `{self.task_mode}` requires `n_frames` to be set, but got None."
            )
            assert self.stride_in_10ps is not None, (
                f"Task mode `{self.task_mode}` requires `stride_in_10ps` to be set, but got None."
            )

    def get_frame_idxs(self):
        if self.task_mode == "iid":
            frame_idxs = np.array([0])
            cond_mask = np.array([0])
        elif self.task_mode == "forward":
            assert isinstance(self.n_frames, int)
            assert isinstance(self.stride_in_10ps, int)
            frame_idxs = np.arange(self.n_frames) * self.stride_in_10ps
            cond_mask = np.zeros_like(frame_idxs)
            cond_mask[0] = 1.0
        elif self.task_mode == "interp":
            assert isinstance(self.n_frames, int)
            assert isinstance(self.stride_in_10ps, int)
            frame_idxs = np.arange(self.n_frames) * self.stride_in_10ps
            frame_idxs = np.concatenate([frame_idxs[-1:], frame_idxs[:-1]])
            cond_mask = np.zeros_like(frame_idxs)
            cond_mask[0] = cond_mask[1] = 1.0
        else:
            raise ValueError(f"Unknown task_mode {self.task_mode}")
        return frame_idxs, cond_mask


@dataclass
class GenDatasetConfig:
    """Config for general generation dataset"""

    name: str

    cases: List[GenCaseConfig]

    # Alternative way to set dataset-level info and later populate into each data case
    task_mode: TASK_MODE_TYPE | None = None
    n_replicates: int | None = None
    n_frames: int | None = None
    stride_in_10ps: int | None = None  # stride

    @classmethod
    def from_json(
        cls,
        json_fpath,
        case_subset: List[str] | None = None,
        relpath_to: str | None = None,
    ) -> "GenDatasetConfig":
        with open(json_fpath, "r") as f:
            cfg_dict = json.load(f)

        ds_name = cfg_dict.pop("name", Path(json_fpath).stem)
        case_list = cfg_dict.pop("cases")
        relpath_to = cfg_dict.pop("relpath_to", relpath_to)
        inst = cls(**cfg_dict, cases=[], name=ds_name)

        for case_dict in case_list:
            if case_subset is not None and case_dict["case_id"] not in case_subset:
                continue
            n_replicates = case_dict.get("n_replicates", inst.n_replicates)
            inst.cases.extend(
                [
                    GenCaseConfig.from_dict(
                        case_dict,
                        rep_id=rep_id,
                        dataset_cfg=inst,
                        relpath_to=relpath_to,
                    )
                    for rep_id in range(n_replicates)
                ]
            )
        # Check if case names (case_id, rep_id) are unique
        case_rep_pairs = [(case.case_id, case.rep_id) for case in inst.cases]
        if len(case_rep_pairs) != len(set(case_rep_pairs)):
            raise ValueError("Case names must be unique")

        return inst


# =============================================================================
# Dataset
# =============================================================================


class GenDataset(torch.utils.data.Dataset):
    """General Dataset for Generation"""

    _allowed_task_modes = [
        "forward",  # forward simulation
        "iid",  # single-frame training
        "interp",  # state_interpolation training
    ]

    def __init__(
        self,
        config: str | GenDatasetConfig,
        repr_loader: Optional[DictConfig] = None,
        case_subset: Optional[Union[str, List[str]]] = None,
        relpath_to: Optional[str] = None,
        sort_by_seqlen: bool = False,
        **loader_kwargs,
    ):
        if isinstance(case_subset, str):
            case_subset = [case_subset]
        if isinstance(config, GenDatasetConfig):
            self.cfg = config
        else:
            self.cfg = GenDatasetConfig.from_json(
                config, case_subset=case_subset, relpath_to=relpath_to
            )
        self.dataset_name = self.cfg.name
        self.loader_cfg = LoaderConfig(**loader_kwargs)

        if sort_by_seqlen:
            self.cfg.cases = sorted(
                self.cfg.cases, key=lambda x: x.seqlen, reverse=True
            )  # Sort from long to short

        self.repr_loader = repr_loader

    def __len__(self):
        return len(self.cfg.cases)

    def process_coords(self, atom_coords, aatype):
        """Process atom37 coordinates with AF2/OpenFold pipeline"""
        all_atom_positions = torch.from_numpy(atom_coords)  # (..., L, 37, 3)
        all_atom_mask = torch.all(
            ~torch.isnan(all_atom_positions), dim=-1
        )  # (..., L, 37)

        all_atom_positions = torch.nan_to_num(
            all_atom_positions, 0.0
        )  # convert NaN to zero

        # OpenFold data transformation
        openfold_feat_dict = {
            "aatype": aatype.long(),
            "all_atom_positions": all_atom_positions.double(),
            "all_atom_mask": all_atom_mask.double(),
        }

        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles("")(
            openfold_feat_dict
        )
        openfold_feat_dict = data_transforms.make_pseudo_beta("")(openfold_feat_dict)

        return openfold_feat_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_cfg = self.cfg.cases[idx]
        seqres = case_cfg.seqres
        seqlen = case_cfg.seqlen
        n_frames = case_cfg.n_frames

        data_dict: dict[str, Any] = {
            "job_info": {
                "dataset": self.cfg.name,
                "case_id": case_cfg.case_id,
                "rep_id": case_cfg.rep_id,
                "n_replicates": case_cfg.n_replicates,
                "task_mode": case_cfg.task_mode,
                "seqres": seqres,
                "seqlen": seqlen,
                "n_frames": n_frames,
                "stride_in_10ps": case_cfg.stride_in_10ps,
            },
            "num_frames": n_frames,
            "task_mode": case_cfg.task_mode,
        }

        #### pretrained representations ####
        if self.repr_loader is not None:
            pretrained_repr = self.repr_loader.load(seqres=seqres)
            data_dict["pretrained_single"] = pretrained_repr.get(
                "pretrained_single", None
            )
            data_dict["pretrained_pair"] = pretrained_repr.get("pretrained_pair", None)

        #### Sequence info ####
        aatype = torch.LongTensor(
            [rc.restype_order_with_x[res] for res in seqres] * n_frames
        )  # (F * L)
        data_dict["aatype"] = rearrange(
            aatype.long(), "(F L) ->  L F", F=n_frames, L=seqlen
        )
        data_dict["torsion_angles_mask"] = rearrange(
            all_angles_mask_with_x[aatype],
            "(F L) ... C ->  L ...(F C)",
            F=n_frames,
            L=seqlen,
            check_inplace=False,
        )
        frame_idxs, cond_mask = case_cfg.get_frame_idxs()
        data_dict["pos_id"] = torch.from_numpy(frame_idxs).long()
        data_dict["ref_mask"] = torch.from_numpy(cond_mask).float()

        #### Load conditioning frames ####
        if case_cfg.conditions is not None:
            batch_atom_coords = case_cfg.conditions.load_coords(seqlen=seqlen)
            num_cond_frames = len(batch_atom_coords)

            # center structures by CA-coords
            batch_atom_coords = np.stack(batch_atom_coords, axis=0)  # (F_ref, L, 37, 3)
            batch_atom_coords -= np.nanmean(
                batch_atom_coords[
                    ..., rc.atom_order["CA"] : rc.atom_order["CA"] + 1, :
                ],
                axis=1,
                keepdims=True,
            )
            atom_coords = np.concatenate(
                batch_atom_coords, axis=0
            )  # [F_ref * L, 37, 3]
            # openfold coords prep pipeline
            cond_aatype = torch.LongTensor(
                [rc.restype_order_with_x[res] for res in seqres] * num_cond_frames
            )

            openfold_feat_dict = self.process_coords(atom_coords, cond_aatype)

            rigids_0 = ru.Rigid.from_tensor_4x4(
                openfold_feat_dict["rigidgroups_gt_frames"]
            )[:, 0].to_tensor_7()
            rigids_0 = rearrange(
                rigids_0,
                "(F L) C -> F L C",
                F=num_cond_frames,
                L=seqlen,
            )  # (F_ref, L, 7)
            pseudo_beta = rearrange(
                openfold_feat_dict["pseudo_beta"].float(),
                "(F L) C -> F L C",
                F=num_cond_frames,
                L=seqlen,
            )  # (F_ref, L, 3)
            pseudo_beta_mask = rearrange(
                openfold_feat_dict["pseudo_beta_mask"].float(),
                "(F L) -> F L",
                F=num_cond_frames,
                L=seqlen,
            )  # (F_ref, L)

            data_dict["rigids_0"] = rearrange(
                rigids_0,
                "F L ... C ->  L ... (F C)",
                F=num_cond_frames,
                L=seqlen,
                check_inplace=False,
            )
            data_dict["pseudo_beta"] = rearrange(
                pseudo_beta,
                "F L ... C ->  L ... (F C)",
                F=num_cond_frames,
                L=seqlen,
                check_inplace=False,
            )
            data_dict["pseudo_beta_mask"] = rearrange(
                pseudo_beta_mask, "F L ->  L F", F=num_cond_frames, L=seqlen
            )
            data_dict["atom14_gt_positions"] = rearrange(
                openfold_feat_dict["atom14_gt_positions"],
                "(F L) ... C ->  L ...(F C)",
                F=num_cond_frames,
                L=seqlen,
                check_inplace=False,
            )

        return data_dict

    @staticmethod
    def collate(batch_list, gt_ref_only=True):
        if all(data is None for data in batch_list):
            # deactivated this batch, used for scheduling valgen dataset
            return None

        batch = {"gt_feat": {}}
        gt_feat_name = [
            "rigids_0",
            "atom14_gt_positions",
            "pseudo_beta",
            "pseudo_beta_mask",
        ]
        lengths = torch.tensor(
            [feat_dict["aatype"].shape[0] for feat_dict in batch_list],
            requires_grad=False,
        )
        max_L = max(lengths)
        padding_mask = torch.arange(max_L).expand(
            len(lengths), max_L
        ) < lengths.unsqueeze(1)
        num_frames = set([feat_dict["num_frames"] for feat_dict in batch_list])
        assert len(num_frames) == 1, "All batch have the same number of frames"
        num_frames = list(num_frames)[0]

        #### Basic configs for padding ####
        __DATA_INFO = [
            "task_mode",
            "num_frames",
            "job_info",
        ]

        for key, val in batch_list[0].items():
            if val is None:
                continue

            # Batch tensors
            if key in __DATA_INFO:
                batched_val = [feat_dict[key] for feat_dict in batch_list]  # (B,)
            elif val.dim() == 0:
                # scalars
                batched_val = torch.stack([feat_dict[key] for feat_dict in batch_list])
            elif (val.dim() < 3) or (key not in ["pretrained_pair"]):
                # 1D or 2D tensors, pad and batch with new first dimension
                if len(batch_list) == 1:
                    batched_val = batch_list[0][key][
                        None, ...
                    ]  # just expand the first dim
                else:
                    batched_val = pad_sequence(
                        [feat_dict[key] for feat_dict in batch_list],
                        batch_first=True,
                        padding_value=0,
                    )
            else:
                assert key == "pretrained_pair"
                if len(batch_list) == 1:
                    batched_val = batch_list[0][key][
                        None, ...
                    ]  # just expand the first dim
                else:
                    batched_val = []
                    C = batch_list[0]["pretrained_pair"].shape[2]
                    for feat_dict in batch_list:
                        edge = feat_dict["pretrained_pair"]
                        L = edge.shape[0]
                        pad = torch.zeros(max_L, max_L, C)
                        pad[:L, :L, :] = edge
                        batched_val.append(pad[None, :])
                    batched_val = torch.cat(batched_val, dim=0)

            if key in ["rigids_0"]:
                # Fix rigids padding from [0, 0, 0, 0] to [1, 0, 0, 0]
                bsz, seqlen = batched_val.shape[:2]
                F = batched_val.shape[-1] // 7
                assert F * 7 == batched_val.shape[-1], (
                    f"Rigids tensor last dim ({batched_val.shape[-1]}) is not 7 * F ({F})"
                )
                batched_val = batched_val + torch.cat(
                    [~padding_mask[..., None], torch.zeros(bsz, seqlen, 6)], dim=-1
                ).repeat(1, 1, F)

            if key in ["ref_mask", "pos_id"]:
                # 1D feture
                batched_val = rearrange(
                    batched_val, "B F -> (B F)", F=num_frames
                ).squeeze()
            elif key not in __DATA_INFO:
                if key in [
                    "rigids_0",
                    "pseudo_beta",
                    "pseudo_beta_mask",
                    "atom14_gt_positions",
                ]:
                    num_frames_this_tensor = int(batch_list[0]["ref_mask"].sum().item())
                elif key in ["pretrained_pair", "pretrained_single"]:
                    num_frames_this_tensor = 1
                else:
                    num_frames_this_tensor = num_frames
                batched_val = rearrange(
                    batched_val,
                    "B L ... (F C) ->  (B F) L ... C",
                    F=num_frames_this_tensor,
                ).squeeze(dim=-1)

            if key in gt_feat_name:
                batch["gt_feat"][key] = batched_val
            else:
                batch[key] = batched_val

        batch["padding_mask"] = padding_mask.repeat_interleave(num_frames, dim=0)
        if "task_mode" in batch:
            assert len(set(batch["task_mode"])) == 1, (
                "All runs in a batch should have the same task mode. Consider set batch_size=1."
            )
            batch["task_mode"] = batch["task_mode"][0]
        if "num_frames" in batch:
            batch["num_frames"] = num_frames

        # Add generation batch tag
        batch["is_inference_batch"] = True
        return batch
