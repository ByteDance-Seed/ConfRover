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

from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path

import mdtraj
import numpy as np
from lightning.pytorch import seed_everything

from confrover.model.utils.writer import Writer
from confrover.utils import get_pylogger
from confrover.utils.test_utils import test_infer_utils

logger = get_pylogger(__name__)


def test_infer_produces_expected_pdb(test_data_dir: pathlib.Path, seed: int = 42):
    logger.warning("Running test_infer_produces_expected_pdb")
    seed_everything(seed, workers=True)  # Ensure reproducibility
    atlas_test_small_json = test_data_dir / "atlasTest_fwd_100ns.json"
    atlas_test_input = test_data_dir / "atlas"
    openfold_repr_path = test_data_dir / "openfold_repr"
    expected_metric_path = Path(__file__).parent / "test_infer_expected_output.npz"

    cfg = test_infer_utils.create_infer_config(
        overrides=[
            f"data.gen_dataset.config='{str(atlas_test_small_json)}'",
            f"data.gen_dataset.relpath_to={str(atlas_test_input)}",
            f"data.gen_dataset.repr_loader.repr_root={str(openfold_repr_path)}",
            f"data.gen_dataset.sort_by_seqlen=True",
            f"sampler.diffusion_steps=2",
        ],
        reset_global_hydra=True,
    )
    print(cfg["data"]["gen_dataset"])

    outputs = test_infer_utils.infer_fast_test_run(cfg)
    atom37, atom37_mask, aatype, padding_mask = (
        outputs["atom37"].cpu().numpy(),
        outputs["atom37_mask"].cpu().numpy(),
        outputs["aatype"].cpu().numpy(),
        outputs["padding_mask"].cpu().numpy(),
    )

    if not expected_metric_path.exists():
        # write to expected metric
        np.savez(
            expected_metric_path,
            atom37=atom37,
            atom37_mask=atom37_mask,
            aatype=aatype,
            padding_mask=padding_mask,
            chain_name=outputs["info"]["chain_name"],
        )
    else:
        expected = np.load(expected_metric_path)

        try:
            expected_prefix = "_".join(expected["chain_name"].item().split("_")[:2])
            output_prefix = "_".join(outputs["info"]["chain_name"].split("_")[:2])
            assert expected_prefix == output_prefix, (
                f"chain_name mismatch: {output_prefix} (expect: {expected_prefix})"
            )
            np.testing.assert_array_equal(expected["atom37"], atom37)
            np.testing.assert_array_equal(expected["atom37_mask"], atom37_mask)
        except AssertionError as e:
            # save new expected metrics
            current_output_path = Path(__file__).parent / "test_infer_actual_output.npz"
            np.savez(
                current_output_path,
                atom37=atom37,
                atom37_mask=atom37_mask,
                aatype=aatype,
                padding_mask=padding_mask,
                chain_name=outputs["info"]["chain_name"],
            )
            raise e


def test_save_xtc():
    expected_metric_path = Path(__file__).parent / "test_infer_expected_output.npz"
    output = np.load(expected_metric_path)

    tmp_dir = tempfile.mkdtemp(prefix="confrover_test_")
    tmp_dir = Path(tmp_dir)

    pdb_writer = Writer(output_dir=tmp_dir / "pdb", output_format="pdb")
    xtc_writer = Writer(output_dir=tmp_dir / "xtc", output_format="xtc")

    bsz = output["atom37_mask"].shape[0]
    for i in range(bsz):
        job_info = {
            "task_mode": "forward",
            "case_id": "test_case_1",
            "rep_id": i,
        }

        pdb_writer.write(
            aatype=output["aatype"][i],
            atom37=output["atom37"][i],
            atom37_mask=output["atom37_mask"][i],
            padding_mask=output["padding_mask"][i],
            job_info=job_info,
        )
        xtc_writer.write(
            aatype=output["aatype"][i],
            atom37=output["atom37"][i],
            atom37_mask=output["atom37_mask"][i],
            padding_mask=output["padding_mask"][i],
            job_info=job_info,
        )

    # Load back and test
    pdb_traj = mdtraj.load(
        [
            str(
                tmp_dir
                / "pdb"
                / "test_case_1"
                / f"test_case_1_sample0"
                / f"test_case_1_sample0_frame{j}.pdb"
            )
            for j in range(3)
        ]
    )
    xtc_traj = mdtraj.load(
        str(tmp_dir / "xtc" / "test_case_1" / f"test_case_1_sample0.xtc"),
        top=str(tmp_dir / "xtc" / "test_case_1" / f"test_case_1_sample0.pdb"),
    )

    xtc_traj = xtc_traj.superpose(xtc_traj, frame=0)
    pdb_traj = pdb_traj.superpose(xtc_traj, frame=0)
    np.testing.assert_almost_equal(pdb_traj.xyz, xtc_traj.xyz, decimal=3)
