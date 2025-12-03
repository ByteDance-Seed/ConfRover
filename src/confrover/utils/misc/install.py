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

"""Installation utilities"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from confrover.env import CachePaths
from confrover.utils import get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Components
# =============================================================================


def check_and_patch_openfold(env: CachePaths):
    from openfold import __file__ as _openfold_path

    stereo_chemical_props_fpath = (
        Path(_openfold_path).parent / "resources/stereo_chemical_props.txt"
    )
    if not stereo_chemical_props_fpath.exists():
        logger.info(
            f"'stereo_chemical_props.txt' not found under {stereo_chemical_props_fpath.parent}"
        )
        logger.info(
            f"Copying 'stereo_chemical_props.txt' to {stereo_chemical_props_fpath}"
        )
        source_fpath = Path(__file__).parent.parent.parent.joinpath(
            "_patch", "openfold", "stereo_chemical_props.txt"
        )
        shutil.copy2(source_fpath, stereo_chemical_props_fpath)


def install_cutlass(env: CachePaths):
    """
    Clone cutlass into $CUTLASS_PATH iff the directory does not already exist.

    Behavior mirrors: [ ! -d $CUTLASS_PATH ] && git clone <repo> -b <branch> --single-branch <path>
    """

    cutclass_path = os.environ.get("CUTLASS_PATH", env.cutlass)

    cutclass_path = Path(cutclass_path).expanduser()
    os.environ["CUTLASS_PATH"] = str(cutclass_path)  # set CUTCLASS_PATH

    # If exists (and is a directory), do nothing.
    if cutclass_path.exists():
        if cutclass_path.is_dir():
            return cutclass_path
        raise RuntimeError(
            f"CUTLASS path exists but is not a directory: {cutclass_path}"
        )

    # Ensure parent exists
    cutclass_path.parent.mkdir(parents=True, exist_ok=True)

    # Require git
    if shutil.which("git") is None:
        raise RuntimeError(
            "`git` not found in PATH; cannot clone https://github.com/NVIDIA/cutlass."
        )

    # Build clone command
    cmd = [
        "git",
        "clone",
        "https://github.com/NVIDIA/cutlass",
        "-b",
        "v3.7.0",
        "--single-branch",
        str(cutclass_path),
    ]

    # Run
    logger.info(f"cutlass not found. Installing cutlass into {cutclass_path}")
    proc = subprocess.run(
        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        # Clean up partial directory if git left one behind
        try:
            if cutclass_path.exists() and not any(cutclass_path.iterdir()):
                cutclass_path.rmdir()
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to clone CUTLASS into {cutclass_path}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return cutclass_path


def check_and_install_dependencies(env: CachePaths):
    check_and_patch_openfold(env)
    install_cutlass(env)
