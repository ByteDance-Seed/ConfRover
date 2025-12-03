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

"""OpenFold utilities"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Union

from lightning.pytorch.utilities import rank_zero_only

from confrover.utils import get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================


# GDrive download
FILE_ID = "1GVzZA2nbdBbz6TKydvzquhfELJ3Movnb"
FILENAME = "openfold_params_07_22.zip"


# =============================================================================
# Functions
# =============================================================================


@rank_zero_only
def run_command(cmd, capture_output: bool = False, sudo=False):
    if sudo and shutil.which("sudo"):
        cmd.insert(0, "sudo")
    try:
        print(f"Running: {' '.join(cmd)}")
        if capture_output:
            out = subprocess.check_output(cmd, text=True)
        else:
            subprocess.check_call(cmd)
            out = ""
        return out
    except subprocess.CalledProcessError:
        print("Command failed:", " ".join(cmd))
        sys.exit(1)


def _parse_confirm_and_uuid(html: str) -> tuple[str | None, str | None]:
    """
    Extract confirm token and uuid from the first HTML.
    Handles current Google Drive large-file page structure.
    """
    # Preferred: hidden inputs in the form
    m_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    confirm = m_confirm.group(1) if m_confirm else None
    uuid = m_uuid.group(1) if m_uuid else None
    return confirm, uuid


@rank_zero_only
def download_openfold_params(
    download_dir: Union[Path, str],
):
    download_par_dir = Path(download_dir).parent

    logger.info(f"Downloading OpenFold params to {download_dir} ...")

    download_par_dir.mkdir(parents=True, exist_ok=True)

    ## 1. Downloading from gdrive with large file check
    cookie_file = Path(tempfile.gettempdir()) / f"cookies_{os.getpid()}.txt"
    try:
        # First request: get the HTML (to stdout) and save cookies.
        first_url = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
        html = run_command(
            [
                "wget",
                "--quiet",
                "--save-cookies",
                str(cookie_file),
                "--keep-session-cookies",
                "--no-check-certificate",
                first_url,
                "-O-",
            ],
            capture_output=True,
        )
        confirm, uuid = _parse_confirm_and_uuid(str(html))

        # Build second URL
        if confirm and uuid:
            second_url = (
                "https://drive.usercontent.google.com/download"
                f"?id={FILE_ID}&export=download&confirm={confirm}&uuid={uuid}"
            )
        else:
            raise NotImplementedError("No confirm/uuid token found in HTML.")

        dest = download_par_dir / FILENAME

        run_command(
            [
                "wget",
                "--load-cookies",
                str(cookie_file),
                "--no-check-certificate",
                second_url,
                "-O",
                str(dest),
            ]
        )
    finally:
        try:
            cookie_file.unlink()
        except FileNotFoundError:
            pass

    ## 2. extract weights and clean up
    with zipfile.ZipFile(dest) as z:
        z.extractall(download_par_dir)
    if Path(download_dir).stem != "openfold_params":
        shutil.move(download_par_dir / "openfold_params", download_dir)
    shutil.move(dest, download_dir)

    print(f"✅ OpenFold parameters downloaded to: {download_dir}")
