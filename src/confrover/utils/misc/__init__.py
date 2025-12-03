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

"""Misc tools"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
import pickle
import random
import shutil
import string
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from torch import Tensor

from .ext_types import ConfigLike, PathLike
from .pylogger import get_pylogger, log_header

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)


def download_file(
    url: str, dest_path: str | None = None, chunk_size: int = 8192, timeout: int = 15
) -> str:
    """
    Download a file from a public URL.

    Args:
        url: Public URL to download from.
        dest_path: Optional path to save to. If None, infer from URL in current directory.
        chunk_size: Bytes per chunk while streaming.
        timeout: Request timeout in seconds.

    Returns:
        The absolute path to the downloaded file.

    Raises:
        RuntimeError: If the download fails for any reason.
    """
    try:
        # Infer filename from URL if not provided
        if dest_path is None:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                raise ValueError(
                    "Could not infer filename from URL; please provide dest_path."
                )
            dest_path = filename

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with dest_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:  # skip keep-alive chunks
                        continue
                    f.write(chunk)

        return str(dest_path.resolve())

    except Exception as e:
        raise RuntimeError(f"Failed to download {url!r} -> {dest_path!r}: {e}") from e


def get_persist_tmp_fpath(suffix=".pdb"):
    """Get a temp file path. The file is not deleted until the end of the session."""
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name


def unbatch_dict(d):
    """unbatch the lenght 1 dict object due to PyG batching"""
    if isinstance(d, dict):
        return {key: unbatch_dict(val) for key, val in d.items()}
    elif isinstance(d, list):
        return d[0]
    elif isinstance(d, Tensor):
        if d.shape == (1,):
            # single value Tensor
            return d[0].cpu().item()
        else:
            # other tensor
            return d[0]
    else:
        raise TypeError(f"Unknown type: {d}")


def gather_all_files(file_root, remove_subfolder=False):
    """Gather all files in sub-folders under the file_root"""
    file_root = Path(file_root)
    all_files = [f for f in file_root.rglob("*.*") if f.is_file()]
    for f in all_files:
        if not file_root.joinpath(f.name).exists():
            shutil.move(str(f), str(file_root))
    if remove_subfolder:
        all_dirs = [f for f in file_root.glob("*") if f.is_dir()]
        for d in all_dirs:
            shutil.rmtree(str(d))


def flatten_list(l):
    """Flatten a list recursively"""
    if isinstance(l, (list, tuple)):
        f_l = []
        for a in l:
            f_l += flatten_list(a)
        return f_l
    else:
        return [l]


def replace_dot_key_val(d, dot_key, replace_to, inplace=True, ignore_error=False):
    """Replace the value in an hierachical dict with dot format key"""
    if not inplace:
        from copy import deepcopy

        d = deepcopy(d)
    key_levels = dot_key.split(".")
    node = d
    try:
        if len(key_levels) > 1:
            for key in key_levels[:-1]:
                node = node[key]
        assert key_levels[-1] in node.keys(), f"{dot_key} not found"
        node[key_levels[-1]] = replace_to
    except Exception as e:
        if ignore_error:
            pass
        else:
            raise e
    return d


def unique_dir(path: str | Path, suffix_len: int = 2) -> Path:
    """
    Ensure a unique directory path by adding a random suffix if it already exists.

    Args:
        path (str | Path): Base directory path.
        suffix_len (int): Length of the random suffix.

    Returns:
        Path: A unique path (does not exist yet).
    """
    path = Path(path)

    # Generate a unique path if it already exists
    new_path = path
    while new_path.exists():
        suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=suffix_len)
        )
        new_path = path.parent / f"{path.name}_{suffix}"

    return new_path
