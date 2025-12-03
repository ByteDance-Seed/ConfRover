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

"""Environmental variable setup"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from confrover.utils import PathLike, get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Components
# =============================================================================

DEFAULT_CACHE_DIR = "./confrover_cache"


@dataclass
class CachePaths:
    """Dataclass to compose default and custom cache paths"""

    root: PathLike = DEFAULT_CACHE_DIR
    confrover_ckpts: PathLike = "{root}/confrover_ckpts"
    msa: PathLike = "{root}/msa"
    folding_repr: PathLike = "{root}/folding_repr"
    openfold_params: PathLike = "{root}/openfold_params"
    igso3: PathLike = "{root}/igso3"
    cutlass: PathLike = "{root}/cutlass"

    def _is_default(self, name, value):
        """Check if the value is the default value"""
        if name == "root":
            return str(value) == str(Path(DEFAULT_CACHE_DIR).resolve())
        else:
            return str(value) == str(Path(DEFAULT_CACHE_DIR).joinpath(name).resolve())

    def __post_init__(self):
        """Ensure all paths are not None, absolute, and coarsed to pathlib.Path"""
        for name, value in self.__dict__.items():
            if name == "root":
                if value is None:
                    value = Path(DEFAULT_CACHE_DIR).resolve()  # revert to default
            else:
                value = str(value)
                if self._is_default(name, value):
                    value = f"{self.root}/{name}"  # default to root/name
                if value.startswith("{root}"):
                    value = value.format(root=self.root)
            setattr(self, name, Path(value).resolve())

    def info(self):
        return [f"{k}: {v}" for k, v in self.__dict__.items()]


CONFROVER_VERSION = version("confrover")
