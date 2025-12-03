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

# tests/conftest.py
from __future__ import annotations

import os
import pathlib
import warnings

import pytest

#### Global setup ####

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
    ":4096:8"  # Setting CUBLAS config to enforce deterministic algorithms
)


def pytest_collection_modifyitems(config, items):
    """Skip redundent tests"""
    pass


#### Session fixtures ####
@pytest.fixture(autouse=True, scope="session")
def setup_warnings():
    warnings.filterwarnings(
        "ignore",
    )


@pytest.fixture(autouse=True, scope="session")
def patch_openfold_deepspeed_bug():
    from confrover.utils import test_utils

    with test_utils.patch_openfold_deepspeed_bug():
        yield


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def test_data_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "tests" / "test_data"
