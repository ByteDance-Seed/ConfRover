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

import contextlib


@contextlib.contextmanager
def patch_openfold_deepspeed_bug():
    """
    Context manager to patch the OpenFold deepspeed bug with cpu testing.
    """
    import openfold.model.primitives as primitives

    original_deepspeed_is_installed = primitives.deepspeed_is_installed
    primitives.deepspeed_is_installed = False
    try:
        yield
    finally:
        primitives.deepspeed_is_installed = original_deepspeed_is_installed
