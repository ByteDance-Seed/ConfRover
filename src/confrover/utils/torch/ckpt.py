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

"""Tools for model checkpoints"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
from huggingface_hub import ModelHubMixin
from lightning.pytorch import LightningModule

from confrover.utils.misc import get_pylogger

from .zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

log = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================


PathLike = Union[str, Path]

# =============================================================================
# Components
# =============================================================================


def _is_deepspeed_ckpt(ckpt_path: PathLike) -> bool:
    sub_ckpt_path = Path(ckpt_path) / "checkpoint"
    mp_rank_00_model_states_fp = sub_ckpt_path / "mp_rank_00_model_states.pt"
    return sub_ckpt_path.exists() and mp_rank_00_model_states_fp.exists()


def _is_torch_ckpt(ckpt_path: PathLike) -> bool:
    ckpt_path = Path(ckpt_path)
    return ckpt_path.is_file() and ckpt_path.suffix in [".pt", ".pth"]


def _is_huggingface_ckpt(ckpt_path: PathLike) -> bool:
    ckpt_path = Path(ckpt_path)
    return len(list(ckpt_path.glob("*.safetensors"))) > 0


def load_state_dict_from_checkpoint(ckpt_path: PathLike):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if _is_deepspeed_ckpt(ckpt_path):
        log.info(f"Loading DeepSpeed checkpoint from {ckpt_path} ...")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = OrderedDict(
            {
                key.replace("_forward_module.", ""): val
                for key, val in state_dict.items()
            }
        )
    elif _is_torch_ckpt(ckpt_path):
        log.info(f"Loading torch checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise NotImplementedError(
            "Can only load state_dict from pytorch or deepspeed checkpoints."
        )
    return state_dict


def load_model_checkpoint(
    model: LightningModule, ckpt_path: PathLike, strict: bool = True
):
    """Load model weights from checkpoint

    Args:
        ckpt_path (Path): Path to checkpoint
    Returns:
        Dict: State dict
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if _is_deepspeed_ckpt(ckpt_path):
        log.info(f"Loading DeepSpeed checkpoint from {ckpt_path} ...")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = OrderedDict(
            {
                key.replace("_forward_module.", ""): val
                for key, val in state_dict.items()
            }
        )
        assert isinstance(model, LightningModule), (
            "Deepspeed checkpoint requires model being a subclass of LightningModule"
        )
        model.on_load_checkpoint(state_dict)
        check_state_dict_matches(state_dict, model)
        model.load_state_dict(state_dict=state_dict, strict=strict)
    elif _is_torch_ckpt(ckpt_path):
        log.info(f"Loading torch checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        assert isinstance(model, LightningModule), (
            "Deepspeed checkpoint requires model being a subclass of LightningModule"
        )
        model.on_load_checkpoint(state_dict)
        check_state_dict_matches(state_dict, model)
        model.load_state_dict(state_dict=state_dict, strict=strict)
    elif _is_huggingface_ckpt(ckpt_path):
        assert isinstance(model, ModelHubMixin), (
            "Huggingface checkpoint requires model being a subclass of ModelHubMixin"
        )
        model.from_pretrained(ckpt_path)
    else:
        raise ValueError(
            f"Unknown checkpoint type: {ckpt_path}. It must be a deepspeed, torch or huggingface checkpoint."
        )
    return model


def check_state_dict_matches(
    state_dict: OrderedDict, model: torch.nn.Module, strict: bool = False
):
    """Check if state_dict matches model weight keys

    Args:
        state_dict (OrderedDict): State dict
        model (nn.Module): Model
    Returns:
        Dict: State dict
    """
    missing_in_model = []
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            missing_in_model.append(key)
            # print(f'[Missing weights in model] {key}')
    missing_in_state_dict = []
    for key in list(model.state_dict().keys()):
        if key not in state_dict.keys():
            missing_in_state_dict.append(key)
            # print(f'[Missing weights in ckpt] {key}')

    log.info("==============> Check model weights <================")
    if len(missing_in_model) > 0:
        if strict:
            raise ValueError(
                f"Missing keys in model:\n\t-" + "\n\t-".join(missing_in_model) + "\n"
            )
        else:
            log.info(
                f"Missing keys in model:\n\t-" + "\n\t-".join(missing_in_model) + "\n"
            )
    else:
        log.info("Missing keys in model: none")

    if len(missing_in_state_dict) > 0:
        if strict:
            raise ValueError(
                f"Missing keys in state_dict:\n\t-"
                + "\n\t-".join(missing_in_state_dict)
                + "\n"
            )
        else:
            log.info(
                f"Missing keys in state_dict:\n\t-"
                + "\n\t-".join(missing_in_state_dict)
                + "\n"
            )
    else:
        log.info("Missing keys in ckpt: none")
    log.info("=====================================================")
