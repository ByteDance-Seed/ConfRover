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

import tempfile
from collections.abc import Iterable
from pathlib import Path

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from lightning import LightningDataModule, LightningModule, seed_everything
from lightning.pytorch.utilities import move_data_to_device
from omegaconf import DictConfig, OmegaConf, open_dict

from confrover.inference import (
    DEFAULT_CONFIG,
    INFER_CONFIG,
    compose_hydra_config,
)
from confrover.utils.misc import get_pylogger

log = get_pylogger(__name__)


def create_infer_config(
    *,
    tmp_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    reset_global_hydra: bool = True,
    disable_model_summary_callback: bool = True,
):
    if reset_global_hydra:
        GlobalHydra.instance().clear()

    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp(prefix="confrover_test_")
    tmp_dir = Path(tmp_dir)

    output_dir = tmp_dir / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg: DictConfig = OmegaConf.load(INFER_CONFIG)
    model_cfg = DEFAULT_CONFIG["confrover"]

    cfg = compose_hydra_config(
        infer_config=infer_cfg,
        model_config=model_cfg,
        cli_overrides=[
            f"model_ckpt=null",
            f"writer.output_dir={output_dir}",
            f"data.gen_dataset.batch_size=1",
            f"model.use_deepspeed_evo_attention=false",
            *(overrides or []),
        ],
    )

    # Disable model summary callback if requested
    if disable_model_summary_callback:
        with open_dict(cfg.callbacks):
            cfg.callbacks.pop("model_summary", None)

    return cfg


def infer_fast_test_run(cfg: DictConfig):
    """A fast single run for inference"""

    # set seed for random number generators in pytorch, numpy and python.random
    assert cfg.seed is not None, "Seed must be specified for testing"
    seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.writer = hydra.utils.instantiate(cfg.writer)
    model.decoder.sampler = hydra.utils.instantiate(cfg.sampler)
    # We don't load ckpt in this unit test

    log.info("Starting testing!")
    # Rerun sampling for the first batch
    assert torch.cuda.is_available(), "Must use CUDA to test inference."
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    batch = next(iter(datamodule.test_dataloader()))
    batch = move_data_to_device(batch, device)
    output = model._ar_sample(**batch)
    return output
