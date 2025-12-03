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

"""Run ConfRover generation"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List

import hydra
import torch
from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from confrover import PACKAGE_ROOT
from confrover.data.infer import GenDatasetConfig
from confrover.data.pretrain_repr.openfold.loader import OpenFoldReprLoader
from confrover.env import CachePaths
from confrover.model import ConfRover, ModelRegistry
from confrover.utils import PathLike, get_pylogger, hydra_utils, log_header
from confrover.utils.misc.cli import str2bool
from confrover.utils.misc.install import check_and_install_dependencies
from confrover.utils.torch.ckpt import load_model_checkpoint

log = get_pylogger(__name__)


# =============================================================================
# Setup
# =============================================================================
os.environ["NCCL_DEBUG"] = "WARN"  # Show less NCCL info

INFER_CONFIG = PACKAGE_ROOT / "configs" / "inference.yaml"

DEFAULT_CONFIG = {
    cfg.stem: cfg for cfg in PACKAGE_ROOT.joinpath("configs", "model").glob("*.yaml")
}

torch.set_float32_matmul_precision("high")

DEFAULT_PATH = CachePaths()

NUM_AVAIL_GPUS = torch.cuda.device_count()


# =============================================================================
# Components
# =============================================================================


def compose_hydra_config(
    infer_config: PathLike | DictConfig,
    model_config: PathLike | dict | DictConfig,
    cli_overrides: List[str] = [],
) -> DictConfig:
    """Compose hydra config from CLI inputs. Keep all generation configs in one .yaml file"""

    # Merge infer_config with model_config. infer_config will overwrite model config
    infer_config = hydra_utils.to_cfg(infer_config)
    model_config = hydra_utils.wrap_under("model", hydra_utils.to_cfg(model_config))
    cfg = OmegaConf.merge(model_config, infer_config)

    # compose model config
    cli_overrides = OmegaConf.from_dotlist(cli_overrides)
    cfg: DictConfig = OmegaConf.merge(cfg, cli_overrides)

    return cfg


def generate(
    cfg: DictConfig,
    env: CachePaths,
    # data preparation control
    num_workers: int,
    msa_max_query_size: int,
) -> Trainer:
    log.info(log_header(log, "Prepare input info"))

    gen_dataset = GenDatasetConfig.from_json(cfg.data.gen_dataset.config)
    seqres_index_pairs = list(
        {case.seqres: case.case_id for case in gen_dataset.cases}.items()
    )

    repr_loader = OpenFoldReprLoader(repr_root=env.folding_repr)
    repr_loader.generate_repr(
        seqres_index_pairs=seqres_index_pairs,
        msa_root=env.msa,
        openfold_params=env.openfold_params,
        num_gpus=num_workers,
        overwrite=False,
        msa_max_query_size=msa_max_query_size,
    )

    log.info(log_header(log, "Run ConfRover"))
    seed_everything(cfg.seed, workers=True)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: ConfRover = hydra.utils.instantiate(cfg.model)
    model.writer = hydra.utils.instantiate(cfg.writer)
    model.decoder.sampler = hydra.utils.instantiate(cfg.sampler)
    assert cfg.model_ckpt is not None, "model_ckpt not found"
    model = load_model_checkpoint(model, cfg.model_ckpt, strict=True)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Setup output
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # model.output_dir = str(output_dir)
    save_cfg = rank_zero_only(OmegaConf.save)
    infer_yaml = output_dir / "inference.yaml"
    if os.path.exists(infer_yaml):
        log.warning(
            f"inference.yaml file already exists, overwriting the file: {infer_yaml}"
        )
    save_cfg(OmegaConf.to_container(cfg, resolve=True), str(infer_yaml))

    # run test
    log.info("Starting sampling ...")
    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=None)
    finish_time = perf_counter()
    peak_mem = torch.cuda.max_memory_allocated(trainer.local_rank) / 1e9  # MB
    print(
        f"[Rank {trainer.local_rank}] finished at {datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')}"
    )
    with open(output_dir / f"rank_{trainer.local_rank}_runtime.log", "w") as handle:
        handle.write(f"start_time: {start_time}\n")
        handle.write(f"finish_time: {finish_time}\n")
        handle.write(f"walltime: {(finish_time - start_time) / 60:.2f} min")
        handle.write(f"walltime: {peak_mem:.2f} GB")

    trainer.strategy.barrier()  # wait until all test subprocess stops
    print(
        f"[Rank {trainer.local_rank}] exited at {datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')}"
    )
    return trainer


def cli(args: argparse.Namespace):
    """Parse args from CLI and overrides default hydra yaml config"""

    #### 1. Setup cache dir ####
    log.info(log_header(log, "ConfRover cache paths"))
    env = CachePaths(
        root=args.cache_dir,
        msa=args.msa_root,
        folding_repr=args.folding_repr,
    )
    log.info("   - " + "\n   - ".join(env.info()))

    #### 2. Job preparation ####
    log.info(log_header(log, "Job preparation"))
    check_and_install_dependencies(env)

    infer_cfg: DictConfig = OmegaConf.load(INFER_CONFIG)

    if Path(args.model).exists():
        # provided a model path
        model_ckpt = str(Path(args.model).resolve())
    else:
        # try registered model
        model_ckpt = ModelRegistry(ckpt_dir=env.confrover_ckpts).get_model_ckpt(
            args.model
        )

    model_cfg = torch.load(model_ckpt)["model_cfg"]
    args.model_ckpt = model_ckpt

    args.output_dir = str(Path(args.output) / Path(args.job_config).stem)

    cfg = compose_hydra_config(
        infer_config=infer_cfg,
        model_config=model_cfg,
        cli_overrides=[
            f"seed={args.seed}",
            f"model_ckpt={args.model_ckpt}",
            f"data.gen_dataset.config={args.job_config}",
            f"data.gen_dataset.repr_loader.repr_root={env.folding_repr}",
            f"data.gen_dataset.batch_size={args.batch_size}",
            f"data.gen_dataset.num_workers={args.num_data_workers}",
            f"data.gen_dataset.sort_by_seqlen={args.sort_by_seqlen}",
            # writer
            f"output_dir={args.output_dir}",
            f"writer.output_format={args.output_format}",
            f"writer.preview_frames={args.output_pdb_preview}",
            # sampling cfg
            f"sampler.diffusion_steps={args.diffusion_steps}",
            f"model.seed={args.seed}",
            f"model.use_deepspeed_evo_attention={args.use_kernel}",
            f"model.kv_cache_type={args.kv_cache_type}",
        ],
    )

    #### 3. Run generation ####
    trainer = generate(
        cfg,
        env,
        num_workers=args.num_workers,
        msa_max_query_size=args.msa_max_query_size,
    )

    #### 4. Post process ####
    trainer.lightning_module.writer.cleanup()

    log.info(log_header(log, "✅ Job completion"))
    log.info(f"Output directory: {args.output_dir}")


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    job_group = parser.add_argument_group(title="Job config")
    # Job config
    job_group.add_argument(
        "--job_config",
        type=str,
        required=True,
        metavar="<path>",
        help="Path to job manifest file for generation",
    )
    job_group.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="<path>",
        help="Path to output directory",
    )
    job_group.add_argument(
        "--output_format",
        type=str,
        metavar="[auto|xtc|pdb]",
        default="auto",
        help="Saving format for generated trajectories. Options: 'auto', 'xtc', 'pdb'. When set to 'auto', 'pdb' will be used for iid task and 'xtc' will be used for trajectory sampling task.",
    )
    job_group.add_argument(
        "--output_pdb_preview",
        type=int,
        default=20,
        metavar="<int>",
        help="If --output_format is 'xtc' or 'auto', save a corresponding PDB preview file with provided number of frames. Set to 0 to disable.",
    )
    job_group.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="<int>",
        help="Random seed to ensure reproducibility",
    )

    # Model config
    model_group = parser.add_argument_group(title="Model config")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="<name|path>",
        help="Model name or path to model ckpt .pt.",
    )
    model_group.add_argument(
        "--use_kernel",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        metavar="true|false",
        help="Use DeepSpeed4Sci's evoformer kernels to reduce GPU memory usage and accelerate the sampling.",
    )
    model_group.add_argument(
        "--kv_cache_type",
        type=str,
        default="offloaded",
        metavar="<str>",
        help="KV cache type to use. Options: 'offloaded' (Offload to memory), sink{n}:{m} (SinkCache with n sink tokens at the beginning of the sequence and m sliding window tokens). Default offloaded.",
    )
    model_group.add_argument(
        "--diffusion_steps",
        type=int,
        default=200,
        metavar="<int>",
        help="Number of diffusion steps to sample each conformation",
    )
    model_group.add_argument(
        "--num_workers",
        type=int,
        default=NUM_AVAIL_GPUS,
        metavar="<int>",
        help="Number of GPU workers",
    )

    # Data
    data_group = parser.add_argument_group(title="Data config")
    data_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        metavar="<int>",
        help="Batch size for sampling. Recommend set to 1 for trajectory sampling task and higher for the iid sampling task",
    )
    data_group.add_argument(
        "--num_data_workers",
        type=int,
        default=4,
        metavar="<int>",
        help="Number of Dataloader workers",
    )
    data_group.add_argument(
        "--msa_max_query_size",
        type=int,
        default=32,
        metavar="<int>",
        help="Max MSA query size for ColabFold's server.",
    )
    data_group.add_argument(
        "--sort_by_seqlen",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Sort case order by sequence length (descending). Can be used for quick OOM test.",
    )

    # Cache paths
    cache_group = parser.add_argument_group(
        title="Args to set relevant cache paths.",
    )
    cache_group.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_PATH.root,
        metavar="<path>",
        help="Path to root cache directory.",
    )
    cache_group.add_argument(
        "--msa_root",
        type=str,
        metavar="<path>",
        default=DEFAULT_PATH.msa,
        help="Path to MSA directory.",
    )
    cache_group.add_argument(
        "--folding_repr",
        type=str,
        metavar="<path>",
        default=DEFAULT_PATH.folding_repr,
        help="Path to folding representation directory.",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ConfRover generation",
        description="Script to run ConfRover generation for 1) forward simulation; 2) conformation interpolation; 3) time-independent ensemble sampling",
    )
    parser = add_args(parser)
    args = parser.parse_args()
    cli(args)
