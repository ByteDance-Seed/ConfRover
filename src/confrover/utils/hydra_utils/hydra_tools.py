# Copyright 2021 ashleve
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under MIT, with the full license text available in the folder.
#
# This modified file is released under the same license.

from __future__ import annotations

import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Mapping

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from confrover.utils.misc import ConfigLike, PathLike

from ..misc.pylogger import get_pylogger
from . import rich_utils

log = get_pylogger(__name__)


def load_hydra_config(config: PathLike, overrides: list[str] = []) -> DictConfig:
    """Load hydra config from .yaml files"""
    GlobalHydra.instance().clear()

    config = Path(config)
    assert config.exists(), f"Hydra config not found: {config}"
    __config_base = config.parent
    __config_name = config.stem
    initialize_config_dir(str(__config_base))
    return compose(
        config_name=__config_name,
        overrides=overrides,
    )


def to_cfg(x: ConfigLike) -> DictConfig:
    """Convert a config like object to DictConfig"""
    if x is None:
        return OmegaConf.create({})
    if isinstance(x, DictConfig):
        return x
    if isinstance(x, (str, Path)):
        return OmegaConf.load(str(x))  # type:ignore
    if isinstance(x, Mapping):
        return OmegaConf.create(x)  # type:ignore
    raise TypeError(f"Unsupported config type: {type(x)}")


def wrap_under(key: str, cfg: DictConfig) -> DictConfig:
    """Ensure the given cfg is nested under top-level `key` (e.g., 'model')."""
    # If already like {'model': ...}, pass through
    if key in cfg and len(cfg.keys()) == 1:
        return cfg
    # Otherwise, put the whole cfg under that key
    return OmegaConf.create({key: cfg})


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb  # type:ignore

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
