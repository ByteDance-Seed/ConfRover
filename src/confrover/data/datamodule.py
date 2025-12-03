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
"""ConfRover DataModule"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig

from confrover.utils import get_pylogger

logger = get_pylogger(__name__)


class ConfRoverDataModule(LightningDataModule):
    """
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        # shared_cfg: DictConfig,
        train_dataset=None,
        val_dataset=None,
        valgen_dataset=None,
        gen_dataset=None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.valgen_dataset = self._make_dataset_dict(valgen_dataset)
        self.pred_dataset = self._make_dataset_dict(gen_dataset)

    def _make_dataset_dict(self, dataset):
        if isinstance(dataset, (dict, DictConfig)):
            return dataset
        elif isinstance(dataset, (list, ListConfig)):
            return {ds.dataset_name: ds for ds in dataset}
        elif dataset is not None:
            return {dataset.dataset_name: dataset}
        else:
            assert dataset is None, "Unrecognized dataset cfg"
            return dataset

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate,
            **self.train_dataset.loader_cfg.to_dict(),
        )
        return train_loader

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate,
            **self.val_dataset.loader_cfg.to_dict(),
        )
        if self.valgen_dataset is not None:
            val_loader = [val_loader]
            for dataset in self.valgen_dataset.values():
                val_loader.append(
                    torch.utils.data.DataLoader(
                        dataset=dataset,
                        collate_fn=dataset.collate,
                        **dataset.loader_cfg.to_dict(),
                    )
                )

        return val_loader

    def test_dataloader(self):
        test_loader = [
            torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                **dataset.loader_cfg.to_dict(),
            )
            for dataset in self.pred_dataset.values()
        ]
        if len(test_loader) == 1:
            return test_loader[0]
        return test_loader

    def predict_dataloader(self):
        pred_loader = [
            torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                **dataset.loader_cfg.to_dict(),
            )
            for dataset in self.pred_dataset.values()
        ]
        if len(pred_loader) == 1:
            return pred_loader[0]
        return pred_loader

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
