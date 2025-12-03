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

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Literal, Optional

import re
import hydra
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.utilities import move_data_to_device
from omegaconf import DictConfig, OmegaConf
from openfold.data.data_transforms import pseudo_beta_fn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from confrover.data.infer import (
    GenCaseConfig,
    GenDataset,
    GenDatasetConfig,
    PDBConditions,
    XTCConditions,
)
from confrover.data.pretrain_repr import OpenFoldReprLoader
from confrover.env import CachePaths
from confrover.model.decoder import Decoder
from confrover.model.decoder.confdiff.sampler.euler import EulerSampler
from confrover.utils import PathLike, get_pylogger
from confrover.utils.misc import download_file
from confrover.utils.misc.install import check_and_install_dependencies
from confrover.utils.torch.tensor import rearrange

from .utils.all_atom import atom14_to_atom37
from .utils.writer import Writer

logger = get_pylogger(__name__)


DEFAULT_PATH = CachePaths()


class ModelRegistry:
    _models = {
        "ConfRover-base-20M-v1.0".lower(): "https://huggingface.co/ByteDance-Seed/ConfRover-base-20M-v1.0/resolve/main/confrover_base_20m_v1_0.pt",
        "ConfRover-interp-20M-v1.0".lower(): "https://huggingface.co/ByteDance-Seed/ConfRover-interp-20M-v1.0/resolve/main/confrover_interp_20m_v1_0.pt",
    }

    def __init__(self, ckpt_dir: PathLike = DEFAULT_PATH.confrover_ckpts) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.model_ckpts = {
            model_name.lower(): self.ckpt_dir / model_url.split("/")[-1]
            for model_name, model_url in self._models.items()
        }

    def get_model_ckpt(self, model_name: str) -> PathLike:
        """Get model ckpt path from model yaml and confrover ckpt dir"""
        if model_name.lower() not in self._models:
            raise ValueError(
                f"Model '{model_name}' not found in {self.ckpt_dir}.\n  - Available models: {list(self.model_ckpts.keys())}"
            )

        ckpt_path = self.model_ckpts[model_name.lower()]
        if not ckpt_path.exists():
            # Download ckpt from huggingface
            url = self._models[model_name.lower()]
            logger.info(f"Download {model_name} from {url} to {ckpt_path}")
            download_file(url, dest_path=str(ckpt_path))
        return ckpt_path


class ConfRover(
    LightningModule,
):
    """ConfRover Inference Model"""

    # Mask token tensors
    mask_token_rigids: torch.Tensor
    mask_token_pseudo_beta: torch.Tensor
    mask_token_pseudo_beta_mask: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        temporal: LlamaPreTrainedModel,
        decoder: Decoder,
        writer: Optional[Writer] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.decoder = decoder
        self.writer = writer
        self.seed = seed

        self.register_buffer(
            "mask_token_rigids",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, None, :],
            persistent=False,
        )
        self.register_buffer(
            "mask_token_pseudo_beta",
            torch.tensor([0.0, 0.0, 0.0])[None, None, :],
            persistent=False,
        )
        self.register_buffer(
            "mask_token_pseudo_beta_mask",
            torch.tensor([1.0])[None, :],
            persistent=False,
        )

    def on_predict_start(self) -> None:
        seed_everything(self.seed + self.local_rank, workers=True)
        logger.info(
            f"[Rank {self.local_rank}] random seed = {self.seed + self.local_rank}"
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.inference_loop(batch)

    def on_predict_end(self):
        self.trainer.strategy.barrier()

    def generate(
        self,
        case_id: str,
        seqres: str,
        output_dir: PathLike,
        task_mode: Literal["forward", "iid", "interp"],
        conditions: PathLike | list[PathLike] | dict[str, PathLike] | None = None,
        n_frames: int | None = None,
        stride_in_10ps: int | None = None,
        n_replicates: int = 1,
        cache_dir: PathLike = DEFAULT_PATH.root,
        msa_root: PathLike = DEFAULT_PATH.msa,
        folding_repr: PathLike = DEFAULT_PATH.folding_repr,
        seed: int = 42,
        diffusion_steps: int = 200,
    ) -> PathLike:
        """ConfRover API to generate conformations or trajectories for single case.

        For batch generation, please use command line tool `confrover generate` instead.

        Args:
            case_id: Case ID.
            seqres: Seqres string.
            num_frames: Number of frames to generate.

        Returns:
            Path to the output directory.
        """

        env = CachePaths(root=cache_dir, msa=msa_root, folding_repr=folding_repr)
        check_and_install_dependencies(env)
        self.eval()

        #### 1. Prepare input dataset ####
        if task_mode == "iid":
            assert conditions is None, (
                "iid mode does not support input frame conditions"
            )
        else:
            if isinstance(conditions, dict):
                conditions = XTCConditions.from_dict(conditions, task_mode=task_mode)
            else:
                # pdb
                conditions = PDBConditions.from_list(conditions, task_mode=task_mode)
        assert isinstance(conditions, XTCConditions | PDBConditions | None)

        dataset_config = GenDatasetConfig(
            name="api_job",
            task_mode=task_mode,
            n_replicates=n_replicates,
            n_frames=n_frames,
            stride_in_10ps=stride_in_10ps,
            cases=[
                GenCaseConfig(
                    case_id=case_id,
                    seqres=seqres,
                    seqlen=len(seqres),
                    task_mode=task_mode,
                    n_replicates=n_replicates,
                    rep_id=rep_id,
                    n_frames=n_frames,
                    stride_in_10ps=stride_in_10ps,
                    conditions=conditions,
                )
                for rep_id in range(n_replicates)
            ],
        )

        # Prepare folding repr
        seqres_index_pairs = list(
            {case.seqres: case.case_id for case in dataset_config.cases}.items()
        )

        repr_loader = OpenFoldReprLoader(repr_root=folding_repr)
        repr_loader.generate_repr(
            seqres_index_pairs=seqres_index_pairs,
            msa_root=msa_root,
            openfold_params=env.openfold_params,
            num_gpus=1,
            overwrite=False,
            msa_max_query_size=12,
        )

        seed_everything(seed, workers=True)
        dataset = GenDataset(config=dataset_config, repr_loader=repr_loader)

        #### 2. Prepare model ####
        self.writer = Writer(
            output_dir=output_dir, output_format="auto", preview_frames=20
        )
        self.writer.output_dir = Path(output_dir)
        if self.decoder.sampler is None:
            self.decoder.sampler = EulerSampler(  # type: ignore
                diffusion_steps=diffusion_steps,
            )
        self.decoder.sampler.diffusion_steps = diffusion_steps

        #### 2. Run inference loop ####
        logger.info("================== Starting sampling ==================")

        for data_ix in tqdm(range(len(dataset)), desc="Sampling", disable=task_mode != 'iid'):
            batch = [dataset[data_ix]]
            batch = dataset.collate(batch)
            batch = move_data_to_device(batch, self.device)
            self.inference_loop(batch)
        
        self.writer.cleanup(remove_pdb=True, n_workers=1)
        logger.info(f"Saved to {self.writer.output_dir}")
        return self.writer.output_dir

    @torch.inference_mode()
    def inference_loop(self, batch):
        """Inference warpper for generation."""
        assert self.writer is not None, "writer is not initialized"

        #### 1. check if all sample has been generated ####
        all_exists = self.writer.check_output_exists(batch)
        if all_exists:
            return

        #### 2. Run generation ####
        start_t = perf_counter()
        output = self._ar_sample(**batch)
        end_t = perf_counter()

        #### 3. Save generated samples ####
        bsz = output["atom37_mask"].shape[0]
        for i in range(bsz):
            job_info = batch["job_info"][i]
            job_info["batch_size"] = bsz
            job_info["batch_walltime_sec"] = end_t - start_t

            self.writer.write(
                aatype=output["aatype"][i],
                atom37=output["atom37"][i],
                atom37_mask=output["atom37_mask"][i],
                padding_mask=output["padding_mask"][i],
                job_info=batch["job_info"][i],
            )

    def _fuse_single_pair(self, feat):
        if isinstance(feat, torch.Tensor):
            # a single hidden tensor
            return feat
        else:
            assert isinstance(feat, (tuple, list)) and len(feat) == 2, (
                "Input feat should be a tuple of tensors (single, pair)"
            )
            single_feat, pair_feat = feat

            pair_feat = rearrange(pair_feat, "N L1 L2 C ->  N (L1 L2)  C")
            return torch.cat([single_feat, pair_feat], dim=-2)  # M = L + L * L

    def _split_single_pair(self, fused_feat, seqlen: int):
        single_feat = fused_feat[:, :seqlen, :]
        pair_feat = rearrange(
            fused_feat[:, seqlen:, :], "N (L1 L2) C -> N L1 L2 C", L1=seqlen
        )  # B L L C
        return single_feat, pair_feat

    @torch.inference_mode()
    def _ar_sample(
        self,
        aatype,
        padding_mask,
        num_frames,
        gt_feat,
        pretrained_single,
        pretrained_pair,
        pos_id,
        job_info,
        task_mode,
        ref_mask=None,
        **kwargs,
    ):
        """Internal function for autoregressive generation"""

        #### Module verification ####
        assert callable(self.temporal.prepare_configs_for_generation), (
            "temporal model must have a method 'prepare_configs_for_generation' "
            "to prepare the configs for generation."
        )

        #### Input verification ####
        start_time = perf_counter()
        if len(pos_id.shape) == 0:
            # single frame, expand dim
            pos_id = pos_id.unsqueeze(dim=0)
        pred_atom14_list = []
        if ref_mask is None:
            # Mask all, uncond generation
            ref_mask = torch.zeros(
                padding_mask.shape[0],
                device=aatype.device,
                dtype=pretrained_single.dtype,
            )
        elif len(ref_mask.shape) == 0:
            # ref_mask has been squeezed into zero shape (e.g., B = F = 1)
            ref_mask = ref_mask.unsqueeze(0)
        batch_frames, seqlen = aatype.shape[:2]
        batch_size = batch_frames // num_frames
        if task_mode == 'iid':
            assert num_frames == 1, "num_frames must be 1 for iid task"
        else:
            # traj generation, check KV cache type
            cache_type = self.temporal.llama_config.cache_type
            if cache_type == "offloaded":
                logger.info("Trajectory generation with OffloadedCache")
            elif cache_type.startswith("sink"):
                # Sink cache defined wth format sink{num_sink}:{sliding_window_length}
                match = re.fullmatch(r"sink(\d+):(\d+)", cache_type)
                if match:
                    num_sink = int(match.group(1))
                    sliding_window_length = int(match.group(2))
                else:
                    raise ValueError(
                        f"String '{cache_type}' is not in the expected format: sink{{sink_num}}:{{sliding_window_length}}"
                    )
                logger.info(
                    f"Trajectory generation with SinkCache(num_sink={num_sink}, sliding_window_length={sliding_window_length})"
                )
            else:
                raise ValueError(
                    "cache_type should be 'offloaded' or 'sink{sink_num}:{sliding_window_length}"
                )

        #### 0. Prepare input tokens ####
        begin_rigids_0 = self.mask_token_rigids.expand(batch_size, seqlen, -1)[
            :, None, ...
        ]  # (B 1 L C)
        begin_pseudo_beta = self.mask_token_pseudo_beta.expand(batch_size, seqlen, -1)[
            :, None, ...
        ]  # (B 1 L 3)
        begin_pseudo_beta_mask = self.mask_token_pseudo_beta_mask.expand(
            batch_size, seqlen
        )[:, None, ...]  # (B 1 L)

        # Conditional tokens
        ref_mask = rearrange(ref_mask, "(B F) -> B F", B=batch_size)
        if "rigids_0" not in gt_feat or ref_mask.sum() == 0:
            # no frame history condition, use begin rigids
            rigids_0 = begin_rigids_0
            pseudo_beta = begin_pseudo_beta
            pseudo_beta_mask = begin_pseudo_beta_mask
        else:
            # condition on frame history
            rigids_0 = rearrange(
                gt_feat["rigids_0"], "(B F) L C -> B F L C", B=batch_size
            )  # [:, ref_mask[0] > 0, ...] # we assum all ref_mask in batch are same # F
            rigids_0 = torch.cat([begin_rigids_0, rigids_0], dim=1)
            pseudo_beta = rearrange(
                gt_feat["pseudo_beta"], "(B F) L C -> B F L C", B=batch_size
            )  # [:, ref_mask[0] > 0, ...]
            pseudo_beta = torch.cat([begin_pseudo_beta, pseudo_beta], dim=1)
            pseudo_beta_mask = rearrange(
                gt_feat["pseudo_beta_mask"], "(B F) L -> B F L", B=batch_size
            )  # [:, ref_mask[0] > 0,...]
            pseudo_beta_mask = torch.cat(
                [begin_pseudo_beta_mask, pseudo_beta_mask], dim=1
            )
            pred_atom14_list.append(
                rearrange(
                    gt_feat["atom14_gt_positions"], "(B F) ... -> B F ...", B=batch_size
                )
            )  # [:,ref_mask[0] > 0,...])

        ### Prepare condition input (src_)
        src_rigids_0 = rearrange(rigids_0, "B F L C -> (B F) L C")
        src_pseudo_beta = rearrange(pseudo_beta, "B F L C -> (B F) L C")
        src_pseudo_beta_mask = rearrange(pseudo_beta_mask, "B F L -> (B F) L")
        num_src_frames = (
            src_rigids_0.shape[0] // batch_size
        )  # num of conditioning frames, including the starting frame

        #### 1. Encoder initial mask token and conditioning tokens ####
        aatype = rearrange(aatype, "(B F) L -> B F L", B=batch_size)
        padding_mask = rearrange(padding_mask, "(B F) L -> B F L", B=batch_size)
        pretrained_single = rearrange(
            pretrained_single, "(B F) ... -> B F ...", B=batch_size
        )
        pretrained_pair = rearrange(
            pretrained_pair, "(B F) ... -> B F ...", B=batch_size
        )

        no_struct_mask = torch.ones_like(
            ref_mask[:, :num_src_frames]
        )  # No masking in autoregressive generation
        single_feat, pair_feat = self.encoder(
            aatype=rearrange(aatype[:, :num_src_frames], "B F1 L -> (B F1) L"),
            padding_mask=rearrange(
                padding_mask[:, :num_src_frames], "B F1 L -> (B F1) L"
            ),
            rigids_0=src_rigids_0,
            batch_size=batch_size,
            struct_mask=rearrange(no_struct_mask, "B F1 -> (B F1)"),
            pseudo_beta=src_pseudo_beta,
            pseudo_beta_mask=src_pseudo_beta_mask,
            pretrained_single=rearrange(
                pretrained_single[:, :num_src_frames, ...], "B F1 ... -> (B F1) ..."
            ),
            pretrained_pair=rearrange(
                pretrained_pair[:, :num_src_frames, ...], "B F1 ... -> (B F1) ..."
            ),
        )  # (B L+L*L) F C
        # Flatten single and pair features into hidden_states
        inputs_embeds = self._fuse_single_pair((single_feat, pair_feat))
        # pair_feat = rearrange(pair_feat, 'N L1 L2 C ->  N (L1 L2)  C')
        # inputs_embeds = torch.cat([single_feat, pair_feat], dim = 1) # M = L + L * L

        #### 2. Prepare AR generation input ####
        inputs_embeds = rearrange(
            inputs_embeds, "(B F) M C ->  (B M) F C", B=batch_size
        )
        pos_id = rearrange(pos_id, "(B F) -> B F", B=batch_size)
        pos_id = rearrange(
            pos_id[:, None, :].expand(-1, seqlen + seqlen * seqlen, -1),
            "B M C -> (B M) C",
            check_inplace=False,
        )
        gen_config: dict[str, Any] = self.temporal.prepare_configs_for_generation(
            inputs=inputs_embeds,
            max_length=num_frames + 1,
            position_ids=pos_id,
            use_cache=True,
        )  # type:ignore
        model_kwargs = gen_config["model_kwargs"]
        stopping_criteria = gen_config["stopping_criteria"]
        synced_gpus = gen_config["synced_gpus"]
        # keep track of which sequences are already finished
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device
        )
        model_kwargs = self.temporal._get_initial_cache_position(
            inputs_embeds[..., 0], model_kwargs
        )

        chain_name = job_info[0]["case_id"]
        # chain_name = kwargs.get("job_info", [None])[0]["case_id"]
        _num_frames_to_generate = num_frames - int(ref_mask.sum().item())
        current_frame = int(ref_mask.sum().item())
        pbar = tqdm(
            desc=f"{chain_name} ({seqlen})",
            total=num_frames,
            initial=int(ref_mask.sum().item()),
            disable=(num_frames == 1),
        )
        if num_frames > 1:
            # trajectory generation. Print traj info.
            print(
                f"[Rank {self.global_rank}] {chain_name}: {_num_frames_to_generate}/{num_frames} frames: {pos_id[0][:5]} ..."
            )

        #### 3. Autoregressive generation ####
        while self.temporal._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=inputs_embeds.device
        ):  # type:ignore
            # prepare model inputs_embeds
            model_inputs = self.temporal.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, **model_kwargs
            )
            # forward pass to get next token
            outputs = self.temporal(
                **model_inputs,
                return_dict=True,
                batch_size=batch_size,
                rigids_mask=rearrange(
                    padding_mask[:, :num_src_frames], "B F1 L -> (B F1) L"
                ),
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            hidden_states = rearrange(
                outputs.last_hidden_state[:, -1, :], "(B L) C -> B L C", B=batch_size
            )  # check
            # split single and pair
            single_feat, pair_feat = self._split_single_pair(
                hidden_states, seqlen=seqlen
            )
            # s = hidden_states[:,:seqlen,:]
            # z = rearrange(hidden_states[:,seqlen:,:],'B (L1 L2) C -> B L1 L2 C', L1= seqlen)

            # Decode one frame
            pred_atom14_one_frame, pred_rigids_0 = self.decoder.sample(
                aatype=aatype[:, 0, :],
                s=single_feat,
                z=pair_feat,
                padding_mask=padding_mask[:, 0, :],
                num_frames=1,
                pretrained_single=pretrained_single[:, 0, ...],
                pretrained_pair=pretrained_pair[:, 0, ...],
            )  # (B L 37), B L
            pred_atom14_list.append(pred_atom14_one_frame[:, None, ...])
            pred_rigids_0 = pred_rigids_0.to_tensor_7().to(single_feat.dtype)  # B L 7
            pbar.update(1)
            current_frame += 1
            # print(f"{chain_name} [{seqlen}]: {pbar.n} frames, KV: {outputs.past_key_values.total_cache_size() /1024/1024/1024:.2f} GB")

            if current_frame == num_frames:
                # Stop Llama properly
                break

            # compute new pseudo beta and structure embedding
            pred_atom37, atom37_mask = atom14_to_atom37(
                pred_atom14_one_frame, aatype[:, 0]
            )
            pseudo_beta = pseudo_beta_fn(
                aatype[:, 0], pred_atom37, all_atom_mask=None
            ).to(dtype=single_feat.dtype)
            single_feat, pair_feat = self.encoder(
                aatype=aatype[:, 0, :],
                padding_mask=padding_mask[:, 0, :],
                rigids_0=pred_rigids_0,
                batch_size=batch_size,
                struct_mask=no_struct_mask[:, 0],
                pseudo_beta=pseudo_beta,
                pseudo_beta_mask=pseudo_beta_mask[:, 0, ...],
                pretrained_single=pretrained_single[:, 0, ...],
                pretrained_pair=pretrained_pair[:, 0, ...],
            )  # (B L+L*L) 1 C
            new_inputs_embeds = self._fuse_single_pair((single_feat, pair_feat))
            new_inputs_embeds = rearrange(
                new_inputs_embeds, "(B F) M C ->  (B M) F C", B=batch_size
            )

            # inputs_embeds = torch.cat([inputs_embeds, new_inputs_embeds], dim=1)
            inputs_embeds = new_inputs_embeds
            # if streamer is not None:
            #     streamer.put(next_tokens.cpu())
            model_kwargs = self.temporal._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )
            num_src_frames = 1

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                inputs_embeds[..., 0], None
            )
            this_peer_finished = unfinished_sequences.max() == 0

        pred_atom14 = torch.cat(pred_atom14_list, dim=1)
        if task_mode == "interp":
            # re-order the frames
            pred_atom14 = torch.cat(
                [pred_atom14[:, 1:, ...], pred_atom14[:, 0:1, ...]], dim=1
            )
            padding_mask = torch.cat(
                [padding_mask[:, 1:, ...], padding_mask[:, 0:1, ...]], dim=1
            )

        pred_atom37, atom37_mask = atom14_to_atom37(pred_atom14, aatype)
        pbar.close()
        end_time = perf_counter()
        if num_frames > 1:
            print(
                f"[{chain_name} ({seqlen}), {num_frames} frames] cost: {(end_time - start_time) / 60:.3f} min"
            )

        return {
            "atom37": pred_atom37,
            "atom37_mask": atom37_mask[:, 0, ...],
            "aatype": aatype[:, 0, ...],
            "padding_mask": padding_mask[:, 0, ...],
            "info": {
                "chain_name": chain_name,
                "seqlen": seqlen,
                "num_frames": num_frames,
                "num_frames_generated": _num_frames_to_generate,
                "time_cost_sec": end_time - start_time,
            },
        }

    @classmethod
    def from_config(
        cls,
        model_cfg: PathLike | dict | DictConfig,
        seed: int = 42,
        use_deepspeed_evo_attention: bool = False,
        kv_cache_type: str = "offloaded",
        return_cfg: bool = False,
    ):
        """Create ConfRover model from model config (dict/DictConfig) or a yaml file"""
        from confrover.utils.hydra_utils import load_hydra_config

        if isinstance(model_cfg, (str, Path)):
            assert model_cfg.endswith(".yaml"), (
                f"Model config must be a yaml file: {model_cfg}"
            )
            model_cfg_ = load_hydra_config(
                model_cfg,
                overrides=[
                    f"seed={seed}",
                    f"use_deepspeed_evo_attention={use_deepspeed_evo_attention}",
                    f"kv_cache_type={kv_cache_type}",
                ],
            )
        else:
            if isinstance(model_cfg, dict):
                model_cfg = OmegaConf.create(model_cfg)
            model_cfg_ = OmegaConf.merge(
                model_cfg,
                {
                    "seed": seed,
                    "use_deepspeed_evo_attention": use_deepspeed_evo_attention,
                    "kv_cache_type": kv_cache_type,
                },
            )

        model = hydra.utils.instantiate(model_cfg_)
        assert isinstance(model, cls), f"Model type is not ConfRover"
        if return_cfg:
            return model, model_cfg_
        else:
            return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str | PathLike,
        ckpt_dir: PathLike = DEFAULT_PATH.confrover_ckpts,
        seed: int = 42,
        use_deepspeed_evo_attention: bool = False,
        kv_cache_type: str = "offloaded",
    ) -> ConfRover:
        """Create ConfRover model from a pretrained model checkpoint"""
        if Path(pretrained_model).exists():
            # provided a model path
            pretrained_model = str(Path(pretrained_model).resolve())
        else:
            # try registered model
            pretrained_model = ModelRegistry(ckpt_dir=ckpt_dir).get_model_ckpt(
                str(pretrained_model)
            )

        model_ckpt = torch.load(pretrained_model)

        model: ConfRover = cls.from_config(
            model_ckpt["model_cfg"],
            seed=seed,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            kv_cache_type=kv_cache_type,
        )  # type: ignore

        model.load_state_dict(model_ckpt["state_dict"])
        logger.info(f"Loaded model from {pretrained_model}")
        return model
