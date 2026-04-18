# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import (get_local_torch_device,
                                   get_sequence_parallel_block_partition,
                                   get_sp_world_size)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.controlnet_union_components import WanControlNetUnionInput
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
)
from fastvideo.utils import FlexibleArgumentParser, get_compute_dtype, shallow_asdict

logger = init_logger(__name__)


def _is_union_controlnet(model) -> bool:
    return "union" in model.__class__.__name__.lower()


def _split_union_control_latent(
    control_latent: torch.Tensor, num_channels_latents: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    c = int(num_channels_latents)
    if control_latent.shape[1] == 3 * c:
        depth = control_latent[:, :c]
        masked = control_latent[:, c:2 * c]
        mask = control_latent[:, 2 * c:3 * c]
        normal = None
        return depth, normal, masked, mask
    if control_latent.shape[1] == 4 * c:
        depth = control_latent[:, :c]
        normal = control_latent[:, c:2 * c]
        masked = control_latent[:, 2 * c:3 * c]
        mask = control_latent[:, 3 * c:4 * c]
        return depth, normal, masked, mask
    raise ValueError(
        f"Union control_latent channel mismatch: got {control_latent.shape[1]}, "
        f"expected 3*C or 4*C (C={c})."
    )


def _build_controlnet_kwargs(
    controlnet, control_latent: torch.Tensor, num_channels_latents: int
) -> dict:
    if not _is_union_controlnet(controlnet):
        return {"controlnet_states": control_latent}
    depth, normal, masked, mask = _split_union_control_latent(
        control_latent, num_channels_latents
    )
    return {
        "controlnet_cond": WanControlNetUnionInput(depth=depth, normal=normal),
        "mask": mask,
        "masked_latent": masked,
    }


def _ensure_first_frame(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    elif x.dim() == 5:
        # [B, F, C, H, W] -> [B, C, F, H, W]
        if x.shape[1] in (1, 3) and x.shape[2] >= 8:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(f"Unsupported first_frame_latent shape: {tuple(x.shape)}")
    if x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must have F==1, got shape={tuple(x.shape)}"
        )
    return x


def _ensure_control_latent(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(0)
    return x


def _ensure_branch_latent_bcfhw(x: torch.Tensor, *,
                                latent_channels: int,
                                name: str) -> torch.Tensor:
    if x.dim() == 4:
        if int(x.shape[0]) != int(latent_channels):
            raise ValueError(
                f"{name} 4D shape must be [C,F,H,W], got {tuple(x.shape)}")
        return x.unsqueeze(0)
    if x.dim() == 5:
        if int(x.shape[1]) == int(latent_channels):
            return x
        if int(x.shape[2]) == int(latent_channels):
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(
        f"{name} must be [C,F,H,W], [B,C,F,H,W], or [B,F,C,H,W], got {tuple(x.shape)}"
    )


def _trim_or_pad_latent_t(x: torch.Tensor, target_t: int, *,
                          name: str) -> torch.Tensor:
    if int(x.shape[2]) == int(target_t):
        return x
    if int(x.shape[2]) > int(target_t):
        return x[:, :, :int(target_t)]
    if int(x.shape[2]) <= 0:
        raise ValueError(f"{name} has empty temporal dimension: {tuple(x.shape)}")
    pad = x[:, :, -1:].repeat(1, 1, int(target_t) - int(x.shape[2]), 1, 1)
    return torch.cat([x, pad], dim=2)


def _parse_matrixcity_record_id(record_id: str) -> tuple[str, int, int, int]:
    parts = [p for p in str(record_id).split("/") if p]
    if len(parts) < 3:
        raise ValueError(f"Invalid MatrixCity record id: {record_id}")
    scene_name = parts[-3]
    window_name = parts[-2]
    clip_name = parts[-1]
    if "_" not in window_name or not clip_name.startswith("clip_start_"):
        raise ValueError(f"Invalid MatrixCity record id layout: {record_id}")
    window_start_str, window_end_str = window_name.split("_", 1)
    clip_start = int(clip_name.split("_")[-1])
    return scene_name, int(window_start_str), int(window_end_str), int(
        clip_start)


@torch.no_grad()
def _encode_control_branch_latent_with_infer_base(
    *,
    infer_base,
    vae,
    branch_tchw: torch.Tensor,
    normalize: bool,
    target_c: int,
    name: str,
    inference_device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if branch_tchw.ndim != 4:
        raise ValueError(
            f"{name} must be [T,C,H,W], got shape={tuple(branch_tchw.shape)}")
    if int(branch_tchw.shape[1]) == 1:
        branch_tchw = branch_tchw.repeat(1, 3, 1, 1)
    elif int(branch_tchw.shape[1]) != 3:
        raise ValueError(
            f"{name} channel count must be 1 or 3 before VAE encode, got {int(branch_tchw.shape[1])}"
        )

    video_bcthw = infer_base._to_vae_input(branch_tchw, normalize=normalize).to(
        device=inference_device,
        dtype=compute_dtype,
    )
    latent = infer_base._encode_video_latents(
        vae,
        video_bcthw,
        sample_mode="mode",
        compute_dtype=compute_dtype,
    )[0]
    latent = infer_base._align_latent_channels(latent, int(target_c), name)
    return latent.unsqueeze(0)


class ARTFControlnetTrainingPipeline(TrainingPipeline):
    """
    Stage-1 AR diffusion (teacher-forcing style) with TI2V + ControlNet inputs.

    Expected parquet schema:
      - `pyarrow_schema_ti2v_controlnet`
      - `vae_latent` must be populated (non-empty)
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]
    trainable_transformer_names = ["transformer", "controlnet"]

    @staticmethod
    def _describe_channel_layout(module: torch.nn.Module, *, name: str) -> str:
        attrs = {
            "in_channels": getattr(module, "in_channels", None),
            "out_channels": getattr(module, "out_channels", None),
            "num_channels_latents": getattr(module, "num_channels_latents", None),
            "num_attention_heads": getattr(module, "num_attention_heads", None),
            "patch_size": getattr(module, "patch_size", None),
        }
        cfg = getattr(module, "config", None)
        arch = getattr(cfg, "arch_config", None) if cfg is not None else None
        if arch is not None:
            for key in ("in_channels", "out_channels", "num_channels_latents",
                        "num_attention_heads", "patch_size",
                        "local_attn_size", "num_frames_per_block"):
                attrs[f"arch_{key}"] = getattr(arch, key, None)
        return f"{name} channel/layout summary: {attrs}"

    def set_schemas(self) -> None:
        self.train_dataset_schema = pyarrow_schema_ti2v_controlnet

    def _debug_timing_enabled(self) -> bool:
        if getattr(self, "global_rank", 0) != 0:
            return False
        try:
            debug_steps = int(os.environ.get("FASTVIDEO_ARTF_DEBUG_STEPS", "1"))
        except ValueError:
            debug_steps = 1
        if debug_steps <= 0:
            return False
        return int(getattr(self, "current_trainstep", 0)) <= debug_steps

    def _sync_device_for_debug(self) -> None:
        if torch.cuda.is_available():
            device = get_local_torch_device()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def _backward_profile_enabled(self) -> bool:
        if not self._debug_timing_enabled():
            return False
        return os.environ.get("FASTVIDEO_ARTF_PROFILE_BACKWARD", "0") == "1"

    def _debug_output_dir(self) -> str:
        return str(getattr(self.training_args, "output_dir", "") or os.getcwd())

    def _dump_backward_profile(self, prof) -> None:
        if getattr(self, "global_rank", 0) != 0:
            return
        output_dir = self._debug_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        step = int(getattr(self, "current_trainstep", 0))
        profile_path = os.path.join(
            output_dir,
            f"artf_backward_profile_step{step:04d}_rank{self.global_rank}.txt",
        )
        tables: list[str] = []
        sort_keys: list[str] = []
        if torch.cuda.is_available():
            sort_keys.extend(["self_cuda_time_total", "cuda_time_total"])
        sort_keys.extend(["self_cpu_time_total", "cpu_time_total"])

        seen: set[str] = set()
        for sort_key in sort_keys:
            if sort_key in seen:
                continue
            seen.add(sort_key)
            try:
                table = prof.key_averages().table(
                    sort_by=sort_key,
                    row_limit=30,
                )
            except Exception:
                continue
            tables.append(f"=== sort_by={sort_key} ===\n{table}\n")

        if not tables:
            return
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tables))
        logger.info(
            "[AR-TF debug] backward profile summary saved to %s",
            profile_path,
        )

    def _log_backward_hook(self, *, tag: str, backward_start_time: float,
                           grad: torch.Tensor | None) -> None:
        if getattr(self, "global_rank", 0) != 0:
            return
        self._sync_device_for_debug()
        elapsed = time.perf_counter() - backward_start_time
        mem_suffix = ""
        if torch.cuda.is_available():
            device = get_local_torch_device()
            if device.type == "cuda":
                mem_mb = torch.cuda.memory_allocated(device) / 1024**2
                max_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                mem_suffix = f" mem={mem_mb:.1f}MB max_mem={max_mem_mb:.1f}MB"
        grad_suffix = ""
        if isinstance(grad, torch.Tensor):
            grad_suffix = (
                f" grad_shape={tuple(grad.shape)} grad_dtype={grad.dtype}"
            )
        logger.info(
            "[AR-TF debug] train_step=%s stage=backward_hook/%s elapsed=%.2fs%s%s",
            int(getattr(self, "current_trainstep", 0)),
            tag,
            elapsed,
            mem_suffix,
            grad_suffix,
        )

    def _log_rank_consistency(self, *, stage: str, values: list[float]) -> None:
        if not dist.is_initialized():
            return
        tensor = torch.tensor(values, device=get_local_torch_device(),
                              dtype=torch.float32)
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        if getattr(self, "global_rank", 0) == 0:
            payload = [item.tolist() for item in gathered]
            logger.info(
                "[AR-TF debug] train_step=%s stage=%s rank_consistency=%s",
                int(getattr(self, "current_trainstep", 0)),
                stage,
                payload,
            )

    def _log_debug_stage(self, *, stage: str, start_time: float,
                         extra: str = "") -> None:
        if not self._debug_timing_enabled():
            return
        self._sync_device_for_debug()
        elapsed = time.perf_counter() - start_time
        msg = (
            "[AR-TF debug] train_step=%s stage=%s elapsed=%.2fs",
            int(getattr(self, "current_trainstep", 0)),
            stage,
            elapsed,
        )
        if torch.cuda.is_available():
            device = get_local_torch_device()
            if device.type == "cuda":
                mem_mb = (torch.cuda.memory_allocated(device) / 1024**2)
                max_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024**2)
                msg = (
                    "[AR-TF debug] train_step=%s stage=%s elapsed=%.2fs "
                    "mem=%.1fMB max_mem=%.1fMB%s",
                    int(getattr(self, "current_trainstep", 0)),
                    stage,
                    elapsed,
                    mem_mb,
                    max_mem_mb,
                    f" {extra}" if extra else "",
                )
                logger.info(*msg)
                return
        logger.info(*(msg if not extra else (
            "[AR-TF debug] train_step=%s stage=%s elapsed=%.2fs%s",
            int(getattr(self, "current_trainstep", 0)),
            stage,
            elapsed,
            f" {extra}",
        )))

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ):
        training_args = cast(TrainingArgs, fastvideo_args)
        modules = super().load_modules(fastvideo_args, loaded_modules)

        if not training_args.controlnet_model_path:
            raise ValueError(
                "controlnet_model_path is required for AR-TF ControlNet training"
            )

        logger.info("Loading student controlnet from: %s", training_args.controlnet_model_path)
        prev_cn_override = getattr(training_args, "override_controlnet_cls_name", None)
        training_args.override_controlnet_cls_name = "CausalWanControlnetUnion3DModel"
        try:
            self.controlnet = PipelineComponentLoader.load_module(
                module_name="controlnet",
                component_model_path=training_args.controlnet_model_path,
                transformers_or_diffusers="diffusers",
                fastvideo_args=training_args,
            )
        finally:
            training_args.override_controlnet_cls_name = prev_cn_override

        if not _is_union_controlnet(self.controlnet):
            raise ValueError(
                "AR-TF student controlnet must be Union. "
                "Please use a Union ControlNet checkpoint/config."
            )
        logger.info("AR-TF student controlnet class: %s", self.controlnet.__class__.__name__)
        modules["controlnet"] = self.controlnet
        return modules

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=fastvideo_args.pipeline_config.flow_shift,
            training=True,
        )

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)

        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=training_args.pipeline_config.flow_shift,
            training=True,
        )
        self.noise_scheduler = self.modules["scheduler"]

        self.transformer = self.get_module("transformer")
        self.controlnet = self.get_module("controlnet")
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)
        # AR-TF joint training: train transformer + controlnet together.
        self.controlnet.requires_grad_(True)
        self.controlnet.train()
        logger.info(self._describe_channel_layout(self.transformer,
                                                 name="transformer"))
        logger.info(self._describe_channel_layout(self.controlnet,
                                                 name="controlnet"))
        logger.info(
            "vae summary: z_dim=%s scaling_factor=%s temporal_compression_ratio=%s spatial_compression_ratio=%s",
            getattr(self.vae, "z_dim", getattr(getattr(self.vae, "config", None),
                                                "z_dim", None)),
            getattr(self.vae, "scaling_factor", None),
            getattr(self.vae, "temporal_compression_ratio", getattr(getattr(self.vae, "config", None), "temporal_compression_ratio", None)),
            getattr(self.vae, "spatial_compression_ratio", getattr(getattr(self.vae, "config", None), "spatial_compression_ratio", None)),
        )

        if training_args.enable_gradient_checkpointing_type is not None:
            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )
            self.controlnet = apply_activation_checkpointing(
                self.controlnet,
                checkpointing_type=training_args.enable_gradient_checkpointing_type,
            )

        betas = tuple(float(x.strip()) for x in training_args.betas.split(","))
        params_to_optimize: list[torch.nn.Parameter] = []
        params_to_optimize.extend(
            p for p in self.transformer.parameters() if p.requires_grad
        )
        params_to_optimize.extend(
            p for p in self.controlnet.parameters() if p.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_args.learning_rate,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        num_train_t = int(self.noise_scheduler.config.num_train_timesteps)
        min_ratio = float(max(0.0, min(1.0, training_args.min_timestep_ratio)))
        max_ratio = float(max(0.0, min(1.0, training_args.max_timestep_ratio)))
        if max_ratio <= min_ratio:
            max_ratio = min(1.0, min_ratio + 0.01)

        self.min_timestep_index = int(min_ratio * num_train_t)
        self.max_timestep_index = max(self.min_timestep_index + 1, int(max_ratio * num_train_t))
        logger.info(
            "AR-TF timestep index range: [%s, %s), num_train_t=%s, num_frame_per_block=%s",
            self.min_timestep_index,
            self.max_timestep_index,
            num_train_t,
            training_args.num_frame_per_block,
        )
        self._matrixcity_phase1_state_initialized = False

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Validation pipeline is not used for AR-TF controlnet training.")

    def _lazy_init_matrixcity_phase1_state(self) -> None:
        if getattr(self, "_matrixcity_phase1_state_initialized", False):
            return

        raw_root = Path(
            str(getattr(self.training_args, "online_warp_raw_root", "")).strip()
        ).expanduser()
        if not raw_root.is_dir():
            raise FileNotFoundError(
                "Phase-1 MatrixCity fallback requires a valid online_warp_raw_root, "
                f"got {raw_root}"
            )

        from tools import infer_wan_controlnet_ti2v as infer_base
        from tools import preprocess_matrixcity_ti2v_controlnet_parquet as mcprep

        street_split = str(
            getattr(self.training_args, "online_warp_street_split",
                    "train_dense"))
        camera_mode = str(
            getattr(self.training_args, "online_warp_camera_mode", "B_inv"))
        pose_index = mcprep._load_matrixcity_pose_index(
            rgb_root=raw_root,
            street_split=street_split,
            camera_mode=camera_mode,
        )
        self._matrixcity_infer_base = infer_base
        self._matrixcity_mcprep = mcprep
        self._matrixcity_raw_root = raw_root
        self._matrixcity_pose_index = pose_index
        self._matrixcity_scene_assets: dict[str, dict[str, Any]] = {}
        self._matrixcity_phase1_state_initialized = True
        logger.info(
            "Initialized MatrixCity AR-TF fallback: raw_root=%s split=%s camera_mode=%s scenes=%d",
            raw_root,
            street_split,
            camera_mode,
            len(pose_index),
        )

    def _resolve_matrixcity_scene_dir(self, scene_name: str) -> Path:
        raw_root = cast(Path, getattr(self, "_matrixcity_raw_root"))
        street_split = str(
            getattr(self.training_args, "online_warp_street_split",
                    "train_dense"))
        candidates = [
            raw_root / "small_city" / "street" / street_split / scene_name,
            raw_root / "street" / street_split / scene_name,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Cannot resolve MatrixCity scene dir for scene={scene_name} under {raw_root}"
        )

    def _get_matrixcity_scene_assets(self, scene_name: str) -> dict[str, Any]:
        self._lazy_init_matrixcity_phase1_state()
        scene_cache = getattr(self, "_matrixcity_scene_assets", {})
        cached = scene_cache.get(scene_name)
        if cached is not None:
            return cached

        from PIL import Image

        mcprep = getattr(self, "_matrixcity_mcprep")
        raw_root = cast(Path, getattr(self, "_matrixcity_raw_root"))
        street_split = str(
            getattr(self.training_args, "online_warp_street_split",
                    "train_dense"))
        scene_dir = self._resolve_matrixcity_scene_dir(scene_name)
        scene_pose_index = getattr(self, "_matrixcity_pose_index")[scene_name]

        rgb_dir = scene_dir / scene_name
        rgb_files = mcprep._sorted_pngs(rgb_dir)
        rgb_map = mcprep._build_numeric_file_map(rgb_files)
        if not rgb_map:
            raise FileNotFoundError(
                f"No RGB files found for scene={scene_name}: {rgb_dir}")

        depth_dir = (raw_root / "small_city_depth" / "street" / street_split /
                     f"{scene_name}_depth" / f"{scene_name}_depth")
        depth_files = mcprep._sorted_depth_files(depth_dir)
        depth_map = mcprep._build_numeric_file_map(depth_files)
        if not depth_map:
            raise FileNotFoundError(
                f"No depth files found for scene={scene_name}: {depth_dir}")

        normal_map: dict[int, Path] | None = None
        normal_dir_candidates = [
            raw_root / "small_city_normal" / "street" / street_split /
            f"{scene_name}_normal" / f"{scene_name}_normal",
            raw_root / "small_city_normal" / "street" / street_split /
            scene_name / scene_name,
            raw_root / "street" / street_split / f"{scene_name}_normal" /
            f"{scene_name}_normal",
            raw_root / "street" / street_split / scene_name / scene_name,
            raw_root / f"{scene_name}_normal" / f"{scene_name}_normal",
            raw_root / scene_name / scene_name,
        ]
        for normal_dir in normal_dir_candidates:
            if not normal_dir.is_dir():
                continue
            nfiles = mcprep._sorted_normal_files(normal_dir)
            if not nfiles:
                continue
            normal_map = mcprep._build_numeric_file_map(nfiles)
            if normal_map:
                break
        if normal_map is None:
            raise FileNotFoundError(
                f"No normal files found for scene={scene_name}")

        ref_img = Image.open(next(iter(rgb_map.values()))).convert("RGB")
        src_w, src_h = ref_img.size
        crop_params = mcprep.infer_base._get_crop_params(
            src_w,
            src_h,
            int(self.training_args.num_width),
            int(self.training_args.num_height),
        )
        camera_k = mcprep._build_intrinsics_from_pose_meta(
            scene_pose_index.intrinsics_meta,
            src_w=src_w,
            src_h=src_h,
        )
        camera_k_aligned = mcprep.infer_base._adjust_intrinsics(
            camera_k,
            crop_params,
            int(self.training_args.num_width),
            int(self.training_args.num_height),
        )

        assets = {
            "rgb_map": rgb_map,
            "depth_map": depth_map,
            "normal_map": normal_map,
            "camera_k_aligned": camera_k_aligned,
            "rt_by_frame_id": scene_pose_index.rt_by_frame_id,
            "crop_params": crop_params,
        }
        scene_cache[scene_name] = assets
        self._matrixcity_scene_assets = scene_cache
        return assets

    def _build_matrixcity_clip_from_info(self, info: dict[str, Any],
                                         total_required_frames: int
                                         ) -> dict[str, Any]:
        self._lazy_init_matrixcity_phase1_state()
        mcprep = getattr(self, "_matrixcity_mcprep")

        record_id = str(info.get("id") or info.get("file_name") or "").strip()
        if not record_id:
            raise KeyError(
                "MatrixCity AR-TF fallback requires info['id'] or info['file_name']"
            )
        scene_name, window_start, window_end, clip_start = _parse_matrixcity_record_id(
            record_id)
        clip_start_global = info.get("clip_start_global_id")
        if clip_start_global in (None, ""):
            clip_start_global, _ = mcprep._resolve_clip_start_global_id_for_sampling(
                clip_start=int(clip_start),
                window_start=int(window_start),
                window_end=int(window_end),
            )
        else:
            clip_start_global = int(clip_start_global)
        max_available_frames = int(window_end - clip_start_global + 1)
        if int(total_required_frames) > int(max_available_frames):
            raise ValueError(
                "Requested MatrixCity clip exceeds available coverage: "
                f"record_id={record_id} requested={int(total_required_frames)} "
                f"available={int(max_available_frames)}")

        assets = self._get_matrixcity_scene_assets(scene_name)
        target_frame_ids = [
            int(clip_start_global) + i for i in range(int(total_required_frames))
        ]
        rgb_paths = mcprep._pick_by_target_ids(assets["rgb_map"], target_frame_ids)
        depth_paths = mcprep._pick_by_target_ids(assets["depth_map"],
                                                 target_frame_ids)
        normal_paths = mcprep._pick_by_target_ids(assets["normal_map"],
                                                  target_frame_ids)
        return {
            "record_id": record_id,
            "scene_name": scene_name,
            "frame_ids": target_frame_ids,
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "normal_paths": normal_paths,
            "camera_k_aligned": assets["camera_k_aligned"],
            "rt_by_frame_id": assets["rt_by_frame_id"],
            "crop_params": assets["crop_params"],
        }

    def _rebuild_matrixcity_phase1_sample(
        self,
        *,
        info: dict[str, Any],
        first_frame_latent: torch.Tensor,
        raw_depth_latent: torch.Tensor | None,
        raw_normal_latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._lazy_init_matrixcity_phase1_state()
        infer_base = getattr(self, "_matrixcity_infer_base")
        mcprep = getattr(self, "_matrixcity_mcprep")

        target_frames = int(self.training_args.num_frames)
        target_t = int(self.training_args.num_latent_t)
        target_h = int(self.training_args.num_height)
        target_w = int(self.training_args.num_width)
        target_c = int(first_frame_latent.shape[1])
        clip = self._build_matrixcity_clip_from_info(info, target_frames)
        compute_dtype = torch.bfloat16

        rgb_tchw = torch.stack([
            mcprep._load_rgb_frame(p, int(target_h), int(target_w))
            for p in clip["rgb_paths"][:target_frames]
        ],
                                dim=0)
        video_bcthw = infer_base._to_vae_input(rgb_tchw,
                                               normalize=True).to(
                                                   device=self.device,
                                                   dtype=compute_dtype)
        vae_latent = infer_base._encode_video_latents(
            self.vae,
            video_bcthw,
            sample_mode="mode",
            compute_dtype=compute_dtype,
        )[0]
        vae_latent = infer_base._align_latent_channels(vae_latent, target_c,
                                                       "vae_latent")
        vae_latent = vae_latent.unsqueeze(0).to(dtype=torch.bfloat16)
        vae_latent = _trim_or_pad_latent_t(vae_latent,
                                           int(target_t),
                                           name="vae_latent")

        if raw_depth_latent is None or raw_normal_latent is None:
            raise ValueError(
                "Phase-1 MatrixCity fallback requires cached full depth_latent and normal_latent "
                "from the parquet dataset."
            )
        depth_latent = _trim_or_pad_latent_t(
            _ensure_branch_latent_bcfhw(raw_depth_latent,
                                        latent_channels=target_c,
                                        name="depth_latent"),
            int(target_t),
            name="depth_latent",
        ).to(device=self.device, dtype=torch.bfloat16)
        normal_latent = _trim_or_pad_latent_t(
            _ensure_branch_latent_bcfhw(raw_normal_latent,
                                        latent_channels=target_c,
                                        name="normal_latent"),
            int(target_t),
            name="normal_latent",
        ).to(device=self.device, dtype=torch.bfloat16)

        depth_path_by_id = {
            int(fid): p
            for fid, p in zip(clip["frame_ids"], clip["depth_paths"])
        }
        global_first_rgb = rgb_tchw[0]
        target_ids = [int(fid) for fid in clip["frame_ids"][1:target_frames]]
        if target_ids:
            warped_masked_rgb_valid, warped_mask_valid = mcprep._warp_maskrgb_from_keyframes_md_aligned_memory(
                keyframe_rgbs_u8=[mcprep._chw_float_to_u8(global_first_rgb)],
                keyframe_frame_ids=[int(clip["frame_ids"][0])],
                target_frame_ids=target_ids,
                depth_path_by_frame_id=depth_path_by_id,
                camera_k_aligned=clip["camera_k_aligned"],
                rt_by_frame_id=clip["rt_by_frame_id"],
                crop_params=clip["crop_params"],
                target_height=int(target_h),
                target_width=int(target_w),
            )
        else:
            warped_masked_rgb_valid = torch.empty(
                (0, 3, int(target_h), int(target_w)), dtype=torch.float32)
            warped_mask_valid = torch.empty(
                (0, 1, int(target_h), int(target_w)), dtype=torch.float32)

        masked_rgb_tchw = torch.cat(
            [global_first_rgb.unsqueeze(0), warped_masked_rgb_valid], dim=0)
        mask_tchw = torch.cat([
            torch.ones((1, 1, int(target_h), int(target_w)),
                       dtype=torch.float32),
            warped_mask_valid,
        ],
                               dim=0)

        masked_latent = _encode_control_branch_latent_with_infer_base(
            infer_base=infer_base,
            vae=self.vae,
            branch_tchw=masked_rgb_tchw,
            normalize=True,
            target_c=target_c,
            name="masked_latent",
            inference_device=self.device,
            compute_dtype=compute_dtype,
        ).to(dtype=torch.bfloat16)
        mask_latent = _encode_control_branch_latent_with_infer_base(
            infer_base=infer_base,
            vae=self.vae,
            branch_tchw=mask_tchw,
            normalize=False,
            target_c=target_c,
            name="mask_latent",
            inference_device=self.device,
            compute_dtype=compute_dtype,
        ).to(dtype=torch.bfloat16)

        control_latent = torch.cat(
            [depth_latent, normal_latent, masked_latent, mask_latent], dim=1)
        control_latent = _trim_or_pad_latent_t(control_latent,
                                               int(target_t),
                                               name="control_latent")
        return vae_latent, control_latent

    def _sample_blockwise_timestep_indices(
        self,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        device: torch.device,
        min_index: int | None = None,
        max_index: int | None = None,
    ) -> torch.Tensor:
        lo = self.min_timestep_index if min_index is None else int(min_index)
        hi = self.max_timestep_index if max_index is None else int(max_index)
        if hi <= lo:
            hi = lo + 1
        nfb = max(1, int(num_frame_per_block))
        num_chunks = (int(num_frame) + nfb - 1) // nfb
        idx = torch.randint(
            lo,
            hi,
            (batch_size, num_chunks),
            device=device,
            dtype=torch.long,
        )
        idx = idx.repeat_interleave(nfb, dim=1)
        return idx[:, :num_frame]

    def _get_next_batch(self, training_batch):
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        vae_latent = batch["vae_latent"]
        encoder_hidden_states = batch["text_embedding"]
        encoder_attention_mask = batch["text_attention_mask"]
        infos = batch["info_list"]
        first_frame_latent = _ensure_first_frame(batch["first_frame_latent"])
        control_latent = _ensure_control_latent(batch["control_latent"])
        raw_depth_latent = batch.get("depth_latent")
        raw_normal_latent = batch.get("normal_latent")

        need_rebuild_vae = vae_latent.numel() == 0 or vae_latent.ndim < 5
        need_rebuild_control = (
            control_latent.numel() == 0 or control_latent.ndim < 5 or
            int(control_latent.shape[2]) < int(self.training_args.num_latent_t))
        if need_rebuild_vae or need_rebuild_control:
            if len(infos) != int(first_frame_latent.shape[0]):
                raise ValueError(
                    "MatrixCity fallback expects infos and first_frame_latent batch sizes to match."
                )
            rebuilt_vae: list[torch.Tensor] = []
            rebuilt_control: list[torch.Tensor] = []
            for idx, info in enumerate(infos):
                info_dict = info if isinstance(info, dict) else shallow_asdict(
                    info)
                sample_vae, sample_control = self._rebuild_matrixcity_phase1_sample(
                    info=info_dict,
                    first_frame_latent=first_frame_latent[idx:idx + 1],
                    raw_depth_latent=None if raw_depth_latent is None else
                    raw_depth_latent[idx:idx + 1],
                    raw_normal_latent=None if raw_normal_latent is None else
                    raw_normal_latent[idx:idx + 1],
                )
                rebuilt_vae.append(sample_vae.cpu())
                rebuilt_control.append(sample_control.cpu())
            vae_latent = torch.cat(rebuilt_vae, dim=0)
            control_latent = torch.cat(rebuilt_control, dim=0)

        vae_latent = vae_latent[:, :, :self.training_args.num_latent_t]
        control_latent = control_latent[:, :, :self.training_args.num_latent_t]

        bsz, c, t, h, w = vae_latent.shape
        training_batch.raw_latent_shape = torch.Size([bsz, c, t, h, w])

        device = get_local_torch_device()
        dtype = torch.bfloat16
        training_batch.latents = vae_latent.to(device, dtype=dtype)
        training_batch.encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(device, dtype=dtype)
        training_batch.first_frame_latent = first_frame_latent.to(device, dtype=dtype)
        training_batch.control_latent = control_latent.to(device, dtype=dtype)
        training_batch.infos = infos
        return training_batch

    def train_one_step(self, training_batch):  # type: ignore[override]
        self.transformer.train()
        self.controlnet.train()
        self.optimizer.zero_grad()
        training_batch.total_loss = 0.0

        args = cast(TrainingArgs, self.training_args)
        device = get_local_torch_device()
        model_dtype = get_compute_dtype()
        # Base TrainingPipeline.train() may overwrite self.noise_scheduler with
        # FlowMatchEulerDiscreteScheduler, which does not implement add_noise.
        # AR-TF requires SelfForcingFlowMatchScheduler APIs.
        scheduler = self.noise_scheduler
        if not hasattr(scheduler, "add_noise"):
            scheduler = self.modules["scheduler"]
            self.noise_scheduler = scheduler

        debug_timing = self._debug_timing_enabled()
        if debug_timing:
            self._sync_device_for_debug()
            if torch.cuda.is_available():
                device_for_stats = get_local_torch_device()
                if device_for_stats.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device_for_stats)
            logger.info(
                "[AR-TF debug] train_step=%s begin train_one_step grad_accum=%s",
                int(getattr(self, "current_trainstep", 0)),
                int(args.gradient_accumulation_steps),
            )

        for micro_step in range(args.gradient_accumulation_steps):
            stage_t0 = time.perf_counter()
            training_batch = self._get_next_batch(training_batch)
            self._log_debug_stage(
                stage=f"micro{micro_step + 1}/get_next_batch",
                start_time=stage_t0,
            )

            clean_latent = training_batch.latents
            assert clean_latent is not None
            first_frame_latent = training_batch.first_frame_latent
            control_latent = training_batch.control_latent
            encoder_hidden_states = training_batch.encoder_hidden_states
            encoder_attention_mask = training_batch.encoder_attention_mask
            assert first_frame_latent is not None
            assert control_latent is not None
            assert encoder_hidden_states is not None
            assert encoder_attention_mask is not None

            bsz, c, num_frames, height, width = clean_latent.shape
            clean_latent_btfhw = clean_latent.permute(0, 2, 1, 3, 4).contiguous()
            local_noisy_partition = None
            if int(get_sp_world_size()) > 1:
                local_noisy_partition = get_sequence_parallel_block_partition(
                    int(num_frames), int(args.num_frame_per_block))

            noise_btfhw = torch.randn(
                clean_latent_btfhw.shape,
                generator=self.noise_gen_cuda,
                device=device,
                dtype=clean_latent.dtype,
            )

            stage_t0 = time.perf_counter()
            timestep_indices = self._sample_blockwise_timestep_indices(
                bsz,
                num_frames,
                int(args.num_frame_per_block),
                device,
            )
            timestep = scheduler.timesteps.to(device).index_select(
                0, timestep_indices.flatten()
            ).reshape(bsz, num_frames)

            noisy_input_btfhw = scheduler.add_noise(
                clean_latent_btfhw.flatten(0, 1),
                noise_btfhw.flatten(0, 1),
                timestep,
            ).unflatten(0, (bsz, num_frames))
            self._log_debug_stage(
                stage=f"micro{micro_step + 1}/noise_and_timestep",
                start_time=stage_t0,
                extra=f"latent_shape={tuple(clean_latent.shape)} timestep_shape={tuple(timestep.shape)}",
            )

            clean_context_btfhw = clean_latent_btfhw
            context_timestep = None
            stage_t0 = time.perf_counter()
            if int(getattr(args, "context_noise", 0)) > 0:
                max_ctx = min(int(args.context_noise),
                              int(scheduler.timesteps.numel()))
                if max_ctx > 0:
                    context_indices = self._sample_blockwise_timestep_indices(
                        bsz,
                        num_frames,
                        int(args.num_frame_per_block),
                        device,
                        min_index=0,
                        max_index=max_ctx,
                    )
                    context_timestep = scheduler.timesteps.to(
                        device).index_select(0, context_indices.flatten()).reshape(
                            bsz, num_frames)
                    clean_context_btfhw = scheduler.add_noise(
                        clean_latent_btfhw.flatten(0, 1),
                        noise_btfhw.flatten(0, 1),
                        context_timestep,
                    ).unflatten(0, (bsz, num_frames))
            self._log_debug_stage(
                stage=f"micro{micro_step + 1}/context_prepare",
                start_time=stage_t0,
            )

            # TI2V teacher-forcing anchor: keep first latent frame fixed to first frame condition.
            noisy_input = noisy_input_btfhw.permute(0, 2, 1, 3, 4).contiguous()
            if noisy_input.shape[2] >= 1:
                noisy_input[:, :, :1] = first_frame_latent
                # Strict alignment with causal ODE trajectory sampling:
                # always force the first latent frame timestep to 0 when
                # first-frame conditioning is injected.
                timestep = timestep.clone()
                timestep[:, 0] = 0.0
            clean_context_bcfhw = clean_context_btfhw.permute(0, 2, 1, 3, 4).contiguous()
            if hasattr(scheduler, "training_target"):
                target_flow = scheduler.training_target(
                    clean_latent_btfhw.flatten(0, 1),
                    noise_btfhw.flatten(0, 1),
                    timestep,
                ).unflatten(0, (bsz, num_frames))
            else:
                target_flow = noise_btfhw - clean_latent_btfhw

            forward_batch = ForwardBatch(data_type="ti2v_controlnet")
            forward_batch.prompt_embeds = [encoder_hidden_states]
            forward_batch.height = int(height) * 8
            forward_batch.width = int(width) * 8
            forward_batch.num_frames = int(num_frames)

            with set_forward_context(
                current_timestep=int(timestep[0, 0].item()),
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                # Keep teacher-forcing clean stream at t=0.
                clean_aug_t = torch.zeros_like(
                    timestep, device=device, dtype=model_dtype
                )
                num_channels_latents = getattr(
                    self.transformer, "num_channels_latents", control_latent.shape[1] // 3
                )
                stage_t0 = time.perf_counter()
                control_res = self.controlnet(
                    hidden_states=noisy_input.to(dtype=model_dtype),
                    encoder_hidden_states=[encoder_hidden_states.to(dtype=model_dtype)],
                    timestep=timestep.to(device, dtype=model_dtype),
                    clean_hidden_states=clean_context_bcfhw.to(dtype=model_dtype),
                    aug_t=clean_aug_t,
                    **_build_controlnet_kwargs(
                        self.controlnet,
                        control_latent.to(dtype=model_dtype),
                        num_channels_latents,
                    ),
                )
                if debug_timing:
                    first_residual_shape = None
                    if isinstance(control_res, (list, tuple)) and len(control_res) > 0:
                        first_residual_shape = tuple(control_res[0].shape)
                    self._log_debug_stage(
                        stage=f"micro{micro_step + 1}/controlnet_forward",
                        start_time=stage_t0,
                        extra=(
                            f"num_residuals={len(control_res) if isinstance(control_res, (list, tuple)) else 'na'} "
                            f"first_residual_shape={first_residual_shape}"
                        ),
                    )
                stage_t0 = time.perf_counter()
                pred_flow = self.transformer(
                    noisy_input.to(dtype=model_dtype),
                    [encoder_hidden_states.to(dtype=model_dtype)],
                    timestep.to(device, dtype=model_dtype),
                    block_controlnet_hidden_states=control_res,
                    clean_x=clean_context_bcfhw.to(dtype=model_dtype),
                    aug_t=clean_aug_t,
                ).permute(0, 2, 1, 3, 4)
                self._log_debug_stage(
                    stage=f"micro{micro_step + 1}/transformer_forward",
                    start_time=stage_t0,
                    extra=f"pred_flow_shape={tuple(pred_flow.shape)}",
                )

            loss_timestep = timestep
            loss_target_flow = target_flow
            if local_noisy_partition is not None:
                loss_timestep = timestep[:, local_noisy_partition.frame_start:
                                         local_noisy_partition.frame_end]
                loss_target_flow = target_flow[:, local_noisy_partition.frame_start:
                                               local_noisy_partition.frame_end]

            if pred_flow.shape != loss_target_flow.shape:
                raise ValueError(
                    "Pred/target shape mismatch after teacher-forcing SP slice: "
                    f"pred={tuple(pred_flow.shape)} target={tuple(loss_target_flow.shape)}"
                )

            stage_t0 = time.perf_counter()
            per_frame_loss = (pred_flow.float() - loss_target_flow.float()).pow(2).mean(
                dim=(2, 3, 4)
            )
            if hasattr(scheduler, "training_weight"):
                weight = scheduler.training_weight(loss_timestep).unflatten(
                    0, (bsz, loss_timestep.shape[1]))
                per_frame_loss = per_frame_loss * weight.to(
                    device=per_frame_loss.device, dtype=per_frame_loss.dtype)
            nonzero_mask = loss_timestep != 0
            if per_frame_loss.numel() == 0:
                local_loss_terms = pred_flow.sum().float().reshape(1) * 0.0
            elif torch.any(nonzero_mask):
                local_loss_terms = per_frame_loss[nonzero_mask]
            else:
                local_loss_terms = per_frame_loss.reshape(-1)

            local_loss_sum = local_loss_terms.sum()
            global_loss_count = torch.tensor(
                [float(local_loss_terms.numel())],
                device=device,
                dtype=torch.float32,
            )
            if dist.is_initialized():
                dist.all_reduce(global_loss_count, op=dist.ReduceOp.SUM)
            global_loss_count = torch.clamp(global_loss_count[0], min=1.0)

            loss = local_loss_sum / global_loss_count
            loss = loss / args.gradient_accumulation_steps
            self._log_debug_stage(
                stage=f"micro{micro_step + 1}/loss_compute",
                start_time=stage_t0,
                extra=f"loss={float(loss.detach().item()):.6f}",
            )
            backward_start_holder = {"value": None}
            if debug_timing:
                def _make_backward_hook(tag: str):
                    def _hook(grad):
                        start_value = backward_start_holder["value"]
                        if start_value is not None:
                            self._log_backward_hook(
                                tag=tag,
                                backward_start_time=start_value,
                                grad=grad,
                            )
                        return grad
                    return _hook

                if pred_flow.requires_grad:
                    pred_flow.register_hook(_make_backward_hook("pred_flow"))
                if isinstance(control_res, (list, tuple)) and len(control_res) > 0:
                    residual_last = control_res[-1]
                    if isinstance(residual_last,
                                  torch.Tensor) and residual_last.requires_grad:
                        residual_last.register_hook(
                            _make_backward_hook("control_residual_last"))
                    residual_mid = control_res[len(control_res) // 2]
                    if isinstance(residual_mid,
                                  torch.Tensor) and residual_mid.requires_grad:
                        residual_mid.register_hook(
                            _make_backward_hook("control_residual_mid"))
                    residual0 = control_res[0]
                    if isinstance(residual0, torch.Tensor) and residual0.requires_grad:
                        residual0.register_hook(
                            _make_backward_hook("control_residual_0"))
            stage_t0 = time.perf_counter()
            backward_start_holder["value"] = stage_t0
            if self._backward_profile_enabled():
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                with profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    loss.backward()
                self._dump_backward_profile(prof)
            else:
                loss.backward()
            self._log_debug_stage(
                stage=f"micro{micro_step + 1}/backward",
                start_time=stage_t0,
            )
            detached_loss_sum = local_loss_sum.detach().float()
            if dist.is_initialized():
                dist.all_reduce(detached_loss_sum, op=dist.ReduceOp.SUM)
            logged_global_loss = detached_loss_sum / global_loss_count
            training_batch.total_loss += float(
                logged_global_loss.item() /
                args.gradient_accumulation_steps)

        grad_norm = None
        if debug_timing:
            self._log_rank_consistency(
                stage="pre_clip_grad_norm",
                values=[
                    float(local_loss_terms.numel()),
                    float(pred_flow.shape[1]),
                    float(loss_target_flow.shape[1]),
                ],
            )
            if dist.is_initialized():
                dist.barrier()
        stage_t0 = time.perf_counter()
        if args.max_grad_norm is not None and float(args.max_grad_norm) > 0.0:
            grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for p in self.transformer.parameters() if p.requires_grad]
                + [p for p in self.controlnet.parameters() if p.requires_grad],
                float(args.max_grad_norm),
            )
        self._log_debug_stage(stage="clip_grad_norm", start_time=stage_t0)
        if debug_timing:
            if dist.is_initialized():
                dist.barrier()
            self._log_rank_consistency(
                stage="pre_optimizer_step",
                values=[0.0 if grad_norm is None else
                        float(grad_norm.detach().float().item()
                              if isinstance(grad_norm, torch.Tensor)
                              else grad_norm)],
            )
        stage_t0 = time.perf_counter()
        self.optimizer.step()
        self.lr_scheduler.step()
        self._log_debug_stage(stage="optimizer_step", start_time=stage_t0)

        if grad_norm is None:
            training_batch.grad_norm = 0.0
        else:
            try:
                training_batch.grad_norm = float(
                    grad_norm.detach().float().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            except Exception:
                training_batch.grad_norm = 0.0
        return training_batch


def main(args) -> None:
    logger.info("Starting Wan ControlNet AR-TF training pipeline...")
    pipeline = ARTFControlnetTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("Wan ControlNet AR-TF training pipeline completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
