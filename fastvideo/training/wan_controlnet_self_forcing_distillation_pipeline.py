# SPDX-License-Identifier: Apache-2.0
import hashlib
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.dataset.utils import pad
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.forward_context import set_forward_context
from fastvideo.training.self_forcing_distillation_pipeline import (
    SelfForcingDistillationPipeline,
)
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_utils import (
    EMA_FSDP,
    clip_grad_norm_while_handling_failing_dtensor_cases,
    get_scheduler,
    normalize_dit_input,
)
from fastvideo.utils import is_vsa_available, shallow_asdict
from fastvideo.models.dits.controlnet_union_components import WanControlNetUnionInput

vsa_available = is_vsa_available()
logger = init_logger(__name__)


def _numpy_dtype_from_name(dtype_name: str | None) -> np.dtype:
    if not dtype_name:
        return np.dtype(np.float32)
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("float16", "fp16"):
        return np.dtype(np.float16)
    if dtype_name in ("float32", "fp32", "bfloat16", "bf16"):
        return np.dtype(np.float32)
    return np.dtype(np.float32)


def _load_fixed_text_condition_from_parquet(
    *,
    data_path: str,
    row_idx: int,
    text_padding_length: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    root = Path(data_path).expanduser()
    if root.is_file():
        parquet_files = [root]
    else:
        parquet_files = sorted(p for p in root.rglob("*.parquet") if p.is_file())
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under fixed_text_embedding_data_path={root}"
        )

    remaining = int(row_idx)
    target_file: Path | None = None
    local_row_idx = 0
    for parquet_file in parquet_files:
        num_rows = int(pq.ParquetFile(parquet_file).metadata.num_rows)
        if remaining < num_rows:
            target_file = parquet_file
            local_row_idx = remaining
            break
        remaining -= num_rows
    if target_file is None:
        raise IndexError(
            f"fixed_text_embedding_row_idx={int(row_idx)} is out of range for {root}"
        )

    table = pq.read_table(
        str(target_file),
        columns=[
            "text_embedding_bytes",
            "text_embedding_shape",
            "text_embedding_dtype",
            "caption",
        ],
        use_threads=False,
    )
    text_embedding_bytes = table["text_embedding_bytes"][local_row_idx].as_py()
    text_embedding_shape = tuple(
        int(v) for v in table["text_embedding_shape"][local_row_idx].as_py())
    text_embedding_dtype = table["text_embedding_dtype"][local_row_idx].as_py()
    caption = table["caption"][local_row_idx].as_py() or ""

    text_embedding_np = np.frombuffer(
        text_embedding_bytes,
        dtype=_numpy_dtype_from_name(text_embedding_dtype),
    ).reshape(text_embedding_shape).copy()
    text_embedding = torch.from_numpy(text_embedding_np)
    text_embedding, text_attention_mask = pad(text_embedding,
                                              int(text_padding_length))
    return text_embedding, text_attention_mask, str(caption), str(target_file)


def _is_union_controlnet(model) -> bool:
    return "union" in model.__class__.__name__.lower()


def _split_union_control_latent(control_latent: torch.Tensor,
                                num_channels_latents: int
                                ) -> tuple[torch.Tensor, torch.Tensor | None,
                                           torch.Tensor, torch.Tensor]:
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
        f"expected 3*C or 4*C (C={c}).")


def _build_controlnet_kwargs(controlnet, control_latent: torch.Tensor,
                             num_channels_latents: int) -> dict:
    if not _is_union_controlnet(controlnet):
        return {"controlnet_states": control_latent}
    depth, normal, masked, mask = _split_union_control_latent(
        control_latent, num_channels_latents)
    return {
        "controlnet_cond": WanControlNetUnionInput(depth=depth, normal=normal),
        "mask": mask,
        "masked_latent": masked,
    }


def _normalize_first_frame_latent(first_frame_latent: torch.Tensor,
                                  vae,
                                  *,
                                  enabled: bool = False) -> torch.Tensor:
    if (not enabled) or first_frame_latent.ndim != 5:
        return first_frame_latent
    latent_bcfhw = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
    latent_bcfhw = normalize_dit_input("wan", latent_bcfhw, vae)
    return latent_bcfhw.permute(0, 2, 1, 3, 4).contiguous()


def _normalize_control_latent(control_latent: torch.Tensor, vae,
                              num_channels_latents: int,
                              *,
                              enabled: bool = False) -> torch.Tensor:
    if (not enabled) or control_latent.ndim != 5:
        return control_latent
    chunks = []
    total_channels = control_latent.shape[1]
    if total_channels % num_channels_latents != 0:
        raise ValueError(
            f"control_latent channels {total_channels} not divisible by latent channels {num_channels_latents}"
        )
    for start in range(0, total_channels, num_channels_latents):
        chunk = control_latent[:, start:start + num_channels_latents]
        chunks.append(normalize_dit_input("wan", chunk, vae))
    return torch.cat(chunks, dim=1)


def _apply_first_frame_latent(
    hidden_states: torch.Tensor,
    first_frame_latent: torch.Tensor | None,
) -> torch.Tensor:
    if first_frame_latent is None:
        return hidden_states
    if hidden_states.ndim != 5 or first_frame_latent.ndim != 5:
        return hidden_states
    image_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
    if hidden_states.shape[1] != image_latent.shape[1] or hidden_states.shape[
            2] < 1:
        return hidden_states
    conditioned = hidden_states.clone()
    conditioned[:, :, :1] = image_latent[:, :, :1]
    return conditioned


def _apply_first_frame_latent_bfchw(
    video_latent: torch.Tensor,
    first_frame_latent: torch.Tensor | None,
) -> torch.Tensor:
    if first_frame_latent is None:
        return video_latent
    if video_latent.ndim != 5 or first_frame_latent.ndim != 5:
        return video_latent
    if video_latent.shape[1] < 1:
        return video_latent
    if (video_latent.shape[0] != first_frame_latent.shape[0]
            or video_latent.shape[2] != first_frame_latent.shape[2]):
        return video_latent
    conditioned = video_latent.clone()
    conditioned[:, :1] = first_frame_latent[:, :1]
    return conditioned


def _with_first_frame_timestep_zero(
    timestep: Any,
    *,
    batch_size: int,
    num_frames: int,
) -> Any:
    if num_frames <= 0 or not isinstance(timestep, torch.Tensor):
        return timestep
    if timestep.ndim == 0:
        model_timestep = timestep.reshape(1, 1).expand(batch_size,
                                                       num_frames).clone()
    elif timestep.ndim == 1:
        if int(timestep.shape[0]) != int(batch_size):
            return timestep
        model_timestep = timestep.view(batch_size, 1).expand(
            batch_size, num_frames).clone()
    elif timestep.ndim == 2:
        if int(timestep.shape[0]) != int(batch_size):
            return timestep
        if int(timestep.shape[1]) == int(num_frames):
            model_timestep = timestep.clone()
        elif int(timestep.shape[1]) == 1:
            model_timestep = timestep.expand(batch_size, num_frames).clone()
        else:
            return timestep
    else:
        return timestep
    model_timestep[:, 0] = 0
    return model_timestep


def _maybe_expand_timestep_to_token_sequence(
    timestep: Any,
    *,
    hidden_states: torch.Tensor,
    model: torch.nn.Module,
) -> Any:
    if not isinstance(timestep, torch.Tensor):
        return timestep
    if hidden_states.ndim != 5 or timestep.ndim != 2:
        return timestep
    if int(timestep.shape[0]) != int(hidden_states.shape[0]):
        return timestep

    model_name = model.__class__.__name__.lower()
    if "causal" in model_name:
        return timestep

    num_frames = int(hidden_states.shape[2])
    if int(timestep.shape[1]) != num_frames:
        return timestep

    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(model, "config", None), "patch_size",
                             None)
    if patch_size is None:
        patch_size = getattr(getattr(getattr(model, "config", None),
                                     "arch_config", None), "patch_size",
                             None)
    if patch_size is None or len(patch_size) < 2:
        return timestep

    patch_h = int(patch_size[-2])
    patch_w = int(patch_size[-1])
    if patch_h <= 0 or patch_w <= 0:
        return timestep

    latent_h = int(hidden_states.shape[3])
    latent_w = int(hidden_states.shape[4])
    if latent_h % patch_h != 0 or latent_w % patch_w != 0:
        return timestep

    frame_seq_len = (latent_h // patch_h) * (latent_w // patch_w)
    if frame_seq_len <= 1:
        return timestep

    return timestep.repeat_interleave(frame_seq_len, dim=1)


def _drop_first_frame_bfchw(
    value: torch.Tensor,
    first_frame_latent: torch.Tensor | None,
) -> torch.Tensor:
    if first_frame_latent is None:
        return value
    if value.ndim != 5 or value.shape[1] <= 1:
        return value
    return value[:, 1:]


def _sample_uniform_score_timestep(*,
                                   batch_size: int,
                                   device: torch.device,
                                   num_train_timestep: int,
                                   timestep_shift: float,
                                   min_timestep: int,
                                   max_timestep: int,
                                   use_rollout_min: bool,
                                   use_rollout_max: bool,
                                   denoised_timestep_from: int | None,
                                   denoised_timestep_to: int | None
                                   ) -> torch.Tensor:
    raw_min_timestep = int(min_timestep)
    raw_max_timestep = int(max_timestep)
    if use_rollout_min and denoised_timestep_to is not None:
        raw_min_timestep = max(raw_min_timestep, int(denoised_timestep_to))
    if use_rollout_max and denoised_timestep_from is not None:
        raw_max_timestep = min(raw_max_timestep, int(denoised_timestep_from))
    if raw_max_timestep < raw_min_timestep:
        raw_min_timestep, raw_max_timestep = min_timestep, max_timestep

    timestep = torch.randint(raw_min_timestep,
                             raw_max_timestep + 1, [batch_size],
                             device=device,
                             dtype=torch.long)
    from fastvideo.training.training_utils import shift_timestep
    timestep = shift_timestep(timestep, timestep_shift, num_train_timestep)
    return timestep.clamp(min_timestep, max_timestep)


def _normalize_student_attention_mode(value: Any) -> str:
    mode = str(value or "causal").strip().lower()
    if mode not in ("causal", "bidirectional"):
        raise ValueError(
            "student_attention_mode must be one of ['causal', 'bidirectional'], "
            f"got {value!r}")
    return mode


def _np_dtype(dtype_str: str | None) -> np.dtype:
    if dtype_str is None or dtype_str == "":
        return np.float32
    s = dtype_str.lower()
    if s in ("float", "float32", "fp32"):
        return np.float32
    if s in ("float16", "fp16"):
        return np.float16
    if s in ("int64", "long"):
        return np.int64
    if s in ("int32",):
        return np.int32
    raise ValueError(f"Unsupported dtype in validation row: {dtype_str}")


def _decode_validation_tensor(row: dict[str, Any], prefix: str) -> torch.Tensor:
    shape = row.get(f"{prefix}_shape")
    blob = row.get(f"{prefix}_bytes")
    dtype_str = row.get(f"{prefix}_dtype")
    if blob is None or shape is None:
        raise KeyError(
            f"Missing {prefix}_bytes/{prefix}_shape in validation sample")
    array = np.frombuffer(blob, dtype=_np_dtype(dtype_str)).reshape(shape).copy()
    return torch.from_numpy(array)


def _samplewise_std(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().flatten(1).std(dim=1, unbiased=False).view(
        tensor.shape[0], 1, 1, 1, 1)


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


def _clone_training_batch_shallow(src: TrainingBatch) -> TrainingBatch:
    dst = TrainingBatch()
    for key, value in src.__dict__.items():
        setattr(dst, key, value)
    return dst


def _pixels_to_unit_range(pixels_bcthw: torch.Tensor) -> torch.Tensor:
    return (pixels_bcthw / 2 + 0.5).clamp(0, 1)


def _compute_num_full_windows(total_frames: int, window_frames: int,
                              overlap_frames: int) -> int:
    if int(window_frames) <= 0:
        raise ValueError(
            f"window_frames must be > 0, got {int(window_frames)}")
    if int(overlap_frames) < 0 or int(overlap_frames) >= int(window_frames):
        raise ValueError(
            "overlap_frames must satisfy 0 <= overlap < window, got "
            f"overlap={int(overlap_frames)} window={int(window_frames)}")
    if int(total_frames) < int(window_frames):
        return 0
    stride = int(window_frames) - int(overlap_frames)
    return 1 + max(0,
                   (int(total_frames) - int(window_frames)) // int(stride))


def _pad_paths(paths: list[Path], target_len: int) -> list[Path]:
    if len(paths) >= int(target_len):
        return list(paths[:int(target_len)])
    if not paths:
        raise ValueError("cannot pad empty path list")
    return list(paths) + [paths[-1]] * (int(target_len) - len(paths))


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


def _pad_branch_latent_t(branch_latent: torch.Tensor,
                         target_t: int) -> torch.Tensor:
    if int(branch_latent.shape[2]) >= int(target_t):
        return branch_latent[:, :, :int(target_t)]
    if int(branch_latent.shape[2]) <= 0:
        raise ValueError("cannot pad empty branch latent")
    pad = branch_latent[:, :, -1:].repeat(1, 1,
                                          int(target_t) - int(branch_latent.shape[2]),
                                          1, 1)
    return torch.cat([branch_latent, pad], dim=2)


def _slice_cached_branch_latent_window(
    branch_latent: torch.Tensor,
    *,
    start_frame: int,
    temporal_ratio: int,
    target_t: int,
    name: str,
) -> torch.Tensor:
    if int(temporal_ratio) <= 0:
        raise ValueError(f"temporal_ratio must be > 0, got {int(temporal_ratio)}")
    if int(start_frame) % int(temporal_ratio) != 0:
        raise ValueError(
            f"{name} cached slicing requires start_frame divisible by temporal_ratio: "
            f"start_frame={int(start_frame)} temporal_ratio={int(temporal_ratio)}")
    start_t = int(start_frame) // int(temporal_ratio)
    sliced = branch_latent[:, :, start_t:start_t + int(target_t)]
    return _pad_branch_latent_t(sliced, int(target_t))


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


def _ensure_first_frame_bcfhw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    elif x.dim() == 5 and x.shape[1] in (1, 3) and x.shape[2] >= 8:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
    if x.dim() != 5 or x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must be [B,C,1,H,W], got {tuple(x.shape)}")
    return x


def _ensure_control_latent_bcfhw(x: torch.Tensor, *,
                                 latent_channels: int) -> torch.Tensor:
    valid_channels = (3 * int(latent_channels), 4 * int(latent_channels))
    if x.dim() == 4:
        if int(x.shape[0]) not in valid_channels:
            raise ValueError(
                f"control_latent 4D shape must be [C_total,F,H,W], got {tuple(x.shape)}"
            )
        return x.unsqueeze(0)
    if x.dim() == 5:
        if int(x.shape[1]) in valid_channels:
            return x
        if int(x.shape[2]) in valid_channels:
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(
        f"control_latent must be [B,C_total,F,H,W] or [B,F,C_total,H,W], got {tuple(x.shape)}"
    )


class WanControlnetSelfForcingDistillationPipeline(SelfForcingDistillationPipeline):
    """
    Self-forcing DMD distillation with an external ControlNet module.

    - Student/generator trains: `transformer` + `controlnet`
    - Critic trains: `fake_score_transformer` + `fake_score_controlnet`
    - Teacher is frozen: `real_score_transformer` + `real_score_controlnet`

    ControlNet residuals are injected into both causal and bidirectional Wan transformers
    via `block_controlnet_hidden_states`.

    Control latent convention:
      If Wan VAE latent channel size is `C_lat` (z_dim), then:
        - video latents: (B, C_lat, T_lat, H_lat, W_lat)
        - control latents: (B, 3*C_lat, T_lat, H_lat, W_lat) = cat(depth, masked_rgb, mask)
    """

    _required_config_modules = ["scheduler", "transformer", "vae"]
    trainable_transformer_names = [
        "transformer",
        "controlnet",
        "fake_score_transformer",
        "fake_score_controlnet",
    ]

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_ti2v_controlnet

    def load_modules(self,
                     fastvideo_args: FastVideoArgs,
                     loaded_modules: dict[str, torch.nn.Module] | None = None):
        training_args = cast(TrainingArgs, fastvideo_args)
        student_attention_mode = _normalize_student_attention_mode(
            getattr(training_args, "student_attention_mode", "causal"))
        if student_attention_mode == "bidirectional":
            student_transformer_cls_name = "WanTransformer3DModel"
            student_controlnet_cls_name = "WanControlnetUnion3DModel"
        else:
            student_transformer_cls_name = "CausalWanTransformer3DModel"
            student_controlnet_cls_name = "CausalWanControlnetUnion3DModel"

        # Teacher/critic transformers are loaded later and explicitly forced to
        # bidirectional in `DistillationPipeline`. The student can optionally
        # switch to a full-sequence bidirectional rollout for RGBN teacher-init
        # experiments.
        prev_override = getattr(training_args, "override_transformer_cls_name",
                                None)
        training_args.override_transformer_cls_name = (
            student_transformer_cls_name)
        try:
            modules = super().load_modules(fastvideo_args, loaded_modules)
        finally:
            training_args.override_transformer_cls_name = prev_override

        # Load student ControlNet (component dir, diffusers format)
        if training_args.controlnet_model_path:
            logger.info("Loading student controlnet from: %s",
                        training_args.controlnet_model_path)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = (
                student_controlnet_cls_name)
            logger.info("Student controlnet override class: %s",
                        training_args.override_controlnet_cls_name)
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
                    "Phase-2 student controlnet must be Union. "
                    "Please use a Union ControlNet checkpoint/config."
                )
            modules["controlnet"] = self.controlnet
        else:
            self.controlnet = None

        # Teacher ControlNet
        if training_args.real_score_controlnet_model_path:
            logger.info("Loading teacher controlnet from: %s",
                        training_args.real_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to teacher.
            setattr(training_args, "_loading_teacher_critic_model", True)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = "WanControlnetUnion3DModel"
            try:
                self.real_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.real_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
            if not _is_union_controlnet(self.real_score_controlnet):
                raise ValueError(
                    "Teacher controlnet must be Union for phase-2 Union training."
                )
            modules["real_score_controlnet"] = self.real_score_controlnet
        else:
            self.real_score_controlnet = None

        # Critic ControlNet
        if training_args.fake_score_controlnet_model_path:
            logger.info("Loading critic controlnet from: %s",
                        training_args.fake_score_controlnet_model_path)
            # Prevent student custom init weights from being applied to critic.
            setattr(training_args, "_loading_teacher_critic_model", True)
            prev_cn_override = getattr(training_args,
                                       "override_controlnet_cls_name", None)
            training_args.override_controlnet_cls_name = "WanControlnetUnion3DModel"
            try:
                self.fake_score_controlnet = PipelineComponentLoader.load_module(
                    module_name="controlnet",
                    component_model_path=training_args.fake_score_controlnet_model_path,
                    transformers_or_diffusers="diffusers",
                    fastvideo_args=training_args,
                )
            finally:
                training_args.override_controlnet_cls_name = prev_cn_override
                if hasattr(training_args, "_loading_teacher_critic_model"):
                    delattr(training_args, "_loading_teacher_critic_model")
            if not _is_union_controlnet(self.fake_score_controlnet):
                raise ValueError(
                    "Critic controlnet must be Union for phase-2 Union training."
                )
            modules["fake_score_controlnet"] = self.fake_score_controlnet
        else:
            self.fake_score_controlnet = None

        return modules

    def _lazy_init_online_warp_training_state(self) -> None:
        if getattr(self, "_online_warp_state_initialized", False):
            return
        if not bool(getattr(self.training_args, "online_warp_training", False)):
            return

        raw_root = Path(
            str(getattr(self.training_args, "online_warp_raw_root", "")).strip()
        ).expanduser()
        if not raw_root.is_dir():
            raise FileNotFoundError(
                "online_warp_training requires a valid online_warp_raw_root, "
                f"got {raw_root}"
            )

        from tools import infer_wan_controlnet_ti2v as infer_base
        from tools import preprocess_matrixcity_ti2v_controlnet_parquet as mcprep

        street_split = str(
            getattr(self.training_args, "online_warp_street_split", "train_dense"))
        camera_mode = str(
            getattr(self.training_args, "online_warp_camera_mode", "B_inv"))
        pose_index = mcprep._load_matrixcity_pose_index(
            rgb_root=raw_root,
            street_split=street_split,
            camera_mode=camera_mode,
        )
        self._online_warp_infer_base = infer_base
        self._online_warp_mcprep = mcprep
        self._online_warp_raw_root = raw_root
        self._online_warp_pose_index = pose_index
        self._online_warp_scene_assets: dict[str, dict[str, Any]] = {}
        self._online_warp_state_initialized = True
        logger.info(
            "Initialized online-warp training state: raw_root=%s split=%s camera_mode=%s depth_mode=%s scenes=%d",
            raw_root,
            street_split,
            camera_mode,
            str(
                getattr(
                    self.training_args,
                    "online_warp_depth_normalization_mode",
                    "md_align",
                )),
            len(pose_index),
        )

    def _lazy_init_fixed_training_text_condition(self) -> None:
        if getattr(self, "_fixed_training_text_condition_initialized", False):
            return
        self._fixed_training_text_condition_initialized = True

        fixed_data_path = str(
            getattr(self.training_args, "fixed_text_embedding_data_path",
                    "")).strip()
        if not fixed_data_path:
            self._fixed_training_text_embedding = None
            self._fixed_training_text_attention_mask = None
            self._fixed_training_text_caption = ""
            return

        text_cfg = self.training_args.pipeline_config.text_encoder_configs[0]
        text_padding_length = int(text_cfg.arch_config.text_len)
        row_idx = int(
            getattr(self.training_args, "fixed_text_embedding_row_idx", 0))
        explicit_caption = str(
            getattr(self.training_args, "fixed_text_embedding_caption",
                    "")).strip()

        (fixed_text_embedding, fixed_text_attention_mask, source_caption,
         source_file) = _load_fixed_text_condition_from_parquet(
             data_path=fixed_data_path,
             row_idx=row_idx,
             text_padding_length=text_padding_length,
         )
        self._fixed_training_text_embedding = fixed_text_embedding.contiguous()
        self._fixed_training_text_attention_mask = fixed_text_attention_mask.contiguous(
        )
        self._fixed_training_text_caption = explicit_caption or source_caption

        logger.info(
            "Loaded fixed training text condition from %s row=%s caption=%r shape=%s mean=%.6f std=%.6f",
            source_file,
            row_idx,
            self._fixed_training_text_caption,
            tuple(self._fixed_training_text_embedding.shape),
            float(self._fixed_training_text_embedding.float().mean().item()),
            float(
                self._fixed_training_text_embedding.float().std(
                    unbiased=False).item()),
        )

    def _resolve_online_warp_scene_dir(self, scene_name: str) -> Path:
        raw_root = cast(Path, getattr(self, "_online_warp_raw_root"))
        street_split = str(
            getattr(self.training_args, "online_warp_street_split", "train_dense"))
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

    def _get_online_warp_scene_assets(self, scene_name: str) -> dict[str, Any]:
        self._lazy_init_online_warp_training_state()
        scene_cache = getattr(self, "_online_warp_scene_assets", {})
        cached = scene_cache.get(scene_name)
        if cached is not None:
            return cached

        from PIL import Image

        mcprep = getattr(self, "_online_warp_mcprep")
        raw_root = cast(Path, getattr(self, "_online_warp_raw_root"))
        street_split = str(
            getattr(self.training_args, "online_warp_street_split", "train_dense"))
        require_normal = bool(
            getattr(self.training_args, "online_warp_require_normal", True))
        scene_dir = self._resolve_online_warp_scene_dir(scene_name)
        scene_pose_index = getattr(self, "_online_warp_pose_index")[scene_name]

        rgb_dir = scene_dir / scene_name
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB dir not found for scene={scene_name}: {rgb_dir}")
        rgb_files = mcprep._sorted_pngs(rgb_dir)
        rgb_map = mcprep._build_numeric_file_map(rgb_files)
        if not rgb_map:
            raise FileNotFoundError(
                f"No RGB files found for scene={scene_name}: {rgb_dir}")

        depth_dir = (
            raw_root
            / "small_city_depth"
            / "street"
            / street_split
            / f"{scene_name}_depth"
            / f"{scene_name}_depth"
        )
        if not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Depth dir not found for scene={scene_name}: {depth_dir}")
        depth_files = mcprep._sorted_depth_files(depth_dir)
        depth_map = mcprep._build_numeric_file_map(depth_files)
        if not depth_map:
            raise FileNotFoundError(
                f"No depth files found for scene={scene_name}: {depth_dir}")

        normal_map: dict[int, Path] | None = None
        normal_dir_candidates = [
            raw_root / "small_city_normal" / "street" / street_split / f"{scene_name}_normal" / f"{scene_name}_normal",
            raw_root / "small_city_normal" / "street" / street_split / scene_name / scene_name,
            raw_root / "street" / street_split / f"{scene_name}_normal" / f"{scene_name}_normal",
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
        if require_normal and not normal_map:
            raise FileNotFoundError(
                f"Normal dir not found or empty for scene={scene_name} under {raw_root}"
            )

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
        self._online_warp_scene_assets = scene_cache
        return assets

    def _build_online_warp_clip_from_info(
        self,
        info: dict[str, Any],
        total_required_frames: int,
    ) -> dict[str, Any]:
        self._lazy_init_online_warp_training_state()
        mcprep = getattr(self, "_online_warp_mcprep")

        record_id = str(info.get("id") or info.get("file_name") or "").strip()
        if not record_id:
            raise KeyError(
                "online_warp_training requires info['id'] or info['file_name'] to reconstruct the raw clip"
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
                "Requested online-warp rollout exceeds clip coverage: "
                f"record_id={record_id} requested={int(total_required_frames)} "
                f"available={int(max_available_frames)}"
            )

        window_frames = int(
            getattr(self.training_args, "online_warp_window_frames", 81))
        overlap_frames = int(
            getattr(self.training_args, "online_warp_overlap_frames", 1))
        max_num_windows = _compute_num_full_windows(
            int(max_available_frames),
            int(window_frames),
            int(overlap_frames),
        )
        if max_num_windows <= 0:
            raise ValueError(
                "online_warp_training requires at least one full window worth of frames: "
                f"record_id={record_id} available={int(max_available_frames)} "
                f"window_frames={int(window_frames)}"
            )

        assets = self._get_online_warp_scene_assets(scene_name)
        target_frame_ids = [
            int(clip_start_global) + i for i in range(int(total_required_frames))
        ]
        rgb_paths = mcprep._pick_by_target_ids(assets["rgb_map"], target_frame_ids)
        depth_paths = mcprep._pick_by_target_ids(assets["depth_map"], target_frame_ids)
        normal_map = assets.get("normal_map")
        normal_paths = (mcprep._pick_by_target_ids(normal_map, target_frame_ids)
                        if normal_map else None)

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
            "window_frames": int(window_frames),
            "overlap_frames": int(overlap_frames),
            "stride_frames": int(window_frames - overlap_frames),
            "max_available_frames": int(max_available_frames),
            "max_num_windows": int(max_num_windows),
        }

    def _sample_online_warp_num_windows(self, max_num_windows: int) -> int:
        min_windows = max(
            1, int(getattr(self.training_args, "online_warp_rollout_min_windows", 2)))
        cfg_max_windows = int(
            getattr(self.training_args, "online_warp_rollout_max_windows", 0))
        if cfg_max_windows > 0:
            max_num_windows = min(int(max_num_windows), int(cfg_max_windows))
        max_num_windows = max(1, int(max_num_windows))
        if max_num_windows < min_windows:
            min_windows = max_num_windows
        if min_windows == max_num_windows:
            return int(max_num_windows)
        candidate_windows = list(
            range(int(min_windows), int(max_num_windows) + 1))
        weights_str = str(
            getattr(self.training_args, "online_warp_rollout_window_weights",
                    "") or "").strip()
        if weights_str:
            try:
                weights = [float(x.strip()) for x in weights_str.split(",")]
            except Exception as exc:
                raise ValueError(
                    "Failed to parse online_warp_rollout_window_weights="
                    f"{weights_str!r}") from exc
            if len(weights) != len(candidate_windows):
                raise ValueError(
                    "online_warp_rollout_window_weights length mismatch: "
                    f"weights={weights} candidates={candidate_windows}")
            if any((not math.isfinite(w)) or w < 0.0 for w in weights):
                raise ValueError(
                    "online_warp_rollout_window_weights must be finite and >= 0: "
                    f"{weights}")
            weight_sum = float(sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(
                    "online_warp_rollout_window_weights must sum to > 0, got "
                    f"{weights}")
            probs = torch.tensor(weights,
                                 device=self.device,
                                 dtype=torch.float32)
            probs = probs / probs.sum()
            sampled_index = torch.multinomial(probs, num_samples=1)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(sampled_index, src=0)
            if not getattr(self, "_logged_online_warp_window_sampling", False):
                logger.info(
                    "online_warp rollout window sampling policy: candidates=%s probs=%s",
                    candidate_windows,
                    [round(float(x), 4) for x in probs.detach().cpu().tolist()],
                )
                self._logged_online_warp_window_sampling = True
            return int(candidate_windows[int(sampled_index.item())])
        sampled = torch.randint(
            low=int(min_windows),
            high=int(max_num_windows) + 1,
            size=(1,),
            device=self.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(sampled, src=0)
        return int(sampled.item())

    def _sample_online_warp_supervised_window_index(
        self,
        *,
        num_windows: int,
        training_batch,
    ) -> int:
        num_windows = max(1, int(num_windows))
        if num_windows <= 1:
            return 0

        policy = str(
            getattr(self.training_args, "online_warp_supervised_window_policy",
                    "final") or "final").strip().lower()
        if policy in ("final", "last"):
            return int(num_windows - 1)
        if policy == "first":
            return 0
        if policy != "random":
            raise ValueError(
                "Unsupported online_warp_supervised_window_policy="
                f"{policy!r}. Expected one of: final, first, random")

        candidate_indices = list(range(int(num_windows)))
        weights_str = str(
            getattr(self.training_args, "online_warp_supervised_window_weights",
                    "") or "").strip()
        if weights_str:
            try:
                weights = [float(x.strip()) for x in weights_str.split(",")]
            except Exception as exc:
                raise ValueError(
                    "Failed to parse online_warp_supervised_window_weights="
                    f"{weights_str!r}") from exc
            if len(weights) != len(candidate_indices):
                raise ValueError(
                    "online_warp_supervised_window_weights length mismatch: "
                    f"weights={weights} candidates={candidate_indices}")
            if any((not math.isfinite(w)) or w < 0.0 for w in weights):
                raise ValueError(
                    "online_warp_supervised_window_weights must be finite and >= 0: "
                    f"{weights}")
            weight_sum = float(sum(weights))
            if weight_sum <= 0.0:
                raise ValueError(
                    "online_warp_supervised_window_weights must sum to > 0, got "
                    f"{weights}")
            probs = [float(w) / weight_sum for w in weights]
        else:
            probs = [1.0 / float(len(candidate_indices))
                     for _ in candidate_indices]

        if not getattr(self, "_logged_online_warp_supervision_policy", False):
            logger.info(
                "online_warp supervision policy: policy=%s candidates=%s probs=%s",
                policy,
                [int(i) + 1 for i in candidate_indices],
                [round(float(x), 4) for x in probs],
            )
            self._logged_online_warp_supervision_policy = True

        sampled = torch.empty(1, device=self.device, dtype=torch.long)
        if (not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0):
            infos = getattr(training_batch, "infos", None) or []
            info0 = infos[0] if infos else {}
            if isinstance(info0, dict):
                sample_key = str(info0.get("id") or info0.get("file_name")
                                 or info0.get("caption") or "")
            else:
                sample_key = str(info0)
            seed_material = (
                "online_warp_supervised_window",
                int(getattr(self, "current_trainstep", 0)),
                sample_key,
                int(num_windows),
                weights_str or "uniform",
            )
            digest = hashlib.sha256(
                "|".join(str(x) for x in seed_material).encode(
                    "utf-8")).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            sampled_idx = int(rng.choices(candidate_indices, weights=probs, k=1)[0])
            sampled.fill_(int(sampled_idx))
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(sampled, src=0)
        return int(sampled.item())

    def _init_additional_simulation_caches(self, batch_size: int,
                                           dtype: torch.dtype,
                                           device: torch.device,
                                           max_num_frames: int) -> None:
        if getattr(self, "controlnet", None) is None:
            self.controlnet_kv_cache1 = None
            self.controlnet_crossattn_cache = None
            return

        num_transformer_blocks = len(self.controlnet.blocks)
        kv_cache_size = max_num_frames * self.frame_seq_length
        num_attention_heads = getattr(self.controlnet, "num_attention_heads",
                                      None)
        attention_head_dim = getattr(self.controlnet, "attention_head_dim",
                                     None)
        text_len = getattr(self.controlnet, "text_len", None)

        kv_cache = []
        for _ in range(num_transformer_blocks):
            kv_cache.append({
                "k":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device)
            })

        crossattn_cache = []
        for _ in range(num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([batch_size, text_len, num_attention_heads, attention_head_dim],
                            dtype=dtype,
                            device=device),
                "is_init":
                False
            })

        self.controlnet_kv_cache1 = kv_cache
        self.controlnet_crossattn_cache = crossattn_cache

    def _reset_additional_simulation_caches(self) -> None:
        if getattr(self, "controlnet_kv_cache1", None) is not None and getattr(
                self, "controlnet_crossattn_cache", None) is not None:
            self._reset_simulation_caches(self.controlnet_kv_cache1,
                                          self.controlnet_crossattn_cache)

    def _simulation_model_forward_raw(
        self,
        *,
        model,
        training_batch_temp,
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        start_frame: int,
        current_num_frames: int,
    ) -> torch.Tensor:
        control_latent = getattr(training_batch_temp, "control_latent", None)
        first_frame_latent = getattr(training_batch_temp, "first_frame_latent",
                                     None)
        if getattr(self, "controlnet", None) is None or control_latent is None:
            return super()._simulation_model_forward_raw(
                model=model,
                training_batch_temp=training_batch_temp,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=current_start_frame,
                start_frame=start_frame,
                current_num_frames=current_num_frames,
            )

        assert self.controlnet_kv_cache1 is not None
        assert self.controlnet_crossattn_cache is not None

        control_frame_offset = int(
            getattr(training_batch_temp, "_control_latent_frame_offset", 0))
        local_start_frame = int(start_frame) - int(control_frame_offset)
        local_end_frame = int(local_start_frame + current_num_frames)
        if local_start_frame < 0 or local_end_frame > int(
                control_latent.shape[2]):
            raise ValueError(
                "control_latent window slice is out of range for shared-cache rollout: "
                f"start_frame={int(start_frame)} offset={int(control_frame_offset)} "
                f"local_start={int(local_start_frame)} local_end={int(local_end_frame)} "
                f"control_t={int(control_latent.shape[2])}")
        control_chunk = control_latent[:, :, local_start_frame:local_end_frame]

        hidden_states = training_batch_temp.input_kwargs["hidden_states"]
        model_timestep = training_batch_temp.input_kwargs.get("timestep", 0)
        # TI2V: enforce the first latent frame of the current window, while the
        # cache position / rotary phase still follows the global latent offset.
        if local_start_frame == 0 and first_frame_latent is not None:
            # first_frame_latent: BFCHW -> BCFHW
            img = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
            if hidden_states.shape[2] >= 1:
                hidden_states = torch.cat([img, hidden_states[:, :, 1:]], dim=2)
                model_timestep = _with_first_frame_timestep_zero(
                    model_timestep,
                    batch_size=int(hidden_states.shape[0]),
                    num_frames=int(hidden_states.shape[2]),
                )

        timestep = training_batch_temp.input_kwargs.get("timestep", 0)
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                context_timestep = 0
            else:
                context_timestep = int(timestep.reshape(-1)[0].item())
        else:
            context_timestep = int(timestep)

        num_channels_latents = getattr(model, "num_channels_latents",
                                       control_chunk.shape[1] // 3)
        with set_forward_context(current_timestep=context_timestep,
                                 attn_metadata=None):
            control_res = self.controlnet(
                hidden_states=hidden_states,
                encoder_hidden_states=training_batch_temp.input_kwargs[
                    "encoder_hidden_states"],
                timestep=model_timestep,
                encoder_hidden_states_image=training_batch_temp.input_kwargs.get(
                    "encoder_hidden_states_image"),
                **_build_controlnet_kwargs(self.controlnet, control_chunk,
                                           num_channels_latents),
                kv_cache=self.controlnet_kv_cache1,
                crossattn_cache=self.controlnet_crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                start_frame=start_frame,
            )

            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=training_batch_temp.input_kwargs[
                    "encoder_hidden_states"],
                timestep=model_timestep,
                encoder_hidden_states_image=training_batch_temp.input_kwargs.get(
                    "encoder_hidden_states_image"),
                block_controlnet_hidden_states=control_res,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                start_frame=start_frame,
            )

    def _simulation_postprocess_chunk_output(self,
                                             denoised_pred: torch.Tensor,
                                             *,
                                             training_batch,
                                             current_start_frame: int,
                                             current_num_frames: int
                                             ) -> torch.Tensor:
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        if current_start_frame == 0 and first_frame_latent is not None:
            # denoised_pred: BFCHW, enforce frame 0
            denoised_pred = denoised_pred.clone()
            denoised_pred[:, :1] = first_frame_latent
        return denoised_pred

    def _predict_noise_with_controlnet(self, *, transformer, controlnet,
                                       input_kwargs: dict[str, Any],
                                       control_latent: torch.Tensor | None,
                                       first_frame_latent: torch.Tensor
                                       | None = None):
        conditioned_input_kwargs = dict(input_kwargs)
        conditioned_hidden_states = _apply_first_frame_latent(
            conditioned_input_kwargs["hidden_states"], first_frame_latent)
        conditioned_input_kwargs["hidden_states"] = conditioned_hidden_states
        if first_frame_latent is not None:
            conditioned_input_kwargs["timestep"] = _with_first_frame_timestep_zero(
                conditioned_input_kwargs.get("timestep", 0),
                batch_size=int(conditioned_hidden_states.shape[0]),
                num_frames=int(conditioned_hidden_states.shape[2]),
            )
            conditioned_input_kwargs["timestep"] = _maybe_expand_timestep_to_token_sequence(
                conditioned_input_kwargs["timestep"],
                hidden_states=conditioned_hidden_states,
                model=transformer,
            )
        if controlnet is None or control_latent is None:
            return transformer(**conditioned_input_kwargs)
        num_channels_latents = getattr(transformer, "num_channels_latents",
                                       control_latent.shape[1] // 3)
        control_res = controlnet(
            hidden_states=conditioned_hidden_states,
            encoder_hidden_states=conditioned_input_kwargs[
                "encoder_hidden_states"],
            timestep=conditioned_input_kwargs["timestep"],
            encoder_hidden_states_image=conditioned_input_kwargs.get(
                "encoder_hidden_states_image"),
            **_build_controlnet_kwargs(controlnet, control_latent,
                                       num_channels_latents),
        )
        return transformer(
            **conditioned_input_kwargs, block_controlnet_hidden_states=control_res)

    def _score_context_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.reshape(-1)[0].item())

    def _build_score_attn_metadata(self, training_batch, timestep: torch.Tensor):
        if getattr(training_batch, "raw_latent_shape", None) is None:
            return getattr(training_batch, "attn_metadata", None)
        original_timesteps = getattr(training_batch, "timesteps", None)
        original_attn_metadata = getattr(training_batch, "attn_metadata", None)
        try:
            training_batch.timesteps = timestep
            self._build_attention_metadata(training_batch)
            return training_batch.attn_metadata
        finally:
            training_batch.timesteps = original_timesteps
            training_batch.attn_metadata = original_attn_metadata

    def _generator_forward(self, training_batch: TrainingBatch) -> torch.Tensor:
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        latents = training_batch.latents
        batch_size, num_frames = latents.shape[:2]

        index = torch.randint(0,
                              len(self.denoising_step_list), [1],
                              device=self.device,
                              dtype=torch.long)
        timestep_value = self.denoising_step_list[index]
        timestep = (torch.ones([batch_size, num_frames],
                               device=self.device,
                               dtype=torch.int64) * timestep_value)
        training_batch.dmd_latent_vis_dict["generator_timestep"] = (
            timestep_value.detach().clone().to(device=self.device,
                                               dtype=torch.float32))

        noise = torch.randn_like(latents)
        noisy_latent = self.noise_scheduler.add_noise(
            latents.flatten(0, 1), noise.flatten(0, 1),
            timestep.flatten(0, 1)).unflatten(0, latents.shape[:2])
        noisy_latent = _apply_first_frame_latent_bfchw(noisy_latent,
                                                       first_frame_latent)
        training_batch = self._build_distill_input_kwargs(
            noisy_latent, timestep, training_batch.conditional_dict,
            training_batch)
        score_attn_metadata = self._build_score_attn_metadata(
            training_batch, timestep)
        score_context_timestep = self._score_context_timestep(timestep)
        current_model = self._select_generator_model_for_timestep(
            float(timestep_value.float().reshape(-1)[0].item()))
        with set_forward_context(current_timestep=score_context_timestep,
                                 attn_metadata=score_attn_metadata):
            pred_noise = self._predict_noise_with_controlnet(
                transformer=current_model,
                controlnet=getattr(self, "controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
                first_frame_latent=first_frame_latent,
            ).permute(0, 2, 1, 3, 4)

        pred_video = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latent.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
            scheduler=self.noise_scheduler).unflatten(0, pred_noise.shape[:2])
        return _apply_first_frame_latent_bfchw(pred_video, first_frame_latent)

    def _generator_multi_step_full_sequence_bidirectional_forward(
            self,
            training_batch: TrainingBatch,
            return_sim_steps: bool = False):
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        latents = training_batch.latents
        batch_size, num_frames = latents.shape[:2]

        forced_exit_index = getattr(self, "_forced_exit_index", None)
        if forced_exit_index is None:
            exit_idx = self._sample_shared_exit_index(device=self.device)
        else:
            exit_idx = int(forced_exit_index)
        exit_idx = max(0, min(int(exit_idx), len(self.denoising_step_list) - 1))

        current_latents = torch.randn_like(latents)
        current_latents = _apply_first_frame_latent_bfchw(
            current_latents, first_frame_latent)
        pred_video = current_latents

        for step_idx, current_timestep in enumerate(self.denoising_step_list):
            timestep = (torch.ones([batch_size, num_frames],
                                   device=self.device,
                                   dtype=torch.int64) * current_timestep)
            current_model = self._select_generator_model_for_timestep(
                float(
                    torch.as_tensor(current_timestep).float().reshape(-1)[0].
                    item()))
            score_attn_metadata = self._build_score_attn_metadata(
                training_batch, timestep)
            score_context_timestep = self._score_context_timestep(timestep)

            def _predict_with_context() -> torch.Tensor:
                batch_with_input = self._build_distill_input_kwargs(
                    current_latents, timestep, training_batch.conditional_dict,
                    training_batch)
                with set_forward_context(current_timestep=score_context_timestep,
                                         attn_metadata=score_attn_metadata):
                    return self._predict_noise_with_controlnet(
                        transformer=current_model,
                        controlnet=getattr(self, "controlnet", None),
                        input_kwargs=batch_with_input.input_kwargs,
                        control_latent=control_latent,
                        first_frame_latent=first_frame_latent,
                    ).permute(0, 2, 1, 3, 4)

            if step_idx < exit_idx:
                with torch.no_grad():
                    pred_noise = _predict_with_context()
            else:
                pred_noise = _predict_with_context()

            pred_video = pred_noise_to_pred_video(
                pred_noise=pred_noise.flatten(0, 1),
                noise_input_latent=current_latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
                scheduler=self.noise_scheduler).unflatten(
                    0, pred_noise.shape[:2])
            pred_video = _apply_first_frame_latent_bfchw(pred_video,
                                                         first_frame_latent)

            if step_idx == exit_idx:
                break

            next_timestep = self.denoising_step_list[step_idx + 1]
            next_timestep_tensor = next_timestep * torch.ones(
                [batch_size * num_frames],
                device=self.device,
                dtype=torch.long)
            current_latents = self.noise_scheduler.add_noise(
                pred_video.flatten(0, 1),
                torch.randn_like(pred_video.flatten(0, 1)),
                next_timestep_tensor).unflatten(0, pred_video.shape[:2])
            current_latents = _apply_first_frame_latent_bfchw(
                current_latents, first_frame_latent)

        training_batch.dmd_latent_vis_dict["generator_timestep"] = (
            self.denoising_step_list[exit_idx].detach().clone().to(
                device=self.device, dtype=torch.float32))

        scheduler_timesteps = self.noise_scheduler.timesteps.to(self.device)
        from_t = self.denoising_step_list[exit_idx]
        denoised_timestep_from = int(self.num_train_timestep - torch.argmin(
            (scheduler_timesteps - from_t).abs()).item())
        if exit_idx == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
        else:
            to_t = self.denoising_step_list[exit_idx + 1]
            denoised_timestep_to = int(self.num_train_timestep -
                                       torch.argmin(
                                           (scheduler_timesteps -
                                            to_t).abs()).item())

        if return_sim_steps:
            return pred_video, denoised_timestep_from, denoised_timestep_to, exit_idx + 1
        return pred_video

    def _generator_multi_step_window_forward_shared_cache(
            self,
            training_batch: TrainingBatch,
            *,
            cache_position_offset_frames: int,
            return_sim_steps: bool = False):
        if self.kv_cache1 is None or self.crossattn_cache is None:
            raise RuntimeError(
                "Shared-cache online_warp rollout requires initialized generator caches."
            )

        latents = training_batch.latents
        dtype = latents.dtype
        batch_size = latents.shape[0]
        initial_latent = getattr(training_batch, "image_latent", None)

        num_training_frames = getattr(self.training_args, "num_latent_t", 21)
        min_num_frames = 20 if self.independent_first_frame else 21
        max_num_frames = (num_training_frames - 1
                          if self.independent_first_frame else
                          num_training_frames)

        sampled_total_frames = torch.randint(min_num_frames,
                                             max_num_frames + 1, (1, ),
                                             device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(sampled_total_frames, src=0)
        num_generated_frames = int(sampled_total_frames.item())
        if self.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        rollout_num_frames = max(0, num_generated_frames - num_input_frames)

        noise_shape = [batch_size, rollout_num_frames, *self.video_latent_shape[2:]]
        noise = torch.randn(noise_shape, device=self.device, dtype=dtype)
        if self.sp_world_size > 1:
            noise = rearrange(noise,
                              "b (n t) c h w -> b n t c h w",
                              n=self.sp_world_size).contiguous()
            noise = noise[:, self.rank_in_sp_group, :, :, :, :]

        batch_size, num_frames, num_channels, height, width = noise.shape
        remainder_frames = num_frames
        all_num_frames: list[int] = []
        if self.independent_first_frame and initial_latent is None:
            all_num_frames.append(1)
            remainder_frames = max(0, remainder_frames - 1)
        full_blocks, tail_frames = divmod(remainder_frames,
                                          self.num_frame_per_block)
        all_num_frames.extend([self.num_frame_per_block] * full_blocks)
        if tail_frames > 0:
            all_num_frames.append(tail_frames)

        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype)

        local_current_start_frame = 0
        frame_offset = int(cache_position_offset_frames)
        previous_control_frame_offset = getattr(training_batch,
                                               "_control_latent_frame_offset",
                                               None)
        training_batch._control_latent_frame_offset = frame_offset
        try:
            if initial_latent is not None:
                timestep = torch.ones(
                    [batch_size, 1], device=noise.device,
                    dtype=torch.int64) * 0
                output[:, :1] = initial_latent
                with torch.no_grad():
                    training_batch_temp = self._build_distill_input_kwargs(
                        initial_latent, timestep * 0,
                        training_batch.conditional_dict, training_batch)
                    current_model = self._select_generator_model_for_timestep(
                        0.0)
                    absolute_start_frame = int(frame_offset +
                                               local_current_start_frame)
                    _ = self._simulation_model_forward_raw(
                        model=current_model,
                        training_batch_temp=training_batch_temp,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start_frame=absolute_start_frame,
                        start_frame=absolute_start_frame,
                        current_num_frames=1,
                    )
                local_current_start_frame += 1

            num_denoising_steps = len(self.denoising_step_list)
            exit_flags = self.generate_and_sync_list(len(all_num_frames),
                                                     num_denoising_steps,
                                                     device=noise.device)
            grad_last_n_frames = int(
                getattr(self.training_args, "gradient_mask_last_n_frames", 21)
                or 21)
            start_gradient_frame_index = max(0,
                                             num_output_frames -
                                             grad_last_n_frames)

            for block_index, current_num_frames in enumerate(all_num_frames):
                noisy_input = noise[:, local_current_start_frame -
                                    num_input_frames:local_current_start_frame +
                                    current_num_frames - num_input_frames]
                absolute_start_frame = int(frame_offset +
                                           local_current_start_frame)

                for index, current_timestep in enumerate(self.denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])

                    timestep = torch.ones([batch_size, current_num_frames],
                                          device=noise.device,
                                          dtype=torch.int64) * current_timestep
                    current_model = self._select_generator_model_for_timestep(
                        float(current_timestep))

                    if not exit_flag:
                        with torch.no_grad():
                            training_batch_temp = self._build_distill_input_kwargs(
                                noisy_input, timestep,
                                training_batch.conditional_dict, training_batch)
                            pred_flow = self._simulation_predict_flow(
                                model=current_model,
                                training_batch_temp=training_batch_temp,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start_frame=absolute_start_frame,
                                start_frame=absolute_start_frame,
                                current_num_frames=current_num_frames,
                            )
                            denoised_pred = pred_noise_to_pred_video(
                                pred_noise=pred_flow.flatten(0, 1),
                                noise_input_latent=noisy_input.flatten(0, 1),
                                timestep=timestep,
                                scheduler=self.noise_scheduler).unflatten(
                                    0, pred_flow.shape[:2])
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.noise_scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(
                                    denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames],
                                    device=noise.device,
                                    dtype=torch.long)).unflatten(
                                        0, denoised_pred.shape[:2])
                    else:
                        if local_current_start_frame < start_gradient_frame_index:
                            with torch.no_grad():
                                training_batch_temp = self._build_distill_input_kwargs(
                                    noisy_input, timestep,
                                    training_batch.conditional_dict,
                                    training_batch)
                                pred_flow = self._simulation_predict_flow(
                                    model=current_model,
                                    training_batch_temp=training_batch_temp,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start_frame=absolute_start_frame,
                                    start_frame=absolute_start_frame,
                                    current_num_frames=current_num_frames,
                                )
                        else:
                            training_batch_temp = self._build_distill_input_kwargs(
                                noisy_input, timestep,
                                training_batch.conditional_dict, training_batch)
                            pred_flow = self._simulation_predict_flow(
                                model=current_model,
                                training_batch_temp=training_batch_temp,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start_frame=absolute_start_frame,
                                start_frame=absolute_start_frame,
                                current_num_frames=current_num_frames,
                            )

                        denoised_pred = pred_noise_to_pred_video(
                            pred_noise=pred_flow.flatten(0, 1),
                            noise_input_latent=noisy_input.flatten(0, 1),
                            timestep=timestep,
                            scheduler=self.noise_scheduler).unflatten(
                                0, pred_flow.shape[:2])
                        break

                denoised_pred = self._simulation_postprocess_chunk_output(
                    denoised_pred,
                    training_batch=training_batch,
                    current_start_frame=local_current_start_frame,
                    current_num_frames=current_num_frames,
                )
                output[:, local_current_start_frame:local_current_start_frame +
                       current_num_frames] = denoised_pred

                context_timestep = torch.ones_like(timestep) * self.context_noise
                if self.rollout_add_context_noise:
                    context_input = self.noise_scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        context_timestep).unflatten(0, denoised_pred.shape[:2])
                else:
                    context_input = denoised_pred

                with torch.no_grad():
                    training_batch_temp = self._build_distill_input_kwargs(
                        context_input, context_timestep,
                        training_batch.conditional_dict, training_batch)
                    current_model = self._select_generator_model_for_timestep(
                        float(self.context_noise))
                    _ = self._simulation_model_forward_raw(
                        model=current_model,
                        training_batch_temp=training_batch_temp,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start_frame=absolute_start_frame,
                        start_frame=absolute_start_frame,
                        current_num_frames=current_num_frames,
                    )

                local_current_start_frame += current_num_frames

            pred_image_or_video = output
            keep_initial_latent_in_output = (
                num_input_frames > 0
                and getattr(training_batch, "first_frame_latent", None) is not None
            )
            if num_input_frames > 0 and not keep_initial_latent_in_output:
                pred_image_or_video = output[:, num_input_frames:]

            gradient_mask = None
            if pred_image_or_video.shape[1] > grad_last_n_frames:
                with torch.no_grad():
                    keep_last = max(0, grad_last_n_frames - 1)
                    latent_to_decode = (
                        pred_image_or_video[:, :-keep_last, ...]
                        if keep_last > 0 else pred_image_or_video)
                    pixels = self._decode_dit_latents_to_pixels(
                        latent_to_decode, dtype)
                    frame = pixels[:, :, -1:, :, :].to(dtype)
                    image_latent = self._encode_pixels_to_dit_latents(
                        frame, dtype)
                suffix = (
                    pred_image_or_video[:, -keep_last:, ...]
                    if keep_last > 0 else pred_image_or_video[:, :0, ...])
                pred_image_or_video_last_n = torch.cat([image_latent, suffix],
                                                       dim=1)
            else:
                pred_image_or_video_last_n = pred_image_or_video

            if num_generated_frames != min_num_frames:
                gradient_mask = torch.ones_like(pred_image_or_video_last_n,
                                                dtype=torch.bool)
                if self.independent_first_frame:
                    gradient_mask[:, :1] = False
                else:
                    gradient_mask[:, :self.num_frame_per_block] = False

            final_output = pred_image_or_video_last_n.to(dtype)
            if gradient_mask is not None:
                final_output = torch.where(
                    gradient_mask,
                    pred_image_or_video_last_n,
                    pred_image_or_video_last_n.detach(),
                )

            training_batch.dmd_latent_vis_dict["generator_timestep"] = (
                self.denoising_step_list[exit_flags[0]].detach().clone().to(
                    device=self.device, dtype=torch.float32))
            if gradient_mask is not None:
                training_batch.dmd_latent_vis_dict["gradient_mask"] = (
                    gradient_mask.float())
                training_batch.dmd_latent_vis_dict["num_generated_frames"] = (
                    torch.tensor(num_generated_frames,
                                 dtype=torch.float32,
                                 device=self.device))
                training_batch.dmd_latent_vis_dict["min_num_frames"] = (
                    torch.tensor(min_num_frames,
                                 dtype=torch.float32,
                                 device=self.device))

            output_tensor = (final_output
                             if gradient_mask is not None else pred_image_or_video)
            denoised_timestep_from: int | None = None
            denoised_timestep_to: int | None = None
            if self.same_step_across_blocks and len(exit_flags) > 0:
                exit_idx = int(exit_flags[0])
                scheduler_timesteps = self.noise_scheduler.timesteps.to(
                    self.device)
                from_t = self.denoising_step_list[exit_idx]
                denoised_timestep_from = int(self.num_train_timestep -
                                             torch.argmin(
                                                 (scheduler_timesteps -
                                                  from_t).abs()).item())
                if exit_idx == len(self.denoising_step_list) - 1:
                    denoised_timestep_to = 0
                else:
                    to_t = self.denoising_step_list[exit_idx + 1]
                    denoised_timestep_to = int(self.num_train_timestep -
                                               torch.argmin(
                                                   (scheduler_timesteps -
                                                    to_t).abs()).item())

            if return_sim_steps:
                return output_tensor, denoised_timestep_from, denoised_timestep_to, (
                    int(exit_flags[0]) + 1 if len(exit_flags) > 0 else 0)
            return output_tensor
        finally:
            if previous_control_frame_offset is None:
                if hasattr(training_batch, "_control_latent_frame_offset"):
                    delattr(training_batch, "_control_latent_frame_offset")
            else:
                training_batch._control_latent_frame_offset = (
                    previous_control_frame_offset)

    def _generator_multi_step_simulation_forward(
            self,
            training_batch: TrainingBatch,
            return_sim_steps: bool = False):
        student_attention_mode = _normalize_student_attention_mode(
            getattr(self.training_args, "student_attention_mode", "causal"))
        if student_attention_mode == "bidirectional":
            return self._generator_multi_step_full_sequence_bidirectional_forward(
                training_batch, return_sim_steps=return_sim_steps)
        if not bool(getattr(self.training_args, "online_warp_training", False)):
            return super()._generator_multi_step_simulation_forward(
                training_batch, return_sim_steps=return_sim_steps)

        infos = getattr(training_batch, "infos", None)
        if not infos:
            raise ValueError(
                "online_warp_training requires training_batch.infos to reconstruct raw clips"
            )
        batch_size = int(training_batch.latents.shape[0])
        if batch_size != 1:
            raise ValueError(
                "online_warp_training currently supports train_batch_size=1 per rank, "
                f"got batch_size={batch_size}"
            )

        self._lazy_init_online_warp_training_state()
        infer_base = getattr(self, "_online_warp_infer_base")
        mcprep = getattr(self, "_online_warp_mcprep")

        overlap_frames = int(
            getattr(self.training_args, "online_warp_overlap_frames", 1))
        if overlap_frames != 1:
            raise ValueError(
                "online_warp_training currently supports overlap_frames=1 so "
                "later windows can keep using the single-frame first_frame_latent interface. "
                f"Got overlap_frames={int(overlap_frames)}")

        info_raw = infos[0]
        info = (info_raw if isinstance(info_raw, dict) else shallow_asdict(info_raw))
        bootstrap_clip = self._build_online_warp_clip_from_info(
            info, int(getattr(self.training_args, "online_warp_window_frames", 81)))
        num_windows = self._sample_online_warp_num_windows(
            int(bootstrap_clip["max_num_windows"]))
        if num_windows <= 0:
            raise ValueError(
                "online_warp_training sampled a non-positive rollout length. "
                f"record_id={bootstrap_clip['record_id']} max_num_windows={int(bootstrap_clip['max_num_windows'])}"
            )

        window_frames = int(bootstrap_clip["window_frames"])
        stride = int(bootstrap_clip["stride_frames"])
        total_required_frames = int(window_frames +
                                    max(0, num_windows - 1) * stride)
        clip = self._build_online_warp_clip_from_info(info,
                                                      total_required_frames)

        if getattr(training_batch, "dmd_latent_vis_dict", None) is None:
            training_batch.dmd_latent_vis_dict = {}

        dtype = training_batch.latents.dtype
        target_c = int(training_batch.latents.shape[2])
        target_h = int(self.training_args.num_height)
        target_w = int(self.training_args.num_width)
        normalize_condition_latents = bool(
            getattr(self.training_args, "normalize_condition_latents", False))

        global_first_frame_latent = getattr(training_batch,
                                            "global_first_frame_latent", None)
        if global_first_frame_latent is None:
            global_first_frame_latent = getattr(training_batch,
                                                "first_frame_latent", None)
        if global_first_frame_latent is None:
            raise ValueError(
                "online_warp_training requires first_frame_latent to be loaded from parquet"
            )
        global_first_frame_latent = global_first_frame_latent.to(
            device=self.device, dtype=dtype)
        bootstrap_control_latent = getattr(training_batch,
                                           "bootstrap_control_latent", None)
        if bootstrap_control_latent is not None:
            bootstrap_control_latent = bootstrap_control_latent.to(
                device=self.device, dtype=dtype)
        use_bootstrap_control_first_window = bool(
            getattr(
                self.training_args,
                "online_warp_use_bootstrap_control_for_first_window",
                False,
            )) and bootstrap_control_latent is not None
        full_depth_latent = getattr(training_batch, "full_depth_latent", None)
        if full_depth_latent is not None:
            full_depth_latent = full_depth_latent.to(device=self.device,
                                                     dtype=dtype)
        full_normal_latent = getattr(training_batch, "full_normal_latent", None)
        if full_normal_latent is not None:
            full_normal_latent = full_normal_latent.to(device=self.device,
                                                       dtype=dtype)
        temporal_ratio = int(
            self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
        latent_window_t = int(training_batch.latents.shape[1])
        overlap_latent_t = int(
            infer_base._latent_frames_from_video_frames(int(overlap_frames)))
        latent_stride_t = int(
            infer_base._latent_frames_from_video_frames(int(stride)))
        total_latent_t = int(latent_window_t + max(0, num_windows - 1) *
                             latent_stride_t)

        global_first_rgb = None
        if not use_bootstrap_control_first_window:
            global_first_rgb = mcprep._load_rgb_frame(
                clip["rgb_paths"][0], int(target_h), int(target_w))
        if (bootstrap_control_latent is not None
                and not use_bootstrap_control_first_window
                and not getattr(self, "_logged_online_warp_ignore_bootstrap", False)):
            logger.info(
                "online_warp_training detected cached first-window control_latent in parquet but will ignore it; "
                "window 0 control is rebuilt from raw first-frame warp + depth/normal."
            )
            self._logged_online_warp_ignore_bootstrap = True
        depth_path_by_id = {
            int(fid): p
            for fid, p in zip(clip["frame_ids"], clip["depth_paths"])
        }
        history_rgbs_u8: dict[int, np.ndarray] = {}
        processed_frame_ids: set[int] = set()
        visibility_map = mcprep._OnlineWarpVisibilityMap(
            voxel_size=float(
                getattr(self.training_args, "online_warp_selection_voxel_size",
                        0.1)))

        carry_prefix_tchw: torch.Tensor | None = None
        warped_masked_rgb_next: torch.Tensor | None = None
        warped_mask_next: torch.Tensor | None = None
        final_rollout = None
        cache_global_start_latent = 0

        self.kv_cache1, self.crossattn_cache = self._initialize_simulation_caches(
            batch_size,
            dtype,
            self.device,
            max_num_frames=int(total_latent_t),
        )
        self._init_additional_simulation_caches(batch_size, dtype, self.device,
                                                int(total_latent_t))
        if not getattr(self, "_logged_online_warp_shared_cache", False):
            logger.info(
                "online_warp_training using shared cross-window caches: latent_window_t=%s latent_stride_t=%s total_latent_t=%s",
                latent_window_t,
                latent_stride_t,
                total_latent_t,
            )
            self._logged_online_warp_shared_cache = True
        supervised_window_idx = self._sample_online_warp_supervised_window_index(
            num_windows=int(num_windows), training_batch=training_batch)
        if int(num_windows) > 1:
            logger.info(
                "online_warp supervised window for this step: index=%s (1-based=%s) out of num_windows=%s. "
                "Only windows before the supervised window run in no-grad warmup mode; later windows are skipped.",
                int(supervised_window_idx),
                int(supervised_window_idx) + 1,
                int(num_windows),
            )

        for win_idx in range(int(num_windows)):
            start_pos = int(win_idx * stride)
            valid_window = min(int(window_frames),
                               int(total_required_frames - start_pos))
            if valid_window <= 0:
                break
            end_pos_valid = int(start_pos + valid_window)

            if win_idx == 0:
                window_first_frame_latent = global_first_frame_latent
            else:
                if carry_prefix_tchw is None or int(carry_prefix_tchw.shape[0]) != 1:
                    raise RuntimeError(
                        "online_warp_training expected a single-frame carry prefix for later windows"
                    )
                with torch.no_grad():
                    first_frame_latent_bcfhw = infer_base._encode_first_frame_latent(
                        vae=self.vae,
                        first_rgb_chw=carry_prefix_tchw[0],
                        target_c=int(target_c),
                        inference_device=self.device,
                        compute_dtype=dtype,
                    ).to(device=self.device, dtype=dtype)
                window_first_frame_latent = first_frame_latent_bcfhw.permute(
                    0, 2, 1, 3, 4).contiguous()
                window_first_frame_latent = _normalize_first_frame_latent(
                    window_first_frame_latent,
                    self.vae,
                    enabled=normalize_condition_latents,
                )

            if win_idx == 0 and use_bootstrap_control_first_window:
                control_latent = bootstrap_control_latent
            else:
                normal_window_paths = None
                if clip["normal_paths"] is not None:
                    normal_window_paths = _pad_paths(
                        clip["normal_paths"][start_pos:end_pos_valid],
                        int(window_frames))

                if win_idx == 0:
                    target_ids = [
                        int(fid) for fid in clip["frame_ids"][start_pos +
                                                              1:end_pos_valid]
                    ]
                    if target_ids:
                        if global_first_rgb is None:
                            raise RuntimeError(
                                "Missing global_first_rgb for uncached online-warp first window"
                            )
                        first_rgb_u8 = mcprep._chw_float_to_u8(global_first_rgb)
                        warped_masked_rgb_valid, warped_mask_valid = mcprep._warp_maskrgb_from_keyframes_md_aligned_memory(
                            keyframe_rgbs_u8=[first_rgb_u8],
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
                            (0, 3, int(target_h), int(target_w)),
                            dtype=torch.float32)
                        warped_mask_valid = torch.empty(
                            (0, 1, int(target_h), int(target_w)),
                            dtype=torch.float32)
                    warped_masked_rgb = mcprep._pad_tchw(
                        warped_masked_rgb_valid, max(int(window_frames) - 1, 1))
                    warped_mask = mcprep._pad_tchw(
                        warped_mask_valid, max(int(window_frames) - 1, 1))
                    mask_tchw = torch.cat([
                        torch.ones((1, 1, int(target_h), int(target_w)),
                                   dtype=torch.float32),
                        warped_mask[:max(int(window_frames) - 1, 0)],
                    ],
                                          dim=0)
                    masked_rgb_tchw = torch.cat([
                        global_first_rgb.unsqueeze(0),
                        warped_masked_rgb[:max(int(window_frames) - 1, 0)],
                    ],
                                                dim=0)
                else:
                    if (carry_prefix_tchw is None or warped_masked_rgb_next is None
                            or warped_mask_next is None):
                        raise RuntimeError(
                            "Missing online-warp carry-over state for window > 0")
                    mask_tchw = torch.cat([
                        torch.ones((int(overlap_frames), 1, int(target_h),
                                    int(target_w)),
                                   dtype=torch.float32),
                        warped_mask_next,
                    ],
                                          dim=0)
                    masked_rgb_tchw = torch.cat(
                        [carry_prefix_tchw, warped_masked_rgb_next], dim=0)
                if full_depth_latent is not None:
                    depth_branch_latent = _slice_cached_branch_latent_window(
                        full_depth_latent,
                        start_frame=int(start_pos),
                        temporal_ratio=int(temporal_ratio),
                        target_t=int(latent_window_t),
                        name="depth_latent",
                    )
                else:
                    depth_window_paths = _pad_paths(
                        clip["depth_paths"][start_pos:end_pos_valid],
                        int(window_frames))
                    depth_tchw = mcprep._load_depth_sequence(
                        depth_window_paths,
                        int(target_h),
                        int(target_w),
                        pmin=5.0,
                        pmax=95.0,
                        invert_depth=False,
                        normalization_mode=str(
                            getattr(
                                self.training_args,
                                "online_warp_depth_normalization_mode",
                                "md_align",
                            )),
                    )
                    depth_branch_latent = _encode_control_branch_latent_with_infer_base(
                        infer_base=infer_base,
                        vae=self.vae,
                        branch_tchw=depth_tchw,
                        normalize=False,
                        target_c=int(target_c),
                        name="depth_lat",
                        inference_device=self.device,
                        compute_dtype=dtype,
                    )
                    depth_branch_latent = _normalize_control_latent(
                        depth_branch_latent,
                        self.vae,
                        int(target_c),
                        enabled=normalize_condition_latents,
                    )

                normal_branch_latent = None
                if full_normal_latent is not None:
                    normal_branch_latent = _slice_cached_branch_latent_window(
                        full_normal_latent,
                        start_frame=int(start_pos),
                        temporal_ratio=int(temporal_ratio),
                        target_t=int(latent_window_t),
                        name="normal_latent",
                    )
                elif normal_window_paths is not None:
                    normal_tchw = torch.stack([
                        mcprep._load_normal_frame(p, int(target_h), int(target_w))
                        for p in normal_window_paths
                    ],
                                              dim=0)
                    normal_branch_latent = _encode_control_branch_latent_with_infer_base(
                        infer_base=infer_base,
                        vae=self.vae,
                        branch_tchw=normal_tchw,
                        normalize=False,
                        target_c=int(target_c),
                        name="normal_lat",
                        inference_device=self.device,
                        compute_dtype=dtype,
                    )
                    normal_branch_latent = _normalize_control_latent(
                        normal_branch_latent,
                        self.vae,
                        int(target_c),
                        enabled=normalize_condition_latents,
                    )

                masked_branch_latent = _encode_control_branch_latent_with_infer_base(
                    infer_base=infer_base,
                    vae=self.vae,
                    branch_tchw=masked_rgb_tchw,
                    normalize=True,
                    target_c=int(target_c),
                    name="masked_lat",
                    inference_device=self.device,
                    compute_dtype=dtype,
                )
                masked_branch_latent = _normalize_control_latent(
                    masked_branch_latent,
                    self.vae,
                    int(target_c),
                    enabled=normalize_condition_latents,
                )
                mask_branch_latent = _encode_control_branch_latent_with_infer_base(
                    infer_base=infer_base,
                    vae=self.vae,
                    branch_tchw=mask_tchw,
                    normalize=False,
                    target_c=int(target_c),
                    name="mask_lat",
                    inference_device=self.device,
                    compute_dtype=dtype,
                )
                mask_branch_latent = _normalize_control_latent(
                    mask_branch_latent,
                    self.vae,
                    int(target_c),
                    enabled=normalize_condition_latents,
                )

                control_chunks = [depth_branch_latent]
                if normal_branch_latent is not None:
                    control_chunks.append(normal_branch_latent)
                control_chunks.extend([masked_branch_latent, mask_branch_latent])
                control_latent = torch.cat(control_chunks, dim=1)

            if win_idx == int(supervised_window_idx):
                training_batch.first_frame_latent = window_first_frame_latent
                training_batch.image_latent = window_first_frame_latent.detach(
                ).clone()
                training_batch.control_latent = control_latent
                final_rollout = self._generator_multi_step_window_forward_shared_cache(
                    training_batch,
                    cache_position_offset_frames=int(cache_global_start_latent),
                    return_sim_steps=return_sim_steps,
                )
                break

            window_batch = _clone_training_batch_shallow(training_batch)
            window_batch.first_frame_latent = window_first_frame_latent
            window_batch.image_latent = window_first_frame_latent.detach(
            ).clone()
            window_batch.control_latent = control_latent
            window_batch.dmd_latent_vis_dict = {}

            previous_forced_exit = getattr(self, "_forced_exit_index", None)
            forced_last_exit = len(self.denoising_step_list) - 1
            try:
                self._forced_exit_index = int(forced_last_exit)
                with torch.no_grad():
                    window_pred = self._generator_multi_step_window_forward_shared_cache(
                        window_batch,
                        cache_position_offset_frames=int(
                            cache_global_start_latent),
                        return_sim_steps=False,
                    )
            finally:
                if previous_forced_exit is None:
                    if hasattr(self, "_forced_exit_index"):
                        delattr(self, "_forced_exit_index")
                else:
                    self._forced_exit_index = previous_forced_exit

            with torch.no_grad():
                decoded_pixels = self._decode_dit_latents_to_pixels(
                    window_pred, torch.float32)
                decoded_window_tchw = _pixels_to_unit_range(
                    decoded_pixels[0]).permute(1, 0, 2, 3).contiguous().cpu()

            current_frame_ids = [
                int(fid) for fid in clip["frame_ids"][start_pos:end_pos_valid]
            ]
            mcprep._update_history_and_visibility_memory(
                frame_ids=current_frame_ids,
                frames_tchw=decoded_window_tchw[:valid_window],
                depth_path_by_id=depth_path_by_id,
                camera_k_aligned=clip["camera_k_aligned"],
                rt_by_frame_id=clip["rt_by_frame_id"],
                crop_params=clip["crop_params"],
                history_rgbs_u8=history_rgbs_u8,
                processed_frame_ids=processed_frame_ids,
                visibility_map=visibility_map,
            )

            last_frame_id = int(clip["frame_ids"][end_pos_valid - 1])
            carry_prefix_tchw = decoded_window_tchw[valid_window -
                                                    int(overlap_frames):
                                                    valid_window].clone()

            next_chunk_start = max(0, int(end_pos_valid - overlap_frames))
            next_chunk_end = min(int(next_chunk_start + window_frames),
                                 int(total_required_frames))
            target_frame_ids_for_chunk = [
                int(fid)
                for fid in clip["frame_ids"][next_chunk_start:next_chunk_end]
            ]
            selected_keyframe_ids = mcprep._select_keyframes_for_next_window_memory(
                visibility_map=visibility_map,
                processed_frame_ids=processed_frame_ids,
                history_rgbs_u8=history_rgbs_u8,
                target_frame_ids_for_chunk=target_frame_ids_for_chunk,
                depth_path_by_id=depth_path_by_id,
                camera_k_aligned=clip["camera_k_aligned"],
                rt_by_frame_id=clip["rt_by_frame_id"],
                crop_params=clip["crop_params"],
                target_height=int(target_h),
                target_width=int(target_w),
                last_frame_id=int(last_frame_id),
                num_keyframes=int(
                    getattr(self.training_args, "online_warp_num_keyframes",
                            4)),
                num_target_samples=int(
                    getattr(self.training_args,
                            "online_warp_selection_num_target_samples", 3)),
            )
            keyframe_rgbs_u8 = [
                history_rgbs_u8[int(fid)] for fid in selected_keyframe_ids
            ]

            next_start = int(end_pos_valid)
            next_valid_new = min(int(stride),
                                 int(total_required_frames - next_start))
            target_ids_next = [
                int(fid)
                for fid in clip["frame_ids"][next_start:next_start +
                                             next_valid_new]
            ]
            if not target_ids_next:
                warped_masked_rgb_next = None
                warped_mask_next = None
                continue

            warped_masked_rgb_next_valid, warped_mask_next_valid = mcprep._warp_maskrgb_from_keyframes_md_aligned_memory(
                keyframe_rgbs_u8=keyframe_rgbs_u8,
                keyframe_frame_ids=selected_keyframe_ids,
                target_frame_ids=target_ids_next,
                depth_path_by_frame_id=depth_path_by_id,
                camera_k_aligned=clip["camera_k_aligned"],
                rt_by_frame_id=clip["rt_by_frame_id"],
                crop_params=clip["crop_params"],
                target_height=int(target_h),
                target_width=int(target_w),
            )
            warped_masked_rgb_next = mcprep._pad_tchw(
                warped_masked_rgb_next_valid, int(stride))
            warped_mask_next = mcprep._pad_tchw(warped_mask_next_valid,
                                                int(stride))
            cache_global_start_latent += int(latent_stride_t)

        if final_rollout is None:
            raise RuntimeError(
                "online_warp_training failed to produce a final rollout window")

        if not torch.is_grad_enabled():
            if self.kv_cache1 is not None and self.crossattn_cache is not None:
                self._reset_simulation_caches(self.kv_cache1,
                                              self.crossattn_cache)
            self._reset_additional_simulation_caches()

        training_batch.dmd_latent_vis_dict.update({
            "online_warp_rollout_num_windows":
            torch.tensor(float(num_windows),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_supervised_window_index":
            torch.tensor(float(int(supervised_window_idx)),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_only_final_window_supervised":
            torch.tensor(
                1.0 if int(supervised_window_idx) == int(num_windows) - 1 else 0.0,
                device=self.device,
                dtype=torch.float32),
            "online_warp_rollout_total_frames":
            torch.tensor(float(total_required_frames),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_rollout_total_latent_t":
            torch.tensor(float(total_latent_t),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_rollout_overlap_latent_t":
            torch.tensor(float(overlap_latent_t),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_rollout_latent_stride_t":
            torch.tensor(float(latent_stride_t),
                         device=self.device,
                         dtype=torch.float32),
            "online_warp_shared_cache_enabled":
            torch.tensor(1.0, device=self.device, dtype=torch.float32),
        })

        return final_rollout

    def _dmd_forward(self, generator_pred_video: torch.Tensor,
                     training_batch) -> torch.Tensor:
        original_latent = generator_pred_video
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        denoised_timestep_from = getattr(training_batch, "denoised_timestep_from",
                                         None)
        denoised_timestep_to = getattr(training_batch, "denoised_timestep_to",
                                       None)
        use_rollout_min = bool(getattr(self.training_args, "ts_schedule", False))
        use_rollout_max = bool(
            getattr(self.training_args, "ts_schedule_max", False))
        if not hasattr(training_batch, "dmd_latent_vis_dict") or training_batch.dmd_latent_vis_dict is None:
            training_batch.dmd_latent_vis_dict = {}
        with torch.no_grad():
            batch_size, num_frames = original_latent.shape[:2]
            timestep = _sample_uniform_score_timestep(
                batch_size=batch_size,
                device=self.device,
                num_train_timestep=self.num_train_timestep,
                timestep_shift=self.timestep_shift,
                min_timestep=self.min_timestep,
                max_timestep=self.max_timestep,
                use_rollout_min=use_rollout_min,
                use_rollout_max=use_rollout_max,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
            timestep_for_noise = timestep.repeat_interleave(num_frames)
            score_attn_metadata = self._build_score_attn_metadata(
                training_batch, timestep)
            score_context_timestep = self._score_context_timestep(timestep)

            noise = torch.randn_like(generator_pred_video)

            noisy_latent = self.noise_scheduler.add_noise(
                generator_pred_video.flatten(0, 1), noise.flatten(0, 1),
                timestep_for_noise).detach().unflatten(
                    0, (batch_size, generator_pred_video.shape[1]))
            score_noisy_latent = _apply_first_frame_latent_bfchw(
                noisy_latent, first_frame_latent)

            # fake_score forward (critic)
            training_batch = self._build_distill_input_kwargs(
                score_noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_fake_score_transformer = self._get_fake_score_transformer(
                timestep)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                fake_score_pred_noise = self._predict_noise_with_controlnet(
                    transformer=current_fake_score_transformer,
                    controlnet=getattr(self, "fake_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            faker_score_pred_video = pred_noise_to_pred_video(
                pred_noise=fake_score_pred_noise.flatten(0, 1),
                noise_input_latent=score_noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, fake_score_pred_noise.shape[:2])

            # real_score forward (teacher) conditional
            training_batch = self._build_distill_input_kwargs(
                score_noisy_latent, timestep, training_batch.conditional_dict,
                training_batch)
            current_real_score_transformer = self._get_real_score_transformer(
                timestep)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                real_score_pred_noise_cond = self._predict_noise_with_controlnet(
                    transformer=current_real_score_transformer,
                    controlnet=getattr(self, "real_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            pred_real_video_cond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_cond.flatten(0, 1),
                noise_input_latent=score_noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_cond.shape[:2])

            # real_score forward (teacher) unconditional
            training_batch = self._build_distill_input_kwargs(
                score_noisy_latent, timestep, training_batch.unconditional_dict,
                training_batch)
            with set_forward_context(current_timestep=score_context_timestep,
                                     attn_metadata=score_attn_metadata):
                real_score_pred_noise_uncond = self._predict_noise_with_controlnet(
                    transformer=current_real_score_transformer,
                    controlnet=getattr(self, "real_score_controlnet", None),
                    input_kwargs=training_batch.input_kwargs,
                    control_latent=control_latent,
                    first_frame_latent=first_frame_latent,
                ).permute(0, 2, 1, 3, 4)

            pred_real_video_uncond = pred_noise_to_pred_video(
                pred_noise=real_score_pred_noise_uncond.flatten(0, 1),
                noise_input_latent=score_noisy_latent.flatten(0, 1),
                timestep=timestep_for_noise,
                scheduler=self.noise_scheduler).unflatten(
                    0, real_score_pred_noise_uncond.shape[:2])

            # Optional bucketed teacher guidance:
            # use a lower guidance scale on high-noise timesteps only.
            timestep_ratio = (
                (timestep.float().view(batch_size, 1, 1, 1, 1) -
                 float(self.min_timestep)) /
                max(float(self.max_timestep - self.min_timestep), 1.0)
            ).clamp(0.0, 1.0)
            teacher_guidance_scale = torch.full_like(
                timestep_ratio, float(self.real_score_guidance_scale))
            high_noise_guidance_mask_ratio = torch.zeros_like(timestep_ratio)
            if bool(
                    getattr(self.training_args,
                            "dmd_teacher_adaptive_guidance", True)):
                guidance_threshold = float(
                    getattr(
                        self.training_args,
                        "dmd_teacher_guidance_high_noise_threshold_ratio",
                        0.7,
                    ))
                guidance_threshold = min(max(guidance_threshold, 0.0), 1.0)
                high_noise_mask = timestep_ratio >= guidance_threshold
                high_noise_guidance_scale = float(
                    getattr(self.training_args,
                            "dmd_teacher_high_noise_guidance_scale", 2.0))
                teacher_guidance_scale = torch.where(
                    high_noise_mask,
                    torch.full_like(teacher_guidance_scale,
                                    high_noise_guidance_scale),
                    teacher_guidance_scale,
                )
                high_noise_guidance_mask_ratio = high_noise_mask.float()
            teacher_cfg_delta = (
                pred_real_video_cond -
                pred_real_video_uncond) * teacher_guidance_scale
            # Timestep-adaptive teacher clamp:
            # high-noise steps use tighter teacher ratios to reduce whitening/drift.
            adaptive_alpha = torch.zeros_like(timestep_ratio)
            if bool(
                    getattr(self.training_args, "dmd_teacher_adaptive_clamp",
                            True)):
                adaptive_start_ratio = float(
                    getattr(self.training_args,
                            "dmd_teacher_adaptive_start_ratio", 0.7))
                adaptive_start_ratio = min(max(adaptive_start_ratio, 0.0),
                                           0.999)
                adaptive_alpha = (
                    (timestep_ratio - adaptive_start_ratio) /
                    max(1e-6, 1.0 - adaptive_start_ratio)
                ).clamp(0.0, 1.0)
            teacher_cfg_delta_std = _samplewise_std(teacher_cfg_delta)
            teacher_cfg_clip_scale = torch.ones_like(teacher_cfg_delta_std)
            teacher_cfg_clip_ratio_base = float(
                getattr(self.training_args,
                        "dmd_teacher_cfg_delta_max_ratio", 1.5))
            teacher_cfg_clip_ratio_high = float(
                getattr(self.training_args,
                        "dmd_teacher_high_noise_cfg_delta_max_ratio",
                        teacher_cfg_clip_ratio_base))
            teacher_cfg_clip_ratio = torch.clamp(
                torch.full_like(teacher_cfg_delta_std, teacher_cfg_clip_ratio_base)
                + (teacher_cfg_clip_ratio_high - teacher_cfg_clip_ratio_base) *
                adaptive_alpha,
                min=0.0,
            )
            if float(teacher_cfg_clip_ratio.max().item()) > 0:
                cond_std = _samplewise_std(pred_real_video_cond).clamp_min(1e-6)
                max_cfg_std = cond_std * teacher_cfg_clip_ratio
                teacher_cfg_clip_scale = torch.clamp(
                    max_cfg_std / teacher_cfg_delta_std.clamp_min(1e-6),
                    max=1.0,
                )
                teacher_cfg_delta = teacher_cfg_delta * teacher_cfg_clip_scale
            teacher_cfg_from_uncond = bool(
                getattr(self.training_args, "dmd_teacher_cfg_use_uncond_base",
                        True))
            if teacher_cfg_from_uncond:
                real_score_pred_video = pred_real_video_uncond + teacher_cfg_delta
            else:
                real_score_pred_video = pred_real_video_cond + teacher_cfg_delta

            teacher_residual = real_score_pred_video - original_latent
            teacher_residual_std = _samplewise_std(teacher_residual)
            teacher_residual_clip_scale = torch.ones_like(
                teacher_residual_std)
            teacher_residual_clip_ratio_base = float(
                getattr(self.training_args,
                        "dmd_teacher_residual_max_ratio", 1.5))
            teacher_residual_clip_ratio_high = float(
                getattr(self.training_args,
                        "dmd_teacher_high_noise_residual_max_ratio",
                        teacher_residual_clip_ratio_base))
            teacher_residual_clip_ratio = torch.clamp(
                torch.full_like(teacher_residual_std,
                                teacher_residual_clip_ratio_base)
                + (teacher_residual_clip_ratio_high -
                   teacher_residual_clip_ratio_base) * adaptive_alpha,
                min=0.0,
            )
            generator_std = _samplewise_std(original_latent).clamp_min(1e-6)
            if float(teacher_residual_clip_ratio.max().item()) > 0:
                max_teacher_residual_std = (
                    generator_std * teacher_residual_clip_ratio)
                teacher_residual_clip_scale = torch.clamp(
                    max_teacher_residual_std /
                    teacher_residual_std.clamp_min(1e-6),
                    max=1.0,
                )
                real_score_pred_video = original_latent + (
                    teacher_residual * teacher_residual_clip_scale)

            # Final safety clamp on teacher output std to avoid high-noise
            # teacher overpowering the student (whitening/drift failure mode).
            teacher_output_std = _samplewise_std(real_score_pred_video)
            teacher_output_clip_scale = torch.ones_like(teacher_output_std)
            teacher_output_clip_ratio_base = float(
                getattr(self.training_args, "dmd_teacher_output_max_ratio",
                        1.15))
            teacher_output_clip_ratio_high = float(
                getattr(self.training_args,
                        "dmd_teacher_high_noise_output_max_ratio",
                        teacher_output_clip_ratio_base))
            teacher_output_clip_ratio = torch.clamp(
                torch.full_like(teacher_output_std, teacher_output_clip_ratio_base)
                + (teacher_output_clip_ratio_high -
                   teacher_output_clip_ratio_base) * adaptive_alpha,
                min=0.0,
            )
            if float(teacher_output_clip_ratio.max().item()) > 0:
                max_teacher_output_std = generator_std * teacher_output_clip_ratio
                teacher_output_clip_scale = torch.clamp(
                    max_teacher_output_std /
                    teacher_output_std.clamp_min(1e-6),
                    max=1.0,
                )
                teacher_delta = real_score_pred_video - original_latent
                real_score_pred_video = original_latent + (
                    teacher_delta * teacher_output_clip_scale)

            # Stabilize DMD normalization when student is close to teacher.
            # Use raw normalizer with EMA-based floor to prevent tiny early-step
            # denominators from amplifying updates.
            grad_normalizer_diff = torch.abs(original_latent -
                                             real_score_pred_video)
            grad_normalizer_diff = _drop_first_frame_bfchw(
                grad_normalizer_diff, first_frame_latent)
            grad_normalizer_raw = grad_normalizer_diff.mean(
                    dim=(1, 2, 3, 4), keepdim=True)
            ema_decay = float(
                getattr(self.training_args, "dmd_grad_normalizer_ema_decay",
                        0.99))
            ema_floor_ratio = float(
                getattr(self.training_args,
                        "dmd_grad_normalizer_ema_floor_ratio", 0.1))
            normalizer_min = float(
                getattr(self.training_args, "dmd_grad_normalizer_min", 1e-3))

            prev_ema = getattr(self, "_dmd_grad_norm_ema", None)
            if prev_ema is None:
                grad_normalizer_ema = grad_normalizer_raw.detach().mean()
            else:
                grad_normalizer_ema = prev_ema * ema_decay + grad_normalizer_raw.detach(
                ).mean() * (1.0 - ema_decay)
            self._dmd_grad_norm_ema = grad_normalizer_ema

            grad_normalizer = torch.maximum(
                grad_normalizer_raw,
                grad_normalizer_ema * ema_floor_ratio,
            ).clamp_min(normalizer_min)
            grad = (faker_score_pred_video -
                    real_score_pred_video) / grad_normalizer
            grad_clip = float(
                getattr(self.training_args, "dmd_grad_value_clip", 10.0))
            if grad_clip > 0:
                grad = grad.clamp(min=-grad_clip, max=grad_clip)
            first_frame_grad_zeroed = 0.0
            if first_frame_latent is not None and grad.ndim == 5 and grad.shape[
                    1] > 0:
                grad = grad.clone()
                grad[:, :1] = 0
                first_frame_grad_zeroed = 1.0
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_for_log = _drop_first_frame_bfchw(grad, first_frame_latent)
            grad_abs_mean = grad_for_log.abs().mean().detach()

        dmd_loss = 0.5 * F.mse_loss(original_latent.float(),
                                    (original_latent.float() -
                                     grad.float()).detach())

        training_batch.dmd_latent_vis_dict.update({
            "training_batch_dmd_fwd_clean_latent": training_batch.latents,
            "generator_pred_video": original_latent.detach(),
            "real_score_pred_video": real_score_pred_video.detach(),
            "faker_score_pred_video": faker_score_pred_video.detach(),
            "dmd_timestep": timestep.detach(),
            "denoised_timestep_from": torch.tensor(
                -1 if denoised_timestep_from is None else denoised_timestep_from,
                device=self.device,
                dtype=torch.float32),
            "denoised_timestep_to": torch.tensor(
                -1 if denoised_timestep_to is None else denoised_timestep_to,
                device=self.device,
                dtype=torch.float32),
            "dmd_grad_normalizer": grad_normalizer.detach(),
            "dmd_grad_normalizer_raw": grad_normalizer_raw.detach(),
            "dmd_grad_normalizer_ema": grad_normalizer_ema.detach(),
            "dmdtrain_gradient_norm": grad_abs_mean,
            "teacher_cfg_delta_std": teacher_cfg_delta_std.detach(),
            "teacher_cfg_clip_scale": teacher_cfg_clip_scale.detach(),
            "teacher_cfg_clip_ratio_used": teacher_cfg_clip_ratio.detach(),
            "teacher_cfg_from_uncond": torch.tensor(
                1.0 if teacher_cfg_from_uncond else 0.0,
                device=self.device,
                dtype=torch.float32),
            "teacher_residual_std": teacher_residual_std.detach(),
            "teacher_residual_clip_scale":
            teacher_residual_clip_scale.detach(),
            "teacher_residual_clip_ratio_used":
            teacher_residual_clip_ratio.detach(),
            "teacher_output_std": teacher_output_std.detach(),
            "teacher_output_clip_scale": teacher_output_clip_scale.detach(),
            "teacher_output_clip_ratio_used":
            teacher_output_clip_ratio.detach(),
            "teacher_guidance_scale_used": teacher_guidance_scale.detach(),
            "teacher_guidance_high_noise_mask_ratio":
            high_noise_guidance_mask_ratio.detach(),
            "teacher_timestep_ratio": timestep_ratio.detach(),
            "teacher_adaptive_alpha": adaptive_alpha.detach(),
            "dmd_first_frame_grad_zeroed": torch.tensor(
                first_frame_grad_zeroed,
                device=self.device,
                dtype=torch.float32),
        })
        return dmd_loss

    def faker_score_forward(self, training_batch):
        control_latent = getattr(training_batch, "control_latent", None)
        first_frame_latent = getattr(training_batch, "first_frame_latent", None)
        denoised_timestep_from = getattr(training_batch, "denoised_timestep_from",
                                         None)
        denoised_timestep_to = getattr(training_batch, "denoised_timestep_to",
                                       None)
        use_rollout_min = bool(getattr(self.training_args, "ts_schedule", False))
        use_rollout_max = bool(
            getattr(self.training_args, "ts_schedule_max", False))
        # The Wan attention stack requires ForwardContext to be set for every
        # forward pass (including ControlNet during simulation).
        with torch.no_grad(), set_forward_context(
                current_timestep=training_batch.timesteps,
                attn_metadata=training_batch.attn_metadata):
            if self.training_args.simulate_generator_forward:
                rollout = self._generator_multi_step_simulation_forward(
                    training_batch, return_sim_steps=True)
                if isinstance(rollout, tuple):
                    (generator_pred_video, denoised_timestep_from,
                     denoised_timestep_to, _) = rollout
                else:
                    generator_pred_video = rollout
            else:
                generator_pred_video = self._generator_forward(training_batch)

        # `online_warp_training` materializes the per-window control latents
        # lazily inside `_generator_multi_step_simulation_forward`. Refresh the
        # local reference here so critic training uses the same conditioning as
        # the generator/teacher paths instead of silently dropping ControlNet.
        control_latent = getattr(training_batch, "control_latent", control_latent)

        batch_size, num_frames = generator_pred_video.shape[:2]
        training_batch.denoised_timestep_from = denoised_timestep_from
        training_batch.denoised_timestep_to = denoised_timestep_to
        fake_score_timestep = _sample_uniform_score_timestep(
            batch_size=batch_size,
            device=self.device,
            num_train_timestep=self.num_train_timestep,
            timestep_shift=self.timestep_shift,
            min_timestep=self.min_timestep,
            max_timestep=self.max_timestep,
            use_rollout_min=use_rollout_min,
            use_rollout_max=use_rollout_max,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        fake_score_timestep_for_noise = fake_score_timestep.repeat_interleave(
            num_frames)
        score_attn_metadata = self._build_score_attn_metadata(
            training_batch, fake_score_timestep)
        score_context_timestep = self._score_context_timestep(
            fake_score_timestep)

        fake_score_noise = torch.randn_like(generator_pred_video)
        noisy_generator_pred_video = self.noise_scheduler.add_noise(
            generator_pred_video.flatten(0, 1), fake_score_noise.flatten(0, 1),
            fake_score_timestep_for_noise).unflatten(
                0, (batch_size, generator_pred_video.shape[1]))
        score_noisy_generator_pred_video = _apply_first_frame_latent_bfchw(
            noisy_generator_pred_video, first_frame_latent)

        training_batch = self._build_distill_input_kwargs(
            score_noisy_generator_pred_video, fake_score_timestep,
            training_batch.conditional_dict, training_batch)

        with set_forward_context(current_timestep=score_context_timestep,
                                 attn_metadata=score_attn_metadata):
            current_fake_score_transformer = self._get_fake_score_transformer(
                fake_score_timestep)
            fake_score_pred_noise = self._predict_noise_with_controlnet(
                transformer=current_fake_score_transformer,
                controlnet=getattr(self, "fake_score_controlnet", None),
                input_kwargs=training_batch.input_kwargs,
                control_latent=control_latent,
                first_frame_latent=first_frame_latent,
            ).permute(0, 2, 1, 3, 4)

        target = fake_score_noise - generator_pred_video
        fake_score_sq_error = (fake_score_pred_noise - target)**2
        fake_score_sq_error = _drop_first_frame_bfchw(fake_score_sq_error,
                                                      first_frame_latent)
        flow_matching_loss = torch.mean(fake_score_sq_error)

        training_batch.fake_score_latent_vis_dict = {
            "generator_pred_video": generator_pred_video,
            "fake_score_timestep": fake_score_timestep,
            "denoised_timestep_from": torch.tensor(
                -1 if denoised_timestep_from is None else denoised_timestep_from,
                device=self.device,
                dtype=torch.float32),
            "denoised_timestep_to": torch.tensor(
                -1 if denoised_timestep_to is None else denoised_timestep_to,
                device=self.device,
                dtype=torch.float32),
        }
        return training_batch, flow_matching_loss

    def _clip_model_grad_norm_(self,
                               training_batch,
                               transformer,
                               *,
                               log_prefix: str | None = None):
        max_grad_norm = self.training_args.max_grad_norm
        if max_grad_norm is None:
            training_batch.grad_norm = None
            if log_prefix is not None:
                setattr(training_batch, f"{log_prefix}_grad_norm_preclip", None)
                setattr(training_batch, f"{log_prefix}_grad_norm_postclip", None)
                setattr(training_batch, f"{log_prefix}_grad_clip_coef", None)
                setattr(training_batch, f"{log_prefix}_grad_was_clipped", None)
            return training_batch

        model_parts = [transformer]
        if transformer is self.transformer or transformer is getattr(
                self, "transformer_2", None):
            controlnet = getattr(self, "controlnet", None)
            if controlnet is not None:
                model_parts.append(controlnet)
        elif transformer is self.fake_score_transformer or transformer is getattr(
                self, "fake_score_transformer_2", None):
            fake_score_controlnet = getattr(self, "fake_score_controlnet", None)
            if fake_score_controlnet is not None:
                model_parts.append(fake_score_controlnet)

        grad_norm = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for m in model_parts for p in m.parameters() if p.requires_grad],
            max_grad_norm,
            foreach=None,
        )
        grad_norm_value = grad_norm.item() if grad_norm is not None else None
        if grad_norm_value is not None and not math.isfinite(grad_norm_value):
            raise ValueError(
                f"Detected non-finite gradient norm: {grad_norm_value}")
        training_batch.grad_norm = grad_norm_value
        if log_prefix is not None:
            setattr(training_batch, f"{log_prefix}_grad_norm_preclip",
                    grad_norm_value)
            if grad_norm_value is None:
                setattr(training_batch, f"{log_prefix}_grad_norm_postclip",
                        None)
                setattr(training_batch, f"{log_prefix}_grad_clip_coef", None)
                setattr(training_batch, f"{log_prefix}_grad_was_clipped", None)
            else:
                max_grad_norm_value = float(max_grad_norm)
                if max_grad_norm_value <= 0.0:
                    postclip_value = 0.0
                    clip_coef = 0.0
                else:
                    postclip_value = min(grad_norm_value, max_grad_norm_value)
                    clip_coef = min(
                        max_grad_norm_value / (grad_norm_value + 1e-6), 1.0)
                setattr(training_batch, f"{log_prefix}_grad_norm_postclip",
                        postclip_value)
                setattr(training_batch, f"{log_prefix}_grad_clip_coef",
                        clip_coef)
                setattr(training_batch, f"{log_prefix}_grad_was_clipped",
                        float(clip_coef < 0.999999))
        return training_batch

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)
        student_attention_mode = _normalize_student_attention_mode(
            getattr(training_args, "student_attention_mode", "causal"))
        logger.info("Student attention mode: %s", student_attention_mode)
        if (student_attention_mode == "bidirectional"
                and bool(getattr(training_args, "online_warp_training",
                                 False))):
            raise ValueError(
                "student_attention_mode=bidirectional does not support "
                "online_warp_training because the shared-cache rollout is "
                "causal-only. Please disable online_warp_training for the "
                "RGBN/full-sequence mode.")

        if getattr(self, "rollout_add_context_noise", True):
            logger.info(
                "Forcing rollout_add_context_noise=False in Wan ControlNet self-forcing to align with causal inference cache update."
            )
        self.rollout_add_context_noise = False
        logger.info(
            "DMD timestep schedule flags: ts_schedule(min)=%s ts_schedule_max(max)=%s",
            bool(getattr(training_args, "ts_schedule", False)),
            bool(getattr(training_args, "ts_schedule_max", False)),
        )
        logger.info(
            "Condition latent normalization: normalize_condition_latents=%s",
            bool(getattr(training_args, "normalize_condition_latents", False)),
        )
        logger.info(
            "Adaptive teacher clamp: enabled=%s start_ratio=%.3f high_noise(cfg=%.3f residual=%.3f output=%.3f)",
            bool(getattr(training_args, "dmd_teacher_adaptive_clamp", True)),
            float(
                getattr(training_args, "dmd_teacher_adaptive_start_ratio",
                        0.7)),
            float(
                getattr(training_args,
                        "dmd_teacher_high_noise_cfg_delta_max_ratio", 1.0)),
            float(
                getattr(training_args,
                        "dmd_teacher_high_noise_residual_max_ratio", 1.0)),
            float(
                getattr(training_args,
                        "dmd_teacher_high_noise_output_max_ratio", 1.0)),
        )
        logger.info(
            "Adaptive teacher guidance: enabled=%s threshold=%.3f base=%.3f high_noise=%.3f",
            bool(
                getattr(training_args, "dmd_teacher_adaptive_guidance", True)),
            float(
                getattr(training_args,
                        "dmd_teacher_guidance_high_noise_threshold_ratio",
                        0.7)),
            float(self.real_score_guidance_scale),
            float(
                getattr(training_args, "dmd_teacher_high_noise_guidance_scale",
                        2.0)),
        )
        logger.info(
            "Teacher CFG composition: use_uncond_base=%s",
            bool(
                getattr(training_args, "dmd_teacher_cfg_use_uncond_base",
                        True)),
        )
        risky_teacher_cfg = not bool(
            getattr(training_args, "dmd_teacher_cfg_use_uncond_base", True))
        risky_teacher_clamp = not bool(
            getattr(training_args, "dmd_teacher_adaptive_clamp", True))
        risky_teacher_guidance = not bool(
            getattr(training_args, "dmd_teacher_adaptive_guidance", True))
        risky_grad_norm = float(
            getattr(training_args, "max_grad_norm", 0.0) or 0.0) > 2.0
        risky_normalizer_floor = float(
            getattr(training_args, "dmd_grad_normalizer_ema_floor_ratio",
                    0.0)) <= 0.0
        risky_normalizer_min = float(
            getattr(training_args, "dmd_grad_normalizer_min", 0.0)) < 1e-4
        if any((
                risky_teacher_cfg,
                risky_teacher_clamp,
                risky_teacher_guidance,
                risky_grad_norm,
                risky_normalizer_floor,
                risky_normalizer_min,
        )):
            logger.warning(
                "Detected a high-risk Phase-3 DMD setup for whitening drift: "
                "use_uncond_base=%s adaptive_clamp=%s adaptive_guidance=%s "
                "max_grad_norm=%.4f grad_norm_floor_ratio=%.4f grad_normalizer_min=%.2e. "
                "If generations wash out or turn overly bright, prefer the Phase-2 style stabilization "
                "(uncond-base CFG, adaptive clamp/guidance, non-zero EMA floor, tighter grad clipping).",
                not risky_teacher_cfg,
                not risky_teacher_clamp,
                not risky_teacher_guidance,
                float(getattr(training_args, "max_grad_norm", 0.0) or 0.0),
                float(
                    getattr(training_args,
                            "dmd_grad_normalizer_ema_floor_ratio", 0.0)),
                float(getattr(training_args, "dmd_grad_normalizer_min", 0.0)),
            )

        # Activation checkpointing for ControlNet modules (important for memory in
        # self-forcing rollout, where the KV-cache path would otherwise retain a
        # large autograd graph).
        if training_args.enable_gradient_checkpointing_type is not None:
            for name in ("controlnet", "fake_score_controlnet",
                         "real_score_controlnet"):
                m = getattr(self, name, None)
                if m is not None:
                    setattr(
                        self,
                        name,
                        apply_activation_checkpointing(
                            m,
                            checkpointing_type=training_args.
                            enable_gradient_checkpointing_type),
                    )

        # Freeze teacher controlnet
        if getattr(self, "real_score_controlnet", None) is not None:
            self.real_score_controlnet.requires_grad_(False)
            self.real_score_controlnet.eval()

        # Make teacher/critic controlnet effectively bidirectional on full sequence (when not using KV cache)
        for name in ("real_score_controlnet", "fake_score_controlnet"):
            m = getattr(self, name, None)
            if m is not None and hasattr(m, "num_frame_per_block"):
                try:
                    m.num_frame_per_block = int(self.training_args.num_latent_t)
                except Exception:
                    pass
                if hasattr(m, "_block_mask_cache"):
                    try:
                        m._block_mask_cache = {}
                    except Exception:
                        pass

        # Make sure student/critic controlnet are trainable (if provided)
        if getattr(self, "controlnet", None) is not None:
            self.controlnet.requires_grad_(True)
            self.controlnet.train()
        if getattr(self, "fake_score_controlnet", None) is not None:
            self.fake_score_controlnet.requires_grad_(True)
            self.fake_score_controlnet.train()

        # Maintain EMA for student ControlNet as well; transformer-only EMA is
        # insufficient for TI2V+ControlNet inference stability.
        self.controlnet_ema = None
        if (getattr(self, "controlnet", None) is not None
                and self.training_args.ema_decay is not None
                and self.training_args.ema_decay > 0.0):
            self.controlnet_ema = EMA_FSDP(
                self.controlnet, decay=self.training_args.ema_decay)
            logger.info("Initialized controlnet EMA with decay=%s",
                        self.training_args.ema_decay)

        # Rebuild generator optimizer to include controlnet params
        if getattr(self, "controlnet", None) is not None:
            gen_params = list(
                filter(lambda p: p.requires_grad,
                       list(self.transformer.parameters()) +
                       list(self.controlnet.parameters())))
            betas_str = training_args.betas
            betas = tuple(float(x.strip()) for x in betas_str.split(","))
            self.optimizer = torch.optim.AdamW(
                gen_params,
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
            if self.transformer_2 is not None:
                gen_params_2 = list(
                    filter(lambda p: p.requires_grad,
                           list(self.transformer_2.parameters()) +
                           list(self.controlnet.parameters())))
                self.optimizer_2 = torch.optim.AdamW(
                    gen_params_2,
                    lr=training_args.learning_rate,
                    betas=betas,
                    weight_decay=training_args.weight_decay,
                    eps=1e-8,
                )
                self.lr_scheduler_2 = get_scheduler(
                    training_args.lr_scheduler,
                    optimizer=self.optimizer_2,
                    num_warmup_steps=training_args.lr_warmup_steps,
                    num_training_steps=training_args.max_train_steps,
                    num_cycles=training_args.lr_num_cycles,
                    power=training_args.lr_power,
                    min_lr_ratio=training_args.min_lr_ratio,
                    last_epoch=self.init_steps - 1,
                )

        # Rebuild critic optimizer to include critic controlnet params
        if getattr(self, "fake_score_controlnet", None) is not None:
            fake_score_lr = training_args.fake_score_learning_rate
            if fake_score_lr == 0.0:
                fake_score_lr = training_args.learning_rate
            betas_str = training_args.fake_score_betas
            betas = tuple(float(x.strip()) for x in betas_str.split(","))
            critic_params = list(
                filter(
                    lambda p: p.requires_grad,
                    list(self.fake_score_transformer.parameters()) +
                    list(self.fake_score_controlnet.parameters()),
                ))
            self.fake_score_optimizer = torch.optim.AdamW(
                critic_params,
                lr=fake_score_lr,
                betas=betas,
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )
            self.fake_score_lr_scheduler = get_scheduler(
                training_args.fake_score_lr_scheduler,
                optimizer=self.fake_score_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )
            if self.fake_score_transformer_2 is not None:
                critic_params_2 = list(
                    filter(
                        lambda p: p.requires_grad,
                        list(self.fake_score_transformer_2.parameters()) +
                        list(self.fake_score_controlnet.parameters()),
                    ))
                self.fake_score_optimizer_2 = torch.optim.AdamW(
                    critic_params_2,
                    lr=fake_score_lr,
                    betas=betas,
                    weight_decay=training_args.weight_decay,
                    eps=1e-8,
                )
                self.fake_score_lr_scheduler_2 = get_scheduler(
                    training_args.fake_score_lr_scheduler,
                    optimizer=self.fake_score_optimizer_2,
                    num_warmup_steps=training_args.lr_warmup_steps,
                    num_training_steps=training_args.max_train_steps,
                    num_cycles=training_args.lr_num_cycles,
                    power=training_args.lr_power,
                    min_lr_ratio=training_args.min_lr_ratio,
                    last_epoch=self.init_steps - 1,
                )

    def _get_next_batch(self, training_batch):
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            self.train_loader_iter = iter(self.train_dataloader)
            batch = next(self.train_loader_iter)

        device = get_local_torch_device()
        dtype = torch.bfloat16

        encoder_hidden_states = batch["text_embedding"]
        encoder_attention_mask = batch["text_attention_mask"]
        infos = batch["info_list"]

        self._lazy_init_fixed_training_text_condition()
        fixed_text_embedding = getattr(self, "_fixed_training_text_embedding",
                                       None)
        fixed_text_attention_mask = getattr(
            self, "_fixed_training_text_attention_mask", None)
        fixed_text_caption = str(
            getattr(self, "_fixed_training_text_caption", "")).strip()
        if fixed_text_embedding is not None and fixed_text_attention_mask is not None:
            batch_size = int(encoder_hidden_states.shape[0])
            encoder_hidden_states = fixed_text_embedding.unsqueeze(0).expand(
                batch_size, -1, -1).contiguous()
            encoder_attention_mask = fixed_text_attention_mask.unsqueeze(
                0).expand(batch_size, -1).contiguous()
            if infos is not None and fixed_text_caption:
                for info in infos:
                    if isinstance(info, dict):
                        info["prompt"] = fixed_text_caption
                        info["caption"] = fixed_text_caption
            if not getattr(self, "_logged_fixed_training_text_override", False):
                logger.info(
                    "Overriding per-sample text conditioning with fixed reference caption=%r for phase-3 training",
                    fixed_text_caption,
                )
                self._logged_fixed_training_text_override = True

        batch_size = encoder_hidden_states.shape[0]
        vae_config = self.training_args.pipeline_config.vae_config.arch_config
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio
        latent_height = self.training_args.num_height // spatial_compression_ratio
        latent_width = self.training_args.num_width // spatial_compression_ratio

        latents = torch.randn(
            batch_size,
            num_channels,
            self.training_args.num_latent_t,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )

        training_batch.latents = latents
        training_batch.encoder_hidden_states = encoder_hidden_states.to(device,
                                                                        dtype=dtype)
        training_batch.encoder_attention_mask = encoder_attention_mask.to(
            device, dtype=dtype)
        training_batch.infos = infos

        normalize_condition_latents = bool(
            getattr(self.training_args, "normalize_condition_latents", False))
        first_frame_latent_bcfhw = _ensure_first_frame_bcfhw(
            batch["first_frame_latent"])
        first_frame_latent = first_frame_latent_bcfhw.permute(
            0, 2, 1, 3, 4).contiguous()
        first_frame_latent = first_frame_latent.to(device, dtype=dtype)
        first_frame_latent = _normalize_first_frame_latent(
            first_frame_latent,
            self.vae,
            enabled=normalize_condition_latents,
        )
        training_batch.first_frame_latent = first_frame_latent
        training_batch.global_first_frame_latent = first_frame_latent.detach(
        ).clone()
        # Keep TI2V rollout aligned with inference/Causal-Forcing by warming the
        # generator cache with a clean t=0 anchor latent before denoising the
        # remaining frames of the window.
        training_batch.image_latent = first_frame_latent.detach().clone()

        if bool(getattr(self.training_args, "online_warp_training", False)):
            num_channels = int(
                self.training_args.pipeline_config.vae_config.arch_config.z_dim)
            bootstrap_control_latent = None
            use_bootstrap_control_first_window = bool(
                getattr(
                    self.training_args,
                    "online_warp_use_bootstrap_control_for_first_window",
                    False,
                ))
            raw_control_latent = batch.get("control_latent")
            if (use_bootstrap_control_first_window
                    and raw_control_latent is not None
                    and isinstance(raw_control_latent, torch.Tensor)
                    and int(raw_control_latent.numel()) > 0
                    and raw_control_latent.dim() >= 4):
                bootstrap_control_latent = _ensure_control_latent_bcfhw(
                    raw_control_latent,
                    latent_channels=int(first_frame_latent_bcfhw.shape[1]))
                if int(bootstrap_control_latent.shape[2]) != int(
                        self.training_args.num_latent_t):
                    raise ValueError(
                        "Online-warp bootstrap control_latent temporal length mismatch: "
                        f"control_latent.shape[2]={int(bootstrap_control_latent.shape[2])}, "
                        f"num_latent_t={int(self.training_args.num_latent_t)}")
                bootstrap_control_latent = bootstrap_control_latent.to(
                    device, dtype=dtype)
                bootstrap_control_latent = _normalize_control_latent(
                    bootstrap_control_latent,
                    self.vae,
                    num_channels,
                    enabled=normalize_condition_latents,
                )

            full_depth_latent = None
            raw_depth_latent = batch.get("depth_latent")
            if (raw_depth_latent is not None and isinstance(raw_depth_latent, torch.Tensor)
                    and int(raw_depth_latent.numel()) > 0 and raw_depth_latent.dim() >= 4):
                full_depth_latent = _ensure_branch_latent_bcfhw(
                    raw_depth_latent,
                    latent_channels=int(first_frame_latent_bcfhw.shape[1]),
                    name="depth_latent",
                ).to(device, dtype=dtype)
                full_depth_latent = _normalize_control_latent(
                    full_depth_latent,
                    self.vae,
                    num_channels,
                    enabled=normalize_condition_latents,
                )

            full_normal_latent = None
            raw_normal_latent = batch.get("normal_latent")
            if (raw_normal_latent is not None and isinstance(raw_normal_latent, torch.Tensor)
                    and int(raw_normal_latent.numel()) > 0 and raw_normal_latent.dim() >= 4):
                full_normal_latent = _ensure_branch_latent_bcfhw(
                    raw_normal_latent,
                    latent_channels=int(first_frame_latent_bcfhw.shape[1]),
                    name="normal_latent",
                ).to(device, dtype=dtype)
                full_normal_latent = _normalize_control_latent(
                    full_normal_latent,
                    self.vae,
                    num_channels,
                    enabled=normalize_condition_latents,
                )
            if full_normal_latent is not None and full_depth_latent is None:
                raise ValueError(
                    "online_warp_training received normal_latent without depth_latent"
                )

            training_batch.bootstrap_control_latent = bootstrap_control_latent
            training_batch.full_depth_latent = full_depth_latent
            training_batch.full_normal_latent = full_normal_latent
            training_batch.control_latent = None
            if not getattr(self, "_logged_condition_latent_stats", False):
                logger.info(
                    "Condition latent stats (online-warp bootstrap): normalize=%s bootstrap_first_window_enabled=%s cached_first_window=%s cached_full_depth=%s cached_full_normal=%s first_frame(mean=%.6f std=%.6f)",
                    normalize_condition_latents,
                    use_bootstrap_control_first_window,
                    bootstrap_control_latent is not None,
                    full_depth_latent is not None,
                    full_normal_latent is not None,
                    float(first_frame_latent.float().mean().item()),
                    float(first_frame_latent.float().std(unbiased=False).item()),
                )
                self._logged_condition_latent_stats = True
            return training_batch

        # Required fields for TI2V + ControlNet
        # Canonicalize condition latent layouts to avoid silent shape-space
        # mismatch across parquet sources:
        # - first_frame_latent: BCFHW (B,C,1,H,W) -> BFCHW (B,1,C,H,W) internally
        # - control_latent: B,C_total,F,H,W
        control_latent = _ensure_control_latent_bcfhw(
            batch["control_latent"],
            latent_channels=int(first_frame_latent_bcfhw.shape[1]))
        if int(control_latent.shape[2]) != int(self.training_args.num_latent_t):
            raise ValueError(
                "Training control_latent temporal length mismatch: "
                f"control_latent.shape[2]={int(control_latent.shape[2])}, "
                f"num_latent_t={int(self.training_args.num_latent_t)}")

        control_latent = control_latent.to(device, dtype=dtype)

        num_channels = int(self.training_args.pipeline_config.vae_config.arch_config.z_dim)
        control_latent = _normalize_control_latent(control_latent, self.vae,
                                                   num_channels,
                                                   enabled=normalize_condition_latents)

        training_batch.control_latent = control_latent
        training_batch.first_frame_latent = first_frame_latent
        if not getattr(self, "_logged_condition_latent_stats", False):
            logger.info(
                "Condition latent stats (post-load): normalize=%s first_frame(mean=%.6f std=%.6f) control(mean=%.6f std=%.6f)",
                normalize_condition_latents,
                float(first_frame_latent.float().mean().item()),
                float(first_frame_latent.float().std(unbiased=False).item()),
                float(control_latent.float().mean().item()),
                float(control_latent.float().std(unbiased=False).item()),
            )
            self._logged_condition_latent_stats = True

        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        required_columns = (
            "first_frame_latent_bytes",
            "first_frame_latent_shape",
            "control_latent_bytes",
            "control_latent_shape",
        )
        missing_columns = [
            key for key in required_columns if validation_batch.get(key) is None
        ]
        if missing_columns:
            raise ValueError(
                "ControlNet validation requires parquet/arrow validation rows with "
                f"{missing_columns}. Current validation sample keys: {sorted(validation_batch.keys())}"
            )

        sampling_param.prompt = validation_batch["prompt"]
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "ti2v_controlnet"
        if training_args.validation_guidance_scale:
            sampling_param.guidance_scale = float(
                training_args.validation_guidance_scale)
        assert self.seed is not None
        sampling_param.seed = self.seed

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        spatial_compression_factor = training_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        sampling_param.num_frames = ((training_args.num_latent_t - 1) *
                                     temporal_compression_factor + 1)
        latents_size = [(sampling_param.num_frames - 1) // temporal_compression_factor + 1,
                        sampling_param.height // spatial_compression_factor,
                        sampling_param.width // spatial_compression_factor]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        first_frame_latent = _ensure_first_frame_bcfhw(
            _decode_validation_tensor(validation_batch, "first_frame_latent"))
        control_latent = _ensure_control_latent_bcfhw(
            _decode_validation_tensor(validation_batch, "control_latent"),
            latent_channels=int(first_frame_latent.shape[1]))
        if int(control_latent.shape[2]) != int(training_args.num_latent_t):
            raise ValueError(
                "Validation control_latent temporal length does not match num_latent_t: "
                f"control_latent.shape[2]={int(control_latent.shape[2])}, "
                f"num_latent_t={int(training_args.num_latent_t)}")

        device = get_local_torch_device()
        dtype = torch.bfloat16
        normalize_condition_latents = bool(
            getattr(training_args, "normalize_condition_latents", False))
        first_frame_latent = _normalize_first_frame_latent(
            first_frame_latent.to(device=device, dtype=dtype),
            self.vae,
            enabled=normalize_condition_latents)
        control_latent = _normalize_control_latent(
            control_latent.to(device=device, dtype=dtype),
            self.vae,
            int(training_args.pipeline_config.vae_config.arch_config.z_dim),
            enabled=normalize_condition_latents)

        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=self.validation_random_generator,
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
            first_frame_latent=first_frame_latent,
            control_latent=control_latent,
        )
        fps = validation_batch.get("fps")
        if fps is not None:
            batch.fps = int(fps)
        return batch

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        # Keep same validation pipeline as WanSelfForcingDistillationPipeline
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)
        args_copy.inference_mode = True
        if (hasattr(args_copy, "pipeline_config")
                and hasattr(args_copy.pipeline_config, "dit_config")
                and hasattr(args_copy.pipeline_config.dit_config,
                            "boundary_ratio")):
            args_copy.pipeline_config.dit_config.boundary_ratio = None
        if hasattr(args_copy, "pipeline_config"):
            # Align checkpoint validation rollout with tools/infer_wan_controlnet_ti2v.py
            # defaults for causal mode.
            if not hasattr(args_copy.pipeline_config,
                           "validation_update_rule"):
                args_copy.pipeline_config.validation_update_rule = "renoise_x0"
            if not hasattr(args_copy.pipeline_config,
                           "validation_full_schedule"):
                args_copy.pipeline_config.validation_full_schedule = False
            logger.info(
                "Validation rollout config: update_rule=%s full_schedule=%s timestep_indices=%s",
                getattr(args_copy.pipeline_config, "validation_update_rule",
                        "renoise_x0"),
                getattr(args_copy.pipeline_config, "validation_full_schedule",
                        False),
                getattr(args_copy.pipeline_config,
                        "validation_timestep_indices", []),
            )
        from fastvideo.pipelines.basic.wan.wan_controlnet_causal_dmd_pipeline import (
            WanControlnetCausalDMDPipeline)

        validation_pipeline = WanControlnetCausalDMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
                "transformer_2": self.get_module("transformer_2", None),
                "controlnet": self.get_module("controlnet"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=True)

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting Wan ControlNet self-forcing distillation pipeline...")
    pipeline = WanControlnetSelfForcingDistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("Wan ControlNet self-forcing distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
