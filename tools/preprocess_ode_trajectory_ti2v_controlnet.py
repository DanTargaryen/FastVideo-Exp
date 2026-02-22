#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Record Phase-1 ODE trajectories (TI2V + ControlNet) using FastVideo modules.

Inputs:
  - A FastVideo TI2V+ControlNet parquet dataset (from v1_preprocess_omnigame_ti2v_controlnet.py)
  - Teacher weights (transformer + controlnet)

Outputs:
  - Parquet dataset with ODE trajectories (pyarrow_schema_ode_trajectory_ti2v_controlnet)

Key behavior:
  - Teacher sampling is bidirectional by default (WanTransformer3DModel + WanControlnet3DModel).
  - Student uses causal masking (handled in training pipeline).
  - Uses FlowMatchEulerDiscreteScheduler on the 1000-step training grid and samples
    an evenly spaced K-step schedule (default K=50).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.record_schema import (
    ode_ti2v_controlnet_record_creator)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_ti2v_controlnet)
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed import (get_world_rank, get_world_size,
                                   maybe_init_distributed_environment_and_model_parallel)
from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs, WorkloadType
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.dits.controlnet_union_components import (
    WanControlNetUnionInput,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


def _compute_negative_prompt_embeddings(
    *,
    tokenizer,
    text_encoder,
    negative_prompt: str,
    max_sequence_length: int,
    dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    encoder_device = next(text_encoder.parameters()).device
    text_encoder.eval()
    with torch.no_grad():
        tokens = tokenizer(
            [negative_prompt],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(
            tokens.input_ids.to(encoder_device),
            tokens.attention_mask.to(encoder_device),
        ).last_hidden_state

    seq_len = int(tokens.attention_mask.sum(dim=1)[0].item())
    seq_len = min(seq_len, prompt_embeds.shape[1])
    prompt_embeds = prompt_embeds[:, :seq_len, :]
    if prompt_embeds.shape[1] < max_sequence_length:
        pad_len = max_sequence_length - prompt_embeds.shape[1]
        pad = prompt_embeds.new_zeros((1, pad_len, prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

    return prompt_embeds.to(dtype=dtype, device=target_device)


def _align_prompt_len(prompt_embeds: torch.Tensor,
                      ref_embeds: torch.Tensor) -> torch.Tensor:
    if prompt_embeds.shape[1] == ref_embeds.shape[1]:
        return prompt_embeds
    if prompt_embeds.shape[1] > ref_embeds.shape[1]:
        return prompt_embeds[:, :ref_embeds.shape[1], :]
    pad_len = ref_embeds.shape[1] - prompt_embeds.shape[1]
    pad = prompt_embeds.new_zeros((prompt_embeds.shape[0], pad_len,
                                   prompt_embeds.shape[2]))
    return torch.cat([prompt_embeds, pad], dim=1)


def _ensure_single_process_env() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _list_parquet_files(root: str | os.PathLike[str]) -> list[str]:
    root = str(root)
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No .parquet files found under: {root}")
    return paths


def _get_parquet_file_lengths(paths: list[str]) -> list[int]:
    lengths: list[int] = []
    for fp in paths:
        pf = pq.ParquetFile(fp)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        lengths.append(n)
    return lengths


def _load_existing_ids(parquet_rank_dir: str | os.PathLike[str]) -> set[str]:
    ids: set[str] = set()
    rank_dir = Path(parquet_rank_dir)
    if not rank_dir.is_dir():
        return ids
    for pf in sorted(rank_dir.rglob("*.parquet")):
        try:
            table = pq.read_table(str(pf), columns=["id"])
            for x in table.column("id").to_pylist():
                if x is not None:
                    ids.add(str(x))
        except Exception as e:
            logger.warning("Failed to read existing ids from %s: %s", pf, e)
    return ids


def _read_row_by_global_index(paths: list[str], lengths: list[int], index: int,
                              columns: list[str]) -> dict:
    if index < 0:
        raise ValueError("--start/--end must define non-negative indices")
    remaining = index
    for fp, n in zip(paths, lengths, strict=True):
        if remaining >= n:
            remaining -= n
            continue
        table = pq.read_table(fp, columns=columns)
        row = table.slice(remaining, 1).to_pydict()
        return {k: (v[0] if isinstance(v, list) else v) for k, v in row.items()}
    raise IndexError(f"Index {index} out of range for dataset")


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
    raise ValueError(f"Unsupported dtype in parquet: {dtype_str}")


def _decode_tensor(row: dict, prefix: str) -> torch.Tensor:
    shape = row.get(f"{prefix}_shape", None)
    blob = row.get(f"{prefix}_bytes", None)
    dtype_str = row.get(f"{prefix}_dtype", None)
    if blob is None or shape is None:
        raise KeyError(f"Missing {prefix}_bytes/{prefix}_shape in parquet row")
    arr = np.frombuffer(blob, dtype=_np_dtype(dtype_str)).reshape(shape).copy()
    return torch.from_numpy(arr)


def _ensure_text_embedding_bld(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"Unsupported text_embedding shape: {tuple(x.shape)}")


def _ensure_bcfhw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    """
    Ensure tensor is shaped [B, C, F, H, W].
    Accepts common variants:
      - [C, F, H, W]
      - [B, C, F, H, W]
      - [B, F, C, H, W]
    """
    channel_like = (1, 3, 16, 32, 48, 64)
    if x.dim() == 4:
        # [C, F, H, W] or [F, C, H, W]
        if x.shape[0] in channel_like and x.shape[1] > 4:
            return x.unsqueeze(0)
        if x.shape[1] in channel_like and x.shape[0] > 4:
            return x.permute(1, 0, 2, 3).contiguous().unsqueeze(0)
        return x.unsqueeze(0)
    if x.dim() == 5:
        # [B, C, F, H, W] or [B, F, C, H, W]
        if x.shape[1] in channel_like and x.shape[2] > 4:
            return x
        if x.shape[2] in channel_like and x.shape[1] > 4:
            return x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")


def _ensure_first_frame_bcfhw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.dim() == 4:
        x = x.unsqueeze(2)
    else:
        x = _ensure_bcfhw(x, name="first_frame_latent")
    if x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must have F==1, got shape={tuple(x.shape)}")
    return x


def _get_train_grid_timesteps(
        scheduler: FlowMatchEulerDiscreteScheduler,
        device: torch.device) -> torch.Tensor:
    ts = getattr(scheduler, "timesteps", None)
    num_train = int(
        getattr(getattr(scheduler, "config", None), "num_train_timesteps",
                1000))
    if ts is None or int(ts.numel()) < num_train:
        raise ValueError(
            "scheduler.timesteps is invalid; do not call scheduler.set_timesteps()"
        )
    return ts[:num_train].to(device=device, dtype=torch.float32)


def _build_schedule_timesteps(
        scheduler: FlowMatchEulerDiscreteScheduler,
        num_steps: int, device: torch.device) -> torch.Tensor:
    if num_steps <= 1:
        raise ValueError("--num_steps must be > 1")
    train_grid_ts = _get_train_grid_timesteps(scheduler, device)
    num_train = int(train_grid_ts.numel())
    idx = torch.linspace(0,
                         num_train - 1,
                         steps=int(num_steps),
                         device=device).round().long()
    return train_grid_ts.index_select(0, idx)


def _parse_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _is_union_controlnet(model) -> bool:
    return "union" in model.__class__.__name__.lower()


def _controlnet_dir_is_union(controlnet_dir: str) -> bool:
    p = Path(str(controlnet_dir))
    if "union" in p.name.lower():
        return True
    cfg = p / "config.json"
    if cfg.exists():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            cls_name = str(obj.get("_class_name", "")).lower()
            if "union" in cls_name:
                return True
        except Exception:
            pass
    return False


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Record ODE trajectories (TI2V + ControlNet) with FlowMatchEuler")
    p.add_argument("--base_model",
                   type=str,
                   required=True,
                   help="Wan diffusers root (for config)")
    p.add_argument("--transformer_dir",
                   type=str,
                   default="",
                   help="Teacher transformer weights dir (defaults to base_model/transformer)")
    p.add_argument("--controlnet_dir",
                   type=str,
                   required=True,
                   help="Teacher ControlNet weights dir")
    p.add_argument("--data_path",
                   type=str,
                   required=True,
                   help="Input TI2V+ControlNet parquet (no ODE)")
    p.add_argument("--out_dir",
                   type=str,
                   required=True,
                   help="Output parquet dir for ODE trajectories")
    p.add_argument("--num_steps",
                   type=int,
                   default=50,
                   help="Number of ODE steps to record (default: 50)")
    p.add_argument("--flow_shift",
                   type=float,
                   default=8.0,
                   help="FlowMatchEuler shift (must match training)")
    p.add_argument("--guidance_scale",
                   type=float,
                   default=1.0,
                   help="CFG scale for teacher sampling (1.0 disables CFG)")
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="bad quality, worst quality",
        help="Negative prompt text used when guidance_scale != 1.0.")
    p.add_argument("--teacher_mode",
                   type=str,
                   default="bidirectional",
                   choices=["bidirectional", "causal"],
                   help="Teacher attention mode for trajectory sampling")
    p.add_argument("--dtype",
                   type=str,
                   default="bf16",
                   choices=["bf16", "fp16", "fp32"],
                   help="Computation dtype for teacher forward")
    p.add_argument("--save_dtype",
                   type=str,
                   default="fp16",
                   choices=["fp16", "fp32"],
                   help="Storage dtype for trajectory latents")
    p.add_argument("--samples_per_file",
                   type=int,
                   default=8,
                   help="Rows per parquet file")
    p.add_argument("--flush_frequency",
                   type=int,
                   default=8,
                   help="How many samples to buffer before flush")
    p.add_argument("--start",
                   type=int,
                   default=0,
                   help="Start global index (inclusive)")
    p.add_argument("--end",
                   type=int,
                   default=-1,
                   help="End global index (exclusive; -1 means dataset end)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--append_clean_latent",
        action="store_true",
        help=(
            "Append clean GT latent (vae_latent) as the last trajectory state so that "
            "-2 is rollout-final and -1 is clean GT, matching Causal-Forcing."
        ),
    )
    p.add_argument(
        "--trajectory_indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated trajectory indices to keep, e.g. "
            "'0,12,24,36,-2,-1'. Applied after optional clean appending."
        ),
    )
    p.add_argument(
        "--skip_existing_ids",
        action="store_true",
        help=
        "Resume mode: skip samples whose id already exists in out_dir/rank_xx parquet files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_single_process_env()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    rank = get_world_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = _parse_dtype(args.dtype)
    save_dtype = np.float16 if str(args.save_dtype).lower() in ("fp16",
                                                                "float16") else np.float32

    base_model = str(args.base_model)
    transformer_dir = args.transformer_dir or str(
        Path(base_model) / "transformer")

    pipeline_config = PipelineConfig.from_pretrained(base_model)
    fastvideo_args = FastVideoArgs(
        model_path=base_model,
        mode=ExecutionMode.INFERENCE,
        workload_type=WorkloadType.I2V,
        pipeline_config=pipeline_config,
        inference_mode=True,
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=True,
    )

    fastvideo_args.pipeline_config.dit_precision = str(args.dtype).lower()
    fastvideo_args.pipeline_config.flow_shift = float(args.flow_shift)
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None

    use_union_controlnet = _controlnet_dir_is_union(args.controlnet_dir)
    if args.teacher_mode == "bidirectional":
        fastvideo_args.override_transformer_cls_name = "WanTransformer3DModel"
        fastvideo_args.override_controlnet_cls_name = (
            "WanControlnetUnion3DModel"
            if use_union_controlnet else "WanControlnet3DModel")
    else:
        fastvideo_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
        fastvideo_args.override_controlnet_cls_name = (
            "CausalWanControlnetUnion3DModel"
            if use_union_controlnet else "CausalWanControlnet3DModel")

    transformer = PipelineComponentLoader.load_module(
        "transformer", transformer_dir, "diffusers", fastvideo_args)
    controlnet = PipelineComponentLoader.load_module(
        "controlnet", args.controlnet_dir, "diffusers", fastvideo_args)

    transformer.eval()
    controlnet.eval()

    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=float(args.flow_shift))
    timesteps_1d = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas_1d = scheduler.sigmas.to(device=device, dtype=torch.float32)
    guidance_scale = float(args.guidance_scale)
    negative_prompt_text = str(args.negative_prompt or "").strip()
    negative_prompt_embeds: torch.Tensor | None = None
    if guidance_scale != 1.0:
        if negative_prompt_text:
            tokenizer = PipelineComponentLoader.load_module(
                "tokenizer",
                str(Path(base_model) / "tokenizer"),
                "transformers",
                fastvideo_args,
            )
            text_encoder = PipelineComponentLoader.load_module(
                "text_encoder",
                str(Path(base_model) / "text_encoder"),
                "transformers",
                fastvideo_args,
            )
            max_text_len = int(getattr(transformer, "text_len", 226))
            negative_prompt_embeds = _compute_negative_prompt_embeddings(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                negative_prompt=negative_prompt_text,
                max_sequence_length=max_text_len,
                dtype=dtype,
                target_device=device,
            )
            del tokenizer, text_encoder
        else:
            logger.warning(
                "--guidance_scale != 1.0 but --negative_prompt is empty; falling back to zeros."
            )

    # Load parquet indices
    parquet_files = _list_parquet_files(args.data_path)
    lengths = _get_parquet_file_lengths(parquet_files)
    total = int(sum(lengths))
    start = max(int(args.start), 0)
    end = total if int(args.end) < 0 else min(int(args.end), total)

    out_dir_rank = os.path.join(args.out_dir, f"rank_{rank:02d}")
    writer = ParquetDatasetWriter(out_dir_rank,
                                  samples_per_file=int(args.samples_per_file))
    buffer: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if bool(args.skip_existing_ids):
        existing_ids = _load_existing_ids(out_dir_rank)
        logger.info(
            "Resume mode enabled on rank_%02d: found %d existing ids in %s",
            rank,
            len(existing_ids),
            out_dir_rank,
        )

    # Distribute indices across ranks
    indices = list(range(start, end))
    if world_size > 1:
        indices = indices[rank::world_size]

    torch.manual_seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    selected_indices: list[int] | None = None
    if str(args.trajectory_indices).strip() != "":
        selected_indices = [
            int(x.strip()) for x in str(args.trajectory_indices).split(",")
            if x.strip() != ""
        ]

    cols = [
        "id",
        "caption",
        "fps",
        "width",
        "height",
        "num_frames",
        "text_embedding_bytes",
        "text_embedding_shape",
        "text_embedding_dtype",
        "first_frame_latent_bytes",
        "first_frame_latent_shape",
        "first_frame_latent_dtype",
        "control_latent_bytes",
        "control_latent_shape",
        "control_latent_dtype",
    ]
    if bool(args.append_clean_latent):
        cols.extend([
            "vae_latent_bytes",
            "vae_latent_shape",
            "vae_latent_dtype",
        ])

    for idx in indices:
        row = _read_row_by_global_index(parquet_files, lengths, idx, cols)
        sample_id = str(row.get("id", f"sample_{idx:06d}"))
        if bool(args.skip_existing_ids) and sample_id in existing_ids:
            continue
        caption = str(row.get("caption", ""))
        fps = float(row.get("fps", 0.0))
        width = int(row.get("width", 0))
        height = int(row.get("height", 0))
        num_frames = int(row.get("num_frames", 0))

        text_embedding = _ensure_text_embedding_bld(
            _decode_tensor(row, "text_embedding")).to(device=device,
                                                      dtype=dtype)
        first_frame_latent = _ensure_first_frame_bcfhw(
            _decode_tensor(row, "first_frame_latent")).to(device=device,
                                                          dtype=dtype)
        control_latent = _ensure_bcfhw(_decode_tensor(row, "control_latent"),
                                       name="control_latent").to(
                                           device=device, dtype=dtype)
        num_channels_latents = int(transformer.num_channels_latents)

        latent_t = int(control_latent.shape[2])
        latent_h = int(control_latent.shape[3])
        latent_w = int(control_latent.shape[4])
        num_steps = int(args.num_steps)

        t_list = _build_schedule_timesteps(scheduler, num_steps, device)
        expand_timesteps = bool(
            getattr(fastvideo_args.pipeline_config, "expand_timesteps", False))
        if args.teacher_mode == "causal" and expand_timesteps:
            logger.info(
                "teacher_mode=causal: using frame-level timesteps for ControlNet/Transformer (disable token-level expanded timesteps)."
            )
        image_latents = first_frame_latent.to(
            device=device, dtype=dtype) if first_frame_latent is not None else None
        first_frame_mask = torch.ones(
            (1, 1, latent_t, latent_h, latent_w),
            device=device,
            dtype=torch.float32,
        )
        if image_latents is not None:
            first_frame_mask[:, :, 0] = 0
        patch_size = getattr(
            getattr(transformer.config, "arch_config", None),
            "patch_size",
            (2, 2),
        )
        patch_h = int(patch_size[-2])
        patch_w = int(patch_size[-1])

        # Prepare forward context (needed by attention backends)
        forward_batch = ForwardBatch(data_type="ti2v_controlnet")
        forward_batch.prompt_embeds = [text_embedding]
        forward_batch.height = int(height) if height > 0 else 0
        forward_batch.width = int(width) if width > 0 else 0
        forward_batch.num_frames = int(latent_t)

        # Init noise
        current_latents = torch.randn(
            (1, transformer.num_channels_latents, latent_t, latent_h,
             latent_w),
            device=device,
            dtype=dtype,
        )
        current_latents[:, :, :1] = first_frame_latent.to(device=device,
                                                          dtype=dtype)

        traj_latents: list[torch.Tensor] = []
        traj_latents.append(current_latents.detach().clone())

        def _predict_flow_at_t(t_cur: torch.Tensor, step_index: int) -> torch.Tensor:
            if expand_timesteps:
                if image_latents is None:
                    latent_model_input = current_latents
                else:
                    latent_model_input = (
                        (1 - first_frame_mask) * image_latents +
                        first_frame_mask * current_latents
                    )
                temp_ts = (first_frame_mask[0, 0] * float(t_cur))
                temp_ts = temp_ts[:, ::patch_h, ::patch_w].flatten()
                timestep = temp_ts.unsqueeze(0).expand(
                    latent_model_input.shape[0], -1)
                if args.teacher_mode == "causal":
                    # Causal Union ControlNet expects frame-level timesteps [B, F].
                    # Token-level expanded timesteps (e.g., [B, F*H*W]) cause
                    # mismatch when fusing global control embeddings.
                    timestep = torch.full((latent_model_input.shape[0], latent_t),
                                          float(t_cur),
                                          device=device,
                                          dtype=torch.float32)
            else:
                if image_latents is None:
                    latent_model_input = current_latents
                else:
                    latent_model_input = torch.cat(
                        [current_latents, image_latents], dim=1)
                timestep = torch.full((latent_model_input.shape[0],),
                                      float(t_cur),
                                      device=device,
                                      dtype=torch.float32)
            latent_model_input = latent_model_input.to(device=device,
                                                       dtype=dtype)

            with set_forward_context(current_timestep=int(step_index),
                                     attn_metadata=None,
                                     forward_batch=forward_batch):
                control_res = controlnet(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=[text_embedding],
                    timestep=timestep,
                    **_build_controlnet_kwargs(controlnet, control_latent,
                                               num_channels_latents),
                )
                pred_flow = transformer(
                    latent_model_input,
                    [text_embedding],
                    timestep,
                    block_controlnet_hidden_states=control_res,
                ).permute(0, 2, 1, 3, 4)

            if guidance_scale == 1.0:
                return pred_flow

            if negative_prompt_embeds is not None:
                negative = _align_prompt_len(negative_prompt_embeds,
                                             text_embedding)
            else:
                negative = torch.zeros_like(text_embedding)
            with set_forward_context(current_timestep=int(step_index),
                                     attn_metadata=None,
                                     forward_batch=forward_batch):
                control_res_uncond = controlnet(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=[negative],
                    timestep=timestep,
                    **_build_controlnet_kwargs(controlnet, control_latent,
                                               num_channels_latents),
                )
                pred_uncond = transformer(
                    latent_model_input,
                    [negative],
                    timestep,
                    block_controlnet_hidden_states=control_res_uncond,
                ).permute(0, 2, 1, 3, 4)
            return pred_uncond + guidance_scale * (pred_flow - pred_uncond)

        # Euler update across timesteps
        for step_i in range(int(t_list.numel()) - 1):
            t_cur = t_list[step_i]
            t_next = t_list[step_i + 1]
            pred_flow_btchw = _predict_flow_at_t(t_cur, step_index=step_i)

            idx_cur = torch.argmin((timesteps_1d - t_cur.float()).abs())
            idx_next = torch.argmin((timesteps_1d - t_next.float()).abs())
            sigma_cur = sigmas_1d[idx_cur]
            sigma_next = sigmas_1d[idx_next]
            dt = (sigma_next - sigma_cur).to(dtype=pred_flow_btchw.dtype)

            current_latents = current_latents + dt * pred_flow_btchw.permute(
                0, 2, 1, 3, 4).contiguous()
            current_latents[:, :, :1] = first_frame_latent.to(device=device,
                                                              dtype=dtype)
            traj_latents.append(current_latents.detach().clone())

        traj_tensor = torch.stack(traj_latents, dim=0).squeeze(1).to(
            dtype=torch.float32)
        t_tensor = t_list.to(dtype=torch.float32)

        # Causal-Forcing alignment: append clean GT latent as the last state.
        if bool(args.append_clean_latent):
            clean_latent = _ensure_bcfhw(
                _decode_tensor(row, "vae_latent"), name="vae_latent"
            ).to(device=device, dtype=torch.float32)
            if clean_latent.shape[0] != 1:
                raise ValueError(
                    f"Expected batch=1 vae_latent for ODE preprocessing, got {tuple(clean_latent.shape)}"
                )
            # traj_tensor: [S, C, F, H, W], clean_latent[0]: [C, F, H, W]
            traj_tensor = torch.cat([traj_tensor, clean_latent[0:1]], dim=0)
            # Keep timesteps aligned in length; the clean target is t=0.
            t_tensor = torch.cat([
                t_tensor,
                torch.tensor([0.0], device=t_tensor.device, dtype=t_tensor.dtype),
            ], dim=0)

        if selected_indices is not None:
            n_states = int(traj_tensor.shape[0])
            idx = torch.tensor(
                [(i + n_states) if i < 0 else i for i in selected_indices],
                device=traj_tensor.device,
                dtype=torch.long,
            )
            if torch.any(idx < 0) or torch.any(idx >= n_states):
                raise ValueError(
                    f"trajectory_indices out of range for {n_states} states: {selected_indices}"
                )
            traj_tensor = traj_tensor.index_select(0, idx)
            t_tensor = t_tensor.index_select(0, idx)

        traj_np = traj_tensor.cpu().numpy().astype(save_dtype, copy=False)
        t_np = t_tensor.cpu().numpy()

        record = ode_ti2v_controlnet_record_creator(
            video_name=sample_id,
            text_embedding=text_embedding[0].to(dtype=torch.float32).cpu().numpy(),
            caption=caption,
            first_frame_latent=_decode_tensor(row, "first_frame_latent").numpy(),
            control_latent=_decode_tensor(row, "control_latent").numpy(),
            trajectory_latents=traj_np,
            trajectory_timesteps=t_np,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
        )
        buffer.append(record)
        if bool(args.skip_existing_ids):
            existing_ids.add(sample_id)

        if len(buffer) >= int(args.flush_frequency):
            table = records_to_table(buffer,
                                     pyarrow_schema_ode_trajectory_ti2v_controlnet)
            writer.append_table(table)
            writer.flush(num_workers=1, write_remainder=False)
            buffer = []

    if buffer:
        table = records_to_table(buffer,
                                 pyarrow_schema_ode_trajectory_ti2v_controlnet)
        writer.append_table(table)
        writer.flush(num_workers=1, write_remainder=True)

    logger.info("Done. Wrote ODE parquet dataset to %s", args.out_dir)


if __name__ == "__main__":
    main()
