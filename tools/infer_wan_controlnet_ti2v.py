#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
TI2V inference for *student* Wan (causal) + ControlNet using exported weight-only checkpoints.

This script:
1) loads student transformer + controlnet from FastVideo weight-only exports
2) reads one (or more) samples from the TI2V+ControlNet parquet dataset
3) runs chunk-wise causal DMD rollout (same re-noising scheme as training's simulation)
4) decodes latents with VAE and saves mp4(s)

Typical usage (single node):

  cd FastVideo
  export PYTHONPATH=$PWD
  export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

  BASE_MODEL=/path/to/Wan2.2-TI2V-5B-Diffusers
  DATA=/path/to/omnigame_ti2v_controlnet_parquet
  CKPT=outputs/wan_controlnet_self_forcing_phase2/checkpoint-800_weight_only

  python tools/infer_wan_controlnet_ti2v.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --transformer_dir "$CKPT/generator_inference_transformer" \
    --controlnet_dir "$CKPT/generator_inference_controlnet" \
    --index 0 \
    --out_dir outputs/infer_ckpt800
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage

logger = init_logger(__name__)


def _ensure_single_process_dist_env() -> None:
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


def _read_row_by_global_index(data_path: str, index: int,
                              columns: list[str]) -> dict:
    if index < 0:
        raise ValueError("--index must be >= 0")
    remaining = index
    for fp in _list_parquet_files(data_path):
        pf = pq.ParquetFile(fp)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        if remaining >= n:
            remaining -= n
            continue
        table = pq.read_table(fp, columns=columns)
        row = table.slice(remaining, 1).to_pydict()
        return {k: (v[0] if isinstance(v, list) else v) for k, v in row.items()}
    raise IndexError(f"--index {index} out of range for dataset: {data_path}")


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
    if x.dim() == 4:
        return x.unsqueeze(0)
    if x.dim() == 5:
        # BFCHW -> BCFHW
        if x.shape[1] in (1, 3) and x.shape[2] >= 8:
            return x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")


def _ensure_first_frame_bcfhw(x: torch.Tensor) -> torch.Tensor:
    # Common variants:
    # - [C, H, W]
    # - [B, C, H, W]
    # - [B, F, C, H, W] (Diff-Factory stores BFCHW)
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
    elif x.dim() == 4:
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
    else:
        x = _ensure_bcfhw(x, name="first_frame_latent")
    # Ensure F==1
    if x.shape[2] != 1:
        raise ValueError(
            f"first_frame_latent must have F==1, got shape={tuple(x.shape)}")
    return x


@dataclass(frozen=True)
class Sample:
    sample_id: str
    caption: str
    fps: int
    text_embedding_bld: torch.Tensor  # [B, L, D]
    first_frame_latent_bcfhw: torch.Tensor  # [B, C, 1, H, W]
    control_latent_bcfhw: torch.Tensor  # [B, 3*C, F, H, W]


def _load_sample(data_path: str, index: int) -> Sample:
    cols = [
        "id",
        "caption",
        "fps",
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
    row = _read_row_by_global_index(data_path, index, cols)
    text_embedding = _ensure_text_embedding_bld(_decode_tensor(row,
                                                               "text_embedding"))
    first_frame_latent = _ensure_first_frame_bcfhw(
        _decode_tensor(row, "first_frame_latent"))
    control_latent = _ensure_bcfhw(_decode_tensor(row, "control_latent"),
                                   name="control_latent")

    sample_id = str(row.get("id", f"index_{index:06d}"))
    caption = str(row.get("caption", ""))
    fps_val = int(row.get("fps", 30) or 30)
    return Sample(sample_id=sample_id,
                  caption=caption,
                  fps=fps_val,
                  text_embedding_bld=text_embedding,
                  first_frame_latent_bcfhw=first_frame_latent,
                  control_latent_bcfhw=control_latent)


def _initialize_kv_cache(*, model, batch_size: int, dtype: torch.dtype,
                         device: torch.device,
                         frame_seq_length: int) -> list[dict]:
    num_blocks = len(model.blocks)
    # Different model variants expose these attributes differently.
    # - CausalWanTransformer3DModel / CausalWanControlnet3DModel: has `num_attention_heads` + `attention_head_dim`
    # - WanTransformer3DModel: has `num_attention_heads` but not `attention_head_dim`
    num_heads = getattr(model, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(getattr(getattr(model, "config", None), "arch_config", None),
                            "num_attention_heads", None)
    if num_heads is None:
        raise AttributeError(f"Cannot determine num_attention_heads for {type(model).__name__}")
    num_heads = int(num_heads)

    head_dim = getattr(model, "attention_head_dim", None)
    if head_dim is None:
        head_dim = getattr(getattr(model, "config", None), "attention_head_dim", None)
    if head_dim is None:
        head_dim = getattr(getattr(getattr(model, "config", None), "arch_config", None),
                           "attention_head_dim", None)
    if head_dim is None:
        # Fallback: infer from inner dim if possible.
        inner_dim = getattr(model, "inner_dim", None)
        if inner_dim is None:
            inner_dim = getattr(model, "hidden_size", None)
        if inner_dim is None:
            raise AttributeError(f"Cannot determine attention_head_dim for {type(model).__name__}")
        head_dim = int(inner_dim) // int(num_heads)
    head_dim = int(head_dim)

    local_attn_size = getattr(model, "local_attn_size", -1)
    sliding_window_num_frames = model.config.arch_config.sliding_window_num_frames
    if local_attn_size != -1:
        kv_cache_size = local_attn_size * frame_seq_length
    else:
        kv_cache_size = frame_seq_length * sliding_window_num_frames

    cache: list[dict] = []
    for _ in range(num_blocks):
        cache.append({
            "k":
            torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device),
            "v":
            torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device),
            "global_end_index":
            torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index":
            torch.tensor([0], dtype=torch.long, device=device),
        })
    return cache


def _initialize_crossattn_cache(*, model, batch_size: int, max_text_len: int,
                                dtype: torch.dtype,
                                device: torch.device) -> list[dict]:
    num_blocks = len(model.blocks)
    num_heads = model.num_attention_heads
    head_dim = model.attention_head_dim
    cache: list[dict] = []
    for _ in range(num_blocks):
        cache.append({
            "k":
            torch.zeros([batch_size, max_text_len, num_heads, head_dim],
                        dtype=dtype,
                        device=device),
            "v":
            torch.zeros([batch_size, max_text_len, num_heads, head_dim],
                        dtype=dtype,
                        device=device),
            "is_init":
            False,
        })
    return cache


@torch.no_grad()
def _causal_dmd_rollout_ti2v_controlnet(
    *,
    transformer,
    controlnet,
    scheduler,
    prompt_embeds_list: list[torch.Tensor],
    first_frame_latent_bcfhw: torch.Tensor | None,
    control_latent_bcfhw: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    dmd_steps: list[int],
    context_noise: int,
    warp_denoising_step: bool,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns: latents [B, C, T_lat, H_lat, W_lat] (BCFHW) where T_lat is latent frames (e.g. 21).
    """
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    generator = torch.Generator(device=device).manual_seed(seed)

    # Make sure scheduler has the 1000-step grid used by warp_denoising_step mapping.
    if hasattr(scheduler, "set_timesteps"):
        # diffusers schedulers have slightly different `set_timesteps` signatures
        # across versions; be permissive here.
        try:
            scheduler.set_timesteps(1000, device=device)  # type: ignore[arg-type]
        except TypeError:
            try:
                scheduler.set_timesteps(1000)
            except TypeError:
                # Some schedulers expect `num_inference_steps` as a named arg.
                scheduler.set_timesteps(num_inference_steps=1000)  # type: ignore[call-arg]

    # Build a minimal ForwardBatch for forward_context bookkeeping.
    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = prompt_embeds_list
    batch.height = height
    batch.width = width
    batch.num_frames = num_frames
    batch.generator = generator

    # Latent shape: derive from ControlNet latent (already VAE-encoded) to avoid
    # guessing VAE compression ratios.
    latent_t = int(control_latent_bcfhw.shape[2])
    latent_h = int(control_latent_bcfhw.shape[3])
    latent_w = int(control_latent_bcfhw.shape[4])
    if first_frame_latent_bcfhw is not None:
        if (first_frame_latent_bcfhw.shape[3] != latent_h
                or first_frame_latent_bcfhw.shape[4] != latent_w):
            raise ValueError(
                "first_frame_latent and control_latent spatial sizes mismatch: "
                f"first_frame={tuple(first_frame_latent_bcfhw.shape)} control={tuple(control_latent_bcfhw.shape)}"
            )

    latents = torch.randn((1, transformer.num_channels_latents, latent_t,
                           latent_h, latent_w),
                          generator=generator,
                          device=device,
                          dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    # DMD timestep list in scheduler coordinate (float) if warp enabled.
    t_list = torch.tensor(dmd_steps, dtype=torch.long).cpu()
    if warp_denoising_step:
        scheduler_timesteps = torch.cat(
            (scheduler.timesteps.cpu(), torch.tensor([0.0])))
        t_list = scheduler_timesteps[1000 - t_list]
    t_list = t_list.to(device=device)

    num_frames_per_block = transformer.config.arch_config.num_frames_per_block
    if latents.shape[2] % num_frames_per_block != 0:
        raise ValueError(
            f"latent_t={latents.shape[2]} must be divisible by num_frames_per_block={num_frames_per_block}"
        )

    # Frame token count per latent frame (used for kv_cache offsets)
    latent_seq_length = latents.shape[-1] * latents.shape[-2]
    patch_ratio = transformer.config.arch_config.patch_size[
        -1] * transformer.config.arch_config.patch_size[-2]
    frame_seq_length = latent_seq_length // patch_ratio

    # Allocate caches (separate for transformer and controlnet)
    kv_cache = _initialize_kv_cache(model=transformer,
                                    batch_size=1,
                                    dtype=dtype,
                                    device=device,
                                    frame_seq_length=frame_seq_length)
    crossattn_cache = _initialize_crossattn_cache(model=transformer,
                                                  batch_size=1,
                                                  max_text_len=transformer.text_len,
                                                  dtype=dtype,
                                                  device=device)
    control_kv_cache = _initialize_kv_cache(model=controlnet,
                                            batch_size=1,
                                            dtype=dtype,
                                            device=device,
                                            frame_seq_length=frame_seq_length)
    control_crossattn_cache = _initialize_crossattn_cache(
        model=controlnet,
        batch_size=1,
        max_text_len=controlnet.text_len,
        dtype=dtype,
        device=device)

    # Main causal chunk loop
    num_blocks = latents.shape[2] // num_frames_per_block
    start_index = 0
    for _block_idx in range(num_blocks):
        current_num_frames = num_frames_per_block
        current_latents = latents[:, :, start_index:start_index +
                                  current_num_frames].contiguous()

        control_chunk = control_latent_bcfhw[:, :, start_index:start_index +
                                             current_num_frames].to(device=device,
                                                                    dtype=dtype)

        # DMD steps: pred -> x0 -> re-noise to next anchor
        for step_i, t_cur in enumerate(t_list):
            # Conversion utilities expect BTCHW
            noise_latents_btchw = current_latents.permute(0, 2, 1, 3,
                                                          4).contiguous()
            latent_model_input = current_latents

            if first_frame_latent_bcfhw is not None and start_index == 0:
                latent_model_input = latent_model_input.clone()
                latent_model_input[:, :, :1] = first_frame_latent_bcfhw.to(
                    device=device, dtype=dtype)

            t_expanded = t_cur * torch.ones((1, 1),
                                            device=device,
                                            dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=dtype,
                                enabled=(dtype != torch.float32)), \
                    set_forward_context(current_timestep=int(step_i),
                                        attn_metadata=None,
                                        forward_batch=batch):
                control_res = controlnet(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds_list,
                    timestep=t_expanded,
                    controlnet_states=control_chunk,
                    kv_cache=control_kv_cache,
                    crossattn_cache=control_crossattn_cache,
                    current_start=start_index * frame_seq_length,
                    start_frame=start_index,
                )

                pred_noise_btchw = transformer(
                    latent_model_input,
                    prompt_embeds_list,
                    t_expanded,
                    block_controlnet_hidden_states=control_res,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start_index * frame_seq_length,
                    start_frame=start_index,
                ).permute(0, 2, 1, 3, 4)

            t_expand_1d = t_cur.repeat(pred_noise_btchw.shape[0])
            pred_video_btchw = pred_noise_to_pred_video(
                pred_noise=pred_noise_btchw.flatten(0, 1),
                noise_input_latent=noise_latents_btchw.flatten(0, 1),
                timestep=t_expand_1d,
                scheduler=scheduler,
            ).unflatten(0, pred_noise_btchw.shape[:2])

            if step_i < len(t_list) - 1:
                next_t = t_list[step_i + 1] * torch.ones([1],
                                                         device=device,
                                                         dtype=torch.float32)
                noise = torch.randn_like(pred_video_btchw,
                                         generator=generator)
                noise_latents_btchw = scheduler.add_noise(
                    pred_video_btchw.flatten(0, 1), noise.flatten(0, 1),
                    next_t).unflatten(0, pred_video_btchw.shape[:2])
                current_latents = noise_latents_btchw.permute(
                    0, 2, 1, 3, 4).contiguous()
            else:
                current_latents = pred_video_btchw.permute(0, 2, 1, 3,
                                                           4).contiguous()

            if first_frame_latent_bcfhw is not None and start_index == 0:
                current_latents[:, :, :1] = first_frame_latent_bcfhw.to(
                    device=device, dtype=dtype)

        # Write back x0 chunk
        latents[:, :, start_index:start_index + current_num_frames] = current_latents

        # Cache update with optional context noise (Self-Forcing style)
        context_t = torch.ones((1, 1),
                               device=device,
                               dtype=torch.float32) * float(context_noise)
        context_btchw = current_latents.permute(0, 2, 1, 3, 4).contiguous()
        if context_noise != 0:
            context_btchw = scheduler.add_noise(
                context_btchw.flatten(0, 1),
                torch.randn_like(context_btchw.flatten(0, 1),
                                 generator=generator), context_t).unflatten(
                                     0, context_btchw.shape[:2])
        context_bcfhw = context_btchw.permute(0, 2, 1, 3, 4).contiguous()

        if first_frame_latent_bcfhw is not None and start_index == 0:
            context_bcfhw = context_bcfhw.clone()
            context_bcfhw[:, :, :1] = first_frame_latent_bcfhw.to(device=device,
                                                                  dtype=dtype)

        with torch.autocast(device_type="cuda", dtype=dtype,
                            enabled=(dtype != torch.float32)), \
                set_forward_context(current_timestep=0,
                                    attn_metadata=None,
                                    forward_batch=batch):
            control_res_ctx = controlnet(
                hidden_states=context_bcfhw,
                encoder_hidden_states=prompt_embeds_list,
                timestep=context_t,
                controlnet_states=control_chunk,
                kv_cache=control_kv_cache,
                crossattn_cache=control_crossattn_cache,
                current_start=start_index * frame_seq_length,
                start_frame=start_index,
            )
            _ = transformer(
                context_bcfhw,
                prompt_embeds_list,
                context_t,
                block_controlnet_hidden_states=control_res_ctx,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start_index * frame_seq_length,
                start_frame=start_index,
            )

        start_index += current_num_frames

    return latents


def _save_mp4(frames_bcthw: torch.Tensor, out_path: str, fps: int) -> None:
    import imageio

    frames = frames_bcthw[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
    frames_u8 = (frames * 255.0).round().astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, list(frames_u8), fps=fps, format="mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--transformer_dir", type=str, required=True)
    parser.add_argument("--controlnet_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dmd_steps",
                        type=str,
                        default="1000,750,500,250",
                        help="Comma-separated DMD anchors in 0..1000 grid.")
    parser.add_argument("--context_noise",
                        type=int,
                        default=0,
                        help="Context timestep used for cache update (0 = clean).")
    parser.add_argument("--warp_denoising_step",
                        action="store_true",
                        default=True,
                        help="Match training: map 0..1000 DMD grid to scheduler.timesteps.")
    parser.add_argument("--dtype",
                        type=str,
                        default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    _ensure_single_process_dist_env()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit(
            f"This inference script is single-process only (WORLD_SIZE={world_size}). "
            "Run with `torchrun --standalone --nproc_per_node=1 ...` or plain `python ...`."
        )
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]

    dmd_steps = [int(x) for x in args.dmd_steps.split(",") if x.strip() != ""]

    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=args.base_model,
        mode="inference",
        workload_type="i2v",
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
    # Ensure student rollout uses the chunk-wise causal transformer (KV cache),
    # even if the exported config.json says "WanTransformer3DModel".
    fastvideo_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    # Ensure we don't trigger Wan2.2 "transformer_2" boundary logic for TI2V.
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    fastvideo_args.pipeline_config.warp_denoising_step = True
    fastvideo_args.pipeline_config.dmd_denoising_steps = dmd_steps
    fastvideo_args.pipeline_config.context_noise = int(args.context_noise)

    transformer = PipelineComponentLoader.load_module(
        "transformer", args.transformer_dir, "diffusers", fastvideo_args)
    controlnet = PipelineComponentLoader.load_module(
        "controlnet", args.controlnet_dir, "diffusers", fastvideo_args)
    scheduler = PipelineComponentLoader.load_module(
        "scheduler", str(Path(args.base_model) / "scheduler"), "diffusers",
        fastvideo_args)
    vae = PipelineComponentLoader.load_module("vae",
                                              str(Path(args.base_model) / "vae"),
                                              "diffusers", fastvideo_args)

    decoding = DecodingStage(vae=vae)

    for i in range(args.num_samples):
        sample_idx = args.index + i
        sample = _load_sample(args.data_path, sample_idx)
        logger.info("sample=%s idx=%s caption=%s", sample.sample_id, sample_idx,
                    sample.caption)

        prompt_embeds = sample.text_embedding_bld.to(device="cuda",
                                                     dtype=dtype)
        first_frame_latent = sample.first_frame_latent_bcfhw.to(device="cuda",
                                                                dtype=dtype)
        control_latent = sample.control_latent_bcfhw.to(device="cuda",
                                                        dtype=dtype)

        latents = _causal_dmd_rollout_ti2v_controlnet(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            prompt_embeds_list=[prompt_embeds],
            first_frame_latent_bcfhw=first_frame_latent,
            control_latent_bcfhw=control_latent,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            dmd_steps=dmd_steps,
            context_noise=args.context_noise,
            warp_denoising_step=True,
            seed=args.seed + i,
            dtype=dtype,
        )

        # Decode to pixels [B, C, T, H, W] in [0,1], then save mp4
        decoded = decoding.decode(latents, fastvideo_args).cpu().float()
        fps = int(sample.fps or args.fps)
        out_path = str(Path(args.out_dir) / f"{sample.sample_id}.mp4")
        _save_mp4(decoded, out_path, fps=fps)
        logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
