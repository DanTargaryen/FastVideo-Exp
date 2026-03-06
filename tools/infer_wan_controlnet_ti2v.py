#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
TI2V inference for *student* Wan (causal) + ControlNet using exported weight-only checkpoints.

This script:
1) loads student transformer + controlnet (either from FastVideo weight-only exports, or by
   loading the base diffusers configs and overriding weights with `*_init.safetensors`)
2) reads one (or more) samples from the TI2V+ControlNet parquet dataset
3) runs chunk-wise causal rollout with FlowMatchEulerDiscreteScheduler (Euler ODE) on a few-step anchor grid
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

Phase-1 init weights (no diffusers-format model folder; use base model for config + override weights):

  BASE_MODEL=/path/to/Wan2.2-TI2V-5B-Diffusers
  TEACHER_CONTROLNET=/path/to/world-renderer-controlnet-warp-mask
  INIT_DIR=/path/to/phase1_fastvideo_inits

  python tools/infer_wan_controlnet_ti2v.py \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --transformer_dir "$BASE_MODEL/transformer" \
    --controlnet_dir "$TEACHER_CONTROLNET" \
    --init_transformer_safetensors "$INIT_DIR/transformer_init.safetensors" \
    --init_controlnet_safetensors "$INIT_DIR/controlnet_init.safetensors" \
    --index 0 \
    --out_dir outputs/infer_phase1_init
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers import UniPCMultistepScheduler as DiffusersUniPCMultistepScheduler
from PIL import Image

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.controlnet_union_components import WanControlNetUnionInput
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage

logger = init_logger(__name__)


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
        except Exception as exc:
            logger.warning(
                "Failed to parse %s for union controlnet detection: %s",
                str(cfg),
                str(exc),
            )
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


def _tensor_or_list_l2(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (list, tuple)):
        sq_sum = 0.0
        for t in x:
            if t is None:
                continue
            sq_sum += float((t.float() * t.float()).sum().item())
        return float(max(sq_sum, 0.0)**0.5)
    return float(torch.linalg.vector_norm(x.float()).item())


def _scale_control_residual(control_res, scale: float):
    if control_res is None:
        return None
    s = float(scale)
    if isinstance(control_res, (list, tuple)):
        return [x * s for x in control_res]
    return control_res * s


def _append_trace_jsonl(path: str | None, record: dict) -> None:
    if path is None or str(path).strip() == "":
        return
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _log_causal_attn_overrides(model: torch.nn.Module, *, name: str) -> None:
    count = 0
    sample = None
    for m in model.modules():
        if m.__class__.__name__ != "CausalWanSelfAttention":
            continue
        count += 1
        if sample is None:
            sample = {
                "local_attn_size": getattr(m, "local_attn_size", None),
                "sink_size": getattr(m, "sink_size", None),
                "max_attention_size": getattr(m, "max_attention_size", None),
            }
    logger.info(
        "%s causal attn: modules=%s model.local_attn_size=%s sample_layer=%s",
        name,
        count,
        getattr(model, "local_attn_size", None),
        sample,
    )

def _override_local_attn_size(model: torch.nn.Module, local_attn_size: int) -> None:
    """
    Force causal self-attention to use a sliding local window (in *latent frames*).

    IMPORTANT:
    - KV cache allocation in this script consults `model.local_attn_size` (top-level).
    - Eviction/windowing during forward consults each `CausalWanSelfAttention.local_attn_size`.
      So we must set BOTH for the override to take effect.
    """
    if int(local_attn_size) <= 0:
        return

    # Top-level attribute used by KV cache initialization.
    try:
        setattr(model, "local_attn_size", int(local_attn_size))
    except Exception:
        pass

    # Per-block attention modules used during forward.
    for m in model.modules():
        if m.__class__.__name__ != "CausalWanSelfAttention":
            continue
        try:
            m.local_attn_size = int(local_attn_size)
            # Keep logic consistent with module __init__.
            if hasattr(m, "max_attention_size"):
                m.max_attention_size = int(local_attn_size) * 1560
        except Exception:
            continue


def _override_sink_size(model: torch.nn.Module, sink_size: int) -> None:
    """
    Keep the first `sink_size` latent frames in KV cache fixed (not evicted) during
    sliding-window causal rollout.

    This maps to `CausalWanSelfAttention.sink_size`, where the eviction logic keeps
    `sink_tokens = sink_size * frame_seq_length` tokens intact at the beginning of the cache.
    """
    if int(sink_size) < 0:
        return
    for m in model.modules():
        if m.__class__.__name__ != "CausalWanSelfAttention":
            continue
        try:
            m.sink_size = int(sink_size)
        except Exception:
            continue


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


def _ensure_control_latent_bcfhw(x: torch.Tensor, *,
                                 latent_channels: int,
                                 name: str) -> torch.Tensor:
    """
    Ensure control latent is [B, C_total, F, H, W] where C_total is 3*C_lat or 4*C_lat.
    Accepts:
      - [C_total, F, H, W]
      - [B, C_total, F, H, W]
      - [B, F, C_total, H, W]
    """
    c3 = 3 * int(latent_channels)
    c4 = 4 * int(latent_channels)
    valid = (c3, c4)
    if x.dim() == 4:
        if int(x.shape[0]) not in valid:
            raise ValueError(
                f"{name} 4D shape must be [C_total,F,H,W] with C_total in {valid}, "
                f"got shape={tuple(x.shape)}")
        return x.unsqueeze(0)
    if x.dim() == 5:
        if int(x.shape[1]) in valid:
            return x
        if int(x.shape[2]) in valid:
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(
        f"Unsupported {name} shape={tuple(x.shape)} for latent_channels={int(latent_channels)}")


def _frame_index_from_stem(stem: str) -> int | None:
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _sorted_files(dir_path: Path, exts: tuple[str, ...]) -> list[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if files:
        parsed = [(_frame_index_from_stem(p.stem), p) for p in files]
        if all(idx is not None for idx, _ in parsed):
            return [p for _, p in sorted(parsed, key=lambda x: (int(x[0]), x[1].name))]
    return sorted(files, key=lambda p: p.name)


def _pad_or_trim_paths(paths: list[Path], target_len: int) -> list[Path]:
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if not paths:
        raise ValueError("input path list is empty")
    if len(paths) >= target_len:
        return paths[:target_len]
    return paths + [paths[-1]] * (target_len - len(paths))


def _resize_for_crop_pil(img: Image.Image, crop_h: int, crop_w: int) -> Image.Image:
    img_w, img_h = img.size
    if (img_h >= crop_h and img_w >= crop_w) or (img_h <= crop_h and img_w <= crop_w):
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    img = img.resize((out_w, out_h), resample=Image.BICUBIC)
    left = max(0, (out_w - crop_w) // 2)
    top = max(0, (out_h - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def _load_rgb_frame(path: Path, height: int, width: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = _resize_for_crop_pil(img, crop_h=int(height), crop_w=int(width))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_mask_frame(path: Path,
                     height: int,
                     width: int,
                     *,
                     threshold: float | None,
                     invert: bool = False) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr_u8 = np.asarray(img).astype(np.uint8)
    h0, w0 = arr_u8.shape[:2]
    target_ratio = float(width) / float(height)
    current_ratio = float(w0) / float(h0)
    if current_ratio > target_ratio:
        new_w = int(h0 * target_ratio)
        left = max(0, (w0 - new_w) // 2)
        arr_u8 = arr_u8[:, left:left + new_w]
    else:
        new_h = int(w0 / target_ratio)
        top = max(0, (h0 - new_h) // 2)
        arr_u8 = arr_u8[top:top + new_h, :]
    arr_u8 = np.array(
        Image.fromarray(arr_u8).resize((int(width), int(height)),
                                       resample=Image.NEAREST))
    denom = 1.0 if int(arr_u8.max()) <= 1 else 255.0
    arr = arr_u8.astype(np.float32) / float(denom)
    # Match md process_mask:
    # - default binary threshold: > 0
    # - optional explicit threshold override if provided
    thr = 0.0 if threshold is None else float(threshold)
    arr = (arr > thr).astype(np.float32)
    if invert:
        arr = 1.0 - arr
    # Return 1xHxW to match md `process_mask`; caller can repeat to 3 channels if needed.
    return torch.from_numpy(arr)[None, ...].contiguous()


def _read_depth_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        try:
            import cv2

            arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr is None:
                raise FileNotFoundError(path)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr.astype(np.float32)
        except Exception:
            pass
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _read_normal_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        try:
            import cv2

            arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr is None:
                raise FileNotFoundError(path)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[..., :3][..., ::-1]
            elif arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            return arr.astype(np.float32)
        except Exception:
            pass
    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr.astype(np.float32)


def _load_normal_frame(path: Path, height: int, width: int) -> torch.Tensor:
    n = _read_normal_any(path)
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError(f"Invalid normal shape from {path}: {tuple(n.shape)}")
    n = n[..., :3].astype(np.float32)

    h0, w0 = n.shape[:2]
    target_ratio = float(width) / float(height)
    current_ratio = float(w0) / float(h0)
    if current_ratio > target_ratio:
        new_w = int(h0 * target_ratio)
        left = max(0, (w0 - new_w) // 2)
        n = n[:, left:left + new_w, :]
    else:
        new_h = int(w0 / target_ratio)
        top = max(0, (h0 - new_h) // 2)
        n = n[top:top + new_h, :, :]

    t = torch.from_numpy(n).permute(2, 0, 1).contiguous().unsqueeze(0)
    t = torch.nn.functional.interpolate(t,
                                        size=(int(height), int(width)),
                                        mode="bilinear",
                                        align_corners=False)
    t = t[0]
    # Match md process_normal:
    # [0,255] -> [-1,1], then OpenCV->OpenGL style coordinate flip (y,z).
    if float(t.max().item()) > 1.5:
        t = t / 127.5 - 1.0
    t[1] = -t[1]
    t[2] = -t[2]
    return t


def _load_depth_sequence(
    depth_paths: list[Path],
    height: int,
    width: int,
    *,
    pmin: float,
    pmax: float,
    invert_depth: bool,
) -> torch.Tensor:
    target_ratio = float(width) / float(height)
    depths: list[np.ndarray] = []
    for p in depth_paths:
        d = _read_depth_any(p)
        if np.isfinite(d).any() and float(np.nanmax(d)) > 1.5:
            d = d / 65535.0
        h0, w0 = d.shape
        current_ratio = float(w0) / float(h0)
        if current_ratio > target_ratio:
            new_w = int(h0 * target_ratio)
            left = max(0, (w0 - new_w) // 2)
            d = d[:, left:left + new_w]
        else:
            new_h = int(w0 / target_ratio)
            top = max(0, (h0 - new_h) // 2)
            d = d[top:top + new_h, :]
        d = np.array(
            Image.fromarray(d).resize((int(width), int(height)),
                                      resample=Image.NEAREST)).astype(np.float32)
        near_mask = d < 0.0015
        far_mask = d > (65500.0 / 65535.0)
        valid = np.isfinite(d) & (~near_mask) & (~far_mask)
        d[~valid] = np.nan
        depths.append(d)

    # Match md process_depth: global min/max (not percentile) over valid pixels.
    stacked = np.stack(depths, axis=0)
    valid_vals = stacked[np.isfinite(stacked)]
    if valid_vals.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(np.nanmin(valid_vals)), float(np.nanmax(valid_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
            lo, hi = 0.0, 1.0

    out = []
    denom = max(hi - lo, 1e-6)
    for d in depths:
        dn = (d - lo) / denom
        dn = np.clip(dn, 0.0, 1.0)
        if invert_depth:
            dn = 1.0 - dn
        # Match md: invalid/NaN -> far(1.0), then map to [-1,1].
        dn = np.nan_to_num(dn, nan=1.0)
        dn = dn * 2.0 - 1.0
        t = torch.from_numpy(dn).float().unsqueeze(0).repeat(3, 1, 1)
        out.append(t)
    return torch.stack(out, dim=0)


def _to_vae_input(video_tchw: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    x = video_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    if normalize:
        x = x * 2.0 - 1.0
    return x


def _postprocess_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
        shift = vae.shift_factor
        if isinstance(shift, torch.Tensor):
            shift = shift.to(latents.device, latents.dtype)
        latents = latents - shift
    scale = getattr(vae, "scaling_factor", None)
    if scale is not None:
        if isinstance(scale, torch.Tensor):
            scale = scale.to(latents.device, latents.dtype)
        latents = latents * scale
    return latents


@torch.no_grad()
def _encode_video_latents(vae,
                          video_bcthw: torch.Tensor,
                          *,
                          sample_mode: str,
                          compute_dtype: torch.dtype = torch.float32
                          ) -> torch.Tensor:
    use_autocast = bool(torch.cuda.is_available() and video_bcthw.is_cuda
                        and compute_dtype != torch.float32)
    with torch.autocast(device_type="cuda",
                        dtype=compute_dtype,
                        enabled=use_autocast):
        out = vae.encode(video_bcthw)
    if sample_mode == "mode":
        latents = out.mode()
    elif sample_mode == "sample":
        latents = out.sample()
    else:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")
    return _postprocess_vae_latents(vae, latents)


@torch.no_grad()
def _build_online_image_latent_from_rgb(
    *,
    first_rgb_chw: torch.Tensor,
    num_frames: int,
    vae,
    target_c: int,
    inference_device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    first_bcthw = _to_vae_input(first_rgb_chw[None, ...], normalize=True).to(
        device=inference_device, dtype=compute_dtype)
    video_condition = torch.cat(
        [
            first_bcthw,
            first_bcthw.new_zeros(first_bcthw.shape[0], first_bcthw.shape[1],
                                  max(int(num_frames) - 1, 0),
                                  first_bcthw.shape[3], first_bcthw.shape[4]),
        ],
        dim=2,
    )
    latent_condition = _encode_video_latents(
        vae,
        video_condition,
        sample_mode="mode",
        compute_dtype=compute_dtype,
    )
    latent_condition_cf = _align_latent_channels(latent_condition[0], target_c,
                                                 "image_latent_condition")
    return latent_condition_cf.unsqueeze(0)


def _align_latent_channels(lat: torch.Tensor, target_c: int, name: str) -> torch.Tensor:
    if lat.ndim != 4:
        raise ValueError(f"{name} must be [C,F,H,W], got shape={tuple(lat.shape)}")
    c = int(lat.shape[0])
    target_c = int(target_c)
    if c == target_c:
        return lat
    if c < target_c and target_c % c == 0:
        k = target_c // c
        logger.warning(
            "%s channel mismatch: got C=%d, target C=%d; repeating channels by x%d",
            name,
            c,
            target_c,
            k,
        )
        return lat.repeat(k, 1, 1, 1)
    if c > target_c and c % target_c == 0:
        logger.warning(
            "%s channel mismatch: got C=%d, target C=%d; truncating to first %d channels",
            name,
            c,
            target_c,
            target_c,
        )
        return lat[:target_c]
    raise ValueError(
        f"{name} channel mismatch: got C={c}, target C={target_c}, cannot auto-align"
    )


def _build_rollout_timesteps(
    *,
    scheduler: FlowMatchEulerDiscreteScheduler,
    schedule_num_inference_steps: int,
    dmd_steps: list[int] | None,
    timestep_indices: list[int] | None,
    warp_denoising_step: bool,
    full_schedule: bool,
    device: torch.device,
) -> torch.Tensor:
    def _get_train_grid_timesteps() -> torch.Tensor:
        ts = getattr(scheduler, "timesteps", None)
        num_train = int(
            getattr(getattr(scheduler, "config", None), "num_train_timesteps",
                    1000))
        if ts is None:
            raise ValueError("scheduler.timesteps is None")
        if int(ts.numel()) < num_train:
            raise ValueError(
                f"scheduler.timesteps too short: {int(ts.numel())} < num_train_timesteps={num_train}. "
                "Do not call scheduler.set_timesteps() for this scheduler in inference."
            )
        return ts[:num_train].to(device=device, dtype=torch.float32)

    if full_schedule:
        return scheduler.timesteps.to(device=device, dtype=torch.float32)
    if timestep_indices is None and dmd_steps is None:
        raise ValueError("Must provide either timestep_indices or dmd_steps.")
    if timestep_indices is not None and dmd_steps is not None:
        raise ValueError("Pass only one of dmd_steps or timestep_indices.")

    train_grid_ts = _get_train_grid_timesteps()
    num_train = int(train_grid_ts.numel())
    logger.info(
        "train_grid: num=%s timesteps[min,max]=[%s,%s]",
        num_train,
        float(train_grid_ts[-1].detach().cpu().item()),
        float(train_grid_ts[0].detach().cpu().item()),
    )

    if timestep_indices is not None:
        if int(schedule_num_inference_steps) <= 0:
            raise ValueError(
                "--schedule_num_inference_steps must be > 0 when using --timestep_indices"
            )
        k = max(1, int(schedule_num_inference_steps))
        teacher_idx = torch.linspace(0,
                                     num_train - 1,
                                     steps=k,
                                     device=device).round().long()
        full_ts = train_grid_ts.index_select(0, teacher_idx)
        idx_list = list(timestep_indices)
        if len(idx_list) > 0:
            last_idx = int(schedule_num_inference_steps) - 1
            if last_idx >= 0 and idx_list[-1] != last_idx:
                idx_list = idx_list + [last_idx]
        idx_t = torch.tensor(idx_list, device=device, dtype=torch.long)
        if torch.any(idx_t < 0) or torch.any(idx_t >= full_ts.numel()):
            raise ValueError(
                f"timestep_indices out of range for schedule_num_inference_steps={schedule_num_inference_steps}: {idx_list}"
            )
        t_list_full = full_ts.index_select(0, idx_t).to(device=device,
                                                        dtype=torch.float32)
        logger.info(
            "rollout schedule: timestep_indices=%s (effective_indices=%s, schedule_num_inference_steps=%s) -> timesteps=%s",
            list(timestep_indices),
            idx_list,
            int(schedule_num_inference_steps),
            [float(x) for x in t_list_full.detach().cpu().tolist()],
        )
        return t_list_full

    dmd_steps_t = torch.tensor(dmd_steps, dtype=torch.long, device=device)
    if warp_denoising_step:
        schedule_ts = train_grid_ts
        schedule_ts = torch.cat(
            (schedule_ts,
             torch.tensor([0.0], device=device, dtype=schedule_ts.dtype)),
            dim=0,
        )
        idx = (num_train - dmd_steps_t).clamp_(0, schedule_ts.numel() - 1)
        t_list_full = schedule_ts.index_select(0, idx)
    else:
        t_list_full = dmd_steps_t.to(dtype=torch.float32)
    logger.info(
        "rollout schedule: dmd_steps=%s warp=%s -> timesteps=%s",
        list(dmd_steps),
        bool(warp_denoising_step),
        [float(x) for x in t_list_full.detach().cpu().tolist()],
    )
    return t_list_full


@dataclass(frozen=True)
class Sample:
    sample_id: str
    caption: str
    fps: int
    text_embedding_bld: torch.Tensor  # [B, L, D]
    first_frame_latent_bcfhw: torch.Tensor  # [B, C, 1, H, W]
    control_latent_bcfhw: torch.Tensor  # [B, 3*C, F, H, W]
    image_latent_bcfhw: torch.Tensor | None = None  # [B, C_img, F, H, W]


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
    control_latent = _ensure_control_latent_bcfhw(
        _decode_tensor(row, "control_latent"),
        latent_channels=int(first_frame_latent.shape[1]),
        name="control_latent",
    )

    sample_id = str(row.get("id", f"index_{index:06d}"))
    caption = str(row.get("caption", ""))
    fps_val = int(row.get("fps", 30) or 30)
    return Sample(sample_id=sample_id,
                  caption=caption,
                  fps=fps_val,
                  text_embedding_bld=text_embedding,
                  first_frame_latent_bcfhw=first_frame_latent,
                  control_latent_bcfhw=control_latent,
                  image_latent_bcfhw=None)


def _read_caption_json(path: Path, caption_key: str) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "captions" in obj and isinstance(obj["captions"], dict):
        caps = obj["captions"]
    else:
        caps = obj

    if caption_key:
        v = caps.get(caption_key, "")
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("Video_Caption", "Short_Caption", "PC_Caption", "caption", "text", "prompt", "description"):
        v = caps.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _resolve_raw_sample_roots(raw_root: str) -> list[Path]:
    root = Path(str(raw_root)).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {root}")
    if (root / "rgb").is_dir():
        return [root]
    sample_roots = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and (p / "rgb").is_dir():
            sample_roots.append(p)
    if not sample_roots:
        raise FileNotFoundError(
            f"No raw samples found under {root}. Expected either <root>/rgb or <root>/*/rgb"
        )
    return sample_roots


def _load_sample_raw(
    *,
    sample_root: Path,
    sample_index: int,
    args,
    transformer,
    vae,
    tokenizer,
    text_encoder,
    inference_device: torch.device,
    dtype: torch.dtype,
) -> Sample:
    rgb_dir = Path(str(args.raw_rgb_dir)).expanduser() if str(
        args.raw_rgb_dir).strip() else (sample_root / "rgb")
    depth_dir = Path(str(args.raw_depth_dir)).expanduser() if str(
        args.raw_depth_dir).strip() else (sample_root / "depth")
    normal_dir = Path(str(args.raw_normal_dir)).expanduser() if str(
        args.raw_normal_dir).strip() else (sample_root / "normal")
    mask_dir = Path(str(args.raw_mask_dir)).expanduser() if str(
        args.raw_mask_dir).strip() else (sample_root / "mask")
    masked_rgb_dir = Path(str(args.raw_masked_rgb_dir)).expanduser() if str(
        args.raw_masked_rgb_dir).strip() else (sample_root / "masked_rgb")

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"raw rgb dir not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"raw depth dir not found: {depth_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"raw mask dir not found: {mask_dir}")

    rgb_paths = _pad_or_trim_paths(
        _sorted_files(rgb_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp")),
        int(args.num_frames),
    )
    depth_paths = _pad_or_trim_paths(
        _sorted_files(depth_dir,
                      (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr")),
        int(args.num_frames),
    )
    mask_paths = _pad_or_trim_paths(
        _sorted_files(mask_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp")),
        int(args.num_frames),
    )
    normal_paths: list[Path] | None = None
    if normal_dir.is_dir():
        nfiles = _sorted_files(normal_dir,
                               (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr"))
        if len(nfiles) > 0:
            normal_paths = _pad_or_trim_paths(nfiles, int(args.num_frames))
    if bool(args.raw_require_normal) and normal_paths is None:
        raise FileNotFoundError(
            f"--raw_require_normal is set but normal dir is missing/empty: {normal_dir}"
        )

    masked_rgb_paths: list[Path] | None = None
    if masked_rgb_dir.is_dir():
        mfiles = _sorted_files(masked_rgb_dir,
                               (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        if len(mfiles) > 0:
            masked_rgb_paths = _pad_or_trim_paths(mfiles, int(args.num_frames))

    prompt = str(args.raw_prompt or "").strip()
    if not prompt:
        caption_path = Path(str(args.raw_caption_path)).expanduser(
        ) if str(args.raw_caption_path).strip() else None
        if caption_path is not None and caption_path.exists():
            if caption_path.suffix.lower() == ".json":
                prompt = _read_caption_json(caption_path,
                                            str(args.raw_caption_key))
            else:
                prompt = caption_path.read_text(encoding="utf-8").strip()
        else:
            text_txt = sample_root / "text.txt"
            if text_txt.is_file():
                prompt = text_txt.read_text(encoding="utf-8").strip()
            else:
                cap_jsons = sorted(sample_root.glob("caption*.json"),
                                   key=lambda p: p.name)
                if cap_jsons:
                    prompt = _read_caption_json(cap_jsons[-1],
                                                str(args.raw_caption_key))
    if not prompt:
        prompt = "A driving scene in city street."

    max_text_len = int(getattr(transformer, "text_len", 226))
    text_embedding = _compute_prompt_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_text_len,
        dtype=dtype,
        target_device=inference_device,
    )

    compute_dtype = dtype
    # Raw mode alignment: use rgb/rgb_0000.png as first-frame anchor.
    # Fallback to the first sorted RGB frame if rgb_0000.png is missing.
    rgb_zero = rgb_dir / "rgb_0000.png"
    if rgb_zero.is_file():
        first_rgb = _load_rgb_frame(rgb_zero, int(args.height), int(args.width))
    else:
        logger.warning(
            "rgb_0000.png not found under %s; fallback to first rgb frame.",
            str(rgb_dir),
        )
        first_rgb = _load_rgb_frame(rgb_paths[0], int(args.height),
                                    int(args.width))

    first_bcthw = _to_vae_input(first_rgb[None, ...], normalize=True).to(
        device=inference_device, dtype=compute_dtype)
    first_lat = _encode_video_latents(
        vae, first_bcthw, sample_mode="mode",
        compute_dtype=compute_dtype)  # [1,C,1,H,W]

    depth_tchw = _load_depth_sequence(
        depth_paths,
        int(args.height),
        int(args.width),
        pmin=float(args.raw_depth_percentile_min),
        pmax=float(args.raw_depth_percentile_max),
        invert_depth=bool(args.raw_depth_invert),
    )
    mask_tchw = torch.stack(
        [
            _load_mask_frame(
                p,
                int(args.height),
                int(args.width),
                threshold=None if float(args.raw_mask_threshold) < 0 else
                float(args.raw_mask_threshold),
                invert=bool(args.raw_mask_invert),
            ) for p in mask_paths
        ],
        dim=0,
    )
    mask3_tchw = mask_tchw.repeat(1, 3, 1, 1)
    if masked_rgb_paths is not None:
        masked_rgb_tchw = torch.stack(
            [_load_rgb_frame(p, int(args.height), int(args.width))
             for p in masked_rgb_paths],
            dim=0,
        )
    else:
        # Match md behavior: if masked RGB is missing, use black frames.
        masked_rgb_tchw = torch.zeros(
            (int(args.num_frames), 3, int(args.height), int(args.width)),
            dtype=torch.float32,
        )

    # Match Diff-Factory/FastVideo preprocess semantics for TI2V:
    # the first control frame should reflect the given image anchor.
    if mask_tchw.shape[0] > 0 and masked_rgb_tchw.shape[0] > 0:
        mask_tchw = mask_tchw.clone()
        masked_rgb_tchw = masked_rgb_tchw.clone()
        mask_tchw[0] = 1.0
        masked_rgb_tchw[0] = first_rgb
        mask3_tchw = mask_tchw.repeat(1, 3, 1, 1)

    normal_tchw = None
    if normal_paths is not None:
        normal_tchw = torch.stack(
            [_load_normal_frame(p, int(args.height), int(args.width))
             for p in normal_paths],
            dim=0,
        )

    if normal_tchw is not None:
        video_n = torch.cat(
            [
                _to_vae_input(depth_tchw, normalize=False),
                _to_vae_input(normal_tchw, normalize=False),
                _to_vae_input(masked_rgb_tchw, normalize=True),
                _to_vae_input(mask3_tchw, normalize=False),
            ],
            dim=0,
        ).to(device=inference_device, dtype=compute_dtype)
        lat_n = _encode_video_latents(vae,
                                      video_n,
                                      sample_mode="mode",
                                      compute_dtype=compute_dtype)
        depth_lat = lat_n[0]
        normal_lat = lat_n[1]
        masked_lat = lat_n[2]
        mask_lat = lat_n[3]
    else:
        video_n = torch.cat(
            [
                _to_vae_input(depth_tchw, normalize=False),
                _to_vae_input(masked_rgb_tchw, normalize=True),
                _to_vae_input(mask3_tchw, normalize=False),
            ],
            dim=0,
        ).to(device=inference_device, dtype=compute_dtype)
        lat_n = _encode_video_latents(vae,
                                      video_n,
                                      sample_mode="mode",
                                      compute_dtype=compute_dtype)
        depth_lat = lat_n[0]
        masked_lat = lat_n[1]
        mask_lat = lat_n[2]

    target_c = int(getattr(transformer, "num_channels_latents", first_lat.shape[1]))
    first_lat_cf = _align_latent_channels(first_lat[0], target_c, "first_frame_lat")
    first_lat_bcfhw = first_lat_cf.unsqueeze(0)
    image_latent_bcfhw = _build_online_image_latent_from_rgb(
        first_rgb_chw=first_rgb,
        num_frames=int(args.num_frames),
        vae=vae,
        target_c=target_c,
        inference_device=inference_device,
        compute_dtype=compute_dtype,
    )

    depth_lat = _align_latent_channels(depth_lat, target_c, "depth_lat")
    masked_lat = _align_latent_channels(masked_lat, target_c, "masked_lat")
    mask_lat = _align_latent_channels(mask_lat, target_c, "mask_lat")
    if normal_tchw is not None:
        normal_lat = _align_latent_channels(normal_lat, target_c, "normal_lat")
        control_lat = torch.cat([depth_lat, normal_lat, masked_lat, mask_lat], dim=0)
    else:
        control_lat = torch.cat([depth_lat, masked_lat, mask_lat], dim=0)
    control_lat_bcfhw = control_lat.unsqueeze(0)

    sample_id = f"{sample_root.name}_idx{sample_index:06d}"
    return Sample(
        sample_id=sample_id,
        caption=prompt,
        fps=int(args.raw_fps),
        text_embedding_bld=text_embedding,
        first_frame_latent_bcfhw=first_lat_bcfhw,
        control_latent_bcfhw=control_lat_bcfhw,
        image_latent_bcfhw=image_latent_bcfhw,
    )


def _initialize_kv_cache(*,
                         model,
                         batch_size: int,
                         dtype: torch.dtype,
                         device: torch.device,
                         frame_seq_length: int,
                         sliding_window_num_frames_override: int | None = None
                         ) -> list[dict]:
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
    sliding_window_num_frames = int(
        sliding_window_num_frames_override
        if sliding_window_num_frames_override is not None else
        getattr(model.config.arch_config, "sliding_window_num_frames", 0))
    if local_attn_size != -1:
        kv_cache_size = local_attn_size * frame_seq_length
    else:
        # Some checkpoints/configs may set sliding_window_num_frames=0. That would allocate
        # a zero-length cache and crash on the first KV write. For causal rollouts with
        # global attention, we need enough cache for the whole latent sequence.
        if sliding_window_num_frames <= 0:
            raise ValueError(
                "Invalid sliding_window_num_frames (<=0) for causal KV cache. "
                "Pass sliding_window_num_frames_override (e.g. latent_t) or set "
                "model.config.arch_config.sliding_window_num_frames to a positive value.")
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
        inner_dim = getattr(model, "inner_dim", None)
        if inner_dim is None:
            inner_dim = getattr(model, "hidden_size", None)
        if inner_dim is None:
            raise AttributeError(f"Cannot determine attention_head_dim for {type(model).__name__}")
        head_dim = int(inner_dim) // int(num_heads)
    head_dim = int(head_dim)
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


def _reset_kv_and_crossattn_caches(
    *,
    kv_cache: list[dict] | None,
    crossattn_cache: list[dict] | None,
) -> None:
    if kv_cache is not None:
        for layer_cache in kv_cache:
            layer_cache["global_end_index"].fill_(0)
            layer_cache["local_end_index"].fill_(0)
            layer_cache["k"].zero_()
            layer_cache["v"].zero_()
    if crossattn_cache is not None:
        for layer_cache in crossattn_cache:
            layer_cache["is_init"] = False
            layer_cache["k"].zero_()
            layer_cache["v"].zero_()


@torch.no_grad()
def _causal_dmd_rollout_ti2v_controlnet(
    *,
    transformer,
    controlnet,
    scheduler: FlowMatchEulerDiscreteScheduler | FlowUniPCMultistepScheduler,
    prompt_embeds_list: list[torch.Tensor],
    negative_prompt_embeds_list: list[torch.Tensor] | None,
    guidance_scale: float,
    first_frame_latent_bcfhw: torch.Tensor | None,
    control_latent_bcfhw: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    schedule_num_inference_steps: int,
    dmd_steps: list[int] | None,
    timestep_indices: list[int] | None,
    context_noise: int,
    warp_denoising_step: bool,
    update_rule: str,
    full_schedule: bool,
    first_frame_timestep_zero: bool,
    expand_timesteps: bool,
    reset_cache_each_block: bool,
    disable_cache_update: bool,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns: latents [B, C, T_lat, H_lat, W_lat] (BCFHW) where T_lat is latent frames (e.g. 21).
    """
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    # Use global RNG seeding for broad torch compatibility (older builds may not
    # support `generator=` for randn_like / randn).
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # Build a minimal ForwardBatch for forward_context bookkeeping.
    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = prompt_embeds_list
    batch.height = height
    batch.width = width
    batch.num_frames = num_frames

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

    is_unipc = isinstance(scheduler, FlowUniPCMultistepScheduler)
    use_step_schedule = is_unipc or bool(full_schedule)
    if use_step_schedule:
        scheduler.set_timesteps(int(schedule_num_inference_steps),
                                device=device)
        t_list_full = scheduler.timesteps.to(device=device)
        logger.info(
            "rollout schedule (%s): num_steps=%s timesteps[0..3]=%s",
            "unipc" if is_unipc else "flowmatch_full",
            int(schedule_num_inference_steps),
            [int(x) for x in t_list_full[:4].detach().cpu().tolist()],
        )
    else:
        t_list_full = _build_rollout_timesteps(
            scheduler=scheduler,
            schedule_num_inference_steps=schedule_num_inference_steps,
            dmd_steps=dmd_steps,
            timestep_indices=timestep_indices,
            warp_denoising_step=warp_denoising_step,
            full_schedule=False,
            device=device,
        )

    # Initialize the full latent sequence from pure noise (sigma_max == 1.0 for this scheduler).
    #
    # IMPORTANT (training alignment):
    # Self-Forcing / causal DMD rollouts start from ONE global noise sample for the whole sequence,
    # then process temporal blocks by slicing this tensor. Do NOT resample per block; that changes the
    # underlying noise "z" between blocks and often collapses to noise.
    latents = torch.randn(
        (1, transformer.num_channels_latents, latent_t, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )

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
                                    frame_seq_length=frame_seq_length,
                                    sliding_window_num_frames_override=latent_t)
    crossattn_cache = _initialize_crossattn_cache(model=transformer,
                                                  batch_size=1,
                                                  max_text_len=transformer.text_len,
                                                  dtype=dtype,
                                                  device=device)
    kv_cache_uncond = _initialize_kv_cache(model=transformer,
                                           batch_size=1,
                                           dtype=dtype,
                                           device=device,
                                           frame_seq_length=frame_seq_length,
                                           sliding_window_num_frames_override=latent_t)
    crossattn_cache_uncond = _initialize_crossattn_cache(model=transformer,
                                                         batch_size=1,
                                                         max_text_len=transformer.text_len,
                                                         dtype=dtype,
                                                         device=device)
    control_kv_cache = _initialize_kv_cache(model=controlnet,
                                            batch_size=1,
                                            dtype=dtype,
                                            device=device,
                                            frame_seq_length=frame_seq_length,
                                            sliding_window_num_frames_override=latent_t)
    control_crossattn_cache = _initialize_crossattn_cache(
        model=controlnet,
        batch_size=1,
        max_text_len=controlnet.text_len,
        dtype=dtype,
        device=device)
    control_kv_cache_uncond = _initialize_kv_cache(model=controlnet,
                                                   batch_size=1,
                                                   dtype=dtype,
                                                   device=device,
                                                   frame_seq_length=frame_seq_length,
                                                   sliding_window_num_frames_override=latent_t)
    control_crossattn_cache_uncond = _initialize_crossattn_cache(
        model=controlnet,
        batch_size=1,
        max_text_len=controlnet.text_len,
        dtype=dtype,
        device=device)

    # Main causal chunk loop
    num_blocks = latents.shape[2] // num_frames_per_block
    start_index = 0
    for _block_idx in range(num_blocks):
        if reset_cache_each_block:
            _reset_kv_and_crossattn_caches(kv_cache=kv_cache, crossattn_cache=crossattn_cache)
            _reset_kv_and_crossattn_caches(kv_cache=kv_cache_uncond, crossattn_cache=crossattn_cache_uncond)
            _reset_kv_and_crossattn_caches(kv_cache=control_kv_cache, crossattn_cache=control_crossattn_cache)
            _reset_kv_and_crossattn_caches(kv_cache=control_kv_cache_uncond, crossattn_cache=control_crossattn_cache_uncond)

        current_num_frames = num_frames_per_block
        # Slice the global noise tensor for this block (do NOT resample).
        current_latents = latents[:, :, start_index:start_index + current_num_frames].clone()

        control_chunk = control_latent_bcfhw[:, :, start_index:start_index +
                                             current_num_frames].to(device=device,
                                                                    dtype=dtype)

        # Timesteps for this chunk.
        t_list = t_list_full

        num_channels_latents = getattr(transformer, "num_channels_latents",
                                       control_chunk.shape[1] // 3)

        def _predict_flow_at_t(t_scalar: torch.Tensor, *, step_index: int) -> torch.Tensor:
            t_scalar = t_scalar.to(dtype=torch.float32)
            latent_model_input = current_latents
            if expand_timesteps and first_frame_latent_bcfhw is not None and start_index == 0:
                first_frame_mask = torch.ones(
                    (1, 1, current_num_frames, latent_h, latent_w),
                    device=device,
                    dtype=dtype,
                )
                first_frame_mask[:, :, 0] = 0
                image_latents = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
                latent_model_input = (1 - first_frame_mask) * image_latents + first_frame_mask * current_latents
            elif first_frame_latent_bcfhw is not None and start_index == 0:
                latent_model_input = latent_model_input.clone()
                latent_model_input[:, :, :1] = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
            latent_model_input = latent_model_input.to(dtype=dtype)

            timestep = torch.ones((1, current_num_frames),
                                  device=device,
                                  dtype=torch.float32) * t_scalar
            # IMPORTANT (training alignment):
            # In FastVideo self-forcing training, TI2V enforces the first latent frame by
            # overwriting `hidden_states[:, :, 0]`, but does NOT override its timestep.
            # Keep the original timestep by default; allow forcing to 0 for Diff-Factory-style
            # expand_timesteps debugging via `--first_frame_timestep_zero`.
            if (first_frame_timestep_zero or
                    (expand_timesteps and first_frame_latent_bcfhw is not None and start_index == 0)):
                timestep = timestep.clone()
                timestep[:, 0] = 0

            with torch.autocast(device_type="cuda",
                                dtype=dtype,
                                enabled=(dtype != torch.float32)):
                with set_forward_context(current_timestep=int(step_index),
                                         attn_metadata=None,
                                         forward_batch=batch):
                    control_res_cond = controlnet(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds_list,
                        timestep=timestep,
                        **_build_controlnet_kwargs(controlnet, control_chunk,
                                                   num_channels_latents),
                        kv_cache=control_kv_cache,
                        crossattn_cache=control_crossattn_cache,
                        current_start=start_index * frame_seq_length,
                        start_frame=start_index,
                    )
                    pred_flow_cond_btchw = transformer(
                        latent_model_input,
                        prompt_embeds_list,
                        timestep,
                        block_controlnet_hidden_states=control_res_cond,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start_index * frame_seq_length,
                        start_frame=start_index,
                    ).permute(0, 2, 1, 3, 4)

                if guidance_scale != 1.0:
                    if negative_prompt_embeds_list is None:
                        raise ValueError("guidance_scale != 1.0 requires negative_prompt_embeds_list.")
                    with set_forward_context(current_timestep=int(step_index),
                                             attn_metadata=None,
                                             forward_batch=batch):
                        control_res_uncond = controlnet(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=negative_prompt_embeds_list,
                            timestep=timestep,
                            **_build_controlnet_kwargs(controlnet, control_chunk,
                                                       num_channels_latents),
                            kv_cache=control_kv_cache_uncond,
                            crossattn_cache=control_crossattn_cache_uncond,
                            current_start=start_index * frame_seq_length,
                            start_frame=start_index,
                        )
                        pred_flow_uncond_btchw = transformer(
                            latent_model_input,
                            negative_prompt_embeds_list,
                            timestep,
                            block_controlnet_hidden_states=control_res_uncond,
                            kv_cache=kv_cache_uncond,
                            crossattn_cache=crossattn_cache_uncond,
                            current_start=start_index * frame_seq_length,
                            start_frame=start_index,
                        ).permute(0, 2, 1, 3, 4)
                    return pred_flow_uncond_btchw + float(guidance_scale) * (pred_flow_cond_btchw - pred_flow_uncond_btchw)

            return pred_flow_cond_btchw

        if use_step_schedule:
            # Full-schedule solver step (FlowMatchEuler or UniPC)
            scheduler.set_timesteps(int(schedule_num_inference_steps),
                                    device=device)
            t_list_full = scheduler.timesteps.to(device=device)
            for step_i, t_cur in enumerate(t_list_full):
                t_cur_f = t_cur.to(dtype=torch.float32)
                pred_flow_btchw = _predict_flow_at_t(t_cur_f, step_index=step_i)
                pred_flow_bcfhw = pred_flow_btchw.permute(0, 2, 1, 3, 4).contiguous()
                current_latents = scheduler.step(
                    pred_flow_bcfhw,
                    t_cur,
                    current_latents,
                ).prev_sample
        elif update_rule == "renoise_x0":
            # Stochastic renoise chain (Self-Forcing simulation style): predict x0 then add noise at the next anchor.
            for step_i, t_cur in enumerate(t_list):
                noisy_input_bfchw = current_latents.permute(0, 2, 1, 3, 4).contiguous()
                pred_flow_btchw = _predict_flow_at_t(t_cur, step_index=step_i)

                timestep = torch.ones((1, current_num_frames),
                                      device=device,
                                      dtype=torch.float32) * t_cur
                if first_frame_timestep_zero and first_frame_latent_bcfhw is not None and start_index == 0:
                    timestep = timestep.clone()
                    timestep[:, 0] = 0

                denoised_pred = pred_noise_to_pred_video(
                    pred_noise=pred_flow_btchw.flatten(0, 1),
                    noise_input_latent=noisy_input_bfchw.flatten(0, 1),
                    timestep=timestep,
                    scheduler=scheduler,
                ).unflatten(0, pred_flow_btchw.shape[:2])

                if step_i < len(t_list) - 1:
                    next_timestep = t_list[step_i + 1]
                    noise = torch.randn_like(denoised_pred.flatten(0, 1))
                    noisy_input_bfchw = scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        noise,
                        next_timestep * torch.ones(
                            (1 * current_num_frames),
                            device=device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                    current_latents = noisy_input_bfchw.permute(0, 2, 1, 3, 4).contiguous()
                else:
                    current_latents = denoised_pred.permute(0, 2, 1, 3, 4).contiguous()


        elif update_rule == "euler_dt":
            # Deterministic Euler stepping in sigma space (Diff-Factory-style):
            #   x_{t_next} = x_{t} + (sigma(t_next) - sigma(t)) * v_theta(x_t, t)
            # NOTE: for few-step distilled students, `t_list` usually includes the last schedule index
            # only as "next timestep" (nxt[i]) rather than an extra model forward. Therefore, we:
            # - predict v at every t_cur (all but the last element)
            # - update x to the last timestep
            # - return x at the final (very low noise) sigma as the output (it is already ~x0).
            if t_list.numel() < 2:
                raise ValueError("euler_dt requires at least 2 timesteps (need a next timestep).")
            timesteps_1d = scheduler.timesteps.to(device=device, dtype=torch.float32)
            sigmas_1d = scheduler.sigmas.to(device=device, dtype=torch.float32)

            # Step through (t_cur -> t_next) pairs without re-noising.
            for step_i in range(int(t_list.numel()) - 1):
                t_cur = t_list[step_i]
                t_next = t_list[step_i + 1]
                pred_flow_btchw = _predict_flow_at_t(t_cur, step_index=step_i)

                idx_cur = torch.argmin((timesteps_1d - t_cur.float()).abs())
                idx_next = torch.argmin((timesteps_1d - t_next.float()).abs())
                sigma_cur = sigmas_1d[idx_cur]
                sigma_next = sigmas_1d[idx_next]
                dt = (sigma_next - sigma_cur).to(dtype=pred_flow_btchw.dtype)

                current_latents = current_latents + dt * pred_flow_btchw.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Unsupported update_rule: {update_rule!r}")

        # Write back the updated chunk.
        latents[:, :, start_index:start_index + current_num_frames] = current_latents

        if not disable_cache_update:
            # Cache update with optional context noise (Self-Forcing style): add context noise then forward once to
            # commit KV cache for the next chunk.
            context_btchw = current_latents.permute(0, 2, 1, 3, 4).contiguous()
            context_timestep = torch.ones((1, current_num_frames),
                                          device=device,
                                          dtype=torch.float32) * float(context_noise)
            if float(context_noise) <= 0.0:
                context_timestep = context_timestep.zero_()
            else:
                if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None and scheduler.timesteps.numel() > 0:
                    schedule_ts = scheduler.timesteps.to(device=device, dtype=context_timestep.dtype)
                    t_flat = context_timestep.flatten()
                    diff = (t_flat[:, None] - schedule_ts[None, :]).abs()
                    nearest_idx = diff.argmin(dim=1)
                    context_timestep = schedule_ts.index_select(0, nearest_idx).view_as(context_timestep)
                # Match Self-Forcing training: always add (small) context noise then commit cache.
                ctx_noise = torch.randn_like(context_btchw.flatten(0, 1))
                context_btchw = scheduler.add_noise(
                    context_btchw.flatten(0, 1),
                    ctx_noise,
                    context_timestep.flatten(),
                ).unflatten(0, context_btchw.shape[:2])

            context_bcfhw = context_btchw.permute(0, 2, 1, 3, 4).contiguous()

            if expand_timesteps and first_frame_latent_bcfhw is not None and start_index == 0:
                first_frame_mask = torch.ones(
                    (1, 1, current_num_frames, latent_h, latent_w),
                    device=device,
                    dtype=dtype,
                )
                first_frame_mask[:, :, 0] = 0
                image_latents = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
                context_bcfhw = (1 - first_frame_mask) * image_latents + first_frame_mask * context_bcfhw
            elif first_frame_latent_bcfhw is not None and start_index == 0:
                context_bcfhw = context_bcfhw.clone()
                context_bcfhw[:, :, :1] = first_frame_latent_bcfhw.to(device=device,
                                                                      dtype=dtype)
            context_bcfhw = context_bcfhw.to(dtype=dtype)

            with torch.autocast(device_type="cuda", dtype=dtype,
                                enabled=(dtype != torch.float32)), \
                    set_forward_context(current_timestep=0,
                                        attn_metadata=None,
                                        forward_batch=batch):
                control_res_ctx = controlnet(
                    hidden_states=context_bcfhw,
                    encoder_hidden_states=prompt_embeds_list,
                    timestep=context_timestep,
                    **_build_controlnet_kwargs(controlnet, control_chunk,
                                               num_channels_latents),
                    kv_cache=control_kv_cache,
                    crossattn_cache=control_crossattn_cache,
                    current_start=start_index * frame_seq_length,
                    start_frame=start_index,
                )
                _ = transformer(
                    context_bcfhw,
                    prompt_embeds_list,
                    context_timestep,
                    block_controlnet_hidden_states=control_res_ctx,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start_index * frame_seq_length,
                    start_frame=start_index,
                )
                if guidance_scale != 1.0:
                    if negative_prompt_embeds_list is None:
                        raise ValueError(
                            "guidance_scale != 1.0 requires negative_prompt_embeds_list."
                        )
                    control_res_ctx_uncond = controlnet(
                        hidden_states=context_bcfhw,
                        encoder_hidden_states=negative_prompt_embeds_list,
                        timestep=context_timestep,
                        **_build_controlnet_kwargs(controlnet, control_chunk,
                                                   num_channels_latents),
                        kv_cache=control_kv_cache_uncond,
                        crossattn_cache=control_crossattn_cache_uncond,
                        current_start=start_index * frame_seq_length,
                        start_frame=start_index,
                    )
                    _ = transformer(
                        context_bcfhw,
                        negative_prompt_embeds_list,
                        context_timestep,
                        block_controlnet_hidden_states=control_res_ctx_uncond,
                        kv_cache=kv_cache_uncond,
                        crossattn_cache=crossattn_cache_uncond,
                        current_start=start_index * frame_seq_length,
                        start_frame=start_index,
                    )

        start_index += current_num_frames

    if expand_timesteps and first_frame_latent_bcfhw is not None:
        first_frame_mask = torch.ones(
            (1, 1, latent_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
        )
        first_frame_mask[:, :, 0] = 0
        image_latents = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
        latents = (1 - first_frame_mask) * image_latents + first_frame_mask * latents
    elif first_frame_latent_bcfhw is not None:
        latents = latents.clone()
        latents[:, :, :1] = first_frame_latent_bcfhw.to(device=device,
                                                        dtype=dtype)
    return latents


@torch.no_grad()
def _bidirectional_dmd_rollout_ti2v_controlnet(
    *,
    transformer,
    controlnet,
    scheduler: FlowMatchEulerDiscreteScheduler | FlowUniPCMultistepScheduler
    | DiffusersUniPCMultistepScheduler,
    prompt_embeds_list: list[torch.Tensor],
    negative_prompt_embeds_list: list[torch.Tensor] | None,
    guidance_scale: float,
    controlnet_weight: float,
    first_frame_latent_bcfhw: torch.Tensor | None,
    image_latent_bcfhw: torch.Tensor | None,
    control_latent_bcfhw: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    schedule_num_inference_steps: int,
    dmd_steps: list[int] | None,
    timestep_indices: list[int] | None,
    context_noise: int,
    warp_denoising_step: bool,
    update_rule: str,
    full_schedule: bool,
    first_frame_timestep_zero: bool,
    expand_timesteps: bool,
    trace_jsonl_path: str | None,
    trace_sample_id: str,
    seed: int,
    dtype: torch.dtype,
    force_first_frame_anchor: bool,
) -> torch.Tensor:
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = prompt_embeds_list
    batch.height = height
    batch.width = width
    batch.num_frames = num_frames

    latent_t = int(control_latent_bcfhw.shape[2])
    latent_h = int(control_latent_bcfhw.shape[3])
    latent_w = int(control_latent_bcfhw.shape[4])

    scheduler.set_timesteps(int(schedule_num_inference_steps), device=device)
    timesteps = scheduler.timesteps.to(device=device)
    logger.info(
        "bidir rollout (diff-factory): num_steps=%s timesteps[0..3]=%s",
        int(schedule_num_inference_steps),
        [int(x) for x in timesteps[:4].detach().cpu().tolist()],
    )

    latents = torch.randn(
        (1, transformer.num_channels_latents, latent_t, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )

    image_latents = None
    first_frame_mask = torch.ones(
        (1, 1, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )
    if not expand_timesteps and image_latent_bcfhw is not None:
        image_latents = image_latent_bcfhw.to(device=device, dtype=dtype)
    elif first_frame_latent_bcfhw is not None:
        image_latents = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
        first_frame_mask[:, :, 0] = 0

    anchor_first_frame = bool(force_first_frame_anchor and not expand_timesteps
                              and image_latent_bcfhw is None
                              and first_frame_latent_bcfhw is not None)
    if anchor_first_frame:
        latents[:, :, :1] = first_frame_latent_bcfhw.to(device=device, dtype=dtype)
        logger.info(
            "bidir parquet mode: forcing first-frame anchor with timestep[0]=0 at every denoising step."
        )

    patch_size = getattr(getattr(transformer.config, "arch_config", None),
                         "patch_size", (2, 2))
    patch_h = int(patch_size[-2])
    patch_w = int(patch_size[-1])
    expected_hidden_channels = int(
        getattr(transformer, "in_channels",
                getattr(transformer, "num_channels_latents", latents.shape[1])))
    if _is_union_controlnet(controlnet):
        expected_hidden_channels = int(
            getattr(controlnet, "in_channels",
                    expected_hidden_channels * 3) // 3)

    latent_input_mode = "latents_only"
    image_latents_concat = None
    if not expand_timesteps and image_latents is not None:
        latent_channels = int(latents.shape[1])
        image_channels = int(image_latents.shape[1])
        concat_channels = latent_channels + image_channels
        if latent_channels == expected_hidden_channels:
            latent_input_mode = "overwrite_first_frame"
            logger.info(
                "bidir alignment: using first-frame overwrite for non-expanded TI2V input "
                "(latent_channels=%s, expected_hidden_channels=%s).",
                latent_channels,
                expected_hidden_channels,
            )
        elif concat_channels == expected_hidden_channels:
            latent_input_mode = "concat_channels"
            image_latents_concat = image_latents
            if int(image_latents_concat.shape[2]) == 1 and latent_t > 1:
                # Diff-Factory non-expanded TI2V expects a full-length image-conditioning latent.
                # Our parquet/raw path only stores the first latent frame, so approximate the
                # original video-conditioned encode with zeros on later latent frames.
                image_latents_full = torch.zeros(
                    (image_latents_concat.shape[0], image_latents_concat.shape[1],
                     latent_t, image_latents_concat.shape[3], image_latents_concat.shape[4]),
                    device=device,
                    dtype=dtype,
                )
                image_latents_full[:, :, :1] = image_latents_concat
                image_latents_concat = image_latents_full
                logger.info(
                    "bidir alignment: expanded first_frame_latent from F=1 to F=%s with zero tail for "
                    "non-expanded TI2V concat input.",
                    latent_t,
                )
            logger.info(
                "bidir alignment: using channel concat for non-expanded TI2V input "
                "(latent_channels=%s, image_channels=%s, expected_hidden_channels=%s).",
                latent_channels,
                image_channels,
                expected_hidden_channels,
            )
        else:
            raise RuntimeError(
                "Unsupported bidirectional TI2V hidden-state channels: "
                f"latent_channels={latent_channels}, image_channels={image_channels}, "
                f"expected_hidden_channels={expected_hidden_channels}.")

    def _build_latent_model_input() -> torch.Tensor:
        if expand_timesteps:
            if image_latents is None:
                return latents
            return (1 - first_frame_mask) * image_latents + first_frame_mask * latents
        if image_latents is None:
            return latents
        if latent_input_mode == "overwrite_first_frame":
            latent_model_input = latents.clone()
            latent_model_input[:, :, :1] = image_latents[:, :, :1]
            return latent_model_input
        if latent_input_mode == "concat_channels":
            return torch.cat([latents, image_latents_concat], dim=1)
        return latents

    def _build_timestep_tokens(t_cur: torch.Tensor) -> torch.Tensor:
        if not expand_timesteps:
            if anchor_first_frame:
                temp_ts = (first_frame_mask[0, 0] * t_cur.to(dtype=torch.float32))
                temp_ts = temp_ts[:, ::patch_h, ::patch_w].flatten()
                return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            if first_frame_timestep_zero and image_latents is not None:
                timestep = torch.ones((latents.shape[0], latent_t),
                                      device=device,
                                      dtype=torch.float32) * t_cur.to(
                                          dtype=torch.float32)
                timestep[:, 0] = 0
                return timestep
            return t_cur.expand(latents.shape[0])
        temp_ts = (first_frame_mask[0, 0] * t_cur)
        if first_frame_timestep_zero and image_latents is not None:
            temp_ts = temp_ts.clone()
            temp_ts[0] = 0
        temp_ts = temp_ts[:, ::patch_h, ::patch_w].flatten()
        return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

    num_channels_latents = getattr(transformer, "num_channels_latents",
                                   control_latent_bcfhw.shape[1] // 3)

    for step_i, t_cur in enumerate(timesteps):
        if float(guidance_scale) != 1.0 and negative_prompt_embeds_list is None:
            raise ValueError(
                "guidance_scale != 1.0 requires negative_prompt_embeds_list."
            )

        latent_model_input = _build_latent_model_input().to(
            device=device, dtype=dtype)
        if int(latent_model_input.shape[1]) != expected_hidden_channels:
            raise RuntimeError(
                "Bidirectional latent_model_input channel mismatch: "
                f"got {int(latent_model_input.shape[1])}, expected {expected_hidden_channels}.")
        timestep = _build_timestep_tokens(t_cur.to(dtype=torch.float32))

        with set_forward_context(current_timestep=int(step_i),
                                 attn_metadata=None,
                                 forward_batch=batch):
            control_res = controlnet(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds_list,
                timestep=timestep,
                **_build_controlnet_kwargs(controlnet, control_latent_bcfhw,
                                           num_channels_latents),
            )
            control_res = _scale_control_residual(control_res,
                                                  scale=controlnet_weight)

            noise_pred = transformer(
                latent_model_input,
                prompt_embeds_list,
                timestep,
                block_controlnet_hidden_states=control_res,
            )
            noise_pred_cond = noise_pred

            noise_uncond = None
            if float(guidance_scale) != 1.0:
                noise_uncond = transformer(
                    latent_model_input,
                    negative_prompt_embeds_list,
                    timestep,
                    block_controlnet_hidden_states=control_res,
                )
                noise_pred = noise_uncond + float(guidance_scale) * (noise_pred - noise_uncond)

        latent_l2_before = _tensor_or_list_l2(latents)
        latents = scheduler.step(noise_pred, t_cur, latents).prev_sample
        if anchor_first_frame:
            latents[:, :, :1] = first_frame_latent_bcfhw.to(device=device,
                                                            dtype=dtype)
        latent_l2_after = _tensor_or_list_l2(latents)
        _append_trace_jsonl(
            trace_jsonl_path,
            {
                "sample_id": str(trace_sample_id),
                "attention_mode": "bidirectional",
                "step_index": int(step_i),
                "num_steps": int(timesteps.numel()),
                "timestep": float(t_cur.detach().float().cpu().item()),
                "control_scale": float(controlnet_weight),
                "latent_l2_before": float(latent_l2_before),
                "control_cond_l2": float(_tensor_or_list_l2(control_res)),
                "control_uncond_l2": float(_tensor_or_list_l2(control_res if float(guidance_scale) != 1.0 else None)),
                "noise_cond_l2": float(_tensor_or_list_l2(noise_pred_cond)),
                "noise_uncond_l2": float(_tensor_or_list_l2(noise_uncond)),
                "noise_final_l2": float(_tensor_or_list_l2(noise_pred)),
                "latent_l2_after": float(latent_l2_after),
            },
        )

    if expand_timesteps and image_latents is not None:
        latents = (1 - first_frame_mask) * image_latents + first_frame_mask * latents

    return latents


def _save_mp4(frames_bcthw: torch.Tensor, out_path: str, fps: int) -> None:
    import imageio

    frames = frames_bcthw[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
    frames_u8 = (frames * 255.0).round().astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, list(frames_u8), fps=fps, format="mp4")


def _save_frames_png(frames_bcthw: torch.Tensor, out_dir: str, *, prefix: str) -> None:
    import imageio

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    frames = frames_bcthw[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
    frames_u8 = (frames * 255.0).round().astype(np.uint8)
    for i, frame in enumerate(frames_u8):
        imageio.imwrite(str(out_root / f"{prefix}_{i:04d}.png"), frame)


def _tensor_stats(name: str, x: torch.Tensor) -> str:
    x_f = x.detach().float()
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x_f.min().item():.4g} max={x_f.max().item():.4g} "
        f"mean={x_f.mean().item():.4g} std={x_f.std(unbiased=False).item():.4g}"
    )


def _compute_negative_prompt_embeddings(
    *,
    tokenizer,
    text_encoder,
    negative_prompt: str,
    max_sequence_length: int,
    dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    """
    Encode a negative prompt into embeddings that align with the transformer's text length.
    """
    assert negative_prompt, "Negative prompt must be provided for classifier-free guidance."
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


def _compute_prompt_embeddings(
    *,
    tokenizer,
    text_encoder,
    prompt: str,
    max_sequence_length: int,
    dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    encoder_device = next(text_encoder.parameters()).device
    text_encoder.eval()
    with torch.no_grad():
        tokens = tokenizer(
            [prompt],
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


def _save_png(frame_chw: torch.Tensor, out_path: str) -> None:
    import imageio

    img = frame_chw.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img_u8 = (img * 255.0).round().astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, img_u8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="parquet",
        choices=["parquet", "raw"],
        help="Input source mode: parquet dataset (default) or raw folders.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help=(
            "For input_mode=parquet: parquet dataset root. "
            "For input_mode=raw: optional raw sample root (if --raw_sample_root is not set)."
        ),
    )
    parser.add_argument(
        "--raw_sample_root",
        type=str,
        default="",
        help="Raw mode root. Supports either <root>/rgb... or <root>/*/rgb...",
    )
    parser.add_argument("--raw_rgb_dir", type=str, default="")
    parser.add_argument("--raw_depth_dir", type=str, default="")
    parser.add_argument("--raw_normal_dir", type=str, default="")
    parser.add_argument("--raw_mask_dir", type=str, default="")
    parser.add_argument("--raw_masked_rgb_dir", type=str, default="")
    parser.add_argument("--raw_caption_path", type=str, default="")
    parser.add_argument("--raw_caption_key", type=str, default="Video_Caption")
    parser.add_argument("--raw_prompt", type=str, default="")
    parser.add_argument("--raw_fps", type=int, default=16)
    parser.add_argument(
        "--conditioning_image_path",
        type=str,
        default="",
        help=(
            "Optional first-frame RGB image for rebuilding full TI2V image_latent online. "
            "Useful for parquet bidirectional inference, where parquet usually stores only first_frame_latent."
        ),
    )
    parser.add_argument("--raw_mask_threshold", type=float, default=-1.0)
    parser.add_argument("--raw_mask_invert", action="store_true")
    parser.add_argument("--raw_require_normal", action="store_true")
    parser.add_argument("--raw_depth_percentile_min", type=float, default=5.0)
    parser.add_argument("--raw_depth_percentile_max", type=float, default=95.0)
    raw_depth_group = parser.add_mutually_exclusive_group()
    raw_depth_group.add_argument("--raw_depth_invert",
                                 dest="raw_depth_invert",
                                 action="store_true")
    raw_depth_group.add_argument("--no_raw_depth_invert",
                                 dest="raw_depth_invert",
                                 action="store_false")
    parser.set_defaults(raw_depth_invert=False)
    parser.add_argument("--raw_first_frame_source",
                        type=str,
                        default="rgb",
                        choices=["rgb", "masked_rgb"])
    parser.add_argument(
        "--transformer_dir",
        type=str,
        required=True,
        help=
        "Diffusers-format folder used for transformer config (contains config.json + *.safetensors). "
        "Can be either (a) consolidated inference export dir, or (b) base model's transformer/ dir.",
    )
    parser.add_argument(
        "--controlnet_dir",
        type=str,
        required=True,
        help=
        "Diffusers-format folder used for ControlNet config (contains config.json + *.safetensors). "
        "Can be either (a) consolidated inference export dir, or (b) teacher ControlNet dir.",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default="causal",
        choices=["causal", "bidirectional"],
        help=
        "Use chunk-wise causal rollout (default) or full bidirectional rollout (no cache).",
    )
    parser.add_argument(
        "--align_run_wan_controlnet_union_md",
        action="store_true",
        help=(
            "Force the bidirectional inference recipe to match FastVideo/run_wan_contorlnet_union.md: "
            "bidirectional + UniPC + full 50-step schedule + fp32 + CFG + controlnet_weight=0.8."
        ),
    )
    parser.add_argument(
        "--controlnet_weight",
        type=float,
        default=1.0,
        help=
        "Scale factor for ControlNet residuals in bidirectional mode only (causal mode unaffected).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["flowmatch_euler", "unipc"],
        help="Scheduler family for inference. Use 'unipc' to match Diff-Factory Wan ControlNet inference defaults.",
    )
    parser.add_argument(
        "--init_transformer_safetensors",
        type=str,
        default="",
        help=
        "Optional: path to a FastVideo init .safetensors for the STUDENT transformer (e.g. phase-1 export). "
        "When set, weights are loaded from this file but config is still read from --transformer_dir.",
    )
    parser.add_argument(
        "--init_controlnet_safetensors",
        type=str,
        default="",
        help=
        "Optional: path to a FastVideo init .safetensors for the STUDENT controlnet (e.g. phase-1 export). "
        "When set, weights are loaded from this file but config is still read from --controlnet_dir.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=1024)
    # Diff-Factory's Wan video export examples use fps=16 for visualization.
    # Treat this as the *output* fps (not the dataset fps stored in parquet),
    # so users can match Diff-Factory behavior even if preprocess wrote fps=30.
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--schedule_num_inference_steps",
        type=int,
        default=50,
        help="Teacher grid size used to define the student timestep anchors (usually 50).",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help=(
            "FlowMatch Euler scheduler shift used during training/inference. "
            "Diff-Factory Wan ControlNet distillation checkpoints expect 5.0, so keep this in sync."
        ),
    )
    parser.add_argument(
        "--timestep_indices",
        type=str,
        default="0,12,24,36",
        help="Comma-separated indices into scheduler.timesteps for few-step rollout (Diff-Factory-style).",
    )
    parser.add_argument(
        "--dmd_steps",
        type=str,
        default="",
        help=(
            "Optional: comma-separated DMD denoising steps in the 0..1000 grid (Self-Forcing/FastVideo training style). "
            "If set, overrides --timestep_indices. When --warp_denoising_step is enabled (default), "
            "these steps are mapped to `scheduler.timesteps[1000 - step]` to match training."
        ),
    )
    parser.add_argument(
        "--update_rule",
        type=str,
        default="euler_dt",
        choices=["renoise_x0", "euler_dt"],
        help=
        "How to propagate between anchors. `euler_dt` matches Diff-Factory / ODE-style Euler integration "
        "(deterministic update in sigma space). `renoise_x0` matches the Self-Forcing *training simulation* "
        "(x0 reconstruction then add_noise with fresh noise), which is usually NOT what you want for inference quality.",
    )
    parser.add_argument(
        "--full_schedule",
        action="store_true",
        help=(
            "Ignore timestep_indices/dmd_steps and run the full scheduler.set_timesteps("
            "schedule_num_inference_steps) loop (Diff-Factory-style inference). "
            "Recommended for teacher baselines or full-step checks."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale. For student inference, 1.0 (no CFG) matches training by default.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bad quality, worst quality",
        help="Negative prompt text used when guidance_scale != 1. Matches Diff-Factory run defaults.",
    )
    parser.add_argument(
        "--first_frame_timestep_zero",
        action="store_true",
        help=(
            "Force the first latent frame's timestep to 0 (Diff-Factory expand_timesteps behavior). "
            "Default OFF to match FastVideo self-forcing training."
        ),
    )
    parser.add_argument(
        "--reset_cache_each_block",
        action="store_true",
        help=(
            "Debug: reset KV/cross-attn caches to zero at the beginning of every block, "
            "so later blocks cannot attend to previous blocks. If outputs are still noisy, "
            "the issue is NOT caused by cross-block KV cache."
        ),
    )
    parser.add_argument(
        "--disable_cache_update",
        action="store_true",
        help=(
            "Debug: do not run the context-noise forward at the end of each block (no explicit cache commit). "
            "This is mainly for diagnosing cache-related issues."
        ),
    )
    parser.add_argument("--context_noise",
                        type=int,
                        default=0,
                        help="Context timestep used for cache update (0 = clean).")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=0,
        help=(
            "Override causal self-attention local window size in *latent frames*. "
            "0 = auto (keep checkpoint default for <=81 frames; use 21 for longer videos). "
            "Example: 21 means attend to a sliding window of ~21 latent frames."
        ),
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help=(
            "KV-cache sink size in *latent frames* (kept fixed and not evicted when using local attention). "
            "Set 1 to keep the first latent frame (TI2V anchor) permanently visible while sliding. "
            "NOTE: if you want '21 sliding frames PLUS 1 sink', set --local_attn_size 22 --sink_size 1."
        ),
    )
    warp_group = parser.add_mutually_exclusive_group()
    warp_group.add_argument(
        "--warp_denoising_step",
        dest="warp_denoising_step",
        action="store_true",
        default=True,
        help=(
            "Snap DMD anchors (0..1000) to the nearest values in scheduler.timesteps. "
            "Enabled by default to match training behavior."
        ),
    )
    warp_group.add_argument(
        "--no_warp_denoising_step",
        dest="warp_denoising_step",
        action="store_false",
        help="Disable warping and treat dmd_steps as real timesteps.",
    )
    parser.add_argument("--dtype",
                        type=str,
                        default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--debug_dump",
        action="store_true",
        help="Dump latent stats and save a decoded first-frame PNG for debugging.",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Also save the decoded video as PNG frames under out_dir/frames/<sample_id>/",
    )
    parser.add_argument(
        "--trace_rollout_jsonl",
        type=str,
        default="",
        help="Optional path to write per-step rollout trace as JSONL (mainly for bidirectional alignment checks).",
    )
    parser.add_argument(
        "--trace_rollout_overwrite",
        action="store_true",
        help="If set with --trace_rollout_jsonl, remove existing trace file before writing.",
    )
    parser.add_argument(
        "--control_depth_only",
        action="store_true",
        help=(
            "Use only the depth chunk from control_latent and zero out masked_rgb/mask chunks. "
            "This is useful to debug mask-conditioned artifacts without regenerating parquet."
        ),
    )
    args = parser.parse_args()
    if bool(args.align_run_wan_controlnet_union_md):
        args.attention_mode = "bidirectional"
        args.scheduler = "unipc"
        args.full_schedule = True
        args.schedule_num_inference_steps = 50
        args.guidance_scale = 6.0
        args.controlnet_weight = 0.8
        args.dtype = "fp32"
        if not str(args.negative_prompt or "").strip():
            args.negative_prompt = "bad quality, worst quality"
    if str(args.input_mode) == "parquet":
        if not str(args.data_path).strip():
            raise ValueError("--data_path is required when --input_mode parquet")
        if bool(args.align_run_wan_controlnet_union_md) and not str(
                args.conditioning_image_path).strip():
            logger.warning(
                "run_wan_contorlnet_union.md alignment is enabled, but --conditioning_image_path is empty. "
                "Parquet mode will fall back to first_frame_latent instead of a rebuilt full image_latent."
            )
    else:
        if not str(args.raw_sample_root).strip() and not str(args.data_path).strip():
            raise ValueError(
                "For --input_mode raw, provide --raw_sample_root or --data_path"
            )

    trace_jsonl_path = str(args.trace_rollout_jsonl).strip()
    if trace_jsonl_path and bool(args.trace_rollout_overwrite):
        tp = Path(trace_jsonl_path)
        if tp.exists():
            tp.unlink()

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
    guidance_scale_value = float(args.guidance_scale)
    if args.attention_mode == "bidirectional" and guidance_scale_value == 1.0:
        logger.warning(
            "bidir alignment: guidance_scale=1.0 differs from the Diff-Factory example default (5.0). "
            "Pass --guidance_scale 5.0 for a closer baseline match."
        )
    if args.attention_mode == "bidirectional" and args.dtype != "fp32":
        logger.warning(
            "bidir alignment: Diff-Factory examples typically run fp32. "
            "Pass --dtype fp32 if memory allows."
        )
    inference_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    timestep_indices = [int(x) for x in args.timestep_indices.split(",") if x.strip() != ""]
    dmd_steps = [int(x) for x in args.dmd_steps.split(",") if x.strip() != ""]
    dmd_steps_list: list[int] | None = dmd_steps if len(dmd_steps) > 0 else None
    timestep_indices_list: list[int] | None = timestep_indices if dmd_steps_list is None else None
    if bool(args.full_schedule):
        dmd_steps_list = None
        timestep_indices_list = None
    if args.scheduler == "unipc" and (dmd_steps_list is not None or timestep_indices_list is not None):
        logger.warning(
            "scheduler=unipc ignores dmd_steps/timestep_indices; using full %s-step schedule.",
            int(args.schedule_num_inference_steps),
        )
        dmd_steps_list = None
        timestep_indices_list = None

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
    use_union_controlnet = _controlnet_dir_is_union(args.controlnet_dir)
    if args.attention_mode == "bidirectional":
        fastvideo_args.override_transformer_cls_name = "WanTransformer3DModel"
        fastvideo_args.override_controlnet_cls_name = (
            "WanControlnetUnion3DModel" if use_union_controlnet else "WanControlnet3DModel"
        )
    else:
        # Ensure student rollout uses the chunk-wise causal transformer (KV cache),
        # even if the exported config.json says "WanTransformer3DModel".
        fastvideo_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
        fastvideo_args.override_controlnet_cls_name = (
            "CausalWanControlnetUnion3DModel" if use_union_controlnet else "CausalWanControlnet3DModel"
        )
    # Ensure we don't trigger Wan2.2 "transformer_2" boundary logic for TI2V.
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    if args.attention_mode == "bidirectional":
        # Diff-Factory Wan Union inference defaults to the non-expanded TI2V path.
        if hasattr(fastvideo_args.pipeline_config, "expand_timesteps"):
            fastvideo_args.pipeline_config.expand_timesteps = False
        if hasattr(fastvideo_args.pipeline_config, "dit_config"):
            fastvideo_args.pipeline_config.dit_config.expand_timesteps = False
        logger.info(
            "bidir alignment: forcing expand_timesteps=False to match Diff-Factory Wan ControlNet inference."
        )
    fastvideo_args.pipeline_config.warp_denoising_step = bool(
        args.warp_denoising_step)
    fastvideo_args.pipeline_config.dmd_denoising_steps = dmd_steps if dmd_steps_list is not None else []
    fastvideo_args.pipeline_config.context_noise = int(args.context_noise)

    if args.init_transformer_safetensors:
        fastvideo_args.init_weights_from_safetensors = args.init_transformer_safetensors
    if args.init_controlnet_safetensors:
        fastvideo_args.init_controlnet_weights_from_safetensors = args.init_controlnet_safetensors

    transformer = PipelineComponentLoader.load_module(
        "transformer", args.transformer_dir, "diffusers", fastvideo_args)
    controlnet = PipelineComponentLoader.load_module(
        "controlnet", args.controlnet_dir, "diffusers", fastvideo_args)
    if controlnet is None:
        logger.warning(
            "ControlNet loader returned None; retrying direct ControlNetLoader for %s",
            args.controlnet_dir,
        )
        from fastvideo.models.loader.component_loader import ControlNetLoader
        controlnet = ControlNetLoader().load(args.controlnet_dir, fastvideo_args)
    if controlnet is None:
        raise RuntimeError(
            f"ControlNet load failed for {args.controlnet_dir}. "
            "Ensure the directory contains config.json and *.safetensors."
        )
    # Keep runtime tensor dtype consistent with loaded model dtype to avoid
    # conv3d type mismatch (e.g., input float32 vs bf16 bias).
    try:
        model_param = next(
            p for p in transformer.parameters() if torch.is_floating_point(p))
    except StopIteration:
        model_param = None
    if model_param is None:
        try:
            model_param = next(
                p for p in controlnet.parameters() if torch.is_floating_point(p))
        except StopIteration:
            model_param = None
    if model_param is not None:
        model_dtype = model_param.dtype
        if dtype != model_dtype:
            logger.info(
                "dtype alignment: overriding runtime dtype from %s to model dtype %s",
                str(dtype),
                str(model_dtype),
            )
            dtype = model_dtype

    # Auto-enable a safe sliding window for long videos when checkpoint config is missing/unstable.
    # This avoids exploding KV cache memory and also avoids the "kv_cache_size==0" crash when
    # some exported configs accidentally set sliding_window_num_frames=0.
    if str(args.attention_mode) == "causal":
        local_attn_override = int(args.local_attn_size)
        if local_attn_override == 0 and int(args.num_frames) > 81:
            local_attn_override = 21
        if local_attn_override > 0:
            logger.info(
                "causal local attention override: local_attn_size=%s latent frames",
                local_attn_override,
            )
            _override_local_attn_size(transformer, local_attn_override)
            _override_local_attn_size(controlnet, local_attn_override)
        if int(args.sink_size) > 0:
            logger.info("causal KV-cache sink override: sink_size=%s latent frames", int(args.sink_size))
            _override_sink_size(transformer, int(args.sink_size))
            _override_sink_size(controlnet, int(args.sink_size))
        _log_causal_attn_overrides(transformer, name="transformer")
        _log_causal_attn_overrides(controlnet, name="controlnet")

    if args.init_transformer_safetensors and not args.init_controlnet_safetensors:
        logger.warning(
            "You set --init_transformer_safetensors but did not set --init_controlnet_safetensors. "
            "If you are testing phase-1 student init, you usually want to initialize BOTH transformer and controlnet "
        "from the same phase-1 export; otherwise you're mixing student transformer with teacher controlnet."
        )
    negative_prompt_embeds_global: torch.Tensor | None = None
    tokenizer = None
    text_encoder = None
    negative_prompt_text = str(args.negative_prompt or "").strip()
    need_text_encoder = (str(args.input_mode) == "raw") or (
        negative_prompt_text and guidance_scale_value != 1.0)
    if need_text_encoder:
        tokenizer = PipelineComponentLoader.load_module(
            "tokenizer", str(Path(args.base_model) / "tokenizer"), "transformers",
            fastvideo_args)
        text_encoder = PipelineComponentLoader.load_module(
            "text_encoder", str(Path(args.base_model) / "text_encoder"), "transformers",
            fastvideo_args)
    if negative_prompt_text and guidance_scale_value != 1.0:
        max_text_len = int(getattr(transformer, "text_len", 226))
        negative_prompt_embeds_global = _compute_negative_prompt_embeddings(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            negative_prompt=negative_prompt_text,
            max_sequence_length=max_text_len,
            dtype=dtype,
            target_device=inference_device,
        )
        logger.info(
            "negative_prompt=%r -> embeddings shape=%s",
            negative_prompt_text,
            tuple(negative_prompt_embeds_global.shape),
        )
    elif guidance_scale_value != 1.0 and not negative_prompt_text:
        logger.warning(
            "--guidance_scale != 1.0 but --negative_prompt is empty; falling back to zeros."
        )
    if str(args.input_mode) != "raw":
        # Raw mode needs tokenizer/text_encoder to build prompt embeddings on the fly.
        # Non-raw mode can release them immediately after negative prompt encoding.
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        if text_encoder is not None:
            del text_encoder
            text_encoder = None
    raw_sample_roots: list[Path] = []
    if str(args.input_mode) == "raw":
        if str(args.raw_rgb_dir).strip():
            raw_root = str(args.raw_sample_root).strip() or str(
                args.data_path).strip() or "."
            raw_sample_roots = [Path(raw_root).expanduser().resolve()]
            logger.info(
                "raw input mode: using explicit raw_*_dir overrides (base root=%s)",
                raw_root,
            )
        else:
            raw_root = str(args.raw_sample_root).strip() or str(
                args.data_path).strip()
            raw_sample_roots = _resolve_raw_sample_roots(raw_root)
            logger.info("raw input mode: found %s sample roots under %s",
                        len(raw_sample_roots), raw_root)
    if args.scheduler == "unipc":
        if args.attention_mode == "bidirectional":
            scheduler = DiffusersUniPCMultistepScheduler.from_pretrained(
                args.base_model, subfolder="scheduler")
            scheduler = DiffusersUniPCMultistepScheduler.from_config(
                scheduler.config, flow_shift=float(args.flow_shift))
        else:
            scheduler = FlowUniPCMultistepScheduler(shift=float(args.flow_shift))
    else:
        # Self-Forcing / DMD training uses Euler ODE (FlowMatchEulerDiscreteScheduler).
        # Do NOT use the base model's UniPC scheduler here. Also, the `shift` MUST match training.
        scheduler = FlowMatchEulerDiscreteScheduler(shift=float(args.flow_shift))
    vae = PipelineComponentLoader.load_module("vae",
                                              str(Path(args.base_model) / "vae"),
                                              "diffusers", fastvideo_args)

    decoding = DecodingStage(vae=vae)

    for i in range(args.num_samples):
        sample_idx = args.index + i
        if str(args.input_mode) == "raw":
            if sample_idx < 0 or sample_idx >= len(raw_sample_roots):
                raise IndexError(
                    f"--index {sample_idx} out of range for raw samples (n={len(raw_sample_roots)})"
                )
            sample = _load_sample_raw(
                sample_root=raw_sample_roots[sample_idx],
                sample_index=sample_idx,
                args=args,
                transformer=transformer,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                inference_device=inference_device,
                dtype=dtype,
            )
        else:
            sample = _load_sample(args.data_path, sample_idx)
        logger.info("sample=%s idx=%s caption=%s", sample.sample_id, sample_idx,
                    sample.caption)

        image_latent = sample.image_latent_bcfhw
        if image_latent is None and str(args.conditioning_image_path).strip():
            cond_img_path = Path(str(args.conditioning_image_path)).expanduser()
            if not cond_img_path.is_file():
                raise FileNotFoundError(
                    f"--conditioning_image_path not found: {str(cond_img_path)}")
            cond_rgb = _load_rgb_frame(cond_img_path, int(args.height),
                                       int(args.width))
            target_c = int(
                getattr(transformer, "num_channels_latents",
                        sample.first_frame_latent_bcfhw.shape[1]))
            image_latent = _build_online_image_latent_from_rgb(
                first_rgb_chw=cond_rgb,
                num_frames=int(args.num_frames),
                vae=vae,
                target_c=target_c,
                inference_device=inference_device,
                compute_dtype=dtype,
            )
            logger.info(
                "bidir alignment: rebuilt image_latent online from %s with shape=%s",
                str(cond_img_path),
                tuple(image_latent.shape),
            )

        prompt_embeds = sample.text_embedding_bld.to(device="cuda",
                                                     dtype=dtype)
        negative_prompt_embeds = None
        if float(args.guidance_scale) != 1.0:
            if negative_prompt_embeds_global is not None:
                negative_prompt_embeds = negative_prompt_embeds_global
            else:
                # Match FastVideo training convention: unconditional prompt embedding is all-zeros.
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        first_frame_latent = sample.first_frame_latent_bcfhw.to(device="cuda",
                                                                dtype=dtype)
        control_latent = sample.control_latent_bcfhw.to(device="cuda",
                                                        dtype=dtype)
        if args.control_depth_only:
            total_c = int(control_latent.shape[1])
            if total_c % 3 != 0:
                raise ValueError(
                    f"control_latent channels must be divisible by 3, got {total_c}"
                )
            base_c = total_c // 3
            control_latent = control_latent.clone()
            control_latent[:, base_c:] = 0
            logger.info(
                "control_depth_only enabled: kept first %s channels, zeroed remaining %s",
                base_c,
                total_c - base_c,
            )
        effective_first_frame_timestep_zero = bool(args.first_frame_timestep_zero)
        force_first_frame_anchor = bool(
            str(args.input_mode) == "parquet"
            and args.attention_mode == "bidirectional"
            and image_latent is None
            and first_frame_latent is not None)
        if force_first_frame_anchor:
            effective_first_frame_timestep_zero = True
            logger.info(
                "bidir parquet mode: no image_latent/image RGB available; enabling conservative first-frame anchor and timestep[0]=0."
            )
        if (args.attention_mode == "bidirectional"
                and effective_first_frame_timestep_zero
                and not force_first_frame_anchor):
            logger.info(
                "bidir alignment: first_frame_timestep_zero=True was requested explicitly."
            )
        elif first_frame_latent is not None and not effective_first_frame_timestep_zero:
            logger.warning(
                "TI2V alignment: first_frame_timestep_zero is OFF. "
                "This matches the current Diff-Factory default when expand_timesteps=False."
            )

        if args.debug_dump:
            logger.info(_tensor_stats("text_embedding", prompt_embeds))
            logger.info(_tensor_stats("first_frame_latent", first_frame_latent))
            logger.info(_tensor_stats("control_latent", control_latent))
            if image_latent is not None:
                logger.info(_tensor_stats("image_latent", image_latent))
            # Sanity-check decode: if this PNG already looks like noise, then the latent space / decode
            # path is mismatched (preprocess vs decode), independent of diffusion sampling quality.
            decoded_first = decoding.decode(first_frame_latent, fastvideo_args)[0]
            _save_png(
                decoded_first[:, 0],
                str(Path(args.out_dir) / f"{sample.sample_id}__first_frame.png"),
            )

        logger.info(
            "sampling: attention_mode=%s update_rule=%s guidance_scale=%s context_noise=%s reset_cache_each_block=%s disable_cache_update=%s",
            str(args.attention_mode),
            str(args.update_rule),
            float(args.guidance_scale),
            int(args.context_noise),
            bool(args.reset_cache_each_block),
            bool(args.disable_cache_update),
        )

        if args.attention_mode == "bidirectional":
            if dmd_steps_list is not None or timestep_indices_list is not None or not bool(
                    args.full_schedule):
                logger.warning(
                    "bidir alignment: ignoring dmd_steps/timestep_indices/update_rule; using full %s-step schedule.",
                    int(args.schedule_num_inference_steps),
                )
            latents = _bidirectional_dmd_rollout_ti2v_controlnet(
                transformer=transformer,
                controlnet=controlnet,
                scheduler=scheduler,
                prompt_embeds_list=[prompt_embeds],
                negative_prompt_embeds_list=([negative_prompt_embeds] if negative_prompt_embeds is not None else None),
                guidance_scale=float(args.guidance_scale),
                controlnet_weight=float(args.controlnet_weight),
                first_frame_latent_bcfhw=first_frame_latent,
                image_latent_bcfhw=(image_latent.to(device="cuda", dtype=dtype)
                                    if image_latent is not None else None),
                control_latent_bcfhw=control_latent,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                schedule_num_inference_steps=int(args.schedule_num_inference_steps),
                dmd_steps=dmd_steps_list,
                timestep_indices=timestep_indices_list,
                context_noise=args.context_noise,
                warp_denoising_step=bool(args.warp_denoising_step),
                update_rule=args.update_rule,
                full_schedule=bool(args.full_schedule),
                first_frame_timestep_zero=effective_first_frame_timestep_zero,
                expand_timesteps=bool(
                    getattr(fastvideo_args.pipeline_config, "expand_timesteps", False)),
                trace_jsonl_path=(trace_jsonl_path if trace_jsonl_path else None),
                trace_sample_id=sample.sample_id,
                seed=args.seed + i,
                dtype=dtype,
                force_first_frame_anchor=force_first_frame_anchor,
            )
        else:
            latents = _causal_dmd_rollout_ti2v_controlnet(
                transformer=transformer,
                controlnet=controlnet,
                scheduler=scheduler,
                prompt_embeds_list=[prompt_embeds],
                negative_prompt_embeds_list=([negative_prompt_embeds] if negative_prompt_embeds is not None else None),
                guidance_scale=float(args.guidance_scale),
                first_frame_latent_bcfhw=first_frame_latent,
                control_latent_bcfhw=control_latent,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                schedule_num_inference_steps=int(args.schedule_num_inference_steps),
                dmd_steps=dmd_steps_list,
                timestep_indices=timestep_indices_list,
                context_noise=args.context_noise,
                warp_denoising_step=bool(args.warp_denoising_step),
                update_rule=args.update_rule,
                full_schedule=bool(args.full_schedule),
                first_frame_timestep_zero=effective_first_frame_timestep_zero,
                expand_timesteps=bool(
                    getattr(fastvideo_args.pipeline_config, "expand_timesteps", False)),
                reset_cache_each_block=bool(args.reset_cache_each_block),
                disable_cache_update=bool(args.disable_cache_update),
                seed=args.seed + i,
                dtype=dtype,
            )

        if args.debug_dump:
            logger.info(_tensor_stats("generated_latents", latents))

        # Decode to pixels [B, C, T, H, W] in [0,1], then save mp4
        decoded = decoding.decode(latents, fastvideo_args).cpu().float()
        fps = int(args.fps)
        logger.info(
            "decoded_video: shape=%s fps=%s seconds=%.2f",
            tuple(decoded.shape),
            fps,
            float(decoded.shape[2]) / float(max(1, fps)),
        )
        out_path = str(Path(args.out_dir) / f"{sample.sample_id}.mp4")
        _save_mp4(decoded, out_path, fps=fps)
        if bool(args.save_frames):
            frames_dir = str(Path(args.out_dir) / "frames" / sample.sample_id)
            _save_frames_png(decoded, frames_dir, prefix=sample.sample_id)
        logger.info("saved: %s", out_path)

    if tokenizer is not None:
        del tokenizer
    if text_encoder is not None:
        del text_encoder


if __name__ == "__main__":
    main()
