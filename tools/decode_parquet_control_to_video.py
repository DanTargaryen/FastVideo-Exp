#!/usr/bin/env python3
"""
Decode control_latent (depth/masked_rgb/mask, optionally normal) from FastVideo parquet
back to PNG sequence and/or mp4 using the Wan VAE.

Example:
  python tools/decode_parquet_control_to_video.py \\
    --data_path /path/to/parquet_dir \\
    --base_model /path/to/Wan2.2-TI2V-5B-Diffusers \\
    --index 0 \\
    --out_dir /path/to/out
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from diffusers.utils import export_to_video

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.stages.decoding import DecodingStage


def _list_parquet_files(root: str | os.PathLike[str]) -> list[str]:
    root = str(root)
    if os.path.isfile(root) and root.endswith(".parquet"):
        return [root]
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {root}")
    return paths


def _read_row_by_global_index(data_path: str, index: int,
                              columns: list[str]) -> dict:
    if index < 0:
        raise ValueError("--index must be >= 0")
    remaining = index
    for fp in _list_parquet_files(data_path):
        pf = pq.ParquetFile(fp)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        if remaining < n:
            # Handle files with multiple row groups.
            rg_count = pf.num_row_groups
            rg_remaining = remaining
            for rg in range(rg_count):
                rg_rows = int(pf.metadata.row_group(rg).num_rows)
                if rg_remaining >= rg_rows:
                    rg_remaining -= rg_rows
                    continue
                table = pf.read_row_group(rg, columns=columns)
                rows = table.slice(rg_remaining, 1).to_pylist()
                if not rows:
                    break
                return rows[0]
            raise IndexError(
                f"failed to locate row index={index} inside parquet file: {fp}"
            )
        remaining -= n
    raise IndexError(f"index out of range: {index}")


def _decode_tensor(row: dict, prefix: str) -> torch.Tensor:
    shape = row.get(f"{prefix}_shape", None)
    blob = row.get(f"{prefix}_bytes", None)
    dtype = row.get(f"{prefix}_dtype", None)
    if shape is None or blob is None or dtype is None:
        raise KeyError(f"Missing {prefix}_shape/bytes/dtype in parquet row")
    arr = np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(arr)


def _ensure_bcfhw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    """
    Ensure tensor is shaped [B, C, F, H, W].
    Accepts [C,F,H,W], [B,C,F,H,W], or [B,C,H,W].
    """
    if x.ndim == 4:
        # [C,F,H,W] -> [1,C,F,H,W]
        x = x.unsqueeze(0)
    elif x.ndim == 5:
        pass
    elif x.ndim == 4 and x.shape[0] != 0:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        # [C,H,W] -> [1,C,1,H,W]
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.ndim == 4 and x.shape[0] == 0:
        pass
    else:
        raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}")
    if x.ndim == 5:
        return x
    raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}")


def _infer_valid_k_from_channels(total_c: int, z_dim: int) -> list[int]:
    cand: list[int] = []
    # Primary check: (k * z_dim) divisibility.
    if int(z_dim) > 0:
        for k in (3, 4):
            if int(total_c) % (k * int(z_dim)) == 0:
                cand.append(k)
    # Fallback for mismatched z_dim metadata.
    if not cand:
        for k in (3, 4):
            if int(total_c) % k == 0:
                cand.append(k)
    return cand


def _looks_like_control_channel_count(total_c: int, z_dim: int) -> bool:
    return len(_infer_valid_k_from_channels(int(total_c), int(z_dim))) > 0


def _ensure_control_latent_bcfhw(
    x: torch.Tensor,
    *,
    z_dim: int,
    name: str,
) -> torch.Tensor:
    """
    Ensure control latent is [B, C_total, F, H, W].

    Supports both channel-first and frame-first layouts:
      - 4D: [C,F,H,W] or [F,C,H,W]
      - 5D: [B,C,F,H,W] or [B,F,C,H,W]
    """
    if x.ndim == 4:
        c0, f0 = int(x.shape[0]), int(x.shape[1])
        c0_ok = _looks_like_control_channel_count(c0, int(z_dim))
        f0_ok = _looks_like_control_channel_count(f0, int(z_dim))
        if c0_ok and not f0_ok:
            return x.unsqueeze(0)
        if f0_ok and not c0_ok:
            return x.permute(1, 0, 2, 3).contiguous().unsqueeze(0)
        # Ambiguous fallback: control channels are usually >= latent frames.
        if c0 >= f0:
            return x.unsqueeze(0)
        return x.permute(1, 0, 2, 3).contiguous().unsqueeze(0)

    if x.ndim == 5:
        c1, f1 = int(x.shape[1]), int(x.shape[2])
        c1_ok = _looks_like_control_channel_count(c1, int(z_dim))
        f1_ok = _looks_like_control_channel_count(f1, int(z_dim))
        if c1_ok and not f1_ok:
            return x
        if f1_ok and not c1_ok:
            return x.permute(0, 2, 1, 3, 4).contiguous()
        if c1 >= f1:
            return x
        return x.permute(0, 2, 1, 3, 4).contiguous()

    if x.ndim == 3:
        # Rare case: [C,H,W] -> [1,C,1,H,W]
        return x.unsqueeze(0).unsqueeze(2)

    raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}")


def _infer_split_k(total_c: int, z_dim: int, *, prefer_three: bool) -> int:
    # Choose 3 or 4 based on z_dim and total channels.
    cand = []
    for k in (3, 4):
        if total_c % (k * z_dim) == 0:
            cand.append(k)
    if not cand:
        # fallback by divisibility only
        if total_c % 3 == 0:
            return 3
        if total_c % 4 == 0:
            return 4
        raise ValueError(f"control_latent channels not divisible by 3 or 4: {total_c}")
    if len(cand) == 1:
        return cand[0]
    # both possible
    return 3 if prefer_three else 4


def _save_frames(decoded_bcthw: torch.Tensor, out_dir: Path, prefix: str,
                 as_gray: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    b, c, t, h, w = decoded_bcthw.shape
    frames = decoded_bcthw[0]  # C,T,H,W
    for i in range(t):
        if as_gray:
            img = frames[0, i]
            arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="L")
        else:
            img = frames[:, i].permute(1, 2, 0)
            arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
        pil.save(out_dir / f"{prefix}_{i:04d}.png")


def _save_video(decoded_bcthw: torch.Tensor, out_path: Path,
                fps: float, as_gray: bool) -> None:
    frames = decoded_bcthw[0]  # C,T,H,W
    pil_list = []
    for i in range(frames.shape[1]):
        if as_gray:
            img = frames[0, i]
            arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = frames[:, i].permute(1, 2, 0)
            arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
        pil_list.append(pil)
    export_to_video(pil_list, str(out_path), fps=float(fps))


def main() -> None:
    p = argparse.ArgumentParser("Decode control_latent from parquet to PNG/MP4")
    p.add_argument("--data_path", type=str, required=True,
                   help="Parquet dir or file")
    p.add_argument("--base_model", type=str, required=True,
                   help="Wan diffusers root (contains vae/)")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--fps", type=float, default=16)
    p.add_argument("--save_frames", action="store_true",
                   help="Also save PNG frames under out_dir/frames_*")
    p.add_argument("--prefer_four", action="store_true",
                   help="Prefer 4-way split when ambiguous")
    p.add_argument(
        "--mask_binarize_threshold",
        type=float,
        default=0.5,
        help=
        "Binarize decoded mask before saving (>=thr -> 1, else 0). Set <0 to disable.",
    )
    args = p.parse_args()

    cols = [
        "id",
        "num_frames",
        "control_latent_bytes",
        "control_latent_shape",
        "control_latent_dtype",
    ]
    row = _read_row_by_global_index(args.data_path, int(args.index), cols)
    raw_control_latent = _decode_tensor(row, "control_latent")

    # Load VAE
    model_path = args.base_model
    pipeline_config = PipelineConfig.from_pretrained(model_path)
    pipeline_config.update_config_from_dict({
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=False, load_decoder=True),
        "text_encoder_cpu_offload": False,
    })
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pipeline_config=pipeline_config,
    )
    vae = PipelineComponentLoader.load_module(
        module_name="vae",
        component_model_path=os.path.join(model_path, "vae"),
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    decoding = DecodingStage(vae=vae)

    # Split control latents
    z_dim = getattr(vae, "z_dim", None)
    if z_dim is None:
        z_dim = getattr(getattr(vae, "config", None), "z_dim", None)
    if z_dim is None:
        z_dim = 16
    control_latent = _ensure_control_latent_bcfhw(
        raw_control_latent,
        z_dim=int(z_dim),
        name="control_latent",
    ).float()

    total_c = int(control_latent.shape[1])
    valid_k = _infer_valid_k_from_channels(total_c, int(z_dim))
    if not valid_k:
        raise ValueError(
            "Invalid control_latent channels from parquet: "
            f"total_c={total_c}, z_dim={int(z_dim)}, raw_shape={tuple(raw_control_latent.shape)}, "
            f"bcfhw_shape={tuple(control_latent.shape)}. "
            "Expected total_c to be divisible by 3*z_dim or 4*z_dim."
        )
    k = _infer_split_k(total_c, int(z_dim), prefer_three=not bool(args.prefer_four))
    c = total_c // k
    print(
        f"[decode] id={row.get('id', '')} raw_shape={tuple(raw_control_latent.shape)} "
        f"-> control_bcfhw={tuple(control_latent.shape)} z_dim={int(z_dim)} split_k={k} chunk_c={c}"
    )
    depth_lat = control_latent[:, :c]
    if k == 4:
        normal_lat = control_latent[:, c:2 * c]
        masked_lat = control_latent[:, 2 * c:3 * c]
        mask_lat = control_latent[:, 3 * c:4 * c]
    else:
        normal_lat = None
        masked_lat = control_latent[:, c:2 * c]
        mask_lat = control_latent[:, 2 * c:3 * c]

    out_dir = Path(os.path.expandvars(os.path.expanduser(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    def _decode(lat: torch.Tensor) -> torch.Tensor:
        return decoding.decode(lat, fastvideo_args).cpu().float()

    decoded_depth = _decode(depth_lat)
    decoded_masked = _decode(masked_lat)
    decoded_mask = _decode(mask_lat)
    if float(args.mask_binarize_threshold) >= 0.0:
        thr = float(args.mask_binarize_threshold)
        # Aggregate channels before thresholding for stable binary masks.
        # mask was encoded from repeated 3-channel input, but decode may have
        # small per-channel deviations.
        decoded_mask_1c = decoded_mask.mean(dim=1, keepdim=True)
        decoded_mask_1c = (decoded_mask_1c >= thr).to(decoded_mask.dtype)
        decoded_mask = decoded_mask_1c.repeat(1, decoded_mask.shape[1], 1, 1, 1)

    _save_video(decoded_depth, out_dir / "depth.mp4", args.fps, as_gray=True)
    _save_video(decoded_masked, out_dir / "masked_rgb.mp4", args.fps, as_gray=False)
    _save_video(decoded_mask, out_dir / "mask.mp4", args.fps, as_gray=True)

    if args.save_frames:
        _save_frames(decoded_depth, out_dir / "frames_depth", "depth", as_gray=True)
        _save_frames(decoded_masked, out_dir / "frames_masked_rgb", "masked_rgb", as_gray=False)
        _save_frames(decoded_mask, out_dir / "frames_mask", "mask", as_gray=True)

    if normal_lat is not None:
        decoded_normal = _decode(normal_lat)
        _save_video(decoded_normal, out_dir / "normal.mp4", args.fps, as_gray=False)
        if args.save_frames:
            _save_frames(decoded_normal, out_dir / "frames_normal", "normal", as_gray=False)

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
