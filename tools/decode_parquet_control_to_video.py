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
            table = pf.read_row_group(0, columns=columns)
            row = table.slice(remaining, 1).to_pylist()[0]
            return row
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

    cols = ["id", "control_latent_bytes", "control_latent_shape",
            "control_latent_dtype"]
    row = _read_row_by_global_index(args.data_path, int(args.index), cols)
    control_latent = _ensure_bcfhw(_decode_tensor(row, "control_latent"),
                                   name="control_latent").float()

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
    total_c = int(control_latent.shape[1])
    k = _infer_split_k(total_c, int(z_dim), prefer_three=not bool(args.prefer_four))
    c = total_c // k
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
        decoded_mask = (decoded_mask >= thr).to(decoded_mask.dtype)

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
