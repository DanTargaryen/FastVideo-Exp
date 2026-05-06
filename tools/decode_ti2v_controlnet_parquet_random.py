#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan


def _np_dtype(dtype_str: str | None) -> np.dtype:
    if not dtype_str:
        return np.float32
    s = str(dtype_str).lower()
    if s in ("float", "float32", "fp32"):
        return np.float32
    if s in ("float16", "fp16"):
        return np.float16
    raise ValueError(f"Unsupported dtype in parquet: {dtype_str}")


def _decode_tensor(row: dict, prefix: str) -> torch.Tensor:
    shape = row.get(f"{prefix}_shape")
    blob = row.get(f"{prefix}_bytes")
    dtype_str = row.get(f"{prefix}_dtype")
    if blob is None or shape is None:
        raise KeyError(f"Missing {prefix}_bytes/{prefix}_shape")
    arr = np.frombuffer(blob, dtype=_np_dtype(dtype_str)).reshape(shape).copy()
    return torch.from_numpy(arr)


def _safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "sample"


def _list_rows(dataset: Path) -> tuple[list[Path], list[int], int]:
    files = sorted(dataset.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset}")
    counts = []
    total = 0
    for fp in files:
        n = int(pq.ParquetFile(fp).metadata.num_rows)
        counts.append(n)
        total += n
    return files, counts, total


def _read_global_row(files: list[Path], counts: list[int], index: int, columns: list[str]) -> dict:
    local = int(index)
    for fp, n in zip(files, counts):
        if local >= int(n):
            local -= int(n)
            continue
        table = pq.read_table(fp, columns=columns)
        row = table.slice(local, 1).to_pydict()
        out = {k: (v[0] if isinstance(v, list) else v) for k, v in row.items()}
        out["_parquet_file"] = str(fp)
        out["_parquet_local_index"] = int(local)
        out["_global_index"] = int(index)
        return out
    raise IndexError(index)


def _ensure_bcfhw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(0)
    if x.ndim == 5:
        return x
    raise ValueError(f"{name} must be [C,F,H,W] or [B,C,F,H,W], got {tuple(x.shape)}")


def _ensure_first_frame_bcfhw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(2)
    if x.ndim == 4:
        if int(x.shape[0]) != 1:
            raise ValueError(f"first_frame_latent [F,C,H,W] must have F=1, got {tuple(x.shape)}")
        return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    if x.ndim == 5:
        if int(x.shape[2]) == 1:
            return x
        if int(x.shape[1]) == 1:
            return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Unsupported first_frame_latent shape: {tuple(x.shape)}")


def _denorm_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
        mean = torch.tensor(
            vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        return latents * std + mean
    scale = getattr(vae, "scaling_factor", None)
    if scale is not None:
        if isinstance(scale, torch.Tensor):
            scale = scale.to(latents.device, latents.dtype)
        latents = latents / scale
    shift = getattr(vae, "shift_factor", None)
    if shift is not None:
        if isinstance(shift, torch.Tensor):
            shift = shift.to(latents.device, latents.dtype)
        latents = latents + shift
    return latents


@torch.no_grad()
def _decode_latent(vae: AutoencoderKLWan, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    latent = _ensure_bcfhw(latent, name="latent").to(device=device, dtype=torch.float32)
    latent = _denorm_latents(vae, latent)
    decoded = vae.decode(latent, return_dict=False)[0]
    return (decoded / 2.0 + 0.5).clamp(0, 1).detach().cpu().float()


def _save_mp4(frames_bcthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames_bcthw[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
    frames_u8 = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    imageio.mimsave(str(path), list(frames_u8), fps=int(fps), format="mp4")


def _make_panel(videos: list[torch.Tensor], *, max_width: int = 416) -> torch.Tensor:
    resized = []
    for v in videos:
        x = v
        if int(x.shape[-1]) != max_width:
            scale = float(max_width) / float(x.shape[-1])
            h = int(round(float(x.shape[-2]) * scale))
            b, c, t, old_h, old_w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, old_h, old_w)
            x_2d = torch.nn.functional.interpolate(
                x_2d,
                size=(h, max_width),
                mode="bilinear",
                align_corners=False,
            )
            x = x_2d.reshape(b, t, c, h, max_width).permute(0, 2, 1, 3, 4)
        resized.append(x)
    top = torch.cat(resized[:3], dim=-1)
    bottom = torch.cat(resized[3:], dim=-1)
    if bottom.shape[-1] < top.shape[-1]:
        pad = top.shape[-1] - bottom.shape[-1]
        bottom = torch.nn.functional.pad(bottom, (0, pad, 0, 0))
    return torch.cat([top, bottom], dim=-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_root = Path(args.out_dir)
    files, counts, total = _list_rows(dataset)
    if total <= 0:
        raise RuntimeError(f"No rows under {dataset}")
    k = min(int(args.num_samples), total)
    rng = random.Random(int(args.seed))
    indices = sorted(rng.sample(range(total), k=k))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    vae = AutoencoderKLWan.from_pretrained(Path(args.base_model) / "vae", torch_dtype=torch.float32)
    vae.eval().to(device)

    columns = [
        "id",
        "caption",
        "vae_latent_bytes",
        "vae_latent_shape",
        "vae_latent_dtype",
        "first_frame_latent_bytes",
        "first_frame_latent_shape",
        "first_frame_latent_dtype",
        "control_latent_bytes",
        "control_latent_shape",
        "control_latent_dtype",
    ]
    manifest = {
        "dataset": str(dataset),
        "total_rows_at_decode": int(total),
        "sample_indices": [int(x) for x in indices],
        "seed": int(args.seed),
        "created_unix": float(time.time()),
        "samples": [],
    }

    for ordinal, idx in enumerate(indices):
        row = _read_global_row(files, counts, idx, columns)
        sample_id = str(row.get("id", f"index_{idx:06d}"))
        sample_dir = out_root / f"sample_{ordinal:02d}_row_{idx:06d}_{_safe_name(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb_lat = _decode_tensor(row, "vae_latent")
        first_lat = _ensure_first_frame_bcfhw(_decode_tensor(row, "first_frame_latent"))
        control_lat = _ensure_bcfhw(_decode_tensor(row, "control_latent"), name="control_latent")
        branch_c = int(rgb_lat.shape[0])
        if int(control_lat.shape[1]) != 4 * branch_c:
            raise ValueError(
                f"Expected 4 control branches with C={branch_c}; got control shape={tuple(control_lat.shape)}"
            )
        depth_lat = control_lat[:, :branch_c]
        normal_lat = control_lat[:, branch_c:2 * branch_c]
        masked_lat = control_lat[:, 2 * branch_c:3 * branch_c]
        mask_lat = control_lat[:, 3 * branch_c:4 * branch_c]

        decoded_rgb = _decode_latent(vae, rgb_lat, device)
        decoded_first = _decode_latent(vae, first_lat, device)
        decoded_depth = _decode_latent(vae, depth_lat, device)
        decoded_normal = _decode_latent(vae, normal_lat, device)
        decoded_masked = _decode_latent(vae, masked_lat, device)
        decoded_mask = _decode_latent(vae, mask_lat, device)

        _save_mp4(decoded_rgb, sample_dir / "rgb_from_vae_latent.mp4", int(args.fps))
        _save_mp4(decoded_first, sample_dir / "first_frame_latent.mp4", int(args.fps))
        _save_mp4(decoded_depth, sample_dir / "control_depth.mp4", int(args.fps))
        _save_mp4(decoded_normal, sample_dir / "control_normal.mp4", int(args.fps))
        _save_mp4(decoded_masked, sample_dir / "control_masked_rgb.mp4", int(args.fps))
        _save_mp4(decoded_mask, sample_dir / "control_mask.mp4", int(args.fps))
        panel = _make_panel([
            decoded_rgb,
            decoded_depth,
            decoded_normal,
            decoded_masked,
            decoded_mask,
            decoded_first.expand(-1, -1, decoded_rgb.shape[2], -1, -1),
        ])
        _save_mp4(panel, sample_dir / "panel_rgb_depth_normal_masked_mask_first.mp4", int(args.fps))

        sample_meta = {
            "global_index": int(idx),
            "id": sample_id,
            "caption": row.get("caption"),
            "parquet_file": row.get("_parquet_file"),
            "parquet_local_index": row.get("_parquet_local_index"),
            "vae_latent_shape": list(rgb_lat.shape),
            "first_frame_latent_shape": list(first_lat.shape),
            "control_latent_shape": list(control_lat.shape),
            "output_dir": str(sample_dir),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(sample_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["samples"].append(sample_meta)
        print(f"decoded row={idx} id={sample_id} -> {sample_dir}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
