#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan


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
    shape = row.get(f"{prefix}_shape")
    blob = row.get(f"{prefix}_bytes")
    dtype_str = row.get(f"{prefix}_dtype")
    if blob is None or shape is None:
        raise KeyError(f"Missing {prefix}_bytes/{prefix}_shape in parquet row")
    arr = np.frombuffer(blob, dtype=_np_dtype(dtype_str)).reshape(shape).copy()
    return torch.from_numpy(arr)


def _read_row(data_path: str, index: int, columns: list[str]) -> dict:
    if index < 0:
        raise ValueError("--sample_index must be >= 0")
    idx = index
    files = sorted(Path(data_path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_path}")
    for fp in files:
        pf = pq.ParquetFile(fp)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        if idx >= n:
            idx -= n
            continue
        table = pq.read_table(fp, columns=columns)
        row = table.slice(idx, 1).to_pydict()
        return {k: (v[0] if isinstance(v, list) else v) for k, v in row.items()}
    raise IndexError(f"sample_index {index} out of range")


def _ensure_bcfhw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(0)
    if x.dim() == 5:
        # BFCHW -> BCFHW
        if x.shape[1] in (1, 3, 16, 48) and x.shape[2] >= 8:
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


def _save_video_frames(video: torch.Tensor, prefix: str, out_dir: Path) -> None:
    video = video.detach().cpu()
    video = (video.clamp(-1, 1) + 1.0) * 0.5
    video = video.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    b, c, f, h, w = video.shape
    assert b >= 1
    video = video[0]
    for frame_idx in range(f):
        frame = video[:, frame_idx].permute(1, 2, 0).numpy()
        path = out_dir / f"{prefix}_frame{frame_idx:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)


def _save_mask_frames(video: torch.Tensor, prefix: str, out_dir: Path) -> None:
    mask = video.detach().cpu()
    mask = (mask.clamp(-1, 1) + 1.0) * 0.5
    mask = mask.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    b, c, f, h, w = mask.shape
    mask = mask[0]
    single = mask[0]
    for frame_idx in range(f):
        frame = single[frame_idx].numpy()
        path = out_dir / f"{prefix}_frame{frame_idx:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)


def _split_control_latents(control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = control.shape[1]
    if c % 3 != 0:
        raise ValueError(f"control channels ({c}) not divisible by 3")
    chunk = c // 3
    return control[:, :chunk], control[:, chunk:chunk * 2], control[:, chunk * 2:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one sample from TI2V ControlNet parquet.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Parquet dataset root (e.g. omnigame_ti2v_controlnet_parquet_0_8).")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/apdcephfs_zwfy2/share_303204533/suanhuang/pretrained_weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Diffusers base path with vae/ folder.",
    )
    parser.add_argument("--sample_index",
                        type=int,
                        default=0,
                        help="Global sample index (row order across shards).")
    parser.add_argument("--out_dir",
                        type=str,
                        default="outputs/parquet_decode",
                        help="Directory to write decoded images.")
    args = parser.parse_args()

    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }

    device = torch.device("cpu")
    vae = AutoencoderKLWan.from_pretrained(
        Path(args.base_model) / "vae", torch_dtype=torch.float32
    )
    vae.eval()
    vae.to(device)

    cols = [
        "id",
        "first_frame_latent_bytes",
        "first_frame_latent_shape",
        "first_frame_latent_dtype",
        "control_latent_bytes",
        "control_latent_shape",
        "control_latent_dtype",
    ]
    row = _read_row(args.dataset, args.sample_index, cols)
    sample_id = str(row.get("id", f"index_{args.sample_index:06d}"))

    first_frame_latent = _decode_tensor(row, "first_frame_latent")
    first_frame_latent = _ensure_first_frame_bcfhw(first_frame_latent)
    control_latent = _decode_tensor(row, "control_latent")
    control_latent = _ensure_bcfhw(control_latent, name="control_latent")

    def _denorm_latents(latents: torch.Tensor) -> torch.Tensor:
        # Match FastVideo DecodingStage: undo preprocessing normalization.
        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            mean = torch.tensor(vae.config.latents_mean,
                                device=latents.device,
                                dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std,
                               device=latents.device,
                               dtype=latents.dtype).view(1, -1, 1, 1, 1)
            latents = latents * std + mean
        else:
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

    def decode_latent(latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(device=device, dtype=torch.float32)
        latent = _denorm_latents(latent)
        with torch.no_grad():
            decoded = vae.decode(latent, return_dict=False)[0]
        return decoded

    first_frame_decoded = decode_latent(first_frame_latent)
    control_depth, control_masked, control_mask = _split_control_latents(control_latent)

    depth_decoded = decode_latent(control_depth)
    masked_decoded = decode_latent(control_masked)
    mask_decoded = decode_latent(control_mask)

    out_root = Path(args.out_dir) / sample_id
    _save_video_frames(first_frame_decoded, "first_frame", out_root / "first_frame")
    _save_video_frames(depth_decoded, "depth", out_root / "depth_frames")
    _save_video_frames(masked_decoded, "masked", out_root / "masked_frames")
    _save_mask_frames(mask_decoded, "mask", out_root / "mask_frames")

    print(f"Decoded sample={sample_id} -> {out_root}")


if __name__ == "__main__":
    main()
