#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode ODE trajectory latents from parquet to mp4 clips.

This script uses FastVideo's DecodingStage so latent de-normalization matches
training/inference behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    for dirpath, _dirnames, filenames in os.walk(root):
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
            table = pq.read_table(fp, columns=columns)
            row = table.slice(remaining, 1).to_pydict()
            return {
                k: (v[0] if isinstance(v, list) else v)
                for k, v in row.items()
            }
        remaining -= n
    raise IndexError(f"index out of range: {index}")


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


def _ensure_traj_scfhw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure trajectory latents are [S, C, F, H, W].
    Accepts:
      - [S, C, F, H, W]
      - [S, F, C, H, W]
      - [1, S, C, F, H, W]
      - [1, S, F, C, H, W]
    """
    channel_like = (1, 3, 16, 32, 48, 64)
    if x.dim() == 6 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 5:
        raise ValueError(f"Unsupported trajectory_latents shape: {tuple(x.shape)}")
    # [S, C, F, H, W]
    if x.shape[1] in channel_like and x.shape[2] > 4:
        return x
    # [S, F, C, H, W] -> [S, C, F, H, W]
    if x.shape[2] in channel_like and x.shape[1] > 4:
        return x.permute(0, 2, 1, 3, 4).contiguous()
    return x


def _ensure_timesteps_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() == 2 and x.shape[0] == 1:
        return x.squeeze(0)
    if x.dim() == 2 and x.shape[1] == 1:
        return x.squeeze(1)
    return x.flatten()


def _parse_indices(spec: str, n_states: int) -> list[int]:
    raw = [s.strip() for s in str(spec).split(",") if s.strip() != ""]
    if not raw:
        return list(range(n_states))
    out: list[int] = []
    for item in raw:
        i = int(item)
        if i < 0:
            i += n_states
        if i < 0 or i >= n_states:
            raise ValueError(
                f"Index {item} out of range for trajectory length {n_states}.")
        out.append(i)
    return out


def _save_video(decoded_bcthw: torch.Tensor, out_path: Path, fps: float) -> None:
    frames = decoded_bcthw[0]  # [C,T,H,W]
    pil_list = []
    for i in range(frames.shape[1]):
        img = frames[:, i].permute(1, 2, 0)
        arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil_list.append(Image.fromarray(arr, mode="RGB"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(pil_list, str(out_path), fps=float(fps))


def main() -> None:
    p = argparse.ArgumentParser("Decode ODE trajectory latents from parquet")
    p.add_argument("--data_path",
                   type=str,
                   required=True,
                   help="ODE parquet dir or parquet file")
    p.add_argument("--base_model",
                   type=str,
                   required=True,
                   help="Wan diffusers root (contains vae/)")
    p.add_argument("--index", type=int, default=0, help="Global sample index")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--indices",
                   type=str,
                   default="0,12,24,36,-2,-1",
                   help="Trajectory indices to decode")
    p.add_argument("--fps", type=float, default=0.0, help="Override fps if >0")
    args = p.parse_args()

    cols = [
        "id",
        "fps",
        "trajectory_latents_bytes",
        "trajectory_latents_shape",
        "trajectory_latents_dtype",
        "trajectory_timesteps_bytes",
        "trajectory_timesteps_shape",
        "trajectory_timesteps_dtype",
    ]
    row = _read_row_by_global_index(args.data_path, int(args.index), cols)
    traj = _ensure_traj_scfhw(_decode_tensor(row, "trajectory_latents")).float()
    traj_ts = _ensure_timesteps_1d(_decode_tensor(row, "trajectory_timesteps")
                                   ).float()

    n_states = int(traj.shape[0])
    sel = _parse_indices(args.indices, n_states)

    model_path = str(args.base_model)
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

    sample_id = str(row.get("id", f"index_{int(args.index):06d}"))
    fps = float(args.fps) if float(args.fps) > 0 else float(row.get("fps", 16.0) or 16.0)
    out_root = Path(os.path.expandvars(os.path.expanduser(args.out_dir))) / sample_id
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"sample_id={sample_id} traj_shape={tuple(traj.shape)} ts_shape={tuple(traj_ts.shape)}")
    print(f"selected_indices={sel}")
    for i in sel:
        timestep_val = float(traj_ts[i].item()) if i < traj_ts.numel() else float("nan")
        lat = traj[i].unsqueeze(0)  # [1,C,F,H,W]
        decoded = decoding.decode(lat, fastvideo_args).cpu().float()
        out_mp4 = out_root / f"traj_idx_{i:03d}_t_{timestep_val:.4f}.mp4"
        _save_video(decoded, out_mp4, fps)
        print(f"saved: {out_mp4}")


if __name__ == "__main__":
    main()

