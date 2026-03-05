#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Patch one parquet sample by inverting depth via:
  decode(depth_latent) -> (1 - depth) -> re-encode -> replace depth_latent

This avoids latent-domain hacks and performs invert in pixel space.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

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


def _locate_row(data_path: str, index: int) -> tuple[str, int]:
    if index < 0:
        raise ValueError("--index must be >= 0")
    remaining = index
    for fp in _list_parquet_files(data_path):
        pf = pq.ParquetFile(fp)
        n = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        if remaining < n:
            return fp, remaining
        remaining -= n
    raise IndexError(f"index out of range: {index}")


def _np_dtype(dtype_str: str | None) -> np.dtype:
    if dtype_str is None or dtype_str == "":
        return np.dtype(np.float32)
    s = dtype_str.lower()
    if s in ("float", "float32", "fp32"):
        return np.dtype(np.float32)
    if s in ("float16", "fp16"):
        return np.dtype(np.float16)
    if s in ("int64", "long"):
        return np.dtype(np.int64)
    if s in ("int32",):
        return np.dtype(np.int32)
    raise ValueError(f"Unsupported dtype: {dtype_str}")


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
def _encode_video_latents(
    vae,
    video_bcthw: torch.Tensor,
    *,
    sample_mode: str,
    compute_dtype: torch.dtype = torch.float32,
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


def _align_latent_channels(lat: torch.Tensor, target_c: int, name: str) -> torch.Tensor:
    if lat.ndim != 4:
        raise ValueError(f"{name} must be [C,F,H,W], got {tuple(lat.shape)}")
    c = int(lat.shape[0])
    if c == target_c:
        return lat
    if c < target_c and target_c % c == 0:
        return lat.repeat(target_c // c, 1, 1, 1)
    if c > target_c and c % target_c == 0:
        return lat[:target_c]
    raise ValueError(
        f"{name} channel mismatch: got C={c}, target C={target_c}, cannot align"
    )


def _pick_control_split(control_c: int, base_c: int) -> int:
    if base_c <= 0:
        raise ValueError(f"Invalid base_c={base_c}")
    if control_c == 4 * base_c:
        return 4
    if control_c == 3 * base_c:
        return 3
    raise ValueError(
        f"control_latent channels mismatch: control_c={control_c}, base_c={base_c}, expected 3*base_c or 4*base_c"
    )


def main() -> None:
    p = argparse.ArgumentParser("patch_parquet_depth_invert_one")
    p.add_argument("--data_path", type=str, required=True, help="Parquet dir/file")
    p.add_argument("--base_model", type=str, required=True, help="Wan diffusers root")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--dtype",
                   type=str,
                   default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--sample_mode",
                   type=str,
                   default="mode",
                   choices=["mode", "sample"])
    p.add_argument("--id_suffix",
                   type=str,
                   default="_depth_reinverted",
                   help="Append suffix to id/file_name; empty to keep original.")
    args = p.parse_args()

    fp, local_idx = _locate_row(args.data_path, int(args.index))
    table = pq.read_table(fp)
    if local_idx >= table.num_rows:
        raise IndexError(
            f"local index {local_idx} out of range for file {fp} with {table.num_rows} rows"
        )
    row = table.slice(local_idx, 1).to_pylist()[0]

    required = [
        "control_latent_bytes",
        "control_latent_shape",
        "control_latent_dtype",
        "first_frame_latent_shape",
    ]
    for k in required:
        if k not in row:
            raise KeyError(f"Missing key in parquet row: {k}")

    control_shape = list(row["control_latent_shape"])
    control_dtype_str = str(row.get("control_latent_dtype", "float32"))
    control_np = np.frombuffer(row["control_latent_bytes"],
                               dtype=_np_dtype(control_dtype_str)).reshape(
                                   control_shape).copy()
    control_t = torch.from_numpy(control_np).float()  # [Ctot,F,H,W]
    if control_t.ndim != 4:
        raise ValueError(
            f"Expected control_latent [C,F,H,W], got {tuple(control_t.shape)}")

    ff_shape = row["first_frame_latent_shape"]
    if len(ff_shape) >= 2:
        base_c = int(ff_shape[1]) if len(ff_shape) == 4 else int(ff_shape[0])
    else:
        raise ValueError(f"Invalid first_frame_latent_shape: {ff_shape}")
    ctot = int(control_t.shape[0])
    k = _pick_control_split(ctot, base_c)

    depth_lat = control_t[:base_c].unsqueeze(0)  # [1,C,F,H,W]

    model_path = str(args.base_model)
    pipeline_config = PipelineConfig.from_pretrained(model_path)
    pipeline_config.update_config_from_dict({
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=True, load_decoder=True),
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

    compute_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[str(args.dtype)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_dec = decoding.decode(depth_lat.to(device=device, dtype=compute_dtype),
                                fastvideo_args).cpu().float()  # [1,3,T,H,W]
    depth_dec_inv = 1.0 - depth_dec

    depth_tchw = depth_dec_inv[0].permute(1, 0, 2, 3).contiguous()  # [T,3,H,W]
    depth_bcthw = _to_vae_input(depth_tchw, normalize=False).to(device=device,
                                                                 dtype=compute_dtype)
    depth_reenc = _encode_video_latents(
        vae,
        depth_bcthw,
        sample_mode=str(args.sample_mode),
        compute_dtype=compute_dtype,
    )[0].detach().cpu().float()  # [C,F,H,W]
    depth_reenc = _align_latent_channels(depth_reenc, base_c, "depth_reenc")

    control_new = control_t.clone()
    control_new[:base_c] = depth_reenc

    out_dtype = _np_dtype(control_dtype_str)
    control_new_np = control_new.numpy().astype(out_dtype, copy=False)
    row["control_latent_bytes"] = control_new_np.tobytes()
    row["control_latent_shape"] = list(control_new_np.shape)
    row["control_latent_dtype"] = str(out_dtype.name)

    sid = str(row.get("id", f"index_{int(args.index):06d}"))
    if str(args.id_suffix):
        sid = f"{sid}{args.id_suffix}"
        row["id"] = sid
        if "file_name" in row:
            row["file_name"] = sid

    out_dir = Path(os.path.expandvars(os.path.expanduser(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "part-00000.parquet"
    out_table = pa.Table.from_pydict({k: [v] for k, v in row.items()})
    pq.write_table(out_table, out_fp)

    print("OK")
    print(f"in_file: {fp}")
    print(f"in_global_index: {int(args.index)} (local={local_idx})")
    print(f"sample_id: {sid}")
    print(f"split_k: {k}, base_c: {base_c}, control_shape: {tuple(control_new_np.shape)}")
    print(f"out_file: {out_fp}")


if __name__ == "__main__":
    main()
