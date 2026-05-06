#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare Diff-Factory-style raw preprocessing against FastVideo raw preprocessing
for the Wan ControlNet Union path.

This checks both pixel/control tensors and VAE latents, because visually close
debug mp4s can still become meaningfully different after VAE encoding.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader

import tools.infer_wan_controlnet_ti2v as base
import tools.infer_wan_controlnet_ti2v_long_firstframe_warp as longwarp


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[str(name)]


def _make_fastvideo_args(args: argparse.Namespace) -> FastVideoArgs:
    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=str(args.base_model),
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
    fastvideo_args.pipeline_config.dit_precision = str(args.dtype)
    return fastvideo_args


def _raw_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        raw_rgb_dir="",
        raw_depth_dir="",
        raw_normal_dir="",
        raw_mask_dir="",
        raw_masked_rgb_dir="",
        raw_require_normal=True,
        raw_prompt=str(args.prompt),
        raw_caption_path="",
        raw_caption_key="Video_Caption",
        raw_fps=int(args.fps),
        cam_k=str(args.cam_k),
        cam_rt_dir=str(args.cam_rt_dir),
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
    )


def _diff_rgb_frame(path: Path, crop_params: tuple[int, int, int, int],
                    height: int, width: int) -> torch.Tensor:
    top, left, ch, cw = crop_params
    img = Image.open(path).convert("RGB")
    img = img.crop((left, top, left + cw, top + ch)).resize(
        (int(width), int(height)), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _diff_depth_sequence(depth_paths: list[Path],
                         crop_params: tuple[int, int, int, int], height: int,
                         width: int) -> torch.Tensor:
    top, left, ch, cw = crop_params
    depths = []
    for p in depth_paths:
        d = base._read_depth_any(p).astype(np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        if p.suffix.lower() != ".exr":
            d = d / 65535.0
        depths.append(d)

    stacked = np.stack(depths, axis=0)
    finite = np.isfinite(stacked)
    if finite.any():
        stacked_for_minmax = np.where(finite, stacked,
                                      float(np.nanmax(stacked[finite])))
        global_min = float(np.nanmin(stacked_for_minmax))
        global_max = float(np.nanmax(stacked_for_minmax))
    else:
        global_min, global_max = 0.0, 1.0

    outputs = []
    denom = global_max - global_min + 1e-8
    for d in depths:
        dn = (d - global_min) / denom
        dn = np.nan_to_num(dn, nan=1.0)
        dn = dn * 2.0 - 1.0
        t = torch.from_numpy(dn).float().unsqueeze(0)
        t = TVF.resized_crop(
            t,
            int(top),
            int(left),
            int(ch),
            int(cw),
            (int(height), int(width)),
            interpolation=InterpolationMode.BILINEAR,
        )
        outputs.append(t.repeat(3, 1, 1))
    return torch.stack(outputs, dim=0)


def _diff_normal_sequence(normal_paths: list[Path],
                          crop_params: tuple[int, int, int, int],
                          height: int, width: int) -> torch.Tensor:
    top, left, ch, cw = crop_params
    outputs = []
    for p in normal_paths:
        n = base._read_normal_any(p).astype(np.float32)[..., :3]
        if float(np.nanmax(n)) > 1.5:
            n = n / 127.5 - 1.0
        n[..., 1] *= -1
        n[..., 2] *= -1
        t = torch.from_numpy(n).permute(2, 0, 1).float()
        t = TVF.resized_crop(
            t,
            int(top),
            int(left),
            int(ch),
            int(cw),
            (int(height), int(width)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        outputs.append(t)
    return torch.stack(outputs, dim=0)


def _stats(x: torch.Tensor) -> dict[str, Any]:
    xf = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "l2": float(torch.linalg.vector_norm(xf).item()),
    }


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    if tuple(af.shape) != tuple(bf.shape):
        return {
            "shape_a": list(af.shape),
            "shape_b": list(bf.shape),
            "shape_mismatch": True,
        }
    d = af - bf
    n = max(int(d.numel()), 1)
    l2_a = float(torch.linalg.vector_norm(af).item())
    return {
        "shape": list(af.shape),
        "mae": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt((d * d).mean()).item()),
        "max_abs": float(d.abs().max().item()),
        "l2_diff": float(torch.linalg.vector_norm(d).item()),
        "rel_l2": float(torch.linalg.vector_norm(d).item()) / max(l2_a, 1e-12),
        "mean_diff": float(d.mean().item()),
        "numel": n,
    }


@torch.no_grad()
def _encode_one(vae, video_tchw: torch.Tensor, *, normalize: bool,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    video = base._to_vae_input(video_tchw, normalize=normalize).to(device=device,
                                                                    dtype=dtype)
    return base._encode_video_latents(
        vae,
        video,
        sample_mode="mode",
        compute_dtype=dtype,
    ).detach().cpu()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    base._ensure_single_process_dist_env()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = _dtype(args.dtype)
    fastvideo_args = _make_fastvideo_args(args)

    tokenizer = PipelineComponentLoader.load_module(
        "tokenizer", str(Path(args.base_model) / "tokenizer"), "transformers",
        fastvideo_args)
    text_encoder = PipelineComponentLoader.load_module(
        "text_encoder", str(Path(args.base_model) / "text_encoder"),
        "transformers", fastvideo_args)
    sequence = longwarp._load_raw_long_sequence_nomask(
        sample_root=Path(args.raw_sample_root).expanduser().resolve(),
        args=_raw_args(args),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=SimpleNamespace(text_len=512),
        inference_device=device,
        dtype=dtype,
    )
    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    h, w = int(args.height), int(args.width)
    n = int(args.num_frames)
    rgb_path = sequence.rgb_paths[0]
    depth_paths = sequence.depth_paths[:n]
    normal_paths = sequence.normal_paths[:n]
    crop_params = tuple(sequence.crop_params)

    diff_rgb = _diff_rgb_frame(rgb_path, crop_params, h, w)
    fast_rgb = base._load_rgb_frame(rgb_path, h, w)
    diff_depth = _diff_depth_sequence(depth_paths, crop_params, h, w)
    fast_depth = base._load_depth_sequence(
        depth_paths,
        h,
        w,
        pmin=2.0,
        pmax=98.0,
        invert_depth=False,
        normalization_mode="md_align",
        crop_params=crop_params,
    )
    diff_normal = _diff_normal_sequence(normal_paths, crop_params, h, w)
    fast_normal = torch.stack(
        [base._load_normal_frame(p, h, w) for p in normal_paths], dim=0)

    depth_path_by_id = {
        int(fid): p
        for fid, p in zip(sequence.frame_ids, sequence.depth_paths)
    }
    first_rgb_u8 = longwarp._chw_float_to_u8(diff_rgb)
    warped_masked_rgb, warped_mask = (
        longwarp._warp_maskrgb_from_keyframes_md_aligned(
            keyframe_rgbs_u8=[first_rgb_u8],
            keyframe_frame_ids=[int(sequence.frame_ids[0])],
            target_frame_ids=sequence.frame_ids[1:n],
            depth_path_by_frame_id=depth_path_by_id,
            camera_k_aligned=sequence.camera_k_aligned,
            camera_rt_dir=sequence.camera_rt_dir,
            crop_params=crop_params,
            target_height=h,
            target_width=w,
        ))
    warped_masked_rgb = longwarp._pad_tchw(warped_masked_rgb, max(n - 1, 1))
    warped_mask = longwarp._pad_tchw(warped_mask, max(n - 1, 1))
    mask_tchw = torch.cat(
        [
            torch.ones((1, 1, h, w), dtype=torch.float32),
            warped_mask[:max(n - 1, 0)],
        ],
        dim=0,
    )
    masked_rgb_tchw = torch.cat(
        [
            diff_rgb.unsqueeze(0),
            warped_masked_rgb[:max(n - 1, 0)],
        ],
        dim=0,
    )
    mask3_tchw = mask_tchw.repeat(1, 3, 1, 1)

    vae = PipelineComponentLoader.load_module(
        "vae", str(Path(args.base_model) / "vae"), "diffusers",
        fastvideo_args).to(device)

    diff_first_lat = _encode_one(vae,
                                 diff_rgb.unsqueeze(0),
                                 normalize=True,
                                 device=device,
                                 dtype=dtype)
    fast_first_lat = _encode_one(vae,
                                 fast_rgb.unsqueeze(0),
                                 normalize=True,
                                 device=device,
                                 dtype=dtype)
    diff_depth_lat = _encode_one(vae,
                                 diff_depth,
                                 normalize=False,
                                 device=device,
                                 dtype=dtype)
    fast_depth_lat = _encode_one(vae,
                                 fast_depth,
                                 normalize=False,
                                 device=device,
                                 dtype=dtype)
    diff_normal_lat = _encode_one(vae,
                                  diff_normal,
                                  normalize=False,
                                  device=device,
                                  dtype=dtype)
    fast_normal_lat = _encode_one(vae,
                                  fast_normal,
                                  normalize=False,
                                  device=device,
                                  dtype=dtype)
    masked_lat = _encode_one(vae,
                             masked_rgb_tchw,
                             normalize=True,
                             device=device,
                             dtype=dtype)
    mask_lat = _encode_one(vae,
                           mask3_tchw,
                           normalize=False,
                           device=device,
                           dtype=dtype)

    report = {
        "settings": {
            "raw_sample_root": str(args.raw_sample_root),
            "height": h,
            "width": w,
            "num_frames": n,
            "dtype": str(args.dtype),
            "crop_params": list(crop_params),
        },
        "pixel_compare": {
            "rgb_first": _compare(diff_rgb, fast_rgb),
            "depth": _compare(diff_depth, fast_depth),
            "normal": _compare(diff_normal, fast_normal),
        },
        "latent_compare": {
            "first_frame": _compare(diff_first_lat, fast_first_lat),
            "depth": _compare(diff_depth_lat, fast_depth_lat),
            "normal": _compare(diff_normal_lat, fast_normal_lat),
        },
        "latent_stats": {
            "masked_rgb": _stats(masked_lat),
            "mask": _stats(mask_lat),
            "diff_first_frame": _stats(diff_first_lat),
            "fast_first_frame": _stats(fast_first_lat),
            "diff_depth": _stats(diff_depth_lat),
            "fast_depth": _stats(fast_depth_lat),
            "diff_normal": _stats(diff_normal_lat),
            "fast_normal": _stats(fast_normal_lat),
        },
    }

    out_path = Path(args.out_dir) / "input_align_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    for group in ("pixel_compare", "latent_compare"):
        print(f"[input-align] {group}")
        for name, stat in report[group].items():
            print(" ", name, "mae=", f"{stat['mae']:.6g}", "rmse=",
                  f"{stat['rmse']:.6g}", "rel_l2=",
                  f"{stat['rel_l2']:.6g}", "max_abs=",
                  f"{stat['max_abs']:.6g}")
    print("[input-align] wrote", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Wan ControlNet raw input alignment")
    p.add_argument("--base_model",
                   default="/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--raw_sample_root",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032")
    p.add_argument("--cam_k",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032/camera/camera_K.txt")
    p.add_argument("--cam_rt_dir",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032/camera")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--prompt", default="A driving scene in city street.")
    p.add_argument("--out_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/input_align_0032_fp32")
    return p.parse_args()


if __name__ == "__main__":
    main()
