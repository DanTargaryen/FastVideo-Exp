# SPDX-License-Identifier: Apache-2.0
"""
Preprocess OmniWorld-Game (pickle manifest) into FastVideo parquet for:
  - TI2V (first-frame latent)
  - ControlNet (control latent = cat(depth, [normal], warped_masked_rgb, warped_mask) in VAE latent space)

Inputs:
  - `--in_pickle`: a list[dict] manifest (same structure as Diff-Factory phase1 recorder)
      keys used: scene_name, frame_indices, caption_path, video_path, control_path
  - `--warp_out_root` (default) or `--no-use-warp-out` + `--mask_root`:
      - warp_out/<scene_name>/warped_masked_rgb/{frame:06d}.png
      - warp_out/<scene_name>/warped_mask/{frame:06d}.png
    If you disable warp_out, the script will read masks from the pickle's
    `mask_path` (if present) or `--mask_root`, and build masked_rgb = rgb * mask.

Output:
  - Parquet dataset directory compatible with `pyarrow_schema_ti2v_controlnet`.

Notes:
  - If Wan VAE latent channel size is `C_lat` (aka z_dim), then each of (depth/normal/masked_rgb/mask)
    encodes to `C_lat` channels. The concatenated control latent has `4*C_lat` channels
    when normal is provided (otherwise `3*C_lat`).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import ftfy
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import (get_local_torch_device, get_world_rank,
                                   get_world_size,
                                   maybe_init_distributed_environment_and_model_parallel)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


def _maybe_expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _read_text_file(path: str) -> str:
    p = _maybe_expand(path)
    text = Path(p).read_text(encoding="utf-8").strip()
    if Path(p).suffix.lower() != ".json":
        return text
    try:
        obj = json.loads(text)
    except Exception:
        return text

    def _pick(d: dict) -> str | None:
        for k in ("prompt", "caption", "text", "description", "caption_en",
                  "caption_zh"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v and isinstance(v[0], str):
                vv = str(v[0]).strip()
                if vv:
                    return vv
            if isinstance(v, dict):
                for kk in ("en", "zh", "text", "caption", "prompt"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
        caps = d.get("captions")
        if isinstance(caps, dict):
            for kk in ("Short_Caption", "Long_Caption", "short", "long",
                       "caption", "text"):
                vv = caps.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
        return None

    if isinstance(obj, dict):
        picked = _pick(obj)
        if picked:
            return picked
    return text


def _prompt_clean(text: str) -> str:
    text = ftfy.fix_text(str(text))
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resize_for_crop_pil(img: Image.Image, crop_h: int,
                         crop_w: int) -> Image.Image:
    img_w, img_h = img.size
    if (img_h >= crop_h and img_w >= crop_w) or (img_h <= crop_h
                                                 and img_w <= crop_w):
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    img = img.resize((out_w, out_h), resample=Image.BICUBIC)
    left = max(0, (out_w - crop_w) // 2)
    top = max(0, (out_h - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def _load_rgb_frame(path: str, height: int, width: int) -> torch.Tensor:
    img = Image.open(_maybe_expand(path)).convert("RGB")
    img = _resize_for_crop_pil(img, crop_h=int(height), crop_w=int(width))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return t


def _load_mask_frame(path: str,
                     height: int,
                     width: int,
                     *,
                     threshold: float | None,
                     invert: bool) -> torch.Tensor:
    img = Image.open(_maybe_expand(path)).convert("L")
    arr_u8 = np.asarray(img).astype(np.uint8)
    # Match Diff-Factory: center-crop to aspect, then resize with NEAREST.
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
    arr = arr_u8.astype(np.float32) / 255.0  # HW in [0,1]
    if threshold is not None:
        arr = (arr > float(threshold)).astype(np.float32)
    if invert:
        arr = 1.0 - arr
    t = torch.from_numpy(arr)[None, ...].repeat(3, 1, 1).contiguous()  # 3HW
    return t


def _load_depth_frames_from_folder(control_dir: str, frame_indices: list[int],
                                   height: int, width: int) -> torch.Tensor:
    """
    Mirrors Diff-Factory `process_control_depth` in run_wan_controlnet_depth_mask.py.
    Returns (T,3,H,W) with valid depth in [0,1] and invalid pixels = -1.
    """
    folder = _maybe_expand(control_dir)
    depths: list[np.ndarray] = []
    target_ratio = float(width) / float(height)
    for idx in frame_indices:
        fname = os.path.join(folder, f"{int(idx):06d}.png")
        depthmap = imageio.imread(fname).astype(np.float32)
        if depthmap.ndim == 3:
            depthmap = depthmap[..., 0]
        depthmap = depthmap / 65535.0

        h0, w0 = depthmap.shape
        current_ratio = float(w0) / float(h0)
        if current_ratio > target_ratio:
            new_w = int(h0 * target_ratio)
            left = max(0, (w0 - new_w) // 2)
            depthmap = depthmap[:, left:left + new_w]
        else:
            new_h = int(w0 / target_ratio)
            top = max(0, (h0 - new_h) // 2)
            depthmap = depthmap[top:top + new_h, :]

        depthmap = cv2.resize(depthmap, (int(width), int(height)),
                              interpolation=cv2.INTER_NEAREST)

        near_mask = depthmap < 0.0015
        far_mask = depthmap > (65500.0 / 65535.0)
        valid = ~(near_mask | far_mask)
        depthmap[~valid] = np.nan
        depths.append(depthmap)

    stacked = np.stack(depths, 0)
    global_min = np.nanmin(stacked)
    global_max = np.nanmax(stacked)
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 0.0, 1.0
    if (global_max - global_min) < 1e-6:
        global_max = global_min + 1e-6

    frames = []
    for depthmap in depths:
        depthmap = (depthmap - global_min) / (global_max - global_min)
        depthmap = 1.0 - depthmap
        depthmap = np.nan_to_num(depthmap, nan=-1.0)
        t = torch.from_numpy(depthmap).float().unsqueeze(0).repeat(3, 1, 1)
        frames.append(t)

    return torch.stack(frames, dim=0)  # TCHW


def _load_normal_frames_from_folder(normal_dir: str, frame_indices: list[int],
                                    height: int, width: int,
                                    normal_format: str = "opencv") -> torch.Tensor:
    """
    Mirrors Diff-Factory `process_normal` in run_wan_controlnet_union.py.
    Returns (T,3,H,W) with values in [-1,1].
    """
    folder = _maybe_expand(normal_dir)
    normals: list[torch.Tensor] = []
    target_ratio = float(width) / float(height)
    for idx in frame_indices:
        fname = os.path.join(folder, f"{int(idx):06d}.png")
        normal = imageio.imread(fname)
        if normal.ndim == 2:
            raise ValueError(f"Normal map must be RGB: {fname}")
        normal = normal[..., :3].astype(np.float32)
        h0, w0, _ = normal.shape
        current_ratio = float(w0) / float(h0)
        if current_ratio > target_ratio:
            new_w = int(h0 * target_ratio)
            left = max(0, (w0 - new_w) // 2)
            normal = normal[:, left:left + new_w]
        else:
            new_h = int(w0 / target_ratio)
            top = max(0, (h0 - new_h) // 2)
            normal = normal[top:top + new_h, :]
        normal = cv2.resize(normal, (int(width), int(height)),
                            interpolation=cv2.INTER_LINEAR)
        if normal.max() > 1.5:
            normal = normal / 127.5 - 1.0
        if normal_format == "opencv":
            normal[..., 1] *= -1
            normal[..., 2] *= -1
        t = torch.from_numpy(normal).permute(2, 0, 1).float()
        normals.append(t)
    return torch.stack(normals, dim=0)


def _to_vae_input(video_tchw: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    # (T,3,H,W) -> (1,3,T,H,W); normalize controls [0,1] -> [-1,1] mapping.
    x = video_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    if normalize:
        x = x * 2.0 - 1.0
    return x


def _postprocess_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    # Match FastVideo preprocess: shift_factor then scaling_factor.
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
                          sample_mode: str) -> torch.Tensor:
    # Returns (B,16,T_lat,H_lat,W_lat) float32 on device.
    with torch.autocast(device_type="cuda",
                        dtype=torch.float32,
                        enabled=torch.cuda.is_available()
                        and video_bcthw.is_cuda):
        out = vae.encode(video_bcthw)
    if sample_mode == "mode":
        latents = out.mode()
    elif sample_mode == "sample":
        latents = out.sample()
    else:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")
    latents = _postprocess_vae_latents(vae, latents)
    return latents


def _infer_latent_repeat(model_path: str, z_dim: int) -> int:
    cfg_path = Path(model_path) / "transformer" / "config.json"
    if not cfg_path.exists():
        return 1
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return 1
    in_channels = cfg.get("in_channels")
    if isinstance(in_channels, (int, float)):
        in_channels = int(in_channels)
        if in_channels > 0 and in_channels % int(z_dim) == 0:
            return in_channels // int(z_dim)
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Preprocess OmniWorld-Game TI2V+ControlNet parquet")
    p.add_argument("--model_path",
                   type=str,
                   required=True,
                   help="Wan diffusers root (contains text_encoder/vae/etc)")
    p.add_argument("--in_pickle",
                   type=str,
                   required=True,
                   help="Input pickle manifest (list[dict])")
    p.add_argument("--warp_out_root",
                   type=str,
                   default="",
                   help="Output root from Diff-Factory/tools/depth_warp.py")
    p.add_argument("--use-warp-out",
                   dest="use_warp_out",
                   action="store_true",
                   help="Use warp_out_root for masked_rgb/mask (default: True).")
    p.add_argument("--no-use-warp-out",
                   dest="use_warp_out",
                   action="store_false",
                   help="Do NOT use warp_out_root; build masked_rgb from rgb*mask.")
    p.set_defaults(use_warp_out=True)
    p.add_argument(
        "--mask_source",
        type=str,
        default="",
        choices=["", "warp_out", "pickle"],
        help=(
            "Override mask/masked_rgb source. "
            "'warp_out' uses warp_out_root; 'pickle' uses mask_path/mask_root with masked_rgb=rgb*mask. "
            "Empty means follow --use-warp-out/--no-use-warp-out."
        ),
    )
    p.add_argument("--mask_root",
                   type=str,
                   default="",
                   help="Mask root dir when not using warp_out (fallback if pickle has no mask_path).")
    p.add_argument("--normal_root",
                   type=str,
                   default="",
                   help="Normal root dir (optional). If set or if pickle has normal_path, "
                        "normal maps will be encoded and concatenated into control_latent.")
    p.add_argument("--normal_format",
                   type=str,
                   default="opencv",
                   choices=["opencv", "opengl"],
                   help="Normal map format. 'opencv' will flip Y/Z to OpenGL.")
    p.add_argument("--mask_threshold",
                   type=float,
                   default=0.5,
                   help="Mask binarization threshold in [0,1]. Set <0 to keep soft mask values.")
    p.add_argument("--mask_invert",
                   action="store_true",
                   help="Invert mask after loading/binarization (1->0, 0->1).")
    p.add_argument("--latent_repeat",
                   type=int,
                   default=0,
                   help="Repeat VAE latent channels to match transformer in_channels. "
                   "0=auto-detect from transformer config; 1=keep as-is; 3=repeat 3x, etc.")
    p.add_argument("--output_dir",
                   type=str,
                   required=True,
                   help="Output parquet dataset directory")
    p.add_argument("--max_height", type=int, default=384)
    p.add_argument("--max_width", type=int, default=512)
    p.add_argument("--clip_length", type=int, default=81)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--samples_per_file", type=int, default=8)
    p.add_argument("--flush_frequency",
                   type=int,
                   default=8,
                   help="How often to flush parquet chunks")
    p.add_argument("--device",
                   type=str,
                   default="cuda",
                   choices=["cuda", "cpu"])
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    model_path = maybe_download_model(args.model_path)

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    world_size = int(get_world_size())
    rank = int(get_world_rank())

    pipeline_config = PipelineConfig.from_pretrained(model_path)
    pipeline_config.update_config_from_dict({
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=True, load_decoder=False),
        "text_encoder_cpu_offload": False,
    })
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=get_world_size(),
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pipeline_config=pipeline_config,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        # Each torchrun process uses its local CUDA device.
        device = get_local_torch_device()
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load modules from the diffusers model folder.
    text_encoder = PipelineComponentLoader.load_module(
        module_name="text_encoder",
        component_model_path=os.path.join(model_path, "text_encoder"),
        transformers_or_diffusers="transformers",
        fastvideo_args=fastvideo_args,
    ).to(device)
    tokenizer = PipelineComponentLoader.load_module(
        module_name="tokenizer",
        component_model_path=os.path.join(model_path, "tokenizer"),
        transformers_or_diffusers="transformers",
        fastvideo_args=fastvideo_args,
    )
    vae = PipelineComponentLoader.load_module(
        module_name="vae",
        component_model_path=os.path.join(model_path, "vae"),
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    ).to(device)

    text_stage = TextEncodingStage(text_encoders=[text_encoder],
                                   tokenizers=[tokenizer])

    z_dim = getattr(vae, "z_dim", None)
    if z_dim is None:
        z_dim = getattr(getattr(vae, "config", None), "z_dim", None)
    if z_dim is None:
        z_dim = getattr(getattr(vae, "config", None), "arch_config", None)
        z_dim = getattr(z_dim, "z_dim", None)
    if z_dim is None:
        z_dim = 16
    if int(args.latent_repeat) > 0:
        latent_repeat = int(args.latent_repeat)
    else:
        latent_repeat = _infer_latent_repeat(model_path, int(z_dim))
    latent_repeat = max(1, int(latent_repeat))
    if latent_repeat != 1:
        logger.info("latent_repeat=%s (z_dim=%s) -> in_channels=%s", latent_repeat, z_dim,
                    int(z_dim) * int(latent_repeat))

    samples = pickle.load(open(_maybe_expand(args.in_pickle), "rb"))
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in pickle, got {type(samples)}")

    start = int(args.start)
    end = len(samples) if int(args.end) < 0 else min(int(args.end), len(samples))
    use_warp_out = bool(args.use_warp_out)
    if str(args.mask_source).strip():
        use_warp_out = str(args.mask_source).strip().lower() == "warp_out"
    if use_warp_out and not args.warp_out_root:
        raise ValueError("--warp_out_root is required when mask_source=warp_out or --use-warp-out is True")
    warp_root = Path(_maybe_expand(args.warp_out_root)) if args.warp_out_root else None

    # Multi-GPU preprocessing: each rank writes to its own subdir to avoid file name collisions.
    out_dir_rank = os.path.join(args.output_dir, f"rank_{rank:02d}")
    writer = ParquetDatasetWriter(out_dir_rank,
                                 samples_per_file=int(args.samples_per_file))
    buffer: list[dict[str, Any]] = []

    indices = list(range(start, end))
    if world_size > 1:
        indices = indices[rank::world_size]
    pbar = tqdm(indices, desc=f"preprocess[r{rank}/{world_size}]")

    for idx in pbar:
        s = samples[idx]
        if not isinstance(s, dict):
            continue

        scene_name = str(s.get("scene_name", f"sample_{idx:06d}"))
        frame_indices = list(s.get("frame_indices") or [])
        if int(args.clip_length) > 0:
            frame_indices = frame_indices[:int(args.clip_length)]
        if not frame_indices:
            continue

        caption_path = s.get("caption_path", "")
        if isinstance(s.get("prompt"), str) and s["prompt"].strip():
            prompt = s["prompt"].strip()
        elif caption_path:
            prompt = _read_text_file(str(caption_path))
        else:
            prompt = ""
        prompt = _prompt_clean(prompt)

        video_dir = _maybe_expand(str(s["video_path"]))  # .../<scene>/color
        control_dir = _maybe_expand(str(s["control_path"]))  # .../<scene>/depth
        first_idx = int(frame_indices[0])
        first_rgb_path = os.path.join(video_dir, f"{first_idx:06d}.png")

        # ---- text embedding (store as float32 [L,D]) ----
        embeds_list = text_stage.encode_text(prompt,
                                             fastvideo_args=fastvideo_args,
                                             encoder_index=0,
                                             return_attention_mask=False,
                                             return_type="list",
                                             device=device,
                                             dtype=torch.float32)
        text_emb = embeds_list[0]
        if text_emb.ndim == 3:
            text_emb = text_emb[0]
        text_emb = text_emb.detach().to("cpu", dtype=torch.float32).numpy()

        # ---- first-frame latent (store as float32 [F=1,C=16,H,W]) ----
        first_rgb = _load_rgb_frame(first_rgb_path, args.max_height,
                                    args.max_width)  # 3HW in [0,1]
        first_bcthw = _to_vae_input(first_rgb[None, ...],
                                    normalize=True)  # 1,3,1,H,W
        first_bcthw = first_bcthw.to(device=device, dtype=torch.float32)
        # Match Diff-Factory inference: use deterministic VAE mode for first-frame.
        first_lat = _encode_video_latents(vae,
                                          first_bcthw,
                                          sample_mode="mode")  # 1,16,1,h,w
        if latent_repeat != 1:
            first_lat = first_lat.repeat(1, latent_repeat, 1, 1, 1)
        first_lat = first_lat[0, :, 0].unsqueeze(0)  # 1,16,h,w (F,C,H,W)
        first_lat_np = first_lat.to("cpu", dtype=torch.float32).numpy()

        # ---- control latents ----
        depth_tchw = _load_depth_frames_from_folder(control_dir, frame_indices,
                                                    args.max_height,
                                                    args.max_width)  # T,3,H,W
        normal_dir = str(s.get("normal_path", "") or s.get("normal_dir", ""))
        if not normal_dir and args.normal_root:
            normal_dir = str(args.normal_root)
        normal_tchw = None
        if normal_dir:
            normal_dir = _maybe_expand(normal_dir)
            if os.path.isdir(normal_dir) and not os.path.exists(
                    os.path.join(normal_dir, f"{int(frame_indices[0]):06d}.png")):
                candidate = os.path.join(normal_dir, scene_name)
                if os.path.isdir(candidate):
                    normal_dir = candidate
            normal_tchw = _load_normal_frames_from_folder(
                normal_dir,
                frame_indices,
                args.max_height,
                args.max_width,
                normal_format=str(args.normal_format),
            )

        if use_warp_out:
            assert warp_root is not None
            masked_rgb_tchw = torch.stack([
                _load_rgb_frame(
                    str(warp_root / scene_name / "warped_masked_rgb" /
                        f"{int(i):06d}.png"), args.max_height, args.max_width)
                for i in frame_indices
            ],
                                           dim=0)

            mask_tchw = torch.stack([
                _load_mask_frame(
                    str(warp_root / scene_name / "warped_mask" /
                        f"{int(i):06d}.png"),
                    args.max_height,
                    args.max_width,
                    threshold=None if float(args.mask_threshold) < 0 else float(args.mask_threshold),
                    invert=bool(args.mask_invert),
                )
                for i in frame_indices
            ],
                                  dim=0)
        else:
            mask_dir = s.get("mask_path", "")
            if not mask_dir:
                if not args.mask_root:
                    raise ValueError(
                        "mask_path missing in pickle; pass --mask_root when --no-use-warp-out")
                mask_dir = args.mask_root
            mask_dir = _maybe_expand(str(mask_dir))
            if os.path.isdir(mask_dir) and not os.path.exists(
                    os.path.join(mask_dir, f"{int(frame_indices[0]):06d}.png")):
                mask_dir = os.path.join(mask_dir, scene_name)

            rgb_tchw = torch.stack([
                _load_rgb_frame(os.path.join(video_dir, f"{int(i):06d}.png"),
                                args.max_height, args.max_width)
                for i in frame_indices
            ],
                                   dim=0)
            mask_tchw = torch.stack([
                _load_mask_frame(
                    os.path.join(mask_dir, f"{int(i):06d}.png"),
                    args.max_height,
                    args.max_width,
                    threshold=None if float(args.mask_threshold) < 0 else float(args.mask_threshold),
                    invert=bool(args.mask_invert),
                )
                for i in frame_indices
            ],
                                  dim=0)
            masked_rgb_tchw = rgb_tchw * mask_tchw

        # Encode 3 or 4 sequences in a single batched call: (N,3,T,H,W)
        # Match Diff-Factory: depth/mask in [0,1], masked_rgb in [-1,1], normal already in [-1,1].
        if normal_tchw is not None:
            video_n = torch.cat([
                _to_vae_input(depth_tchw, normalize=False),
                _to_vae_input(normal_tchw, normalize=False),
                _to_vae_input(masked_rgb_tchw, normalize=True),
                _to_vae_input(mask_tchw, normalize=False),
            ],
                                dim=0).to(device=device, dtype=torch.float32)
            lat_n = _encode_video_latents(vae, video_n,
                                          sample_mode="mode")  # 4,16,T_lat,h,w
            lat_n = lat_n.to("cpu", dtype=torch.float32)
            depth_lat = lat_n[0]
            normal_lat = lat_n[1]
            masked_lat = lat_n[2]
            mask_lat = lat_n[3]
        else:
            video_n = torch.cat([
                _to_vae_input(depth_tchw, normalize=False),
                _to_vae_input(masked_rgb_tchw, normalize=True),
                _to_vae_input(mask_tchw, normalize=False),
            ],
                                dim=0).to(device=device, dtype=torch.float32)
            lat_n = _encode_video_latents(vae, video_n,
                                          sample_mode="mode")  # 3,16,T_lat,h,w
            lat_n = lat_n.to("cpu", dtype=torch.float32)
            depth_lat = lat_n[0]
            masked_lat = lat_n[1]
            mask_lat = lat_n[2]
        if latent_repeat != 1:
            depth_lat = depth_lat.repeat(latent_repeat, 1, 1, 1)
            if normal_tchw is not None:
                normal_lat = normal_lat.repeat(latent_repeat, 1, 1, 1)
            masked_lat = masked_lat.repeat(latent_repeat, 1, 1, 1)
            mask_lat = mask_lat.repeat(latent_repeat, 1, 1, 1)
        if normal_tchw is not None:
            control_lat = torch.cat([depth_lat, normal_lat, masked_lat, mask_lat],
                                    dim=0)  # 64,T_lat,h,w
        else:
            control_lat = torch.cat([depth_lat, masked_lat, mask_lat],
                                    dim=0)  # 48,T_lat,h,w
        control_lat_np = control_lat.numpy()

        # Optional: no GT video latents for self-forcing (simulate_generator_forward=True)
        empty_lat = np.zeros((0,), dtype=np.float32)

        record_id = f"{scene_name}__{idx:06d}"
        record = {
            "id": record_id,
            "text_embedding_bytes": text_emb.tobytes(),
            "text_embedding_shape": list(text_emb.shape),
            "text_embedding_dtype": "float32",
            "vae_latent_bytes": empty_lat.tobytes(),
            "vae_latent_shape": [],
            "vae_latent_dtype": "",
            "first_frame_latent_bytes": first_lat_np.tobytes(),
            "first_frame_latent_shape": list(first_lat_np.shape),
            "first_frame_latent_dtype": "float32",
            "control_latent_bytes": control_lat_np.tobytes(),
            "control_latent_shape": list(control_lat_np.shape),
            "control_latent_dtype": "float32",
            "file_name": record_id,
            "caption": prompt,
            "media_type": "video",
            "width": int(args.max_width),
            "height": int(args.max_height),
            "num_frames": int(control_lat_np.shape[1])
            if len(control_lat_np.shape) > 1 else 0,
            "duration_sec": float(len(frame_indices)) / float(args.fps)
            if args.fps > 0 else 0.0,
            "fps": float(args.fps),
        }
        buffer.append(record)

        if len(buffer) >= int(args.flush_frequency):
            table = records_to_table(buffer, pyarrow_schema_ti2v_controlnet)
            writer.append_table(table)
            writer.flush(num_workers=1, write_remainder=False)
            buffer = []

    if buffer:
        table = records_to_table(buffer, pyarrow_schema_ti2v_controlnet)
        writer.append_table(table)
        writer.flush(num_workers=1, write_remainder=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        logger.info("Done. Wrote parquet dataset to: %s (per-rank shards under rank_XX/)", args.output_dir)


if __name__ == "__main__":
    main(parse_args())
