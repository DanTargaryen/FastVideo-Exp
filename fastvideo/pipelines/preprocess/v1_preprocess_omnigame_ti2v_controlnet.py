# SPDX-License-Identifier: Apache-2.0
"""
Preprocess OmniWorld-Game (pickle manifest) into FastVideo parquet for:
  - TI2V (first-frame latent)
  - ControlNet (control latent = cat(depth, warped_masked_rgb, warped_mask) in VAE latent space)

Inputs:
  - `--in_pickle`: a list[dict] manifest (same structure as Diff-Factory phase1 recorder)
      keys used: scene_name, frame_indices, caption_path, video_path, control_path
  - `--warp_out_root`: output from Diff-Factory `tools/depth_warp.py`
      warp_out/<scene_name>/warped_masked_rgb/{frame:06d}.png
      warp_out/<scene_name>/warped_mask/{frame:06d}.png

Output:
  - Parquet dataset directory compatible with `pyarrow_schema_ti2v_controlnet`.

Notes:
  - If Wan VAE latent channel size is `C_lat` (aka z_dim), then each of (depth/masked_rgb/mask)
    encodes to `C_lat` channels, and the concatenated control latent has `3*C_lat` channels.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.dataset.dataloader.parquet_io import (ParquetDatasetWriter,
                                                     records_to_table)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ti2v_controlnet
from fastvideo.distributed import (get_local_torch_device, get_world_size,
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


def _load_mask_frame(path: str, height: int, width: int) -> torch.Tensor:
    img = Image.open(_maybe_expand(path)).convert("L")
    img = _resize_for_crop_pil(img, crop_h=int(height), crop_w=int(width))
    arr = (np.asarray(img).astype(np.float32) / 255.0)  # HW in [0,1]
    arr = (arr > 0.5).astype(np.float32)
    t = torch.from_numpy(arr)[None, ...].repeat(3, 1, 1).contiguous()  # 3HW
    return t


def _load_depth_frames_from_folder(control_dir: str, frame_indices: list[int],
                                   height: int, width: int) -> torch.Tensor:
    """
    Mirrors Diff-Factory `wan_controlnet_phase2_pickle.py::_load_depth_frames_from_folder`.
    Returns (T,3,H,W) in [0,1].
    """

    def _center_crop_to_aspect_2d(arr2d: np.ndarray) -> np.ndarray:
        h0, w0 = arr2d.shape[:2]
        target_ratio = float(width) / float(height)
        current_ratio = float(w0) / float(h0)
        if current_ratio > target_ratio:
            new_w = int(h0 * target_ratio)
            left = max(0, (w0 - new_w) // 2)
            return arr2d[:, left:left + new_w]
        new_h = int(w0 / target_ratio)
        top = max(0, (h0 - new_h) // 2)
        return arr2d[top:top + new_h, :]

    def _resize_2d(arr2d: np.ndarray) -> np.ndarray:
        img = Image.fromarray(arr2d)
        img = img.resize((int(width), int(height)), resample=Image.NEAREST)
        return np.array(img)

    folder = _maybe_expand(control_dir)
    hist = np.zeros(65536, dtype=np.int64)
    raws: list[np.ndarray] = []
    for idx in frame_indices:
        fname = os.path.join(folder, f"{int(idx):06d}.png")
        arr = np.array(Image.open(fname))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        hist += np.bincount(arr.reshape(-1), minlength=65536).astype(np.int64)
        raws.append(arr)

    uniq_vals = np.flatnonzero(hist)
    if uniq_vals.size == 0:
        low, high = 0.0, 1.0
    else:
        low = float(np.percentile(uniq_vals, 5))
        high = float(np.percentile(uniq_vals, 95))
        if high - low < 1e-6:
            high = low + 1e-6

    frames = []
    for raw in raws:
        depth = raw.astype(np.float32)
        depth = (depth - low) / (high - low)
        depth = np.clip(depth, 0.0, 1.0)
        depth = _center_crop_to_aspect_2d(depth)
        depth = _resize_2d(depth)
        depth = 1.0 - depth
        depth3 = np.repeat(depth[..., None], 3, axis=2)
        frames.append(torch.from_numpy(depth3).permute(2, 0, 1).contiguous())

    return torch.stack(frames, dim=0)  # TCHW


def _to_vae_input(video_tchw: torch.Tensor) -> torch.Tensor:
    # (T,3,H,W) in [0,1] -> (1,3,T,H,W) in [-1,1]
    x = video_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
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
def _encode_video_latents(vae, video_bcthw: torch.Tensor) -> torch.Tensor:
    # Returns (B,16,T_lat,H_lat,W_lat) float32 on device.
    with torch.autocast(device_type="cuda",
                        dtype=torch.float32,
                        enabled=torch.cuda.is_available()
                        and video_bcthw.is_cuda):
        out = vae.encode(video_bcthw)
    latents = out.mean
    latents = _postprocess_vae_latents(vae, latents)
    return latents


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
                   required=True,
                   help="Output root from Diff-Factory/tools/depth_warp.py")
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
    if int(get_world_size()) != 1:
        raise RuntimeError("This preprocess script only supports 1 GPU/process.")

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

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

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

    samples = pickle.load(open(_maybe_expand(args.in_pickle), "rb"))
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    if not isinstance(samples, list):
        raise TypeError(f"Expected list in pickle, got {type(samples)}")

    start = int(args.start)
    end = len(samples) if int(args.end) < 0 else min(int(args.end), len(samples))
    warp_root = Path(_maybe_expand(args.warp_out_root))

    writer = ParquetDatasetWriter(args.output_dir,
                                 samples_per_file=int(args.samples_per_file))
    buffer: list[dict[str, Any]] = []

    for idx in tqdm(range(start, end), desc="preprocess"):
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
        first_bcthw = _to_vae_input(first_rgb[None, ...])  # 1,3,1,H,W
        first_bcthw = first_bcthw.to(device=device, dtype=torch.float32)
        first_lat = _encode_video_latents(vae, first_bcthw)  # 1,16,1,h,w
        first_lat = first_lat[0, :, 0].unsqueeze(0)  # 1,16,h,w (F,C,H,W)
        first_lat_np = first_lat.to("cpu", dtype=torch.float32).numpy()

        # ---- control latents ----
        depth_tchw = _load_depth_frames_from_folder(control_dir, frame_indices,
                                                    args.max_height,
                                                    args.max_width)  # T,3,H,W

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
                    f"{int(i):06d}.png"), args.max_height, args.max_width)
            for i in frame_indices
        ],
                              dim=0)

        # Encode 3 sequences in a single batched call: (3,3,T,H,W)
        video_3 = torch.cat([
            _to_vae_input(depth_tchw),
            _to_vae_input(masked_rgb_tchw),
            _to_vae_input(mask_tchw),
        ],
                            dim=0).to(device=device, dtype=torch.float32)
        lat_3 = _encode_video_latents(vae, video_3)  # 3,16,T_lat,h,w
        lat_3 = lat_3.to("cpu", dtype=torch.float32)

        depth_lat = lat_3[0]  # 16,T_lat,h,w
        masked_lat = lat_3[1]
        mask_lat = lat_3[2]
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

    logger.info("Done. Wrote parquet dataset to: %s", args.output_dir)


if __name__ == "__main__":
    main(parse_args())
