#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader

import tools.infer_wan_controlnet_ti2v as base
import tools.infer_wan_controlnet_ti2v_long_firstframe_warp as longwarp

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export first long-warp window mask/masked_rgb preview")
    p.add_argument("--base_model", required=True)
    p.add_argument("--raw_sample_root", required=True)
    p.add_argument("--raw_rgb_dir", default="")
    p.add_argument("--raw_depth_dir", default="")
    p.add_argument("--raw_normal_dir", default="")
    p.add_argument("--raw_mask_dir", default="")
    p.add_argument("--raw_masked_rgb_dir", default="")
    p.add_argument("--raw_require_normal", action="store_true")
    p.add_argument("--raw_prompt", default="")
    p.add_argument("--raw_caption_path", default="")
    p.add_argument("--raw_caption_key", default="Video_Caption")
    p.add_argument("--raw_fps", type=int, default=16)
    p.add_argument("--cam_k", default="")
    p.add_argument("--cam_rt_dir", default="")
    p.add_argument("--raw_depth_percentile_min", type=float, default=2.0)
    p.add_argument("--raw_depth_percentile_max", type=float, default=98.0)
    p.add_argument("--raw_depth_invert", action="store_true")
    p.add_argument("--no_raw_depth_invert", action="store_false", dest="raw_depth_invert")
    p.set_defaults(raw_depth_invert=False)

    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=45)
    p.add_argument("--causal_window_frames", type=int, default=45)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_frames", action="store_true")
    return p.parse_args()


def _save_mask_video(mask_tchw: torch.Tensor, out_path: str, fps: int) -> None:
    mask_rgb = mask_tchw.repeat(1, 3, 1, 1)
    base._save_mp4(mask_rgb.permute(1, 0, 2, 3).unsqueeze(0).contiguous(), out_path, fps=fps)


def main() -> None:
    args = parse_args()
    base._ensure_single_process_dist_env()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit("This script is single-process only.")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    inference_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

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

    transformer = PipelineComponentLoader.load_module(
        "transformer", str(Path(args.base_model) / "transformer"), "diffusers", fastvideo_args
    )
    tokenizer = PipelineComponentLoader.load_module(
        "tokenizer", str(Path(args.base_model) / "tokenizer"), "transformers", fastvideo_args
    )
    text_encoder = PipelineComponentLoader.load_module(
        "text_encoder", str(Path(args.base_model) / "text_encoder"), "transformers", fastvideo_args
    )

    sample_root = Path(str(args.raw_sample_root)).expanduser().resolve()
    sequence = longwarp._load_raw_long_sequence_nomask(
        sample_root=sample_root,
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        inference_device=inference_device,
        dtype=dtype,
    )

    window_frames = int(args.causal_window_frames)
    H = int(args.height)
    W = int(args.width)
    valid_window = min(window_frames, int(args.num_frames), len(sequence.frame_ids))
    depth_path_by_id = {int(fid): p for fid, p in zip(sequence.frame_ids, sequence.depth_paths)}

    first_rgb = base._load_rgb_frame(sequence.rgb_paths[0], H, W)
    first_rgb_u8 = longwarp._chw_float_to_u8(first_rgb)
    target_ids = sequence.frame_ids[1:valid_window]
    if target_ids:
        warped_masked_rgb_valid, warped_mask_valid = longwarp._warp_maskrgb_from_keyframes_md_aligned(
            keyframe_rgbs_u8=[first_rgb_u8],
            keyframe_frame_ids=[int(sequence.frame_ids[0])],
            target_frame_ids=target_ids,
            depth_path_by_frame_id=depth_path_by_id,
            camera_k_aligned=sequence.camera_k_aligned,
            camera_rt_dir=sequence.camera_rt_dir,
            crop_params=sequence.crop_params,
            target_height=H,
            target_width=W,
        )
    else:
        warped_masked_rgb_valid = torch.empty((0, 3, H, W), dtype=torch.float32)
        warped_mask_valid = torch.empty((0, 1, H, W), dtype=torch.float32)

    warped_masked_rgb = longwarp._pad_tchw(warped_masked_rgb_valid, max(window_frames - 1, 1))
    warped_mask = longwarp._pad_tchw(warped_mask_valid, max(window_frames - 1, 1))
    mask_tchw = torch.cat(
        [
            torch.ones((1, 1, H, W), dtype=torch.float32),
            warped_mask[:max(window_frames - 1, 0)],
        ],
        dim=0,
    )[:valid_window]
    masked_rgb_tchw = torch.cat(
        [
            first_rgb.unsqueeze(0),
            warped_masked_rgb[:max(window_frames - 1, 0)],
        ],
        dim=0,
    )[:valid_window]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base._save_mp4(masked_rgb_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous(), str(out_dir / "first_window_masked_rgb.mp4"), fps=int(sequence.fps))
    _save_mask_video(mask_tchw, str(out_dir / "first_window_mask.mp4"), fps=int(sequence.fps))

    if bool(args.save_frames):
        base._save_frames_png(
            masked_rgb_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
            str(out_dir / "masked_rgb_frames"),
            prefix="masked_rgb",
        )
        _save_mask_video(mask_tchw, str(out_dir / "first_window_mask_frames_preview.mp4"), fps=int(sequence.fps))
        (out_dir / "mask_frames").mkdir(parents=True, exist_ok=True)
        for i in range(int(mask_tchw.shape[0])):
            mask_rgb = (mask_tchw[i].repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255.0).astype(base.np.uint8)
            base.Image.fromarray(mask_rgb).save(out_dir / "mask_frames" / f"mask_{i:04d}.png")

    logger.info("saved: %s", str(out_dir))


if __name__ == "__main__":
    main()
