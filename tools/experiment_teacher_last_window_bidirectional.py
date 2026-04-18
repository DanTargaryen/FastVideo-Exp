#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export the target long-rollout window controls from the current student rollout,
save them as per-frame PNGs, then run a teacher bidirectional 50-step inference
on that isolated window.

Default setup targets the 5th window (1-based) of Matrixcity_sample_401 with:
- student: checkpoint-100 causal student + controlnet
- teacher: base Wan transformer + teacher union controlnet
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.pipelines.stages.decoding import DecodingStage

import tools.infer_wan_controlnet_ti2v as base
import tools.infer_wan_controlnet_ti2v_long_firstframe_warp as longwarp

logger = init_logger(__name__)


def _save_rgb_png(x_chw: torch.Tensor, path: Path) -> None:
    arr = (
        x_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_gray_png(x_chw: torch.Tensor, path: Path) -> None:
    arr = (
        x_chw.detach().cpu().float().clamp(0, 1).squeeze(0).numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_minus1_1_rgb_png(x_chw: torch.Tensor, path: Path) -> None:
    arr = (
        ((x_chw.detach().cpu().float().clamp(-1, 1) + 1.0) * 0.5)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _prepare_teacher_raw_sample(
    *,
    out_root: Path,
    sample_id: str,
    prompt: str,
    anchor_rgb_chw: torch.Tensor,
    raw_rgb_paths: list[Path],
    raw_depth_paths: list[Path],
    raw_normal_paths: list[Path] | None,
    mask_tchw: torch.Tensor,
    masked_rgb_tchw: torch.Tensor,
    depth_tchw: torch.Tensor,
    normal_tchw: torch.Tensor | None,
) -> Path:
    raw_root = out_root / f"{sample_id}_teacher_raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    (raw_root / "rgb").mkdir(parents=True, exist_ok=True)
    (raw_root / "depth").mkdir(parents=True, exist_ok=True)
    (raw_root / "normal").mkdir(parents=True, exist_ok=True)
    (raw_root / "mask").mkdir(parents=True, exist_ok=True)
    (raw_root / "masked_rgb").mkdir(parents=True, exist_ok=True)
    (raw_root / "control_frames" / "depth").mkdir(parents=True, exist_ok=True)
    (raw_root / "control_frames" / "normal").mkdir(parents=True, exist_ok=True)
    (raw_root / "control_frames" / "mask").mkdir(parents=True, exist_ok=True)
    (raw_root / "control_frames" / "masked_rgb").mkdir(parents=True, exist_ok=True)

    # Frame-0 anchor for bidirectional raw TI2V should match the carry prefix
    # used by the student rollout for this window.
    _save_rgb_png(anchor_rgb_chw, raw_root / "rgb" / "rgb_0000.png")

    num_frames = int(mask_tchw.shape[0])
    for idx in range(num_frames):
        stem = f"{idx:04d}"
        if idx > 0:
            # Save remaining RGB frames for completeness/debugging. They are
            # not strictly required by the loader beyond frame0.
            rgb_src = raw_rgb_paths[idx]
            _link_or_copy(rgb_src, raw_root / "rgb" / f"rgb_{stem}{rgb_src.suffix.lower()}")
        depth_src = raw_depth_paths[idx]
        _link_or_copy(depth_src, raw_root / "depth" / f"depth_{stem}{depth_src.suffix.lower()}")
        if raw_normal_paths is not None:
            normal_src = raw_normal_paths[idx]
            _link_or_copy(
                normal_src,
                raw_root / "normal" / f"normal_{stem}{normal_src.suffix.lower()}",
            )
        _save_gray_png(mask_tchw[idx], raw_root / "mask" / f"mask_{stem}.png")
        _save_rgb_png(masked_rgb_tchw[idx], raw_root / "masked_rgb" / f"masked_rgb_{stem}.png")
        _save_minus1_1_rgb_png(depth_tchw[idx], raw_root / "control_frames" / "depth" / f"{stem}.png")
        _save_gray_png(mask_tchw[idx], raw_root / "control_frames" / "mask" / f"{stem}.png")
        _save_rgb_png(masked_rgb_tchw[idx], raw_root / "control_frames" / "masked_rgb" / f"{stem}.png")
        if normal_tchw is not None:
            _save_minus1_1_rgb_png(
                normal_tchw[idx],
                raw_root / "control_frames" / "normal" / f"{stem}.png",
            )

    (raw_root / "text.txt").write_text(prompt + "\n", encoding="utf-8")
    return raw_root


def _capture_target_window_controls(
    *,
    sequence: longwarp.RawLongSequenceNoMask,
    args,
    transformer,
    controlnet,
    scheduler,
    vae,
    decoding: DecodingStage,
    fastvideo_args: FastVideoArgs,
    dtype: torch.dtype,
    inference_device: torch.device,
    dmd_steps_list: list[int] | None,
    timestep_indices_list: list[int] | None,
    target_window_index: int,
):
    condition_mode = str(args.first_frame_condition_mode).lower()
    if condition_mode == "md_align":
        condition_mode = "hard_replace"

    total_required = int(args.num_frames)
    window_frames = int(args.causal_window_frames)
    overlap_frames = int(args.causal_overlap_frames)
    stride = int(window_frames - overlap_frames)
    num_windows = longwarp._compute_num_windows(total_required, window_frames, overlap_frames)
    if int(target_window_index) < 0 or int(target_window_index) >= int(num_windows):
        raise ValueError(
            f"target_window_index={int(target_window_index)} out of range for num_windows={int(num_windows)}"
        )

    prompt_embeds = sequence.text_embedding_bld.to(device="cuda", dtype=dtype)
    target_c = int(getattr(transformer, "num_channels_latents", 16))
    H = int(args.height)
    W = int(args.width)
    rollout_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    global_first_rgb = base._load_rgb_frame(sequence.rgb_paths[0], H, W)
    global_first_frame_latent = base._encode_first_frame_latent(
        vae=vae,
        first_rgb_chw=global_first_rgb,
        target_c=target_c,
        inference_device=inference_device,
        compute_dtype=dtype,
    ).to(device="cuda", dtype=dtype)

    depth_path_by_id = {int(fid): p for fid, p in zip(sequence.frame_ids, sequence.depth_paths)}
    history_rgbs_u8: dict[int, np.ndarray] = {}
    processed_frame_ids: set[int] = set()
    visibility_map = longwarp._GlobalVoxelVisibilityMap(voxel_size=float(args.selection_voxel_size))

    carry_prefix_tchw: torch.Tensor | None = None
    warped_masked_rgb_next: torch.Tensor | None = None
    warped_mask_next: torch.Tensor | None = None

    batch = base.ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = [prompt_embeds]
    batch.height = H
    batch.width = W
    batch.num_frames = window_frames

    frame_seq_length: int | None = None
    cache_global_start_latent = 0
    latent_stride_t: int | None = None
    total_latent_frames: int | None = None
    kv_cache = None
    crossattn_cache = None
    kv_cache_uncond = None
    crossattn_cache_uncond = None
    control_kv_cache = None
    control_crossattn_cache = None
    control_kv_cache_uncond = None
    control_crossattn_cache_uncond = None

    for win_idx in range(num_windows):
        start_pos = int(win_idx * stride)
        valid_window = min(window_frames, total_required - start_pos)
        end_pos_valid = int(start_pos + valid_window)

        window_prefix_tchw = global_first_rgb.unsqueeze(0) if win_idx == 0 else carry_prefix_tchw
        if window_prefix_tchw is None:
            raise RuntimeError("Missing prefix frames for target-window capture")
        window_first_frame_latent = (
            global_first_frame_latent
            if win_idx == 0
            else longwarp._encode_rgb_prefix_latent(
                vae=vae,
                prefix_rgb_tchw=window_prefix_tchw,
                target_c=target_c,
                inference_device=inference_device,
                compute_dtype=dtype,
            ).to(device="cuda", dtype=dtype)
        )
        window_anchor_latent_frames = max(
            int(args.first_frame_anchor_latent_frames),
            int(window_first_frame_latent.shape[2]),
        )

        depth_window_paths = longwarp._pad_paths(sequence.depth_paths[start_pos:end_pos_valid], window_frames)
        normal_window_paths = (
            longwarp._pad_paths(sequence.normal_paths[start_pos:end_pos_valid], window_frames)
            if sequence.normal_paths is not None
            else None
        )
        raw_rgb_window_paths = sequence.rgb_paths[start_pos:end_pos_valid]

        if win_idx == 0:
            first_rgb_u8 = longwarp._chw_float_to_u8(global_first_rgb)
            target_ids = sequence.frame_ids[start_pos + 1:end_pos_valid]
            if len(target_ids) > 0:
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
                [torch.ones((1, 1, H, W), dtype=torch.float32), warped_mask[: max(window_frames - 1, 0)]],
                dim=0,
            )
            masked_rgb_tchw = torch.cat(
                [global_first_rgb.unsqueeze(0), warped_masked_rgb[: max(window_frames - 1, 0)]],
                dim=0,
            )
        else:
            if carry_prefix_tchw is None or warped_masked_rgb_next is None or warped_mask_next is None:
                raise RuntimeError("Missing carry-over state while capturing target window")
            mask_tchw = torch.cat(
                [torch.ones((overlap_frames, 1, H, W), dtype=torch.float32), warped_mask_next],
                dim=0,
            )
            masked_rgb_tchw = torch.cat([carry_prefix_tchw, warped_masked_rgb_next], dim=0)

        depth_tchw = base._load_depth_sequence(
            depth_window_paths,
            H,
            W,
            pmin=float(args.raw_depth_percentile_min),
            pmax=float(args.raw_depth_percentile_max),
            invert_depth=bool(args.raw_depth_invert),
            normalization_mode=str(args.raw_depth_normalization_mode),
        )
        normal_tchw = None
        if normal_window_paths is not None:
            normal_tchw = torch.stack([base._load_normal_frame(p, H, W) for p in normal_window_paths], dim=0)

        control_latent = base._encode_control_latent_from_tchw(
            vae=vae,
            depth_tchw=depth_tchw,
            normal_tchw=normal_tchw,
            masked_rgb_tchw=masked_rgb_tchw,
            mask_tchw=mask_tchw,
            target_c=target_c,
            inference_device=inference_device,
            compute_dtype=dtype,
        ).to(device="cuda", dtype=dtype)

        if win_idx == int(target_window_index):
            target_frame_ids = sequence.frame_ids[start_pos:end_pos_valid]
            return {
                "window_index": int(win_idx),
                "start_pos": int(start_pos),
                "end_pos_valid": int(end_pos_valid),
                "target_frame_ids": [int(x) for x in target_frame_ids],
                "window_prefix_tchw": window_prefix_tchw.detach().cpu().float(),
                "window_first_frame_latent": window_first_frame_latent.detach().cpu().float(),
                "mask_tchw": mask_tchw.detach().cpu().float(),
                "masked_rgb_tchw": masked_rgb_tchw.detach().cpu().float(),
                "depth_tchw": depth_tchw.detach().cpu().float(),
                "normal_tchw": (normal_tchw.detach().cpu().float() if normal_tchw is not None else None),
                "control_latent": control_latent.detach().cpu().float(),
                "raw_rgb_window_paths": list(raw_rgb_window_paths),
                "depth_window_paths": list(depth_window_paths),
                "normal_window_paths": (list(normal_window_paths) if normal_window_paths is not None else None),
            }

        # Run the student rollout for earlier windows to obtain carry-over state.
        if frame_seq_length is None:
            latent_t = int(control_latent.shape[2])
            latent_h = int(control_latent.shape[3])
            latent_w = int(control_latent.shape[4])
            overlap_latent_t = base._latent_frames_from_video_frames(overlap_frames)
            latent_stride_t = max(1, latent_t - overlap_latent_t)
            latent_seq_length = latent_h * latent_w
            patch_ratio = int(
                transformer.config.arch_config.patch_size[-1]
                * transformer.config.arch_config.patch_size[-2]
            )
            frame_seq_length = latent_seq_length // patch_ratio
            total_latent_frames = latent_t + max(0, num_windows - 1) * latent_stride_t
            (
                kv_cache,
                crossattn_cache,
                kv_cache_uncond,
                crossattn_cache_uncond,
                control_kv_cache,
                control_crossattn_cache,
                control_kv_cache_uncond,
                control_crossattn_cache_uncond,
            ) = longwarp._initialize_long_rollout_caches(
                transformer=transformer,
                controlnet=controlnet,
                prompt_embeds=prompt_embeds,
                dtype=dtype,
                rollout_device=rollout_device,
                frame_seq_length=int(frame_seq_length),
                total_latent_frames=int(total_latent_frames),
                use_guidance=False,
            )
            cache_global_start_latent = 0

        latents = base._causal_dmd_rollout_one_window_with_cache(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            prompt_embeds_list=[prompt_embeds],
            negative_prompt_embeds_list=None,
            guidance_scale=float(args.guidance_scale),
            first_frame_latent_bcfhw=window_first_frame_latent,
            control_latent_bcfhw=control_latent,
            num_frames=window_frames,
            schedule_num_inference_steps=int(args.schedule_num_inference_steps),
            dmd_steps=dmd_steps_list,
            timestep_indices=timestep_indices_list,
            context_noise=args.context_noise,
            warp_denoising_step=bool(args.warp_denoising_step),
            update_rule=args.update_rule,
            full_schedule=bool(args.full_schedule),
            first_frame_timestep_zero=bool(args.first_frame_timestep_zero),
            expand_timesteps=bool(getattr(fastvideo_args.pipeline_config, "expand_timesteps", False)),
            disable_cache_update=bool(args.disable_cache_update),
            first_frame_anchor_latent_frames=int(window_anchor_latent_frames),
            first_frame_condition_mode=condition_mode,
            seed=int(args.seed + win_idx),
            dtype=dtype,
            global_start_latent=int(cache_global_start_latent),
            frame_seq_length=int(frame_seq_length),
            batch=batch,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            kv_cache_uncond=kv_cache_uncond,
            crossattn_cache_uncond=crossattn_cache_uncond,
            control_kv_cache=control_kv_cache,
            control_crossattn_cache=control_crossattn_cache,
            control_kv_cache_uncond=control_kv_cache_uncond,
            control_crossattn_cache_uncond=control_crossattn_cache_uncond,
        )
        cache_global_start_latent += int(latent_stride_t)

        decoded_window = decoding.decode(latents, fastvideo_args).cpu().float()
        decoded_window_tchw = decoded_window[0].permute(1, 0, 2, 3).contiguous()

        current_frame_ids = [int(fid) for fid in sequence.frame_ids[start_pos:end_pos_valid]]
        longwarp._update_history_and_visibility(
            sequence=sequence,
            frame_ids=current_frame_ids,
            frames_tchw=decoded_window_tchw[:valid_window],
            history_rgbs_u8=history_rgbs_u8,
            processed_frame_ids=processed_frame_ids,
            visibility_map=visibility_map,
        )

        last_frame_id = int(sequence.frame_ids[end_pos_valid - 1])
        carry_prefix_tchw = decoded_window_tchw[valid_window - overlap_frames : valid_window].clone()

        next_chunk_start = max(0, int(end_pos_valid - overlap_frames))
        next_chunk_end = min(next_chunk_start + window_frames, total_required)
        target_frame_ids_for_chunk = [int(fid) for fid in sequence.frame_ids[next_chunk_start:next_chunk_end]]
        selected_keyframe_ids = longwarp._select_keyframes_for_next_window(
            sequence=sequence,
            visibility_map=visibility_map,
            processed_frame_ids=processed_frame_ids,
            history_rgbs_u8=history_rgbs_u8,
            target_frame_ids_for_chunk=target_frame_ids_for_chunk,
            depth_path_by_id=depth_path_by_id,
            target_height=H,
            target_width=W,
            last_frame_id=last_frame_id,
            num_keyframes=int(args.warp_num_keyframes),
            num_target_samples=int(args.selection_num_target_samples),
        )
        keyframe_rgbs_u8 = [history_rgbs_u8[int(fid)] for fid in selected_keyframe_ids]

        next_start = int(end_pos_valid)
        next_valid_new = min(stride, total_required - next_start)
        target_ids_next = sequence.frame_ids[next_start : next_start + next_valid_new]
        warped_masked_rgb_next_valid, warped_mask_next_valid = longwarp._warp_maskrgb_from_keyframes_md_aligned(
            keyframe_rgbs_u8=keyframe_rgbs_u8,
            keyframe_frame_ids=selected_keyframe_ids,
            target_frame_ids=target_ids_next,
            depth_path_by_frame_id=depth_path_by_id,
            camera_k_aligned=sequence.camera_k_aligned,
            camera_rt_dir=sequence.camera_rt_dir,
            crop_params=sequence.crop_params,
            target_height=H,
            target_width=W,
        )
        warped_masked_rgb_next = longwarp._pad_tchw(warped_masked_rgb_next_valid, stride)
        warped_mask_next = longwarp._pad_tchw(warped_mask_next_valid, stride)

    raise RuntimeError("Failed to capture target window controls")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export last rollout window controls and run teacher bidirectional inference")
    p.add_argument("--base_model", required=True)
    p.add_argument("--student_transformer_dir", required=True)
    p.add_argument("--student_controlnet_dir", required=True)
    p.add_argument("--teacher_transformer_dir", required=True)
    p.add_argument("--teacher_controlnet_dir", required=True)
    p.add_argument("--raw_sample_root", required=True)
    p.add_argument("--raw_rgb_dir", default="")
    p.add_argument("--raw_depth_dir", default="")
    p.add_argument("--raw_normal_dir", default="")
    p.add_argument("--raw_mask_dir", default="")
    p.add_argument("--raw_masked_rgb_dir", default="")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--raw_fps", type=int, default=16)
    p.add_argument("--raw_prompt", type=str, default="")
    p.add_argument("--raw_require_normal", action="store_true", default=True)
    p.add_argument("--cam_k", required=True)
    p.add_argument("--cam_rt_dir", required=True)
    p.add_argument("--causal_window_frames", type=int, default=81)
    p.add_argument("--causal_overlap_frames", type=int, default=1)
    p.add_argument("--local_attn_size", type=int, default=21)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--warp_num_keyframes", type=int, default=4)
    p.add_argument("--selection_num_target_samples", type=int, default=3)
    p.add_argument("--selection_voxel_size", type=float, default=0.1)
    p.add_argument("--schedule_num_inference_steps", type=int, default=50)
    p.add_argument("--dmd_steps", type=str, default="1000,750,500,250")
    p.add_argument("--update_rule", type=str, default="renoise_x0")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--context_noise", type=int, default=0)
    p.add_argument("--warp_denoising_step", action="store_true", default=True)
    p.add_argument("--full_schedule", action="store_true", default=False)
    p.add_argument("--disable_cache_update", action="store_true", default=False)
    p.add_argument("--teacher_guidance_scale", type=float, default=3.0)
    p.add_argument("--teacher_dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--raw_depth_percentile_min", type=float, default=5.0)
    p.add_argument("--raw_depth_percentile_max", type=float, default=95.0)
    p.add_argument("--raw_depth_normalization_mode", choices=["md_align", "percentile"], default="percentile")
    p.add_argument("--raw_depth_invert", action="store_true", default=False)
    p.add_argument("--first_frame_timestep_zero", action="store_true", default=True)
    p.add_argument("--first_frame_condition_mode", choices=["hard_replace", "noise_init", "md_align"], default="hard_replace")
    p.add_argument("--first_frame_anchor_latent_frames", type=int, default=1)
    p.add_argument("--cache_reset_interval_windows", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_window", type=int, default=5, help="1-based window index to export")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.target_window) <= 0:
        raise ValueError("--target_window must be 1-based and > 0")

    base._ensure_single_process_dist_env()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    inference_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

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
    fastvideo_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    fastvideo_args.override_controlnet_cls_name = "CausalWanControlnetUnion3DModel"
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    fastvideo_args.pipeline_config.warp_denoising_step = True
    dmd_steps_list = [int(x) for x in str(args.dmd_steps).split(",") if x.strip()]
    fastvideo_args.pipeline_config.dmd_denoising_steps = dmd_steps_list

    transformer = PipelineComponentLoader.load_module("transformer", args.student_transformer_dir, "diffusers", fastvideo_args)
    controlnet = PipelineComponentLoader.load_module("controlnet", args.student_controlnet_dir, "diffusers", fastvideo_args)
    tokenizer = PipelineComponentLoader.load_module("tokenizer", str(Path(args.base_model) / "tokenizer"), "transformers", fastvideo_args)
    text_encoder = PipelineComponentLoader.load_module("text_encoder", str(Path(args.base_model) / "text_encoder"), "transformers", fastvideo_args)
    vae = PipelineComponentLoader.load_module("vae", str(Path(args.base_model) / "vae"), "diffusers", fastvideo_args)
    decoding = DecodingStage(vae=vae)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence = longwarp._load_raw_long_sequence_nomask(
        sample_root=Path(args.raw_sample_root).expanduser().resolve(),
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        inference_device=inference_device,
        dtype=dtype,
    )

    target = _capture_target_window_controls(
        sequence=sequence,
        args=args,
        transformer=transformer,
        controlnet=controlnet,
        scheduler=scheduler,
        vae=vae,
        decoding=decoding,
        fastvideo_args=fastvideo_args,
        dtype=dtype,
        inference_device=inference_device,
        dmd_steps_list=dmd_steps_list,
        timestep_indices_list=None,
        target_window_index=int(args.target_window) - 1,
    )

    target_name = f"{sequence.sample_id}_window{int(args.target_window):02d}"
    raw_teacher_root = _prepare_teacher_raw_sample(
        out_root=out_dir,
        sample_id=target_name,
        prompt=sequence.prompt,
        anchor_rgb_chw=target["window_prefix_tchw"][0],
        raw_rgb_paths=target["raw_rgb_window_paths"],
        raw_depth_paths=target["depth_window_paths"],
        raw_normal_paths=target["normal_window_paths"],
        mask_tchw=target["mask_tchw"],
        masked_rgb_tchw=target["masked_rgb_tchw"],
        depth_tchw=target["depth_tchw"],
        normal_tchw=target["normal_tchw"],
    )
    torch.save(target["control_latent"], out_dir / f"{target_name}_control_latent.pt")
    logger.info("Saved target-window control latent tensor and control frames under %s", str(raw_teacher_root))

    teacher_out_dir = out_dir / "teacher_bidirectional"
    teacher_out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dtype = str(args.teacher_dtype)
    teacher_cmd = [
        os.environ.get("PYTHON", "python3"),
        "tools/infer_wan_controlnet_ti2v.py",
        "--base_model",
        str(args.base_model),
        "--input_mode",
        "raw",
        "--attention_mode",
        "bidirectional",
        "--raw_sample_root",
        str(raw_teacher_root),
        "--raw_rgb_dir",
        str(raw_teacher_root / "rgb"),
        "--raw_depth_dir",
        str(raw_teacher_root / "depth"),
        "--raw_normal_dir",
        str(raw_teacher_root / "normal"),
        "--raw_mask_dir",
        str(raw_teacher_root / "mask"),
        "--raw_masked_rgb_dir",
        str(raw_teacher_root / "masked_rgb"),
        "--raw_prompt",
        str(sequence.prompt),
        "--raw_require_normal",
        "--transformer_dir",
        str(args.teacher_transformer_dir),
        "--controlnet_dir",
        str(args.teacher_controlnet_dir),
        "--scheduler",
        "unipc",
        "--full_schedule",
        "--schedule_num_inference_steps",
        "50",
        "--guidance_scale",
        str(float(args.teacher_guidance_scale)),
        "--dtype",
        teacher_dtype,
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        str(int(args.causal_window_frames)),
        "--fps",
        str(int(args.raw_fps)),
        "--out_dir",
        str(teacher_out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[1]}:{env.get('PYTHONPATH', '')}"
    logger.info("Running teacher bidirectional inference: %s", " ".join(teacher_cmd))
    subprocess.run(teacher_cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]), env=env)
    logger.info("Teacher bidirectional result saved under %s", str(teacher_out_dir))


if __name__ == "__main__":
    main()
