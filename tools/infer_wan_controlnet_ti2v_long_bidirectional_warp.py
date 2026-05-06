#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Long TI2V + ControlNet inference with windowed bidirectional rollout.

Supported modes:
- `input_mode=raw`:
  Rebuild mask/masked_rgb online between windows using the same first-frame /
  multi-keyframe warp logic as `infer_wan_controlnet_ti2v_long_firstframe_warp.py`,
  but run each window with bidirectional denoising instead of the causal KV-cache
  rollout.
- `input_mode=parquet`:
  Read a full-length parquet sample whose `control_latent` already spans the
  long clip, slice it window-by-window, and stitch the decoded windows using
  the same overlap writeback format as the long causal script.

Window stitch format:
- First window writes all `causal_window_frames`.
- Later windows overwrite the local overlap prefix for continuity but only
  append the non-overlap suffix (`window_frames - overlap_frames`) to the final
  video, matching the original long-rollout layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from fastvideo.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines.stages.decoding import DecodingStage

import tools.infer_wan_controlnet_ti2v as base
import tools.infer_wan_controlnet_ti2v_long_firstframe_warp as longwarp

logger = init_logger(__name__)


def _write_run_manifest(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    sample_id: str,
    prompt: str,
    output_path: Path,
) -> None:
    manifest = {
        "sample_id": str(sample_id),
        "prompt": str(prompt),
        "output_path": str(output_path),
        "argv": [str(x) for x in sys.argv],
        "paths": {
            "base_model": str(args.base_model),
            "transformer_dir": str(args.transformer_dir),
            "controlnet_dir": str(args.controlnet_dir),
            "raw_sample_root": str(args.raw_sample_root),
            "data_path": str(args.data_path),
            "cam_k": str(args.cam_k),
            "cam_rt_dir": str(args.cam_rt_dir),
        },
        "sampling": {
            "input_mode": str(args.input_mode),
            "scheduler": str(args.scheduler),
            "schedule_num_inference_steps": int(args.schedule_num_inference_steps),
            "full_schedule": bool(args.full_schedule),
            "guidance_scale": float(args.guidance_scale),
            "negative_prompt": str(args.negative_prompt),
            "flow_shift": float(args.flow_shift),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
        },
        "shape": {
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": int(args.fps),
            "window_frames": int(args.causal_window_frames),
            "overlap_frames": int(args.causal_overlap_frames),
        },
        "raw_preprocess": {
            "raw_depth_normalization_mode": str(args.raw_depth_normalization_mode),
            "raw_depth_invert": bool(args.raw_depth_invert),
            "raw_require_normal": bool(args.raw_require_normal),
        },
        "bidir": {
            "bidir_first_frame_timestep_zero": bool(args.bidir_first_frame_timestep_zero),
            "bidir_sync_first_frame_state": bool(args.bidir_sync_first_frame_state),
        },
    }
    manifest_path = out_dir / f"{sample_id}_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    logger.info("saved run manifest: %s", str(manifest_path))


def _slice_or_pad_latent_t(latent_bcfhw: torch.Tensor, *, start_t: int,
                           target_t: int, name: str) -> torch.Tensor:
    if latent_bcfhw.ndim != 5:
        raise ValueError(
            f"{name} must be [B,C,T,H,W], got {tuple(latent_bcfhw.shape)}")
    sliced = latent_bcfhw[:, :, int(start_t):int(start_t) + int(target_t)]
    if int(sliced.shape[2]) == int(target_t):
        return sliced
    if int(sliced.shape[2]) <= 0:
        raise ValueError(
            f"{name} slice is empty for start_t={int(start_t)} target_t={int(target_t)} "
            f"full_shape={tuple(latent_bcfhw.shape)}")
    pad = sliced[:, :, -1:].repeat(1, 1,
                                   int(target_t) - int(sliced.shape[2]), 1,
                                   1)
    return torch.cat([sliced, pad], dim=2)


def _align_runtime_dtype(transformer, controlnet,
                         requested_dtype: torch.dtype) -> torch.dtype:
    try:
        model_param = next(p for p in transformer.parameters()
                           if torch.is_floating_point(p))
    except StopIteration:
        model_param = next(p for p in controlnet.parameters()
                           if torch.is_floating_point(p))
    model_dtype = model_param.dtype
    if requested_dtype != model_dtype:
        logger.info("dtype alignment: overriding runtime dtype from %s to model dtype %s",
                    str(requested_dtype), str(model_dtype))
    return model_dtype


def _build_image_latent_from_first_frame(
    *,
    first_frame_latent_bcfhw: torch.Tensor,
    target_t: int,
) -> torch.Tensor:
    return base._build_image_latent_from_first_frame_latent(
        first_frame_latent_bcfhw=first_frame_latent_bcfhw,
        target_frames=int(target_t),
    )


@torch.no_grad()
def _run_bidirectional_window(
    *,
    transformer,
    controlnet,
    scheduler,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    guidance_scale: float,
    first_frame_latent_bcfhw: torch.Tensor,
    control_latent_bcfhw: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    schedule_num_inference_steps: int,
    dmd_steps_list: list[int] | None,
    timestep_indices_list: list[int] | None,
    context_noise: int,
    warp_denoising_step: bool,
    update_rule: str,
    full_schedule: bool,
    bidir_first_frame_timestep_zero: bool,
    bidir_sync_first_frame_state: bool,
    seed: int,
    dtype: torch.dtype,
    trace_sample_id: str,
) -> torch.Tensor:
    image_latent = _build_image_latent_from_first_frame(
        first_frame_latent_bcfhw=first_frame_latent_bcfhw,
        target_t=int(control_latent_bcfhw.shape[2]),
    )
    return base._bidirectional_dmd_rollout_ti2v_controlnet(
        transformer=transformer,
        controlnet=controlnet,
        scheduler=scheduler,
        prompt_embeds_list=[prompt_embeds],
        negative_prompt_embeds_list=([negative_prompt_embeds]
                                     if negative_prompt_embeds is not None else
                                     None),
        guidance_scale=float(guidance_scale),
        controlnet_weight=1.0,
        first_frame_latent_bcfhw=first_frame_latent_bcfhw,
        image_latent_bcfhw=image_latent,
        control_latent_bcfhw=control_latent_bcfhw,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        schedule_num_inference_steps=int(schedule_num_inference_steps),
        dmd_steps=dmd_steps_list,
        timestep_indices=timestep_indices_list,
        context_noise=int(context_noise),
        warp_denoising_step=bool(warp_denoising_step),
        update_rule=str(update_rule),
        full_schedule=bool(full_schedule),
        first_frame_timestep_zero=bool(bidir_first_frame_timestep_zero),
        bidir_sync_first_frame_state=bool(bidir_sync_first_frame_state),
        expand_timesteps=True,
        trace_jsonl_path=None,
        trace_sample_id=str(trace_sample_id),
        seed=int(seed),
        dtype=dtype,
    )


@torch.no_grad()
def _run_windowed_bidirectional_parquet(
    *,
    sample: base.Sample,
    args,
    transformer,
    controlnet,
    scheduler,
    decoding: DecodingStage,
    fastvideo_args: FastVideoArgs,
    negative_prompt_embeds_global: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    total_required = int(args.num_frames)
    window_frames = int(args.causal_window_frames)
    overlap_frames = int(args.causal_overlap_frames)
    if overlap_frames <= 0 or overlap_frames >= window_frames:
        raise ValueError(
            f"Invalid overlap/window settings: overlap={overlap_frames}, window={window_frames}"
        )
    stride = int(window_frames - overlap_frames)
    num_windows = longwarp._compute_num_windows(total_required, window_frames,
                                                overlap_frames)

    prompt_embeds = sample.text_embedding_bld.to(device="cuda", dtype=dtype)
    first_frame_latent_global = sample.first_frame_latent_bcfhw.to(device="cuda",
                                                                   dtype=dtype)
    control_latent_full = sample.control_latent_bcfhw.to(device="cuda",
                                                         dtype=dtype)

    latent_window_t = int(base._latent_frames_from_video_frames(window_frames))
    latent_overlap_t = int(base._latent_frames_from_video_frames(overlap_frames))
    latent_stride_t = max(1, latent_window_t - latent_overlap_t)

    available_frames = (int(control_latent_full.shape[2]) - 1) * 4 + 1
    if int(total_required) > int(available_frames):
        raise ValueError(
            f"Requested num_frames={int(total_required)} exceeds parquet control coverage "
            f"(available={int(available_frames)})")

    H = int(args.height)
    W = int(args.width)
    final_tchw = torch.empty((int(total_required), 3, H, W), dtype=torch.float32)
    write_ptr = 0
    carry_prefix_tchw: torch.Tensor | None = None

    for win_idx in range(int(num_windows)):
        start_frame = int(win_idx * stride)
        valid_window = min(int(window_frames),
                           int(total_required) - int(start_frame))
        if valid_window <= 0:
            break
        end_frame = int(start_frame + valid_window)
        start_t = int(win_idx * latent_stride_t)
        control_window = _slice_or_pad_latent_t(control_latent_full,
                                                start_t=start_t,
                                                target_t=int(latent_window_t),
                                                name="control_latent")

        if win_idx == 0:
            window_prefix_tchw = None
            window_first_frame_latent = first_frame_latent_global
        else:
            if carry_prefix_tchw is None or int(carry_prefix_tchw.shape[0]) <= 0:
                raise RuntimeError(
                    "Missing carry prefix for parquet bidirectional window > 0")
            window_prefix_tchw = carry_prefix_tchw
            window_first_frame_latent = base._encode_first_frame_latent(
                vae=decoding.vae,
                first_rgb_chw=window_prefix_tchw[0].to(device="cuda",
                                                       dtype=torch.float32),
                target_c=int(first_frame_latent_global.shape[1]),
                inference_device=torch.device(
                    f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"),
                compute_dtype=dtype,
            ).to(device="cuda", dtype=dtype)

        latents = _run_bidirectional_window(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds_global,
            guidance_scale=float(args.guidance_scale),
            first_frame_latent_bcfhw=window_first_frame_latent,
            control_latent_bcfhw=control_window,
            height=H,
            width=W,
            num_frames=int(window_frames),
            schedule_num_inference_steps=int(args.schedule_num_inference_steps),
            dmd_steps_list=None if bool(args.full_schedule) else
            [int(x) for x in str(args.dmd_steps).split(",") if x.strip()],
            timestep_indices_list=None if bool(args.full_schedule) else
            [int(x) for x in str(args.timestep_indices).split(",") if x.strip()],
            context_noise=int(args.context_noise),
            warp_denoising_step=bool(args.warp_denoising_step),
            update_rule=str(args.update_rule),
            full_schedule=bool(args.full_schedule),
            bidir_first_frame_timestep_zero=bool(
                args.bidir_first_frame_timestep_zero),
            bidir_sync_first_frame_state=bool(
                args.bidir_sync_first_frame_state),
            seed=int(args.seed + win_idx),
            dtype=dtype,
            trace_sample_id=f"{sample.sample_id}_win{win_idx:02d}",
        )
        decoded_window = decoding.decode(latents, fastvideo_args).cpu().float()
        decoded_window_tchw = decoded_window[0].permute(1, 0, 2, 3).contiguous()
        decoded_window_tchw = decoded_window_tchw.clone()
        if window_prefix_tchw is not None:
            prefix_keep = min(int(window_prefix_tchw.shape[0]),
                              int(decoded_window_tchw.shape[0]))
            decoded_window_tchw[:prefix_keep] = window_prefix_tchw[:prefix_keep]

        if win_idx == 0:
            write_frames = decoded_window_tchw[:valid_window]
        else:
            valid_new = max(0, int(valid_window) - int(overlap_frames))
            write_frames = decoded_window_tchw[int(overlap_frames):int(
                overlap_frames) + valid_new]
        final_tchw[write_ptr:write_ptr + int(write_frames.shape[0])] = write_frames
        write_ptr += int(write_frames.shape[0])
        carry_prefix_tchw = decoded_window_tchw[int(valid_window) -
                                                int(overlap_frames):int(
                                                    valid_window)].clone()
        logger.info(
            "window=%s frame_range=(%s,%s) latent_range=(%s,%s) write_ptr=%s",
            int(win_idx),
            int(start_frame),
            int(end_frame),
            int(start_t),
            int(min(start_t + latent_window_t, control_latent_full.shape[2])),
            int(write_ptr),
        )

    if int(write_ptr) != int(total_required):
        raise RuntimeError(
            f"Final parquet frame count mismatch: wrote {int(write_ptr)}, expected {int(total_required)}"
        )
    return final_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@torch.no_grad()
def _run_windowed_bidirectional_raw(
    *,
    sequence: longwarp.RawLongSequenceNoMask,
    args,
    transformer,
    controlnet,
    scheduler,
    decoding: DecodingStage,
    fastvideo_args: FastVideoArgs,
    negative_prompt_embeds_global: torch.Tensor | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    total_required = int(args.num_frames)
    window_frames = int(args.causal_window_frames)
    overlap_frames = int(args.causal_overlap_frames)
    if overlap_frames <= 0 or overlap_frames >= window_frames:
        raise ValueError(
            f"Invalid overlap/window settings: overlap={overlap_frames}, window={window_frames}"
        )
    stride = int(window_frames - overlap_frames)
    num_windows = longwarp._compute_num_windows(total_required, window_frames,
                                                overlap_frames)

    prompt_embeds = sequence.text_embedding_bld.to(device="cuda", dtype=dtype)
    target_c = int(getattr(transformer, "num_channels_latents", 16))
    H = int(args.height)
    W = int(args.width)
    inference_device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    use_precomputed_mask = sequence.mask_paths is not None

    global_first_rgb = base._load_rgb_frame(sequence.rgb_paths[0], H, W)
    global_first_frame_latent = base._encode_first_frame_latent(
        vae=decoding.vae,
        first_rgb_chw=global_first_rgb,
        target_c=target_c,
        inference_device=inference_device,
        compute_dtype=dtype,
    ).to(device="cuda", dtype=dtype)

    depth_path_by_id = {
        int(fid): p
        for fid, p in zip(sequence.frame_ids, sequence.depth_paths)
    }
    final_tchw = torch.empty((int(total_required), 3, H, W), dtype=torch.float32)
    saved_controls: dict[str, torch.Tensor] | None = None
    if bool(args.save_control_outputs):
        saved_controls = {
            "depth": torch.empty((int(total_required), 3, H, W),
                                  dtype=torch.float32),
            "mask": torch.empty((int(total_required), 1, H, W),
                                 dtype=torch.float32),
            "masked_rgb": torch.empty((int(total_required), 3, H, W),
                                       dtype=torch.float32),
        }
        if sequence.normal_paths is not None:
            saved_controls["normal"] = torch.empty((int(total_required), 3, H, W),
                                                    dtype=torch.float32)

    write_ptr = 0
    history_rgbs_u8: dict[int, base.np.ndarray] = {}
    processed_frame_ids: set[int] = set()
    visibility_map = longwarp._GlobalVoxelVisibilityMap(
        voxel_size=float(args.selection_voxel_size))

    carry_prefix_tchw: torch.Tensor | None = None
    warped_masked_rgb_next: torch.Tensor | None = None
    warped_mask_next: torch.Tensor | None = None

    dmd_steps_list = None if bool(args.full_schedule) else [
        int(x) for x in str(args.dmd_steps).split(",") if x.strip()
    ]
    timestep_indices_list = None if bool(args.full_schedule) else [
        int(x) for x in str(args.timestep_indices).split(",") if x.strip()
    ]

    for win_idx in range(int(num_windows)):
        start_pos = int(win_idx * stride)
        valid_window = min(int(window_frames),
                           int(total_required) - int(start_pos))
        if valid_window <= 0:
            break
        end_pos_valid = int(start_pos + valid_window)

        window_prefix_tchw = (global_first_rgb.unsqueeze(0) if win_idx == 0 else
                              carry_prefix_tchw)
        if window_prefix_tchw is None or int(window_prefix_tchw.shape[0]) <= 0:
            raise RuntimeError("Missing prefix frames for raw window > 0")

        window_first_frame_latent = (
            global_first_frame_latent if win_idx == 0 else
            longwarp._encode_rgb_prefix_latent(
                vae=decoding.vae,
                prefix_rgb_tchw=window_prefix_tchw,
                target_c=target_c,
                inference_device=inference_device,
                compute_dtype=dtype,
            ).to(device="cuda", dtype=dtype))

        depth_window_paths = longwarp._pad_paths(
            sequence.depth_paths[start_pos:end_pos_valid], int(window_frames))
        normal_window_paths = (
            longwarp._pad_paths(sequence.normal_paths[start_pos:end_pos_valid],
                                int(window_frames))
            if sequence.normal_paths is not None else None)

        if use_precomputed_mask:
            assert sequence.mask_paths is not None
            mask_window_paths = longwarp._pad_paths(
                sequence.mask_paths[start_pos:end_pos_valid],
                int(window_frames),
            )
            mask_tchw = torch.stack(
                [
                    base._load_mask_frame(
                        p,
                        H,
                        W,
                        threshold=(
                            None if float(getattr(args, "raw_mask_threshold", -1.0)) < 0
                            else float(getattr(args, "raw_mask_threshold", -1.0))
                        ),
                        invert=bool(getattr(args, "raw_mask_invert", False)),
                    )
                    for p in mask_window_paths
                ],
                dim=0,
            )
            if sequence.masked_rgb_paths is not None:
                masked_rgb_window_paths = longwarp._pad_paths(
                    sequence.masked_rgb_paths[start_pos:end_pos_valid],
                    int(window_frames),
                )
                masked_rgb_tchw = torch.stack(
                    [base._load_rgb_frame(p, H, W) for p in masked_rgb_window_paths],
                    dim=0,
                )
            else:
                rgb_window_paths = longwarp._pad_paths(
                    sequence.rgb_paths[start_pos:end_pos_valid],
                    int(window_frames),
                )
                rgb_tchw_for_mask = torch.stack(
                    [base._load_rgb_frame(p, H, W) for p in rgb_window_paths],
                    dim=0,
                )
                masked_rgb_tchw = rgb_tchw_for_mask * mask_tchw.repeat(1, 3, 1, 1)
        elif win_idx == 0:
            first_rgb_u8 = longwarp._chw_float_to_u8(global_first_rgb)
            target_ids = sequence.frame_ids[start_pos + 1:end_pos_valid]
            if len(target_ids) > 0:
                warped_masked_rgb_valid, warped_mask_valid = (
                    longwarp._warp_maskrgb_from_keyframes_md_aligned(
                        keyframe_rgbs_u8=[first_rgb_u8],
                        keyframe_frame_ids=[int(sequence.frame_ids[0])],
                        target_frame_ids=target_ids,
                        depth_path_by_frame_id=depth_path_by_id,
                        camera_k_aligned=sequence.camera_k_aligned,
                        camera_rt_dir=sequence.camera_rt_dir,
                        crop_params=sequence.crop_params,
                        target_height=H,
                        target_width=W,
                    ))
            else:
                warped_masked_rgb_valid = torch.empty((0, 3, H, W),
                                                      dtype=torch.float32)
                warped_mask_valid = torch.empty((0, 1, H, W),
                                                dtype=torch.float32)
            warped_masked_rgb = longwarp._pad_tchw(
                warped_masked_rgb_valid,
                max(int(window_frames) - 1, 1))
            warped_mask = longwarp._pad_tchw(warped_mask_valid,
                                             max(int(window_frames) - 1, 1))
            mask_tchw = torch.cat([
                torch.ones((1, 1, H, W), dtype=torch.float32),
                warped_mask[:max(int(window_frames) - 1, 0)],
            ],
                                  dim=0)
            masked_rgb_tchw = torch.cat([
                global_first_rgb.unsqueeze(0),
                warped_masked_rgb[:max(int(window_frames) - 1, 0)],
            ],
                                        dim=0)
        else:
            if (carry_prefix_tchw is None or warped_masked_rgb_next is None
                    or warped_mask_next is None):
                raise RuntimeError(
                    "Missing carry-over warp state for raw window > 0")
            mask_tchw = torch.cat([
                torch.ones((int(overlap_frames), 1, H, W),
                           dtype=torch.float32),
                warped_mask_next,
            ],
                                  dim=0)
            masked_rgb_tchw = torch.cat([carry_prefix_tchw, warped_masked_rgb_next],
                                        dim=0)

        depth_tchw = base._load_depth_sequence(
            depth_window_paths,
            H,
            W,
            pmin=float(args.raw_depth_percentile_min),
            pmax=float(args.raw_depth_percentile_max),
            invert_depth=bool(args.raw_depth_invert),
            normalization_mode=str(args.raw_depth_normalization_mode),
            crop_params=sequence.crop_params,
        )
        normal_tchw = None
        if normal_window_paths is not None:
            normal_tchw = torch.stack(
                [base._load_normal_frame(p, H, W) for p in normal_window_paths],
                dim=0)

        control_latent = base._encode_control_latent_from_tchw(
            vae=decoding.vae,
            depth_tchw=depth_tchw,
            normal_tchw=normal_tchw,
            masked_rgb_tchw=masked_rgb_tchw,
            mask_tchw=mask_tchw,
            target_c=target_c,
            inference_device=inference_device,
            compute_dtype=dtype,
        ).to(device="cuda", dtype=dtype)

        if bool(args.control_depth_only):
            total_c = int(control_latent.shape[1])
            if total_c % 3 != 0:
                raise ValueError(
                    f"control_latent channels must be divisible by 3, got {total_c}")
            base_c = total_c // 3
            control_latent = control_latent.clone()
            control_latent[:, base_c:] = 0

        latents = _run_bidirectional_window(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds_global,
            guidance_scale=float(args.guidance_scale),
            first_frame_latent_bcfhw=window_first_frame_latent,
            control_latent_bcfhw=control_latent,
            height=H,
            width=W,
            num_frames=int(window_frames),
            schedule_num_inference_steps=int(args.schedule_num_inference_steps),
            dmd_steps_list=dmd_steps_list,
            timestep_indices_list=timestep_indices_list,
            context_noise=int(args.context_noise),
            warp_denoising_step=bool(args.warp_denoising_step),
            update_rule=str(args.update_rule),
            full_schedule=bool(args.full_schedule),
            bidir_first_frame_timestep_zero=bool(
                args.bidir_first_frame_timestep_zero),
            bidir_sync_first_frame_state=bool(
                args.bidir_sync_first_frame_state),
            seed=int(args.seed + win_idx),
            dtype=dtype,
            trace_sample_id=f"{sequence.sample_id}_win{win_idx:02d}",
        )
        decoded_window = decoding.decode(latents, fastvideo_args).cpu().float()
        decoded_window_tchw = decoded_window[0].permute(1, 0, 2, 3).contiguous()
        decoded_window_tchw = decoded_window_tchw.clone()
        prefix_keep = min(int(window_prefix_tchw.shape[0]),
                          int(decoded_window_tchw.shape[0]))
        decoded_window_tchw[:prefix_keep] = window_prefix_tchw[:prefix_keep]

        if win_idx == 0:
            write_frames = decoded_window_tchw[:valid_window]
            depth_write = depth_tchw[:valid_window]
            mask_write = mask_tchw[:valid_window]
            masked_rgb_write = masked_rgb_tchw[:valid_window]
            normal_write = (normal_tchw[:valid_window]
                            if normal_tchw is not None else None)
        else:
            valid_new = max(0, int(valid_window) - int(overlap_frames))
            write_frames = decoded_window_tchw[int(overlap_frames):int(
                overlap_frames) + valid_new]
            depth_write = depth_tchw[int(overlap_frames):int(overlap_frames) +
                                     valid_new]
            mask_write = mask_tchw[int(overlap_frames):int(overlap_frames) +
                                   valid_new]
            masked_rgb_write = masked_rgb_tchw[int(overlap_frames):int(
                overlap_frames) + valid_new]
            normal_write = (normal_tchw[int(overlap_frames):int(overlap_frames)
                                        + valid_new]
                            if normal_tchw is not None else None)

        write_count = int(write_frames.shape[0])
        final_tchw[write_ptr:write_ptr + write_count] = write_frames
        if saved_controls is not None:
            saved_controls["depth"][write_ptr:write_ptr +
                                    write_count] = depth_write.cpu().float()
            saved_controls["mask"][write_ptr:write_ptr +
                                   write_count] = mask_write.cpu().float()
            saved_controls["masked_rgb"][write_ptr:write_ptr +
                                         write_count] = masked_rgb_write.cpu().float(
                                         )
            if "normal" in saved_controls and normal_write is not None:
                saved_controls["normal"][write_ptr:write_ptr +
                                         write_count] = normal_write.cpu().float(
                                         )
        write_ptr += write_count

        if win_idx < num_windows - 1:
            last_frame_id = int(sequence.frame_ids[end_pos_valid - 1])
            prefix_keep = min(int(overlap_frames), int(valid_window))
            carry_prefix_tchw = decoded_window_tchw[int(valid_window) -
                                                    prefix_keep:int(
                                                        valid_window)].clone()
            if use_precomputed_mask:
                continue

            current_frame_ids = [
                int(fid) for fid in sequence.frame_ids[start_pos:end_pos_valid]
            ]
            longwarp._update_history_and_visibility(
                sequence=sequence,
                frame_ids=current_frame_ids,
                frames_tchw=decoded_window_tchw[:valid_window],
                history_rgbs_u8=history_rgbs_u8,
                processed_frame_ids=processed_frame_ids,
                visibility_map=visibility_map,
            )

            next_chunk_start = max(0, int(end_pos_valid - overlap_frames))
            next_chunk_end = min(int(next_chunk_start + window_frames),
                                 int(total_required))
            target_frame_ids_for_chunk = [
                int(fid)
                for fid in sequence.frame_ids[next_chunk_start:next_chunk_end]
            ]
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
            logger.info("window=%s next_chunk_start=%s selected_keyframes=%s",
                        int(win_idx), int(next_chunk_start),
                        [int(x) for x in selected_keyframe_ids])
            keyframe_rgbs_u8 = [
                history_rgbs_u8[int(fid)] for fid in selected_keyframe_ids
            ]

            next_start = int(end_pos_valid)
            next_valid_new = min(int(stride),
                                 int(total_required) - int(next_start))
            target_ids_next = sequence.frame_ids[next_start:next_start +
                                                 next_valid_new]
            if target_ids_next:
                (warped_masked_rgb_next_valid,
                 warped_mask_next_valid) = longwarp._warp_maskrgb_from_keyframes_md_aligned(
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
                warped_masked_rgb_next = longwarp._pad_tchw(
                    warped_masked_rgb_next_valid, int(stride))
                warped_mask_next = longwarp._pad_tchw(warped_mask_next_valid,
                                                      int(stride))
            else:
                warped_masked_rgb_next = None
                warped_mask_next = None

    if int(write_ptr) != int(total_required):
        raise RuntimeError(
            f"Final raw frame count mismatch: wrote {int(write_ptr)}, expected {int(total_required)}"
        )
    return final_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous(), saved_controls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Long windowed bidirectional inference with parquet or raw online warp")
    p.add_argument("--base_model", required=True)
    p.add_argument("--transformer_dir", required=True)
    p.add_argument("--controlnet_dir", required=True)
    p.add_argument("--init_transformer_safetensors", default="")
    p.add_argument("--init_controlnet_safetensors", default="")

    p.add_argument("--input_mode",
                   choices=["parquet", "raw"],
                   default="parquet")
    p.add_argument("--data_path", default="")
    p.add_argument("--index", type=int, default=0)

    p.add_argument("--raw_sample_root", default="")
    p.add_argument("--raw_rgb_dir", default="")
    p.add_argument("--raw_depth_dir", default="")
    p.add_argument("--raw_normal_dir", default="")
    p.add_argument("--raw_mask_dir", default="")
    p.add_argument("--raw_masked_rgb_dir", default="")
    p.add_argument("--raw_require_normal", action="store_true")
    p.add_argument("--raw_mask_threshold", type=float, default=-1.0)
    p.add_argument("--raw_mask_invert", action="store_true")
    p.add_argument("--raw_prompt", default="")
    p.add_argument("--raw_caption_path", default="")
    p.add_argument("--raw_caption_key", default="Video_Caption")
    p.add_argument("--raw_fps", type=int, default=16)
    p.add_argument("--cam_k", default="")
    p.add_argument("--cam_rt_dir", default="")
    p.add_argument("--raw_depth_percentile_min", type=float, default=2.0)
    p.add_argument("--raw_depth_percentile_max", type=float, default=98.0)
    p.add_argument("--raw_depth_normalization_mode",
                   type=str,
                   default="md_align",
                   choices=["md_align", "percentile"])
    raw_depth_group = p.add_mutually_exclusive_group()
    raw_depth_group.add_argument("--raw_depth_invert",
                                 dest="raw_depth_invert",
                                 action="store_true")
    raw_depth_group.add_argument("--no_raw_depth_invert",
                                 dest="raw_depth_invert",
                                 action="store_false")
    p.set_defaults(raw_depth_invert=False)

    p.add_argument("--scheduler",
                   choices=["flowmatch_euler", "unipc"],
                   default="unipc")
    p.add_argument("--schedule_num_inference_steps", type=int, default=50)
    p.add_argument("--timestep_indices", type=str, default="")
    p.add_argument("--dmd_steps", type=str, default="1000,750,500,250")
    p.add_argument("--update_rule",
                   choices=["renoise_x0", "euler_dt"],
                   default="euler_dt")
    p.add_argument("--full_schedule", action="store_true")
    p.add_argument("--warp_denoising_step", action="store_true", default=True)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--negative_prompt",
                   type=str,
                   default="bad quality, worst quality")
    p.add_argument("--control_depth_only", action="store_true")

    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--causal_window_frames", type=int, default=81)
    p.add_argument("--causal_overlap_frames", type=int, default=1)
    p.add_argument("--warp_num_keyframes", type=int, default=4)
    p.add_argument("--selection_num_target_samples", type=int, default=3)
    p.add_argument("--selection_voxel_size", type=float, default=0.1)
    p.add_argument("--context_noise", type=int, default=0)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--dtype",
                   choices=["fp32", "bf16", "fp16"],
                   default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--save_control_outputs", action="store_true")

    bidir_t0_group = p.add_mutually_exclusive_group()
    bidir_t0_group.add_argument("--bidir_first_frame_timestep_zero",
                                dest="bidir_first_frame_timestep_zero",
                                action="store_true")
    bidir_t0_group.add_argument("--no_bidir_first_frame_timestep_zero",
                                dest="bidir_first_frame_timestep_zero",
                                action="store_false")
    p.set_defaults(bidir_first_frame_timestep_zero=True)

    bidir_sync_group = p.add_mutually_exclusive_group()
    bidir_sync_group.add_argument("--bidir_sync_first_frame_state",
                                  dest="bidir_sync_first_frame_state",
                                  action="store_true")
    bidir_sync_group.add_argument("--no_bidir_sync_first_frame_state",
                                  dest="bidir_sync_first_frame_state",
                                  action="store_false")
    p.set_defaults(bidir_sync_first_frame_state=True)

    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.causal_overlap_frames) <= 0 or int(
            args.causal_overlap_frames) >= int(args.causal_window_frames):
        raise ValueError(
            "causal_overlap_frames must satisfy 0 < overlap < window for the windowed bidirectional script"
        )
    if str(args.input_mode) == "parquet" and not str(args.data_path).strip():
        raise ValueError("--data_path is required for input_mode=parquet")
    if str(args.input_mode) == "raw" and not str(args.raw_sample_root).strip():
        raise ValueError("--raw_sample_root is required for input_mode=raw")

    base._ensure_single_process_dist_env()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    requested_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    inference_device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    dmd_steps_list = [int(x) for x in str(args.dmd_steps).split(",") if x.strip()
                      ] or None
    timestep_indices_list = [
        int(x) for x in str(args.timestep_indices).split(",") if x.strip()
    ] or None
    if bool(args.full_schedule):
        dmd_steps_list = None
        timestep_indices_list = None

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
    use_union_controlnet = base._controlnet_dir_is_union(args.controlnet_dir)
    # Match Diff-Factory fp32 pipeline behavior when --dtype fp32 is requested.
    # Without this, FastVideo's loader instantiates DiT/ControlNet in the pipeline
    # default bf16 and runtime tensors are later forced back to bf16.
    fastvideo_args.pipeline_config.dit_precision = str(args.dtype)
    fastvideo_args.override_transformer_cls_name = "WanTransformer3DModel"
    fastvideo_args.override_controlnet_cls_name = (
        "WanControlnetUnion3DModel" if use_union_controlnet else
        "WanControlnet3DModel")
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    if hasattr(fastvideo_args.pipeline_config, "expand_timesteps"):
        fastvideo_args.pipeline_config.expand_timesteps = True
    if hasattr(fastvideo_args.pipeline_config, "dit_config"):
        fastvideo_args.pipeline_config.dit_config.expand_timesteps = True
    fastvideo_args.pipeline_config.warp_denoising_step = bool(
        args.warp_denoising_step)
    fastvideo_args.pipeline_config.dmd_denoising_steps = dmd_steps_list or []
    fastvideo_args.pipeline_config.context_noise = int(args.context_noise)

    if args.init_transformer_safetensors:
        fastvideo_args.init_weights_from_safetensors = args.init_transformer_safetensors
    if args.init_controlnet_safetensors:
        fastvideo_args.init_controlnet_weights_from_safetensors = args.init_controlnet_safetensors

    transformer = PipelineComponentLoader.load_module("transformer",
                                                      args.transformer_dir,
                                                      "diffusers",
                                                      fastvideo_args)
    controlnet = PipelineComponentLoader.load_module("controlnet",
                                                     args.controlnet_dir,
                                                     "diffusers",
                                                     fastvideo_args)
    dtype = _align_runtime_dtype(transformer, controlnet, requested_dtype)

    tokenizer = None
    text_encoder = None
    if str(args.input_mode) == "raw" or float(args.guidance_scale) != 1.0:
        tokenizer = PipelineComponentLoader.load_module(
            "tokenizer",
            str(Path(args.base_model) / "tokenizer"),
            "transformers",
            fastvideo_args,
        )
        text_encoder = PipelineComponentLoader.load_module(
            "text_encoder",
            str(Path(args.base_model) / "text_encoder"),
            "transformers",
            fastvideo_args,
        )

    negative_prompt_embeds_global = None
    if (float(args.guidance_scale) != 1.0 and tokenizer is not None
            and text_encoder is not None):
        max_text_len = int(getattr(transformer, "text_len", 226))
        negative_prompt_embeds_global = base._compute_negative_prompt_embeddings(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            negative_prompt=str(args.negative_prompt),
            max_sequence_length=max_text_len,
            dtype=dtype,
            target_device=inference_device,
        )

    if args.scheduler == "unipc":
        scheduler = base.DiffusersUniPCMultistepScheduler.from_pretrained(
            args.base_model, subfolder="scheduler")
        scheduler = base.DiffusersUniPCMultistepScheduler.from_config(
            scheduler.config, flow_shift=float(args.flow_shift))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=float(args.flow_shift))

    vae = PipelineComponentLoader.load_module(
        "vae", str(Path(args.base_model) / "vae"), "diffusers",
        fastvideo_args)
    decoding = DecodingStage(vae=vae)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.input_mode) == "raw":
        assert tokenizer is not None and text_encoder is not None
        sequence = longwarp._load_raw_long_sequence_nomask(
            sample_root=Path(args.raw_sample_root).expanduser().resolve(),
            args=args,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            inference_device=inference_device,
            dtype=dtype,
        )
        logger.info(
            "sample=%s caption=%s long-bidir-online-warp num_frames=%s window=%s overlap=%s",
            sequence.sample_id,
            sequence.prompt,
            int(args.num_frames),
            int(args.causal_window_frames),
            int(args.causal_overlap_frames),
        )
        decoded, saved_controls = _run_windowed_bidirectional_raw(
            sequence=sequence,
            args=args,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            decoding=decoding,
            fastvideo_args=fastvideo_args,
            negative_prompt_embeds_global=negative_prompt_embeds_global,
            dtype=dtype,
        )
        out_path = out_dir / f"{sequence.sample_id}_windowed_bidir.mp4"
        base._save_mp4(decoded, str(out_path), fps=int(sequence.fps))
        _write_run_manifest(
            out_dir=out_dir,
            args=args,
            sample_id=sequence.sample_id,
            prompt=sequence.prompt,
            output_path=out_path,
        )
        if bool(args.save_frames):
            base._save_frames_png(decoded,
                                  str(out_dir / "frames" / sequence.sample_id),
                                  prefix=sequence.sample_id)
        if saved_controls is not None:
            longwarp._save_control_outputs(
                out_dir=out_dir,
                sample_id=sequence.sample_id,
                fps=int(sequence.fps),
                depth_tchw=saved_controls["depth"],
                normal_tchw=saved_controls.get("normal"),
                mask_tchw=saved_controls["mask"],
                masked_rgb_tchw=saved_controls["masked_rgb"],
                save_frames=bool(args.save_frames),
            )
    else:
        sample = base._load_sample(str(args.data_path), int(args.index))
        logger.info(
            "sample=%s idx=%s caption=%s long-bidir-parquet num_frames=%s window=%s overlap=%s",
            sample.sample_id,
            int(args.index),
            sample.caption,
            int(args.num_frames),
            int(args.causal_window_frames),
            int(args.causal_overlap_frames),
        )
        decoded = _run_windowed_bidirectional_parquet(
            sample=sample,
            args=args,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            decoding=decoding,
            fastvideo_args=fastvideo_args,
            negative_prompt_embeds_global=negative_prompt_embeds_global,
            dtype=dtype,
        )
        out_path = out_dir / (
            f"{sample.sample_id.replace('/', '__')}_windowed_bidir_o{int(args.causal_overlap_frames)}.mp4"
        )
        base._save_mp4(decoded, str(out_path), fps=int(args.fps))
        _write_run_manifest(
            out_dir=out_dir,
            args=args,
            sample_id=sample.sample_id.replace("/", "__"),
            prompt=sample.caption,
            output_path=out_path,
        )
        if bool(args.save_frames):
            base._save_frames_png(decoded,
                                  str(out_dir / "frames" / sample.sample_id),
                                  prefix=sample.sample_id)

    logger.info("saved: %s", str(out_path))


if __name__ == "__main__":
    main()
