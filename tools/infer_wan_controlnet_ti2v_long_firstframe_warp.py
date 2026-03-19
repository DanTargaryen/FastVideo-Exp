#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Long causal TI2V + ControlNet inference with online warp-generated mask/masked_rgb.

Behavior:
- First window: use only frame0 RGB as the source keyframe and warp to the first
  `causal_window_frames` targets. Frame0 itself is forced visible.
- Later windows: keep a 1-frame visual overlap. The overlap frame is inherited
  directly from the previous chunk output, while the remaining
  `causal_window_frames - 1` frames are warped from the previous chunk's last
  4 generated frames and merged.
- Temporal continuity is carried by the overlap frame, multi-keyframe warp, and
  the causal KV-cache.

This keeps the rollout semantics aligned with long causal inference while
removing the dependency on precomputed mask/masked_rgb clips.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage

import tools.infer_wan_controlnet_ti2v as base

logger = init_logger(__name__)


@dataclass(frozen=True)
class RawLongSequenceNoMask:
    sample_id: str
    prompt: str
    fps: int
    text_embedding_bld: torch.Tensor
    rgb_paths: list[Path]
    depth_paths: list[Path]
    normal_paths: list[Path] | None
    frame_ids: list[int]
    camera_k_aligned: base.np.ndarray
    camera_rt_dir: Path
    crop_params: tuple[int, int, int, int]


class _GlobalVoxelVisibilityMap:

    def __init__(self, voxel_size: float = 0.1) -> None:
        self.voxel_size = float(voxel_size)
        self.voxel_vis: dict[tuple[int, int, int], set[int]] = defaultdict(set)

    def points_to_voxels(self, points_world: base.np.ndarray) -> set[tuple[int, int, int]]:
        if int(points_world.shape[0]) == 0:
            return set()
        voxels = base.np.floor(points_world / float(self.voxel_size)).astype(base.np.int32)
        return set(map(tuple, voxels.tolist()))

    def add_view(self, frame_idx: int, points_world: base.np.ndarray) -> None:
        voxels = self.points_to_voxels(points_world)
        for voxel in voxels:
            self.voxel_vis[voxel].add(int(frame_idx))

    def query_top_k_complementary_views(
        self,
        *,
        target_voxels: set[tuple[int, int, int]],
        k: int,
        valid_history_ids: set[int] | None = None,
    ) -> list[int]:
        remaining_voxels = set(target_voxels)
        selected_views: list[int] = []
        for _ in range(int(k)):
            view_counts: dict[int, int] = defaultdict(int)
            for voxel in remaining_voxels:
                for frame_idx in self.voxel_vis.get(voxel, ()):
                    if valid_history_ids is not None and int(frame_idx) not in valid_history_ids:
                        continue
                    if int(frame_idx) in selected_views:
                        continue
                    view_counts[int(frame_idx)] += 1
            if not view_counts:
                break
            best_view = max(view_counts, key=view_counts.get)
            selected_views.append(int(best_view))
            covered = {
                voxel for voxel in remaining_voxels
                if int(best_view) in self.voxel_vis.get(voxel, ())
            }
            remaining_voxels -= covered
        return selected_views


def _pad_paths(paths: list[Path], target_len: int) -> list[Path]:
    if len(paths) >= target_len:
        return list(paths[:target_len])
    if not paths:
        raise ValueError("cannot pad empty path list")
    return list(paths) + [paths[-1]] * (target_len - len(paths))


def _pad_tchw(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if int(x.shape[0]) >= int(target_len):
        return x[:target_len]
    if int(x.shape[0]) <= 0:
        raise ValueError("cannot pad empty frame tensor")
    pad = x[-1:].repeat(int(target_len) - int(x.shape[0]), 1, 1, 1)
    return torch.cat([x, pad], dim=0)


def _compute_num_windows(total_frames: int, window_frames: int, overlap_frames: int) -> int:
    if total_frames <= window_frames:
        return 1
    stride = window_frames - overlap_frames
    return 1 + int(math.ceil(float(total_frames - window_frames) / float(stride)))


def _chw_float_to_u8(x_chw: torch.Tensor) -> base.np.ndarray:
    return (
        x_chw.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    ).round().astype(base.np.uint8)


def _select_keyframes_for_next_window(
    *,
    sequence: RawLongSequenceNoMask,
    visibility_map: _GlobalVoxelVisibilityMap,
    processed_frame_ids: set[int],
    history_rgbs_u8: dict[int, base.np.ndarray],
    target_frame_ids_for_chunk: list[int],
    depth_path_by_id: dict[int, Path],
    target_height: int,
    target_width: int,
    last_frame_id: int,
    num_keyframes: int,
    num_target_samples: int,
) -> list[int]:
    if int(last_frame_id) not in history_rgbs_u8:
        raise KeyError(f"Missing RGB history for overlap frame {int(last_frame_id)}")

    warper = base._PointCloudWarper()
    target_voxels: set[tuple[int, int, int]] = set()
    if target_frame_ids_for_chunk:
        sample_n = max(1, min(int(num_target_samples), len(target_frame_ids_for_chunk)))
        sample_positions = base.np.linspace(
            0, len(target_frame_ids_for_chunk) - 1, num=sample_n, dtype=int
        ).tolist()
        sampled_ids = [int(target_frame_ids_for_chunk[i]) for i in sample_positions]
        dummy_rgb = base.np.zeros((int(target_height), int(target_width), 3), dtype=base.np.uint8)
        for tgt_id in sampled_ids:
            tgt_depth = base._aligned_physical_depth(
                depth_path_by_id[int(tgt_id)],
                sequence.crop_params,
                int(target_width),
                int(target_height),
            )
            tgt_rt = base._load_camera_matrix(
                base._resolve_camera_rt_path(sequence.camera_rt_dir, int(tgt_id))
            )
            points_world, _ = warper.back_project(
                dummy_rgb, tgt_depth, sequence.camera_k_aligned, tgt_rt
            )
            target_voxels.update(visibility_map.points_to_voxels(points_world))

    k_query = max(0, int(num_keyframes) - 1)
    best_keys = visibility_map.query_top_k_complementary_views(
        target_voxels=target_voxels,
        k=k_query,
        valid_history_ids=processed_frame_ids,
    )
    selected = [int(fid) for fid in best_keys if int(fid) != int(last_frame_id)]

    if len(selected) < k_query:
        recent_candidates = sorted(
            [int(fid) for fid in processed_frame_ids if int(fid) != int(last_frame_id)],
            reverse=True,
        )
        for fid in recent_candidates:
            if fid in selected or fid not in history_rgbs_u8:
                continue
            selected.append(fid)
            if len(selected) >= k_query:
                break

    selected = selected[:k_query]
    selected.append(int(last_frame_id))
    return selected


def _update_history_and_visibility(
    *,
    sequence: RawLongSequenceNoMask,
    frame_ids: list[int],
    frames_tchw: torch.Tensor,
    history_rgbs_u8: dict[int, base.np.ndarray],
    processed_frame_ids: set[int],
    visibility_map: _GlobalVoxelVisibilityMap,
) -> None:
    warper = base._PointCloudWarper()
    H = int(frames_tchw.shape[-2])
    W = int(frames_tchw.shape[-1])
    frame_id_to_depth_path = {int(fid): p for fid, p in zip(sequence.frame_ids, sequence.depth_paths)}
    for local_idx, frame_id in enumerate(frame_ids):
        frame_id = int(frame_id)
        rgb_u8 = _chw_float_to_u8(frames_tchw[local_idx])
        history_rgbs_u8[frame_id] = rgb_u8
        processed_frame_ids.add(frame_id)
        depth_path = frame_id_to_depth_path.get(frame_id)
        if depth_path is None:
            continue
        depth_aligned = base._aligned_physical_depth(
            depth_path,
            sequence.crop_params,
            W,
            H,
        )
        rt = base._load_camera_matrix(base._resolve_camera_rt_path(sequence.camera_rt_dir, frame_id))
        points_world, _ = warper.back_project(rgb_u8, depth_aligned, sequence.camera_k_aligned, rt)
        visibility_map.add_view(frame_id, points_world)


def _warp_maskrgb_from_keyframes_md_aligned(
    *,
    keyframe_rgbs_u8: list[base.np.ndarray],
    keyframe_frame_ids: list[int],
    target_frame_ids: list[int],
    depth_path_by_frame_id: dict[int, Path],
    camera_k_aligned: base.np.ndarray,
    camera_rt_dir: Path,
    crop_params: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align the warp kernel to run_wan_controlnet_union_long_w_selection.md:
    - each source keyframe is independently back-projected and projected
    - per-target results are merged in keyframe order
    - mask is a hard visibility union over all warped sources
    """
    warper = base._PointCloudWarper()
    source_clouds: list[tuple[base.np.ndarray, base.np.ndarray]] = []
    for src_rgb, src_id in zip(keyframe_rgbs_u8, keyframe_frame_ids):
        if int(src_id) not in depth_path_by_frame_id:
            raise KeyError(f"Missing depth path for source frame id {int(src_id)}")
        src_depth = base._aligned_physical_depth(
            depth_path_by_frame_id[int(src_id)],
            crop_params,
            int(target_width),
            int(target_height),
        )
        src_rt = base._load_camera_matrix(
            base._resolve_camera_rt_path(camera_rt_dir, int(src_id))
        )
        points_world, colors = warper.back_project(
            src_rgb,
            src_depth,
            camera_k_aligned,
            src_rt,
        )
        source_clouds.append((points_world, colors))

    merged_rgb_tensors: list[torch.Tensor] = []
    merged_mask_tensors: list[torch.Tensor] = []
    for tgt_id in target_frame_ids:
        if int(tgt_id) not in depth_path_by_frame_id:
            raise KeyError(f"Missing depth path for target frame id {int(tgt_id)}")
        tgt_rt = base._load_camera_matrix(
            base._resolve_camera_rt_path(camera_rt_dir, int(tgt_id))
        )
        tgt_depth = base._aligned_physical_depth(
            depth_path_by_frame_id[int(tgt_id)],
            crop_params,
            int(target_width),
            int(target_height),
        )

        merged_rgb = base.np.zeros((int(target_height), int(target_width), 3), dtype=base.np.uint8)
        merged_mask = base.np.zeros((int(target_height), int(target_width)), dtype=base.np.uint8)
        for points_world, colors in source_clouds:
            w_rgb, w_mask = warper.project_to_view(
                points_world,
                colors,
                camera_k_aligned,
                tgt_rt,
                int(target_height),
                int(target_width),
                target_depth=tgt_depth,
            )
            valid_area = w_mask > 127.5
            merged_rgb[valid_area] = w_rgb[valid_area]
            merged_mask[valid_area] = 255

        merged_rgb_tensors.append(
            torch.from_numpy(merged_rgb.astype(base.np.float32) / 255.0).permute(2, 0, 1).contiguous()
        )
        merged_mask_tensors.append(
            torch.from_numpy((merged_mask > 127.5).astype(base.np.float32))[None, ...].contiguous()
        )

    return torch.stack(merged_rgb_tensors, dim=0), torch.stack(merged_mask_tensors, dim=0)


def _load_raw_long_sequence_nomask(
    *,
    sample_root: Path,
    args,
    tokenizer,
    text_encoder,
    transformer,
    inference_device: torch.device,
    dtype: torch.dtype,
) -> RawLongSequenceNoMask:
    rgb_dir, depth_dir, normal_dir, _mask_dir, _masked_rgb_dir = base._resolve_raw_dirs(
        sample_root=sample_root, args=args
    )

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"raw rgb dir not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"raw depth dir not found: {depth_dir}")

    rgb_all = base._sorted_files(
        rgb_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    )
    depth_all = base._sorted_files(
        depth_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr")
    )
    normal_all: list[Path] | None = None
    if normal_dir.is_dir():
        nfiles = base._sorted_files(
            normal_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr")
        )
        if nfiles:
            normal_all = nfiles
    if bool(args.raw_require_normal) and normal_all is None:
        raise FileNotFoundError(
            f"--raw_require_normal is set but normal dir is missing/empty: {normal_dir}"
        )

    total_required = int(args.num_frames)
    if len(rgb_all) < total_required:
        raise ValueError(
            f"raw rgb frames are insufficient: need {total_required}, got {len(rgb_all)}"
        )
    if len(depth_all) < total_required:
        raise ValueError(
            f"raw depth frames are insufficient: need {total_required}, got {len(depth_all)}"
        )
    if normal_all is not None and len(normal_all) < total_required:
        raise ValueError(
            f"raw normal frames are insufficient: need {total_required}, got {len(normal_all)}"
        )

    rgb_paths = rgb_all[:total_required]
    depth_paths = depth_all[:total_required]
    frame_ids = base._extract_frame_ids(depth_paths, name="depth")
    normal_paths = normal_all[:total_required] if normal_all is not None else None

    prompt = base._resolve_raw_prompt(sample_root, args)
    max_text_len = int(getattr(transformer, "text_len", 226))
    text_embedding = base._compute_prompt_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_text_len,
        dtype=dtype,
        target_device=inference_device,
    )

    cam_k_path = Path(str(args.cam_k)).expanduser() if str(args.cam_k).strip() else (
        sample_root / "camera" / "camera_K.txt"
    )
    camera_rt_dir = Path(str(args.cam_rt_dir)).expanduser() if str(args.cam_rt_dir).strip() else (
        sample_root / "camera"
    )
    if not camera_rt_dir.is_dir():
        raise FileNotFoundError(f"camera RT dir not found: {camera_rt_dir}")
    camera_k = base._load_camera_matrix(cam_k_path).astype(base.np.float32)

    ref_img = base.Image.open(rgb_paths[0]).convert("RGB")
    src_w, src_h = ref_img.size
    crop_params = base._get_crop_params(src_w, src_h, int(args.width), int(args.height))
    camera_k_aligned = base._adjust_intrinsics(
        camera_k, crop_params, int(args.width), int(args.height)
    )

    return RawLongSequenceNoMask(
        sample_id=sample_root.name,
        prompt=prompt,
        fps=int(args.raw_fps),
        text_embedding_bld=text_embedding,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        normal_paths=normal_paths,
        frame_ids=frame_ids,
        camera_k_aligned=camera_k_aligned,
        camera_rt_dir=camera_rt_dir,
        crop_params=crop_params,
    )


@torch.no_grad()
def _run_long_rollout_firstframe_warp(
    *,
    sequence: RawLongSequenceNoMask,
    args,
    transformer,
    controlnet,
    scheduler,
    vae,
    decoding: DecodingStage,
    fastvideo_args: FastVideoArgs,
    negative_prompt_embeds_global: torch.Tensor | None,
    dtype: torch.dtype,
    inference_device: torch.device,
    dmd_steps_list: list[int] | None,
    timestep_indices_list: list[int] | None,
) -> torch.Tensor:
    condition_mode = str(args.first_frame_condition_mode).lower()
    if condition_mode == "md_align":
        logger.warning(
            "first_frame_condition_mode=md_align is unstable for long causal rollout; "
            "falling back to hard_replace in this branch."
        )
        condition_mode = "hard_replace"

    total_required = int(args.num_frames)
    window_frames = int(args.causal_window_frames)
    overlap_frames = int(args.causal_overlap_frames)
    if overlap_frames != 1:
        raise ValueError(
            f"This script currently expects causal_overlap_frames=1, got {overlap_frames}"
        )
    stride = int(window_frames - overlap_frames)
    num_windows = _compute_num_windows(total_required, window_frames, overlap_frames)

    prompt_embeds = sequence.text_embedding_bld.to(device="cuda", dtype=dtype)
    negative_prompt_embeds = None
    if float(args.guidance_scale) != 1.0:
        negative_prompt_embeds = (
            negative_prompt_embeds_global
            if negative_prompt_embeds_global is not None
            else torch.zeros_like(prompt_embeds)
        )

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
    final_tchw = torch.empty((total_required, 3, H, W), dtype=torch.float32)
    write_ptr = 0
    history_rgbs_u8: dict[int, base.np.ndarray] = {}
    processed_frame_ids: set[int] = set()
    visibility_map = _GlobalVoxelVisibilityMap(voxel_size=float(args.selection_voxel_size))

    carry_keyframes_tchw: torch.Tensor | None = None
    carry_keyframe_ids: list[int] | None = None
    warped_masked_rgb_next: torch.Tensor | None = None
    warped_mask_next: torch.Tensor | None = None

    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = [prompt_embeds]
    batch.height = H
    batch.width = W
    batch.num_frames = window_frames

    frame_seq_length: int | None = None
    global_start_latent = 0
    latent_stride_t: int | None = None
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
        if valid_window <= 0:
            break
        end_pos_valid = int(start_pos + valid_window)
        window_first_rgb = global_first_rgb if win_idx == 0 else carry_keyframes_tchw[-1]
        window_first_frame_latent = (
            global_first_frame_latent
            if win_idx == 0
            else base._encode_first_frame_latent(
                vae=vae,
                first_rgb_chw=window_first_rgb,
                target_c=target_c,
                inference_device=inference_device,
                compute_dtype=dtype,
            ).to(device="cuda", dtype=dtype)
        )

        depth_window_paths = _pad_paths(sequence.depth_paths[start_pos:end_pos_valid], window_frames)
        if sequence.normal_paths is not None:
            normal_window_paths = _pad_paths(sequence.normal_paths[start_pos:end_pos_valid], window_frames)
        else:
            normal_window_paths = None

        if win_idx == 0:
            first_rgb_u8 = (
                global_first_rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
            ).round().astype(base.np.uint8)
            target_ids = sequence.frame_ids[start_pos + 1:end_pos_valid]
            if len(target_ids) > 0:
                warped_masked_rgb_valid, warped_mask_valid = _warp_maskrgb_from_keyframes_md_aligned(
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

            warped_masked_rgb = _pad_tchw(warped_masked_rgb_valid, max(window_frames - 1, 1))
            warped_mask = _pad_tchw(warped_mask_valid, max(window_frames - 1, 1))
            mask_tchw = torch.cat(
                [
                    torch.ones((1, 1, H, W), dtype=torch.float32),
                    warped_mask[:max(window_frames - 1, 0)],
                ],
                dim=0,
            )
            masked_rgb_tchw = torch.cat(
                [
                    global_first_rgb.unsqueeze(0),
                    warped_masked_rgb[:max(window_frames - 1, 0)],
                ],
                dim=0,
            )
        else:
            if (
                carry_keyframes_tchw is None
                or carry_keyframe_ids is None
                or warped_masked_rgb_next is None
                or warped_mask_next is None
            ):
                raise RuntimeError("Missing carry-over warp state for window > 0")
            mask_tchw = torch.cat(
                [
                    torch.ones((1, 1, H, W), dtype=torch.float32),
                    warped_mask_next,
                ],
                dim=0,
            )
            masked_rgb_tchw = torch.cat(
                [
                    carry_keyframes_tchw[-1:],
                    warped_masked_rgb_next,
                ],
                dim=0,
            )

        depth_tchw = base._load_depth_sequence(
            depth_window_paths,
            H,
            W,
            pmin=float(args.raw_depth_percentile_min),
            pmax=float(args.raw_depth_percentile_max),
            invert_depth=bool(args.raw_depth_invert),
        )
        normal_tchw = None
        if normal_window_paths is not None:
            normal_tchw = torch.stack(
                [base._load_normal_frame(p, H, W) for p in normal_window_paths],
                dim=0,
            )

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

        if bool(args.control_depth_only):
            total_c = int(control_latent.shape[1])
            if total_c % 3 != 0:
                raise ValueError(f"control_latent channels must be divisible by 3, got {total_c}")
            base_c = total_c // 3
            control_latent = control_latent.clone()
            control_latent[:, base_c:] = 0

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
            if patch_ratio <= 0 or latent_seq_length % patch_ratio != 0:
                raise ValueError(
                    f"Invalid patch_ratio={patch_ratio} for latent HxW={latent_h}x{latent_w}"
                )
            frame_seq_length = latent_seq_length // patch_ratio
            total_latent_frames = latent_t + max(0, num_windows - 1) * latent_stride_t
            kv_cache = base._initialize_kv_cache(
                model=transformer,
                batch_size=1,
                dtype=dtype,
                device=rollout_device,
                frame_seq_length=frame_seq_length,
                sliding_window_num_frames_override=total_latent_frames,
            )
            crossattn_cache = base._initialize_crossattn_cache(
                model=transformer,
                batch_size=1,
                max_text_len=int(getattr(transformer, "text_len", prompt_embeds.shape[1])),
                dtype=dtype,
                device=rollout_device,
            )
            control_kv_cache = base._initialize_kv_cache(
                model=controlnet,
                batch_size=1,
                dtype=dtype,
                device=rollout_device,
                frame_seq_length=frame_seq_length,
                sliding_window_num_frames_override=total_latent_frames,
            )
            control_crossattn_cache = base._initialize_crossattn_cache(
                model=controlnet,
                batch_size=1,
                max_text_len=int(getattr(controlnet, "text_len", prompt_embeds.shape[1])),
                dtype=dtype,
                device=rollout_device,
            )
            if float(args.guidance_scale) != 1.0:
                kv_cache_uncond = base._initialize_kv_cache(
                    model=transformer,
                    batch_size=1,
                    dtype=dtype,
                    device=rollout_device,
                    frame_seq_length=frame_seq_length,
                    sliding_window_num_frames_override=total_latent_frames,
                )
                crossattn_cache_uncond = base._initialize_crossattn_cache(
                    model=transformer,
                    batch_size=1,
                    max_text_len=int(getattr(transformer, "text_len", prompt_embeds.shape[1])),
                    dtype=dtype,
                    device=rollout_device,
                )
                control_kv_cache_uncond = base._initialize_kv_cache(
                    model=controlnet,
                    batch_size=1,
                    dtype=dtype,
                    device=rollout_device,
                    frame_seq_length=frame_seq_length,
                    sliding_window_num_frames_override=total_latent_frames,
                )
                control_crossattn_cache_uncond = base._initialize_crossattn_cache(
                    model=controlnet,
                    batch_size=1,
                    max_text_len=int(getattr(controlnet, "text_len", prompt_embeds.shape[1])),
                    dtype=dtype,
                    device=rollout_device,
                )

        latents = base._causal_dmd_rollout_one_window_with_cache(
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            prompt_embeds_list=[prompt_embeds],
            negative_prompt_embeds_list=(
                [negative_prompt_embeds] if negative_prompt_embeds is not None else None
            ),
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
            first_frame_timestep_zero=bool(args.first_frame_timestep_zero) or condition_mode != "md_align",
            expand_timesteps=bool(getattr(fastvideo_args.pipeline_config, "expand_timesteps", False)),
            disable_cache_update=bool(args.disable_cache_update),
            first_frame_anchor_latent_frames=int(args.first_frame_anchor_latent_frames),
            first_frame_condition_mode=condition_mode,
            seed=int(args.seed + win_idx),
            dtype=dtype,
            global_start_latent=int(global_start_latent),
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
        global_start_latent += int(latent_stride_t)

        decoded_window = decoding.decode(latents, fastvideo_args).cpu().float()
        decoded_window_tchw = decoded_window[0].permute(1, 0, 2, 3).contiguous()
        decoded_window_tchw = decoded_window_tchw.clone()
        if condition_mode == "hard_replace":
            decoded_window_tchw[0] = window_first_rgb
        if win_idx == 0:
            blend_frames = int(max(0, int(args.first_frame_blend_frames)))
            if blend_frames > 0:
                n_blend = min(blend_frames, int(window_frames) - 1)
                for k in range(1, n_blend + 1):
                    alpha = 1.0 - float(k) / float(n_blend + 1)
                    decoded_window_tchw[k] = alpha * global_first_rgb + (
                        1.0 - alpha
                    ) * decoded_window_tchw[k]
        if win_idx == 0:
            write_frames = decoded_window_tchw[:valid_window]
        else:
            valid_new = max(0, valid_window - overlap_frames)
            write_frames = decoded_window_tchw[overlap_frames:overlap_frames + valid_new]

        write_count = int(write_frames.shape[0])
        final_tchw[write_ptr:write_ptr + write_count] = write_frames
        write_ptr += write_count

        current_frame_ids = [int(fid) for fid in sequence.frame_ids[start_pos:end_pos_valid]]
        _update_history_and_visibility(
            sequence=sequence,
            frame_ids=current_frame_ids,
            frames_tchw=decoded_window_tchw[:valid_window],
            history_rgbs_u8=history_rgbs_u8,
            processed_frame_ids=processed_frame_ids,
            visibility_map=visibility_map,
        )

        if win_idx < num_windows - 1:
            last_frame_id = int(sequence.frame_ids[end_pos_valid - 1])

            next_chunk_start = max(0, int(end_pos_valid - overlap_frames))
            next_chunk_end = min(next_chunk_start + window_frames, total_required)
            target_frame_ids_for_chunk = [
                int(fid) for fid in sequence.frame_ids[next_chunk_start:next_chunk_end]
            ]
            selected_keyframe_ids = _select_keyframes_for_next_window(
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
            logger.info(
                "window=%s next_chunk_start=%s selected_keyframes=%s",
                int(win_idx),
                int(next_chunk_start),
                [int(x) for x in selected_keyframe_ids],
            )
            keyframe_rgbs_u8 = [history_rgbs_u8[int(fid)] for fid in selected_keyframe_ids]
            carry_keyframe_ids = [int(fid) for fid in selected_keyframe_ids]
            carry_keyframes_tchw = torch.stack(
                [
                    torch.from_numpy(rgb.astype(base.np.float32) / 255.0).permute(2, 0, 1).contiguous()
                    for rgb in keyframe_rgbs_u8
                ],
                dim=0,
            )

            next_start = int(end_pos_valid)
            next_valid_new = min(stride, total_required - next_start)
            target_ids_next = sequence.frame_ids[next_start:next_start + next_valid_new]

            warped_masked_rgb_next_valid, warped_mask_next_valid = _warp_maskrgb_from_keyframes_md_aligned(
                keyframe_rgbs_u8=keyframe_rgbs_u8,
                keyframe_frame_ids=carry_keyframe_ids,
                target_frame_ids=target_ids_next,
                depth_path_by_frame_id=depth_path_by_id,
                camera_k_aligned=sequence.camera_k_aligned,
                camera_rt_dir=sequence.camera_rt_dir,
                crop_params=sequence.crop_params,
                target_height=H,
                target_width=W,
            )
            warped_masked_rgb_next = _pad_tchw(warped_masked_rgb_next_valid, stride)
            warped_mask_next = _pad_tchw(warped_mask_next_valid, stride)

    if write_ptr != total_required:
        raise RuntimeError(f"Final frame count mismatch: wrote {write_ptr}, expected {total_required}")
    return final_tchw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Long causal raw inference with first-frame warp")
    p.add_argument("--base_model", required=True)
    p.add_argument("--transformer_dir", required=True)
    p.add_argument("--controlnet_dir", required=True)
    p.add_argument("--init_transformer_safetensors", default="")
    p.add_argument("--init_controlnet_safetensors", default="")

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

    p.add_argument("--scheduler", choices=["flowmatch_euler", "unipc"], default="flowmatch_euler")
    p.add_argument("--schedule_num_inference_steps", type=int, default=50)
    p.add_argument("--timestep_indices", type=str, default="")
    p.add_argument("--dmd_steps", type=str, default="1000,750,500,250")
    p.add_argument("--update_rule", choices=["renoise_x0", "euler_dt"], default="renoise_x0")
    p.add_argument("--full_schedule", action="store_true")
    p.add_argument("--warp_denoising_step", action="store_true", default=True)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    p.add_argument("--control_depth_only", action="store_true")

    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--causal_window_frames", type=int, default=45)
    p.add_argument("--causal_overlap_frames", type=int, default=1)
    p.add_argument("--local_attn_size", type=int, default=21)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--warp_num_keyframes", type=int, default=4)
    p.add_argument("--selection_num_target_samples", type=int, default=3)
    p.add_argument("--selection_voxel_size", type=float, default=0.1)
    p.add_argument("--context_noise", type=int, default=0)
    p.add_argument("--disable_cache_update", action="store_true")
    p.add_argument("--first_frame_timestep_zero", action="store_true")
    p.add_argument("--first_frame_condition_mode", default="hard_replace", choices=["hard_replace", "noise_init", "md_align"])
    p.add_argument("--first_frame_anchor_latent_frames", type=int, default=1)
    p.add_argument("--first_frame_blend_frames", type=int, default=0)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_frames", action="store_true")
    p.set_defaults(raw_depth_invert=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.causal_overlap_frames) < 0 or int(args.causal_overlap_frames) >= int(args.causal_window_frames):
        raise ValueError("causal_overlap_frames must be >=0 and smaller than causal_window_frames")

    base._ensure_single_process_dist_env()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise SystemExit("This script is single-process only.")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    inference_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")

    dmd_steps_list = [int(x) for x in str(args.dmd_steps).split(",") if x.strip()] or None
    timestep_indices_list = [int(x) for x in str(args.timestep_indices).split(",") if x.strip()] or None
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
    fastvideo_args.override_transformer_cls_name = "CausalWanTransformer3DModel"
    fastvideo_args.override_controlnet_cls_name = (
        "CausalWanControlnetUnion3DModel" if use_union_controlnet else "CausalWanControlnet3DModel"
    )
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    fastvideo_args.pipeline_config.warp_denoising_step = bool(args.warp_denoising_step)
    fastvideo_args.pipeline_config.dmd_denoising_steps = dmd_steps_list or []
    fastvideo_args.pipeline_config.context_noise = int(args.context_noise)
    if args.init_transformer_safetensors:
        fastvideo_args.init_weights_from_safetensors = args.init_transformer_safetensors
    if args.init_controlnet_safetensors:
        fastvideo_args.init_controlnet_weights_from_safetensors = args.init_controlnet_safetensors

    transformer = PipelineComponentLoader.load_module(
        "transformer", args.transformer_dir, "diffusers", fastvideo_args
    )
    controlnet = PipelineComponentLoader.load_module(
        "controlnet", args.controlnet_dir, "diffusers", fastvideo_args
    )
    try:
        model_param = next(p for p in transformer.parameters() if torch.is_floating_point(p))
    except StopIteration:
        model_param = next(p for p in controlnet.parameters() if torch.is_floating_point(p))
    model_dtype = model_param.dtype
    if dtype != model_dtype:
        logger.info("dtype alignment: overriding runtime dtype from %s to model dtype %s", str(dtype), str(model_dtype))
        dtype = model_dtype

    if int(args.local_attn_size) > 0:
        base._override_local_attn_size(transformer, int(args.local_attn_size))
        base._override_local_attn_size(controlnet, int(args.local_attn_size))
    if int(args.sink_size) > 0:
        base._override_sink_size(transformer, int(args.sink_size))
        base._override_sink_size(controlnet, int(args.sink_size))
    base._log_causal_attn_overrides(transformer, name="transformer")
    base._log_causal_attn_overrides(controlnet, name="controlnet")

    tokenizer = PipelineComponentLoader.load_module(
        "tokenizer", str(Path(args.base_model) / "tokenizer"), "transformers", fastvideo_args
    )
    text_encoder = PipelineComponentLoader.load_module(
        "text_encoder", str(Path(args.base_model) / "text_encoder"), "transformers", fastvideo_args
    )
    negative_prompt_embeds_global = None
    if float(args.guidance_scale) != 1.0 and str(args.negative_prompt).strip():
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
        scheduler = FlowUniPCMultistepScheduler(shift=float(args.flow_shift))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=float(args.flow_shift))

    vae = PipelineComponentLoader.load_module(
        "vae", str(Path(args.base_model) / "vae"), "diffusers", fastvideo_args
    )
    decoding = DecodingStage(vae=vae)

    sample_root = Path(str(args.raw_sample_root)).expanduser().resolve()
    sequence = _load_raw_long_sequence_nomask(
        sample_root=sample_root,
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        inference_device=inference_device,
        dtype=dtype,
    )
    logger.info(
        "sample=%s caption=%s long-firstframe-warp num_frames=%s window=%s overlap=%s",
        sequence.sample_id,
        sequence.prompt,
        int(args.num_frames),
        int(args.causal_window_frames),
        int(args.causal_overlap_frames),
    )

    decoded = _run_long_rollout_firstframe_warp(
        sequence=sequence,
        args=args,
        transformer=transformer,
        controlnet=controlnet,
        scheduler=scheduler,
        vae=vae,
        decoding=decoding,
        fastvideo_args=fastvideo_args,
        negative_prompt_embeds_global=negative_prompt_embeds_global,
        dtype=dtype,
        inference_device=inference_device,
        dmd_steps_list=dmd_steps_list,
        timestep_indices_list=timestep_indices_list,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{sequence.sample_id}.mp4")
    base._save_mp4(decoded, out_path, fps=int(sequence.fps))
    if bool(args.save_frames):
        base._save_frames_png(decoded, str(out_dir / "frames" / sequence.sample_id), prefix=sequence.sample_id)
    logger.info("saved: %s", out_path)


if __name__ == "__main__":
    main()
