#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _save_rgb_png(x_chw, path: Path) -> None:
    arr = (x_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() *
           255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_gray_png(x_chw, path: Path) -> None:
    arr = (x_chw.detach().cpu().float().clamp(0, 1).squeeze(0).numpy() *
           255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _frame_stem(i: int) -> str:
    return f"{int(i):04d}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=
        "Build one MatrixCity sample with mask/masked_rgb generated directly from raw RGB warp."
    )
    p.add_argument("--raw_root",
                   default="/vePFS-buaa/yinli/datasets/matrixcity")
    p.add_argument("--street_split", default="train_dense")
    p.add_argument("--scene_name",
                   default="small_city_road_down_dense")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--chunk_frames",
                   type=int,
                   default=81,
                   help="Chunk size in raw video frames. 81 means stride 80.")
    p.add_argument("--camera_mode", default="B_inv")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--output_root",
        default="/vePFS-buaa/linming/workspace/worldrender",
    )
    p.add_argument(
        "--output_name",
        default="Matrixcity_sample_401_rawrgbwarp",
    )
    p.add_argument("--prompt",
                   default="A continuous driving view through a city street.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from tools import preprocess_matrixcity_ti2v_controlnet_parquet as mcprep

    raw_root = Path(str(args.raw_root)).expanduser().resolve()
    output_root = Path(str(args.output_root)).expanduser().resolve()
    out_dir = output_root / str(args.output_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "depth", "normal", "mask", "masked_rgb", "camera"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    pose_index = mcprep._load_matrixcity_pose_index(
        rgb_root=raw_root,
        street_split=str(args.street_split),
        camera_mode=str(args.camera_mode),
    )
    scene_name = str(args.scene_name)
    if scene_name not in pose_index:
        raise KeyError(
            f"scene_name={scene_name} not found in pose index for split={args.street_split} camera_mode={args.camera_mode}"
        )
    scene_pose_index = pose_index[scene_name]

    scene_dir_candidates = [
        raw_root / "small_city" / "street" / str(args.street_split) / scene_name,
        raw_root / "street" / str(args.street_split) / scene_name,
    ]
    scene_dir = None
    for candidate in scene_dir_candidates:
        if candidate.is_dir():
            scene_dir = candidate
            break
    if scene_dir is None:
        raise FileNotFoundError(
            f"Could not resolve scene dir for {scene_name} under {raw_root}")

    rgb_dir = scene_dir / scene_name
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    rgb_map = mcprep._build_numeric_file_map(mcprep._sorted_pngs(rgb_dir))
    if not rgb_map:
        raise FileNotFoundError(f"No RGB files found: {rgb_dir}")

    depth_dir = (raw_root / "small_city_depth" / "street" /
                 str(args.street_split) / f"{scene_name}_depth" /
                 f"{scene_name}_depth")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth dir not found: {depth_dir}")
    depth_map = mcprep._build_numeric_file_map(
        mcprep._sorted_depth_files(depth_dir))
    if not depth_map:
        raise FileNotFoundError(f"No depth files found: {depth_dir}")

    normal_dir_candidates = [
        raw_root / "small_city_normal" / "street" / str(args.street_split) /
        f"{scene_name}_normal" / f"{scene_name}_normal",
        raw_root / "small_city_normal" / "street" / str(args.street_split) /
        scene_name / scene_name,
        raw_root / "street" / str(args.street_split) / f"{scene_name}_normal" /
        f"{scene_name}_normal",
        raw_root / "street" / str(args.street_split) / scene_name /
        scene_name,
    ]
    normal_dir = None
    for candidate in normal_dir_candidates:
        if candidate.is_dir():
            normal_dir = candidate
            break
    if normal_dir is None:
        raise FileNotFoundError(f"Normal dir not found for scene={scene_name}")
    normal_map = mcprep._build_numeric_file_map(
        mcprep._sorted_normal_files(normal_dir))
    if not normal_map:
        raise FileNotFoundError(f"No normal files found: {normal_dir}")

    start_frame = int(args.start_frame)
    num_frames = int(args.num_frames)
    frame_ids = [start_frame + i for i in range(num_frames)]
    rgb_paths = mcprep._pick_by_target_ids(rgb_map, frame_ids)
    depth_paths = mcprep._pick_by_target_ids(depth_map, frame_ids)
    normal_paths = mcprep._pick_by_target_ids(normal_map, frame_ids)

    ref_img = Image.open(next(iter(rgb_map.values()))).convert("RGB")
    src_w, src_h = ref_img.size
    crop_params = mcprep.infer_base._get_crop_params(
        src_w,
        src_h,
        int(args.width),
        int(args.height),
    )
    camera_k = mcprep._build_intrinsics_from_pose_meta(
        scene_pose_index.intrinsics_meta,
        src_w=src_w,
        src_h=src_h,
    )
    camera_k_aligned = mcprep.infer_base._adjust_intrinsics(
        camera_k,
        crop_params,
        int(args.width),
        int(args.height),
    )
    np.savetxt(out_dir / "camera" / "camera_K.txt",
               camera_k_aligned,
               fmt="%.8f")

    for local_idx, (fid, rgb_p, depth_p,
                    normal_p) in enumerate(zip(frame_ids, rgb_paths, depth_paths,
                                               normal_paths)):
        target_rgb = out_dir / "rgb" / f"rgb_{_frame_stem(local_idx)}{rgb_p.suffix.lower()}"
        target_depth = out_dir / "depth" / f"depth_{_frame_stem(local_idx)}{depth_p.suffix.lower()}"
        target_normal = out_dir / "normal" / f"normal_{_frame_stem(local_idx)}{normal_p.suffix.lower()}"
        if target_rgb.exists() or target_rgb.is_symlink():
            target_rgb.unlink()
        if target_depth.exists() or target_depth.is_symlink():
            target_depth.unlink()
        if target_normal.exists() or target_normal.is_symlink():
            target_normal.unlink()
        os.symlink(rgb_p, target_rgb)
        os.symlink(depth_p, target_depth)
        os.symlink(normal_p, target_normal)

        if int(fid) not in scene_pose_index.rt_by_frame_id:
            raise KeyError(f"Missing RT for frame_id={int(fid)}")
        np.savetxt(
            out_dir / "camera" / f"camera_RT_{_frame_stem(local_idx)}.txt",
            scene_pose_index.rt_by_frame_id[int(fid)],
            fmt="%.8f",
        )

    depth_path_by_id = {int(fid): p for fid, p in zip(frame_ids, depth_paths)}
    mask_tensors = [None for _ in range(num_frames)]
    masked_rgb_tensors = [None for _ in range(num_frames)]
    anchor_segments: list[dict[str, int]] = []

    stride = max(1, int(args.chunk_frames) - 1)
    segment_span = max(1, int(args.chunk_frames) - 1)

    for anchor_local_idx in range(0, num_frames, stride):
        anchor_global_id = int(frame_ids[anchor_local_idx])
        anchor_rgb = mcprep._load_rgb_frame(rgb_paths[anchor_local_idx],
                                            int(args.height),
                                            int(args.width))
        mask_tensors[anchor_local_idx] = anchor_rgb.new_ones(
            (1, int(args.height), int(args.width)))
        masked_rgb_tensors[anchor_local_idx] = anchor_rgb

        target_local_start = anchor_local_idx + 1
        target_local_end = min(anchor_local_idx + segment_span, num_frames - 1)
        target_local_ids = list(range(target_local_start, target_local_end + 1))
        target_global_ids = [int(frame_ids[i]) for i in target_local_ids]
        if target_global_ids:
            warped_masked_rgb_valid, warped_mask_valid = mcprep._warp_maskrgb_from_keyframes_md_aligned_memory(
                keyframe_rgbs_u8=[mcprep._chw_float_to_u8(anchor_rgb)],
                keyframe_frame_ids=[anchor_global_id],
                target_frame_ids=target_global_ids,
                depth_path_by_frame_id=depth_path_by_id,
                camera_k_aligned=camera_k_aligned,
                rt_by_frame_id=scene_pose_index.rt_by_frame_id,
                crop_params=crop_params,
                target_height=int(args.height),
                target_width=int(args.width),
            )
            for offset, local_idx in enumerate(target_local_ids):
                masked_rgb_tensors[local_idx] = warped_masked_rgb_valid[offset]
                mask_tensors[local_idx] = warped_mask_valid[offset]

        anchor_segments.append({
            "anchor_local_idx": int(anchor_local_idx),
            "anchor_global_id": int(anchor_global_id),
            "target_local_start": int(target_local_start),
            "target_local_end": int(target_local_end),
        })

    for local_idx in range(num_frames):
        if mask_tensors[local_idx] is not None and masked_rgb_tensors[local_idx] is not None:
            continue
        rgb = mcprep._load_rgb_frame(rgb_paths[local_idx], int(args.height),
                                     int(args.width))
        mask_tensors[local_idx] = rgb.new_ones(
            (1, int(args.height), int(args.width)))
        masked_rgb_tensors[local_idx] = rgb

    for local_idx in range(num_frames):
        _save_gray_png(mask_tensors[local_idx],
                       out_dir / "mask" / f"mask_{_frame_stem(local_idx)}.png")
        _save_rgb_png(
            masked_rgb_tensors[local_idx],
            out_dir / "masked_rgb" / f"masked_rgb_{_frame_stem(local_idx)}.png",
        )

    (out_dir / "text.txt").write_text(str(args.prompt).strip() + "\n",
                                       encoding="utf-8")
    meta = {
        "scene_name": scene_name,
        "street_split": str(args.street_split),
        "camera_mode": str(args.camera_mode),
        "start_frame": int(start_frame),
        "num_frames": int(num_frames),
        "chunk_frames": int(args.chunk_frames),
        "frame_ids": [int(x) for x in frame_ids],
        "rgb_source_paths": [str(p) for p in rgb_paths],
        "depth_source_paths": [str(p) for p in depth_paths],
        "normal_source_paths": [str(p) for p in normal_paths],
        "camera_k_path": str(out_dir / "camera" / "camera_K.txt"),
        "anchor_segments": anchor_segments,
        "warp_rule":
        "Each anchor uses the raw RGB of its anchor frame. mask/masked_rgb for subsequent frames are generated by single-keyframe RGB warp. Later anchor frames overwrite their own local entry with identity-visible raw RGB.",
    }
    (out_dir / "sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"output_dir: {out_dir}")
    print(f"scene={scene_name} start={start_frame} num_frames={num_frames}")
    print(f"saved meta: {out_dir / 'sample_meta.json'}")


if __name__ == "__main__":
    main()
