#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Build an OmniWorld-Game manifest for
fastvideo/pipelines/preprocess/v1_preprocess_omnigame_ti2v_controlnet.py.

Raw data layout:
  <data_root>/<scene>/
      color/%06d.png
      depth/%06d.png
      text/<window_start>_<window_end>.json

Mask layout:
  <mask_root>/<scene>/<window_start>_<window_end>/<instance_id>/%06d.png

Each mask instance directory becomes one manifest sample. The downstream
preprocess script reads RGB/depth from data_root and builds masked_rgb as
rgb * mask when launched with --no-use-warp-out.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _parse_window(name: str) -> tuple[int, int] | None:
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _has_nonzero_mask(mask_dir: Path, frame_indices: list[int]) -> bool:
    for idx in frame_indices:
        p = mask_dir / f"{idx:06d}.png"
        if not p.is_file():
            return False
        try:
            extrema = Image.open(p).convert("L").getextrema()
        except Exception:
            return False
        if extrema[1] > 0:
            return True
    return False


def _has_all_required_files(
    *,
    scene_data_dir: Path,
    mask_dir: Path,
    caption_path: Path,
    frame_indices: list[int],
    validation_mode: str,
) -> bool:
    if not caption_path.is_file():
        return False
    color_dir = scene_data_dir / "color"
    depth_dir = scene_data_dir / "depth"
    if not color_dir.is_dir() or not depth_dir.is_dir() or not mask_dir.is_dir():
        return False
    if validation_mode == "fast":
        first_idx = frame_indices[0]
        last_idx = frame_indices[-1]
        if not (color_dir / f"{first_idx:06d}.png").is_file():
            return False
        if not (color_dir / f"{last_idx:06d}.png").is_file():
            return False
        if not (depth_dir / f"{first_idx:06d}.png").is_file():
            return False
        if not (depth_dir / f"{last_idx:06d}.png").is_file():
            return False
        if not (mask_dir / f"{first_idx:06d}.png").is_file():
            return False
        if not (mask_dir / f"{last_idx:06d}.png").is_file():
            return False
        return True
    for idx in frame_indices:
        if not (color_dir / f"{idx:06d}.png").is_file():
            return False
        if not (depth_dir / f"{idx:06d}.png").is_file():
            return False
        if not (mask_dir / f"{idx:06d}.png").is_file():
            return False
    return True


def _has_required_window_files_fast(
    *,
    scene_data_dir: Path,
    caption_path: Path,
    frame_indices: list[int],
) -> bool:
    if not caption_path.is_file():
        return False
    color_dir = scene_data_dir / "color"
    depth_dir = scene_data_dir / "depth"
    if not color_dir.is_dir() or not depth_dir.is_dir():
        return False
    first_idx = frame_indices[0]
    last_idx = frame_indices[-1]
    return (
        (color_dir / f"{first_idx:06d}.png").is_file()
        and (color_dir / f"{last_idx:06d}.png").is_file()
        and (depth_dir / f"{first_idx:06d}.png").is_file()
        and (depth_dir / f"{last_idx:06d}.png").is_file()
    )


def _has_required_mask_files_fast(mask_dir: Path, frame_indices: list[int]) -> bool:
    if not mask_dir.is_dir():
        return False
    first_idx = frame_indices[0]
    last_idx = frame_indices[-1]
    return (
        (mask_dir / f"{first_idx:06d}.png").is_file()
        and (mask_dir / f"{last_idx:06d}.png").is_file()
    )


def _iter_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_root = Path(args.data_root).expanduser()
    mask_root = Path(args.mask_root).expanduser()

    scene_filter = {x.strip() for x in str(args.scene).split(",") if x.strip()}
    manifest: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "clip_length": int(args.clip_length),
        "skip_empty_masks": bool(args.skip_empty_masks),
        "validation_mode": str(args.validation_mode),
        "total_mask_instance_dirs_seen": 0,
        "kept": 0,
        "skipped": {
            "scene_filtered": 0,
            "bad_window_name": 0,
            "bad_window_length": 0,
            "missing_required_files": 0,
            "empty_mask": 0,
        },
    }

    if not mask_root.is_dir():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    for scene_mask_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        scene = scene_mask_dir.name
        if scene_filter and scene not in scene_filter:
            stats["skipped"]["scene_filtered"] += 1
            continue
        scene_data_dir = data_root / scene
        for window_dir in sorted(p for p in scene_mask_dir.iterdir() if p.is_dir()):
            parsed = _parse_window(window_dir.name)
            if parsed is None:
                stats["skipped"]["bad_window_name"] += 1
                continue
            start, end = parsed
            frame_indices = list(range(start, end + 1))
            if int(args.clip_length) > 0:
                if len(frame_indices) < int(args.clip_length):
                    stats["skipped"]["bad_window_length"] += 1
                    continue
                frame_indices = frame_indices[:int(args.clip_length)]

            caption_path = scene_data_dir / "text" / f"{window_dir.name}.json"
            inst_dirs = sorted(p for p in window_dir.iterdir() if p.is_dir())
            if str(args.validation_mode) == "fast" and not _has_required_window_files_fast(
                scene_data_dir=scene_data_dir,
                caption_path=caption_path,
                frame_indices=frame_indices,
            ):
                stats["total_mask_instance_dirs_seen"] += len(inst_dirs)
                stats["skipped"]["missing_required_files"] += len(inst_dirs)
                continue

            for inst_dir in inst_dirs:
                stats["total_mask_instance_dirs_seen"] += 1
                if str(args.validation_mode) == "fast":
                    if not _has_required_mask_files_fast(inst_dir, frame_indices):
                        stats["skipped"]["missing_required_files"] += 1
                        continue
                else:
                    if not _has_all_required_files(
                        scene_data_dir=scene_data_dir,
                        mask_dir=inst_dir,
                        caption_path=caption_path,
                        frame_indices=frame_indices,
                        validation_mode=str(args.validation_mode),
                    ):
                        stats["skipped"]["missing_required_files"] += 1
                        continue
                if bool(args.skip_empty_masks) and not _has_nonzero_mask(inst_dir, frame_indices):
                    stats["skipped"]["empty_mask"] += 1
                    continue

                sample_name = f"{scene}_{window_dir.name}_inst{inst_dir.name}"
                manifest.append(
                    {
                        "scene_name": sample_name,
                        "frame_indices": frame_indices,
                        "caption_path": str(caption_path),
                        "video_path": str(scene_data_dir / "color"),
                        "control_path": str(scene_data_dir / "depth"),
                        "mask_path": str(inst_dir),
                    }
                )
                if int(args.max_samples) > 0 and len(manifest) >= int(args.max_samples):
                    stats["kept"] = len(manifest)
                    return manifest, stats

    stats["kept"] = len(manifest)
    return manifest, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build OmniWorld-Game raw-mask pickle manifest")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--mask_root", type=str, required=True)
    p.add_argument("--out_pickle", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--clip_length", type=int, default=81)
    p.add_argument(
        "--scene",
        type=str,
        default="",
        help="Optional comma-separated scene id filter.",
    )
    p.add_argument(
        "--skip_empty_masks",
        action="store_true",
        help="Skip mask instance dirs whose selected frames are all zero.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Stop after this many kept samples. 0 means no limit.",
    )
    p.add_argument(
        "--validation_mode",
        type=str,
        default="full",
        choices=["full", "fast"],
        help=(
            "'full' checks every RGB/depth/mask frame before writing the manifest. "
            "'fast' checks dirs, caption, and first/last RGB/depth/mask frames."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest, stats = _iter_samples(args)

    out_pickle = Path(args.out_pickle).expanduser()
    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    with out_pickle.open("wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.out_summary:
        out_summary = Path(args.out_summary).expanduser()
    else:
        out_summary = out_pickle.with_suffix(out_pickle.suffix + ".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote_pickle={out_pickle}")
    print(f"wrote_summary={out_summary}")


if __name__ == "__main__":
    main()
