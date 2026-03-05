#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Extract one MatrixCity scene sample with aligned RGB/Depth/Normal frames.

This script follows the same frame sorting/slicing semantics as
tools/preprocess_matrixcity_ti2v_controlnet_parquet.py:
- numeric suffix parsing (e.g. 0001, rgb_0001, depth_0001)
- index-range if valid, otherwise id-range
- per-modality repeat-last padding when insufficient

It writes one sample directory containing:
- rgb/*.png (or source suffix)
- depth/*.png|*.exr|...
- normal/*.png|*.exr|...
- camera/ (if found)
- video/  (if found) or video file
- text.txt (if found)
- sample_meta.json (source mapping + chosen frame ids)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path


def _frame_index_from_stem(stem: str) -> int | None:
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _sorted_files(dir_path: Path, exts: tuple[str, ...]) -> list[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if files:
        parsed = [(_frame_index_from_stem(p.stem), p) for p in files]
        if all(idx is not None for idx, _ in parsed):
            return [p for _, p in sorted(parsed, key=lambda x: (int(x[0]), x[1].name))]
    return sorted(files, key=lambda p: p.name)


def _slice_by_id_or_index(files: list[Path], start: int, end: int) -> list[Path]:
    if not files:
        return []
    n = len(files)
    if 0 <= start <= end < n:
        return files[start:end + 1]

    parsed_ids = [_frame_index_from_stem(p.stem) for p in files]
    if all(fid is not None for fid in parsed_ids):
        ids = [int(fid) for fid in parsed_ids]
        lo = None
        hi = None
        for i, fid in enumerate(ids):
            if lo is None and fid >= start:
                lo = i
            if fid <= end:
                hi = i
        if lo is not None and hi is not None and lo <= hi:
            return files[lo:hi + 1]
    return []


def _pad_or_trim(seq: list[Path], target_len: int) -> list[Path]:
    if not seq:
        return []
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [seq[-1]] * (target_len - len(seq))


def _link_or_copy(
    src: Path,
    dst: Path,
    copy_mode: bool,
    *,
    allow_copy_fallback: bool,
    stats: dict[str, int],
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy_mode:
        shutil.copy2(src, dst)
        stats["file_copy"] += 1
        return
    try:
        os.symlink(src, dst)
        stats["file_link"] += 1
    except OSError:
        if not allow_copy_fallback:
            raise
        shutil.copy2(src, dst)
        stats["file_copy"] += 1


def _copy_tree_or_link(
    src_dir: Path,
    dst_dir: Path,
    copy_mode: bool,
    *,
    allow_copy_fallback: bool,
    stats: dict[str, int],
) -> None:
    if not src_dir.exists():
        return
    if dst_dir.exists() or dst_dir.is_symlink():
        if dst_dir.is_dir() and not dst_dir.is_symlink():
            shutil.rmtree(dst_dir)
        else:
            dst_dir.unlink()
    if copy_mode:
        shutil.copytree(src_dir, dst_dir)
        stats["dir_copy"] += 1
        return
    try:
        os.symlink(src_dir, dst_dir, target_is_directory=True)
        stats["dir_link"] += 1
    except OSError:
        if not allow_copy_fallback:
            raise
        shutil.copytree(src_dir, dst_dir)
        stats["dir_copy"] += 1


def _resolve_rgb_dir(rgb_root: Path, street_split: str, street_name: str, rgb_dir: str) -> Path:
    if rgb_dir.strip():
        p = Path(os.path.expanduser(os.path.expandvars(rgb_dir.strip())))
        return p
    return rgb_root / "small_city" / "street" / street_split / street_name / street_name


def _resolve_depth_dir(depth_root: Path, street_split: str, street_name: str, depth_dir: str) -> Path | None:
    if depth_dir.strip():
        p = Path(os.path.expanduser(os.path.expandvars(depth_dir.strip())))
        return p if p.is_dir() else None
    candidates = [
        depth_root / "small_city_depth" / "street" / street_split / f"{street_name}_depth" / f"{street_name}_depth",
        depth_root / "street" / street_split / f"{street_name}_depth" / f"{street_name}_depth",
        depth_root / f"{street_name}_depth" / f"{street_name}_depth",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _resolve_normal_dir(normal_root: Path, street_split: str, street_name: str, normal_dir: str) -> Path | None:
    if normal_dir.strip():
        p = Path(os.path.expanduser(os.path.expandvars(normal_dir.strip())))
        return p if p.is_dir() else None
    candidates = [
        normal_root / "small_city_normal" / "street" / street_split / f"{street_name}_normal" / f"{street_name}_normal",
        normal_root / "small_city_normal" / "street" / street_split / street_name / street_name,
        normal_root / "street" / street_split / f"{street_name}_normal" / f"{street_name}_normal",
        normal_root / "street" / street_split / street_name / street_name,
        normal_root / f"{street_name}_normal" / f"{street_name}_normal",
        normal_root / street_name / street_name,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _first_existing_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("extract_matrixcity_scene_sample")
    p.add_argument("--rgb_root", type=str, required=True)
    p.add_argument("--depth_root", type=str, required=True)
    p.add_argument("--normal_root", type=str, required=True)
    p.add_argument("--street_split", type=str, default="train_dense_half")
    p.add_argument("--street_name", type=str, required=True, help="e.g. small_city_road_down_dense")
    p.add_argument("--start", type=int, default=0, help="range start (index or frame-id)")
    p.add_argument("--num_frames", type=int, default=401)
    p.add_argument("--rgb_dir", type=str, default="")
    p.add_argument("--depth_dir", type=str, default="")
    p.add_argument("--normal_dir", type=str, default="")
    p.add_argument(
        "--meta_scene_dir",
        type=str,
        default="",
        help="Optional directory containing camera/, video (dir or file), text.txt",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/vePFS-buaa/linming/workspace/worldrender",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Default: <street_name>_f<num_frames>_start<start>",
    )
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink.")
    p.add_argument(
        "--no_copy_fallback",
        action="store_true",
        help="When symlink fails, raise error instead of silently copying.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    num_frames = int(args.num_frames)
    if num_frames <= 0:
        raise ValueError("--num_frames must be > 0")

    rgb_root = Path(os.path.expanduser(os.path.expandvars(args.rgb_root)))
    depth_root = Path(os.path.expanduser(os.path.expandvars(args.depth_root)))
    normal_root = Path(os.path.expanduser(os.path.expandvars(args.normal_root)))
    street_split = str(args.street_split)
    street_name = str(args.street_name)
    start = int(args.start)
    end = start + num_frames - 1
    allow_copy_fallback = not bool(args.no_copy_fallback)
    stats = {"file_link": 0, "file_copy": 0, "dir_link": 0, "dir_copy": 0}
    t0 = time.perf_counter()

    rgb_dir = _resolve_rgb_dir(rgb_root, street_split, street_name, str(args.rgb_dir))
    depth_dir = _resolve_depth_dir(depth_root, street_split, street_name, str(args.depth_dir))
    normal_dir = _resolve_normal_dir(normal_root, street_split, street_name, str(args.normal_dir))

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if depth_dir is None or not depth_dir.is_dir():
        raise FileNotFoundError("Depth dir not found. Pass --depth_dir explicitly.")
    if normal_dir is None or not normal_dir.is_dir():
        raise FileNotFoundError("Normal dir not found. Pass --normal_dir explicitly.")

    rgb_files = _sorted_files(rgb_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    depth_files = _sorted_files(depth_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr"))
    normal_files = _sorted_files(normal_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr"))
    print(
        f"[scan] rgb={len(rgb_files)} depth={len(depth_files)} normal={len(normal_files)} "
        f"(elapsed={time.perf_counter() - t0:.2f}s)"
    )
    if not rgb_files:
        raise RuntimeError(f"No RGB frames under: {rgb_dir}")
    if not depth_files:
        raise RuntimeError(f"No depth frames under: {depth_dir}")
    if not normal_files:
        raise RuntimeError(f"No normal frames under: {normal_dir}")

    rgb_sel = _pad_or_trim(_slice_by_id_or_index(rgb_files, start, end), num_frames)
    depth_sel = _pad_or_trim(_slice_by_id_or_index(depth_files, start, end), num_frames)
    normal_sel = _pad_or_trim(_slice_by_id_or_index(normal_files, start, end), num_frames)
    if not rgb_sel:
        raise RuntimeError(
            f"Failed to slice rgb frames with start={start}, end={end}. "
            "Range not valid as index-range or id-range."
        )
    if not depth_sel:
        raise RuntimeError(
            f"Failed to slice depth frames with start={start}, end={end}. "
            "Try --depth_dir explicit path."
        )
    if not normal_sel:
        raise RuntimeError(
            f"Failed to slice normal frames with start={start}, end={end}. "
            "Try --normal_dir explicit path."
        )

    out_root = Path(os.path.expanduser(os.path.expandvars(args.output_root)))
    out_name = str(args.output_name).strip() or f"{street_name}_f{num_frames}_start{start}"
    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_write = time.perf_counter()
    for i, src in enumerate(rgb_sel):
        _link_or_copy(
            src,
            out_dir / "rgb" / f"{i:06d}{src.suffix.lower()}",
            bool(args.copy),
            allow_copy_fallback=allow_copy_fallback,
            stats=stats,
        )
    for i, src in enumerate(depth_sel):
        _link_or_copy(
            src,
            out_dir / "depth" / f"{i:06d}{src.suffix.lower()}",
            bool(args.copy),
            allow_copy_fallback=allow_copy_fallback,
            stats=stats,
        )
    for i, src in enumerate(normal_sel):
        _link_or_copy(
            src,
            out_dir / "normal" / f"{i:06d}{src.suffix.lower()}",
            bool(args.copy),
            allow_copy_fallback=allow_copy_fallback,
            stats=stats,
        )
    print(f"[write-frames] elapsed={time.perf_counter() - t_write:.2f}s")

    meta_scene_dir = Path(os.path.expanduser(os.path.expandvars(args.meta_scene_dir))) if str(args.meta_scene_dir).strip() else None
    if meta_scene_dir is not None and meta_scene_dir.exists():
        camera_dir = meta_scene_dir / "camera"
        if camera_dir.is_dir():
            _copy_tree_or_link(
                camera_dir,
                out_dir / "camera",
                bool(args.copy),
                allow_copy_fallback=allow_copy_fallback,
                stats=stats,
            )

        video_dir = meta_scene_dir / "video"
        if video_dir.is_dir():
            _copy_tree_or_link(
                video_dir,
                out_dir / "video",
                bool(args.copy),
                allow_copy_fallback=allow_copy_fallback,
                stats=stats,
            )
        else:
            video_file = _first_existing_file(
                [
                    meta_scene_dir / "video.mp4",
                    meta_scene_dir / "video.mov",
                    meta_scene_dir / "video.mkv",
                    meta_scene_dir / "video.avi",
                ]
            )
            if video_file is not None:
                _link_or_copy(
                    video_file,
                    out_dir / video_file.name,
                    bool(args.copy),
                    allow_copy_fallback=allow_copy_fallback,
                    stats=stats,
                )

        text_file = _first_existing_file([meta_scene_dir / "text.txt", meta_scene_dir / "caption.txt"])
        if text_file is not None:
            _link_or_copy(
                text_file,
                out_dir / "text.txt",
                bool(args.copy),
                allow_copy_fallback=allow_copy_fallback,
                stats=stats,
            )

    def _src_ids(paths: list[Path]) -> list[int | None]:
        return [_frame_index_from_stem(p.stem) for p in paths]

    meta = {
        "street_name": street_name,
        "street_split": street_split,
        "start": start,
        "end": end,
        "num_frames": num_frames,
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "normal_dir": str(normal_dir),
        "meta_scene_dir": str(meta_scene_dir) if meta_scene_dir is not None else "",
        "output_dir": str(out_dir),
        "rgb_source_paths": [str(p) for p in rgb_sel],
        "depth_source_paths": [str(p) for p in depth_sel],
        "normal_source_paths": [str(p) for p in normal_sel],
        "rgb_source_frame_ids": _src_ids(rgb_sel),
        "depth_source_frame_ids": _src_ids(depth_sel),
        "normal_source_frame_ids": _src_ids(normal_sel),
    }
    (out_dir / "sample_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"output_dir: {out_dir}")
    print(f"rgb_count: {len(rgb_sel)} depth_count: {len(depth_sel)} normal_count: {len(normal_sel)}")
    print(
        "link/copy stats:",
        f"file_link={stats['file_link']}, file_copy={stats['file_copy']}, "
        f"dir_link={stats['dir_link']}, dir_copy={stats['dir_copy']}",
    )
    print(f"total_elapsed={time.perf_counter() - t0:.2f}s")
    print("saved meta:", out_dir / "sample_meta.json")


if __name__ == "__main__":
    main()
