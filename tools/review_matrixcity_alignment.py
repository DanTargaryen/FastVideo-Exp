#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Review MatrixCity preprocessing alignment and write a text report.

This script is designed to mirror the frame-selection logic used in:
  tools/preprocess_matrixcity_ti2v_controlnet_parquet.py

It does NOT encode VAE latents. It only checks frame-path alignment behavior.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.preprocess_matrixcity_ti2v_controlnet_parquet import (
    _frame_index_from_stem,
    _resolve_clip_start_offset,
    _slice_by_id_or_index,
    _sorted_depth_files,
    _sorted_normal_files,
    _sorted_pngs,
)


def _pick_indexed_files_debug(
    files: dict[int, Path],
    length: int,
    *,
    offset_candidates: list[int] | None = None,
) -> tuple[list[Path], int, int]:
    """
    Same behavior as preprocess _pick_indexed_files, plus:
    - chosen offset
    - direct hit count
    """
    if not files:
        raise ValueError("Empty files map")
    if int(length) <= 0:
        raise ValueError("length must be > 0")

    available = sorted(files.keys())
    cand = list(dict.fromkeys(offset_candidates or [0]))
    if not cand:
        cand = [0]

    best_off = cand[0]
    best_hits = -1
    for off in cand:
        hits = 0
        base = int(off)
        for i in range(int(length)):
            if (base + i) in files:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_off = base

    if best_hits <= 0:
        seq = [files[k] for k in available]
        if len(seq) >= int(length):
            return seq[:int(length)], int(best_off), int(best_hits)
        return seq + [seq[-1]] * (int(length) - len(seq)), int(best_off), int(best_hits)

    out: list[Path] = []
    for i in range(int(length)):
        target = int(best_off) + i
        if target in files:
            out.append(files[target])
            continue
        j = None
        for a in reversed(available):
            if a <= target:
                j = a
                break
        if j is None:
            j = available[0]
        out.append(files[j])
    return out, int(best_off), int(best_hits)


def _ids(paths: list[Path]) -> list[int | None]:
    return [_frame_index_from_stem(p.stem) for p in paths]


def _pair_mismatch(a: list[int | None], b: list[int | None]) -> int:
    n = min(len(a), len(b))
    bad = 0
    for i in range(n):
        if a[i] is None or b[i] is None:
            continue
        if int(a[i]) != int(b[i]):
            bad += 1
    return bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("review_matrixcity_alignment")
    p.add_argument("--mask_root", type=str, required=True)
    p.add_argument("--rgb_root", type=str, required=True)
    p.add_argument("--depth_root", type=str, default="")
    p.add_argument("--normal_root", type=str, default="")
    p.add_argument("--street_split", type=str, default="train_dense_half")
    p.add_argument("--street_dir", type=str, default="")
    p.add_argument("--clip_length", type=int, default=81)
    p.add_argument("--window_len", type=int, default=243)
    p.add_argument("--max_clips", type=int, default=200, help="0 means all clips")
    p.add_argument("--output_txt", type=str, default="data_review.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mask_root = Path(os.path.expanduser(os.path.expandvars(args.mask_root)))
    rgb_root = Path(os.path.expanduser(os.path.expandvars(args.rgb_root)))
    depth_root = Path(os.path.expanduser(os.path.expandvars(args.depth_root))) if str(
        args.depth_root).strip() else rgb_root
    normal_root = Path(os.path.expanduser(os.path.expandvars(args.normal_root))) if str(
        args.normal_root).strip() else None
    out_txt = Path(os.path.expanduser(os.path.expandvars(args.output_txt)))
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    street_dirs = [mask_root / args.street_dir] if args.street_dir else [
        p for p in sorted(mask_root.iterdir()) if p.is_dir()
    ]
    all_clips: list[Path] = []
    for street_dir in street_dirs:
        for window_dir in sorted(street_dir.iterdir()):
            if not window_dir.is_dir() or "_" not in window_dir.name:
                continue
            for clip_dir in sorted(window_dir.iterdir()):
                if clip_dir.is_dir() and clip_dir.name.startswith("clip_start_"):
                    all_clips.append(clip_dir)

    if int(args.max_clips) > 0:
        all_clips = all_clips[:int(args.max_clips)]

    rgb_cache: dict[tuple[str, str], list[Path]] = {}
    depth_cache: dict[tuple[str, str], list[Path] | None] = {}
    normal_seq_cache: dict[str, list[Path]] = {}

    lines: list[str] = []
    lines.append("=== MatrixCity Data Review ===")
    lines.append(f"mask_root={mask_root}")
    lines.append(f"rgb_root={rgb_root}")
    lines.append(f"depth_root={depth_root}")
    lines.append(f"normal_root={normal_root}")
    lines.append(f"street_split={args.street_split}")
    lines.append(f"clip_length={int(args.clip_length)} window_len={int(args.window_len)}")
    lines.append(f"total_scanned_clips={len(all_clips)}")
    lines.append("")

    bad_total = 0
    checked = 0
    suspicious: list[str] = []

    for clip_dir in all_clips:
        window_dir = clip_dir.parent
        street_dir = window_dir.parent
        street_name = street_dir.name

        try:
            w_start_str, w_end_str = window_dir.name.split("_", 1)
            window_start = int(w_start_str)
            window_end = int(w_end_str)
            clip_start = int(clip_dir.name.split("_")[-1])
        except Exception:
            continue

        rgb_key = (str(rgb_root), street_name)
        if rgb_key not in rgb_cache:
            rgb_dir = rgb_root / "small_city" / "street" / str(args.street_split) / street_name / street_name
            rgb_cache[rgb_key] = _sorted_pngs(rgb_dir) if rgb_dir.is_dir() else []
        rgb_files = rgb_cache[rgb_key]
        if not rgb_files:
            continue

        depth_key = (str(depth_root), street_name)
        if depth_key not in depth_cache:
            ddir = depth_root / "small_city_depth" / "street" / str(args.street_split) / f"{street_name}_depth" / f"{street_name}_depth"
            depth_cache[depth_key] = _sorted_depth_files(ddir) if ddir.is_dir() else None
        depth_files = depth_cache[depth_key]
        if depth_files is None or len(depth_files) == 0:
            continue

        window_rgb = _slice_by_id_or_index(rgb_files, window_start, window_end)
        if not window_rgb:
            continue
        if len(window_rgb) < int(args.window_len):
            window_rgb = window_rgb + [window_rgb[-1]] * (int(args.window_len) - len(window_rgb))

        window_depth = _slice_by_id_or_index(depth_files, window_start, window_end)
        if not window_depth:
            window_depth = [depth_files[min(max(window_start, 0), len(depth_files) - 1)]]
        if len(window_depth) < len(window_rgb):
            window_depth = window_depth + [window_depth[-1]] * (len(window_rgb) - len(window_depth))

        start_off = _resolve_clip_start_offset(
            window_files=window_rgb,
            clip_start=clip_start,
            window_start=window_start,
        )
        need = int(start_off) + int(args.clip_length)
        if need > len(window_rgb):
            window_rgb = window_rgb + [window_rgb[-1]] * (need - len(window_rgb))
        if need > len(window_depth):
            window_depth = window_depth + [window_depth[-1]] * (need - len(window_depth))

        rgb_paths = window_rgb[start_off:start_off + int(args.clip_length)]
        depth_paths = window_depth[start_off:start_off + int(args.clip_length)]

        mask_dir = clip_dir / "mask"
        mr_dir = clip_dir / "masked_rgb"
        local_n_dir = clip_dir / "normal"

        mask_map: dict[int, Path] = {}
        if mask_dir.is_dir():
            for p in mask_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    idx = _frame_index_from_stem(p.stem)
                    if idx is not None:
                        mask_map[idx] = p

        mr_map: dict[int, Path] = {}
        if mr_dir.is_dir():
            for p in mr_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    idx = _frame_index_from_stem(p.stem)
                    if idx is not None:
                        mr_map[idx] = p

        local_n_map: dict[int, Path] = {}
        if local_n_dir.is_dir():
            for p in local_n_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".exr"):
                    idx = _frame_index_from_stem(p.stem)
                    if idx is not None:
                        local_n_map[idx] = p

        idx_cands = [0, int(clip_start), int(window_start), int(start_off)]

        mask_paths, mask_off, mask_hits = ([], 0, -1)
        if mask_map:
            mask_paths, mask_off, mask_hits = _pick_indexed_files_debug(
                mask_map, int(args.clip_length), offset_candidates=idx_cands)

        mr_paths, mr_off, mr_hits = ([], 0, -1)
        if mr_map:
            mr_paths, mr_off, mr_hits = _pick_indexed_files_debug(
                mr_map, int(args.clip_length), offset_candidates=idx_cands)

        normal_paths: list[Path] = []
        normal_src = "none"
        normal_off = 0
        normal_hits = -1
        if local_n_map:
            normal_paths, normal_off, normal_hits = _pick_indexed_files_debug(
                local_n_map, int(args.clip_length), offset_candidates=idx_cands)
            normal_src = "clip_local"
        elif normal_root is not None:
            ext1 = normal_root / street_name / window_dir.name / clip_dir.name / "normal"
            ext2 = normal_root / street_name / window_dir.name / clip_dir.name
            found = False
            for ext_dir in (ext1, ext2):
                if not ext_dir.is_dir():
                    continue
                n_map: dict[int, Path] = {}
                for p in ext_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".exr"):
                        idx = _frame_index_from_stem(p.stem)
                        if idx is not None:
                            n_map[idx] = p
                if n_map:
                    normal_paths, normal_off, normal_hits = _pick_indexed_files_debug(
                        n_map, int(args.clip_length), offset_candidates=idx_cands)
                    normal_src = f"ext_clip:{ext_dir}"
                    found = True
                    break
            if not found:
                normal_candidates = [
                    normal_root / "small_city_normal" / "street" / str(args.street_split) / f"{street_name}_normal" / f"{street_name}_normal",
                    normal_root / "small_city_normal" / "street" / str(args.street_split) / street_name / street_name,
                    normal_root / "street" / str(args.street_split) / f"{street_name}_normal" / f"{street_name}_normal",
                    normal_root / "street" / str(args.street_split) / street_name / street_name,
                    normal_root / f"{street_name}_normal" / f"{street_name}_normal",
                    normal_root / street_name / street_name,
                ]
                for nd in normal_candidates:
                    if not nd.is_dir():
                        continue
                    nd_key = str(nd)
                    if nd_key not in normal_seq_cache:
                        normal_seq_cache[nd_key] = _sorted_normal_files(nd)
                    nfiles = normal_seq_cache[nd_key]
                    if not nfiles:
                        continue
                    w_n = _slice_by_id_or_index(nfiles, window_start, window_end)
                    if not w_n:
                        continue
                    if len(w_n) < len(window_rgb):
                        w_n = w_n + [w_n[-1]] * (len(window_rgb) - len(w_n))
                    if need > len(w_n):
                        w_n = w_n + [w_n[-1]] * (need - len(w_n))
                    normal_paths = w_n[start_off:start_off + int(args.clip_length)]
                    normal_src = f"seq:{nd}"
                    break

        rgb_ids = _ids(rgb_paths)
        depth_ids = _ids(depth_paths)
        mask_ids = _ids(mask_paths) if mask_paths else []
        mr_ids = _ids(mr_paths) if mr_paths else []
        normal_ids = _ids(normal_paths) if normal_paths else []

        mm_depth = _pair_mismatch(rgb_ids, depth_ids)
        mm_mask = _pair_mismatch(rgb_ids, mask_ids) if mask_ids else -1
        mm_mr = _pair_mismatch(rgb_ids, mr_ids) if mr_ids else -1
        mm_n = _pair_mismatch(rgb_ids, normal_ids) if normal_ids else -1

        clip_tag = f"{street_name}/{window_dir.name}/{clip_dir.name}"
        checked += 1

        suspicious_reasons: list[str] = []
        if mm_depth > 0:
            suspicious_reasons.append(f"rgb-depth mismatch={mm_depth}")
        if mm_mask > 0:
            suspicious_reasons.append(f"rgb-mask mismatch={mm_mask}")
        if mm_mr > 0:
            suspicious_reasons.append(f"rgb-masked mismatch={mm_mr}")
        if mm_n > 0:
            suspicious_reasons.append(f"rgb-normal mismatch={mm_n}")
        if mask_hits >= 0 and mask_hits < int(args.clip_length) // 2:
            suspicious_reasons.append(f"mask_hits_low={mask_hits}")
        if mr_hits >= 0 and mr_hits < int(args.clip_length) // 2:
            suspicious_reasons.append(f"masked_hits_low={mr_hits}")

        if suspicious_reasons:
            bad_total += 1
            suspicious.append(
                f"{clip_tag} | " + "; ".join(suspicious_reasons) +
                f" | offs(start={start_off},mask={mask_off},mr={mr_off},n={normal_off}) "
                f"| hits(mask={mask_hits},mr={mr_hits},n={normal_hits}) "
                f"| normal_src={normal_src}"
            )

        lines.append(f"[{checked:05d}] {clip_tag}")
        lines.append(
            f"  offs: start={start_off} mask={mask_off} mr={mr_off} n={normal_off} | "
            f"hits: mask={mask_hits} mr={mr_hits} n={normal_hits} | normal_src={normal_src}")
        lines.append(
            f"  mismatch: rgb-depth={mm_depth} rgb-mask={mm_mask} rgb-masked={mm_mr} rgb-normal={mm_n}")
        lines.append(
            f"  rgb[0,last]={rgb_paths[0].name if rgb_paths else 'NA'}, {rgb_paths[-1].name if rgb_paths else 'NA'}")
        lines.append(
            f"  depth[0,last]={depth_paths[0].name if depth_paths else 'NA'}, {depth_paths[-1].name if depth_paths else 'NA'}")
        lines.append(
            f"  mask[0,last]={mask_paths[0].name if mask_paths else 'NA'}, {mask_paths[-1].name if mask_paths else 'NA'}")
        lines.append(
            f"  m_rgb[0,last]={mr_paths[0].name if mr_paths else 'NA'}, {mr_paths[-1].name if mr_paths else 'NA'}")
        lines.append(
            f"  normal[0,last]={normal_paths[0].name if normal_paths else 'NA'}, {normal_paths[-1].name if normal_paths else 'NA'}")
        lines.append("")

    lines.append("=== Summary ===")
    lines.append(f"checked_clips={checked}")
    lines.append(f"suspicious_clips={bad_total}")
    lines.append("")
    lines.append("=== Suspicious Top 200 ===")
    for s in suspicious[:200]:
        lines.append(s)

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote review: {out_txt}")
    print(f"checked={checked} suspicious={bad_total}")


if __name__ == "__main__":
    main()

