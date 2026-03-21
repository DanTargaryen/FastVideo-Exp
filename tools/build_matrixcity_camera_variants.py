#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Build multiple camera variants for a copied MatrixCity sample from transforms.json.

This is a debug utility for long-warp inference. It reads:
- sample_root/rgb, sample_meta.json
- original transforms.json

And writes 4 camera variants under:
- sample_root/camera_A_raw
- sample_root/camera_B_inv
- sample_root/camera_C_scale_inv
- sample_root/camera_D_scale_raw

Variant definitions:
- A_raw:        Rt = rot_mat
- B_inv:        Rt = inv(rot_mat)
- C_scale_inv:  c2w = rot_mat; c2w[:3,:3] *= 100; c2w[:3,3] /= 100; Rt = inv(c2w)
- D_scale_raw:  Rt = rot_mat; Rt[:3,:3] *= 100; Rt[:3,3] /= 100
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _frame_index_from_stem(stem: str) -> int | None:
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return None
    return int(m.group(1))


def _sorted_files(dir_path: Path, exts: tuple[str, ...]) -> list[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    parsed = [(_frame_index_from_stem(p.stem), p) for p in files]
    if files and all(idx is not None for idx, _ in parsed):
        return [p for _, p in sorted(parsed, key=lambda x: (int(x[0]), x[1].name))]
    return sorted(files, key=lambda p: p.name)


def _build_intrinsics(transforms: dict, sample_root: Path) -> np.ndarray:
    first_rgb = _sorted_files(sample_root / "rgb", (".png", ".jpg", ".jpeg", ".webp", ".bmp"))[0]
    w, h = Image.open(first_rgb).size

    fl_x = transforms.get("fl_x")
    fl_y = transforms.get("fl_y")
    cx = transforms.get("cx")
    cy = transforms.get("cy")

    if fl_x is None:
        camera_angle_x = transforms.get("camera_angle_x")
        if camera_angle_x is None:
            raise ValueError("transforms.json must contain fl_x or camera_angle_x")
        fl_x = 0.5 * float(w) / math.tan(0.5 * float(camera_angle_x))

    if fl_y is None:
        camera_angle_y = transforms.get("camera_angle_y")
        if camera_angle_y is not None:
            fl_y = 0.5 * float(h) / math.tan(0.5 * float(camera_angle_y))
        else:
            fl_y = float(fl_x)

    if cx is None:
        cx = float(w) / 2.0
    if cy is None:
        cy = float(h) / 2.0

    return np.array(
        [
            [float(fl_x), 0.0, float(cx)],
            [0.0, float(fl_y), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _variant_rt(rot_mat: np.ndarray, mode: str) -> np.ndarray:
    x = np.array(rot_mat, dtype=np.float32).copy()
    if x.shape != (4, 4):
        raise ValueError(f"Expected 4x4 rot_mat, got {x.shape}")

    if mode == "A_raw":
        return x
    if mode == "B_inv":
        return np.linalg.inv(x)
    if mode == "C_scale_inv":
        x[:3, :3] *= 100.0
        x[:3, 3] /= 100.0
        return np.linalg.inv(x)
    if mode == "D_scale_raw":
        x[:3, :3] *= 100.0
        x[:3, 3] /= 100.0
        return x
    raise KeyError(f"Unknown mode: {mode}")


def main() -> None:
    p = argparse.ArgumentParser("build_matrixcity_camera_variants")
    p.add_argument("--sample_root", required=True)
    p.add_argument("--transforms_json", required=True)
    p.add_argument("--num_frames", type=int, default=401)
    args = p.parse_args()

    sample_root = Path(args.sample_root)
    transforms_json = Path(args.transforms_json)
    meta_path = sample_root / "sample_meta.json"

    if not sample_root.is_dir():
        raise FileNotFoundError(f"sample_root not found: {sample_root}")
    if not transforms_json.is_file():
        raise FileNotFoundError(f"transforms_json not found: {transforms_json}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"sample_meta.json not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(transforms_json, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    src_ids = meta.get("rgb_source_frame_ids", [])
    if not src_ids:
        raise RuntimeError("rgb_source_frame_ids missing in sample_meta.json")
    if len(src_ids) < int(args.num_frames):
        raise RuntimeError(
            f"rgb_source_frame_ids has {len(src_ids)} entries, need {int(args.num_frames)}"
        )
    src_ids = src_ids[: int(args.num_frames)]

    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError("No frames found in transforms.json")

    frame_map: dict[int, np.ndarray] = {}
    for fr in frames:
        frame_index = fr.get("frame_index")
        rot_mat = fr.get("rot_mat") or fr.get("transform_matrix")
        if frame_index is None or rot_mat is None:
            continue
        frame_map[int(frame_index)] = np.array(rot_mat, dtype=np.float32)

    K = _build_intrinsics(transforms, sample_root)

    modes = ["A_raw", "B_inv", "C_scale_inv", "D_scale_raw"]
    for mode in modes:
        cam_dir = sample_root / f"camera_{mode}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(cam_dir / "camera_K.txt", K, fmt="%.8f")

        for local_idx, src_id in enumerate(src_ids):
            if src_id is None:
                raise RuntimeError(f"rgb_source_frame_ids[{local_idx}] is None")
            src_id_i = int(src_id)
            if src_id_i not in frame_map:
                raise KeyError(f"frame_index {src_id_i} not found in transforms.json")
            Rt = _variant_rt(frame_map[src_id_i], mode)
            np.savetxt(cam_dir / f"camera_RT_{local_idx:04d}.txt", Rt, fmt="%.8f")

        print(f"[done] {mode}: {cam_dir}")


if __name__ == "__main__":
    main()
