#!/usr/bin/env python3
import json
import math
from pathlib import Path
import numpy as np
import re
import argparse


def frame_key(p: Path):
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else 10**18


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transforms_json", required=True)
    p.add_argument("--sample_root", required=True)
    p.add_argument("--num_frames", type=int, default=401)
    args = p.parse_args()

    transforms_json = Path(args.transforms_json)
    sample_root = Path(args.sample_root)
    rgb_dir = sample_root / "rgb"
    cam_dir = sample_root / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)

    with open(transforms_json, "r", encoding="utf-8") as f:
        tj = json.load(f)

    frames = tj["frames"]

    # matrixcity_dnm.py: street uses 512x512
    w = 512.0
    h = 512.0
    angle_x = float(tj["camera_angle_x"])
    fl_x = 0.5 * w / math.tan(0.5 * angle_x)
    fl_y = fl_x
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [fl_x, 0.0, cx],
        [0.0, fl_y, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    np.savetxt(cam_dir / "camera_K.txt", K, fmt="%.8f")

    rgb_files = sorted(
        [x for x in rgb_dir.iterdir() if x.is_file()],
        key=frame_key,
    )
    if len(rgb_files) < args.num_frames:
        raise RuntimeError(f"rgb frames insufficient: have {len(rgb_files)}, need {args.num_frames}")

    if len(frames) < args.num_frames:
        raise RuntimeError(f"transforms frames insufficient: have {len(frames)}, need {args.num_frames}")

    # Align by local sequential order, matching copied sample files 0000..0400
    for local_idx in range(args.num_frames):
        fr = frames[local_idx]

        c2w = np.array(fr["rot_mat"], dtype=np.float32)
        if c2w.shape != (4, 4):
            raise RuntimeError(f"bad rot_mat shape at idx={local_idx}: {c2w.shape}")

        # matrixcity_dnm.py logic
        c2w = c2w.copy()
        c2w[:3, :3] *= 100.0
        c2w[:3, 3] /= 100.0
        w2c = np.linalg.inv(c2w)

        np.savetxt(cam_dir / f"camera_RT_{local_idx:04d}.txt", w2c, fmt="%.8f")

    print(f"saved K + {args.num_frames} RT files to: {cam_dir}")


if __name__ == "__main__":
    main()
