#!/usr/bin/env python3
"""
Build a video from a contiguous PNG frame range like 000001.png ... 000120.png.

Example:
  python tools/png_seq_to_video.py --input_dir /path/to/rgb \
      --start 1 --end 81 --fps 16 --output /path/out.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
from diffusers.utils import export_to_video


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("PNG sequence to MP4")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Folder with 000001.png style frames")
    p.add_argument("--start", type=int, required=True,
                   help="Start frame index (inclusive)")
    p.add_argument("--end", type=int, required=True,
                   help="End frame index (inclusive)")
    p.add_argument("--fps", type=float, default=16,
                   help="Output FPS (default: 16)")
    p.add_argument("--output", type=str, required=True,
                   help="Output mp4 path")
    p.add_argument("--width", type=int, default=0,
                   help="Optional resize width (0=keep)")
    p.add_argument("--height", type=int, default=0,
                   help="Optional resize height (0=keep)")
    return p.parse_args()


def _load_frame(path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if width > 0 and height > 0:
        img = img.resize((int(width), int(height)), resample=Image.BICUBIC)
    return img


def main() -> None:
    args = _parse_args()
    input_dir = Path(os.path.expandvars(os.path.expanduser(args.input_dir)))
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    start = int(args.start)
    end = int(args.end)
    if end < start:
        raise ValueError(f"end must be >= start, got start={start} end={end}")

    frames = []
    for idx in range(start, end + 1):
        frame_path = input_dir / f"{idx:06d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"missing frame: {frame_path}")
        frames.append(_load_frame(frame_path, args.width, args.height))

    output_path = Path(os.path.expandvars(os.path.expanduser(args.output)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=float(args.fps))


if __name__ == "__main__":
    main()
