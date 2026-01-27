#!/usr/bin/env python3
"""
Create a minimal 1-sample pickle manifest + optional warp_out-style folders
from a local clip directory, so you can reuse FastVideo's existing preprocessing
pipeline to write TI2V+ControlNet parquet.

Expected source layout (under --src):
  - rgb/      (0..N png frames)
  - depth/    (0..N png frames, 16-bit depth)
  - mask/     (0..N png frames, 0/255 or soft)
  - maskrgb/  (0..N png frames, RGB already masked)
  - text/     (one caption file; optional)

Outputs (under --stage):
  - rgb/depth/mask/maskrgb/  (symlinks or copies)
  - warp_out/<scene>/warped_masked_rgb/*.png  (links to maskrgb)
  - warp_out/<scene>/warped_mask/*.png        (links to mask)
  - <scene>_<start>_<end>.pickle
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path


def _resolve_frame(d: Path, i: int) -> Path:
    for name in (f"{i:06d}.png", f"{i}.png"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"missing frame {i} in {d}")


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _pick_caption(text_dir: Path, start: int, end: int) -> str:
    if not text_dir.exists():
        return ""
    patterns = [
        f"{start}-{end}.*",
        f"{start+1}-{end+1}.*",
        f"{start:06d}_{end:06d}.*",
        f"{start:06d}-{end:06d}.*",
        "*.json",
        "*.txt",
    ]
    for pat in patterns:
        matches = sorted(text_dir.glob(pat))
        if matches:
            return str(matches[0])
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("make_custom_clip_stage")
    p.add_argument("--src", type=str, required=True, help="Clip root directory")
    p.add_argument("--stage",
                   type=str,
                   required=True,
                   help="Staging output dir (links/copies + pickle)")
    p.add_argument("--scene_name",
                   type=str,
                   default="scene",
                   help="scene_name stored in pickle and warp_out/<scene_name>")
    p.add_argument("--start", type=int, default=0, help="First frame index")
    p.add_argument("--end",
                   type=int,
                   default=80,
                   help="Last frame index (inclusive)")
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (symlink is default).",
    )
    p.add_argument(
        "--caption_path",
        type=str,
        default="",
        help="Optional explicit caption path; otherwise pick from src/text.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    stage = Path(args.stage)
    scene = str(args.scene_name)
    start = int(args.start)
    end = int(args.end)
    frames = list(range(start, end + 1))

    for sub in ("depth", "mask", "maskrgb", "rgb"):
        if not (src / sub).exists():
            if sub == "rgb":
                raise FileNotFoundError(
                    f"Missing required folder: {src/sub}. "
                    "TI2V parquet needs rgb frames to build first_frame_latent.")
            raise FileNotFoundError(f"Missing required folder: {src/sub}")

        for i in frames:
            src_p = _resolve_frame(src / sub, i)
            dst_p = stage / sub / f"{i:06d}.png"
            _link_or_copy(src_p, dst_p, copy=bool(args.copy))

    # Build warp_out layout so v1_preprocess_omnigame_ti2v_controlnet.py can use maskrgb directly.
    warp_root = stage / "warp_out" / scene
    warped_masked_rgb = warp_root / "warped_masked_rgb"
    warped_mask = warp_root / "warped_mask"
    warped_masked_rgb.mkdir(parents=True, exist_ok=True)
    warped_mask.mkdir(parents=True, exist_ok=True)
    for i in frames:
        _link_or_copy(stage / "maskrgb" / f"{i:06d}.png",
                      warped_masked_rgb / f"{i:06d}.png",
                      copy=False)
        _link_or_copy(stage / "mask" / f"{i:06d}.png",
                      warped_mask / f"{i:06d}.png",
                      copy=False)

    caption_path = str(args.caption_path).strip()
    if not caption_path:
        caption_path = _pick_caption(src / "text", start, end)

    sample = {
        "scene_name": scene,
        "frame_indices": frames,
        "video_path": str(stage / "rgb"),
        "control_path": str(stage / "depth"),
        "mask_path": str(stage / "mask"),
        "caption_path": caption_path,
        "prompt": "",
    }
    pickle_path = stage / f"{scene}_{start}_{end}.pickle"
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump({"samples": [sample]}, f)

    print("OK")
    print("stage:", stage)
    print("warp_out_root:", stage / "warp_out")
    print("pickle:", pickle_path)
    if caption_path:
        print("caption:", caption_path)
    else:
        print("caption: (empty)")


if __name__ == "__main__":
    main()

