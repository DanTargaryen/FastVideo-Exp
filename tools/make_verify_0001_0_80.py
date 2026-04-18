from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

DEFAULT_SRC = "/vePFS-MLP/buaa/wangyuzhen/Dataset/verify/0001"
DEFAULT_STAGE = "/vePFS-buaa/linming/workspace/worldrender/remote_data/verify_0001_0_80_stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a verify clip by symlinking RGB/depth/mask inputs into a compact layout."
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"Source verify scene dir (default: {DEFAULT_SRC})")
    parser.add_argument("--stage", default=DEFAULT_STAGE, help=f"Staging directory (default: {DEFAULT_STAGE})")
    parser.add_argument("--scene-name", default="0001", help="Scene name stored in the generated pickle.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to stage.")
    parser.add_argument("--end-frame", type=int, default=80, help="Last frame index to stage.")
    parser.add_argument(
        "--pickle-name",
        default="verify_0001_0_80.pickle",
        help="Output pickle filename inside the stage directory.",
    )
    return parser.parse_args()


def resolve_frame(directory: Path, frame_idx: int) -> Path:
    for name in (f"{frame_idx:06d}.png", f"{frame_idx}.png"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing frame {frame_idx} in {directory}")


def relink(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    os.symlink(src_path, dst_path)


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser()
    stage = Path(args.stage).expanduser()
    frames = list(range(args.start_frame, args.end_frame + 1))

    for sub_dir in ("depth", "mask", "maskrgb", "rgb"):
        for frame_idx in frames:
            src_path = resolve_frame(src / sub_dir, frame_idx)
            dst_path = stage / sub_dir / f"{frame_idx:06d}.png"
            relink(src_path, dst_path)

    warp_root = stage / "warp_out" / args.scene_name
    for frame_idx in frames:
        relink(stage / "maskrgb" / f"{frame_idx:06d}.png",
               warp_root / "warped_masked_rgb" / f"{frame_idx:06d}.png")
        relink(stage / "mask" / f"{frame_idx:06d}.png",
               warp_root / "warped_mask" / f"{frame_idx:06d}.png")

    text_dir = src / "text"
    captions = list(text_dir.glob("1-81.*"))
    caption_path = str(captions[0]) if captions else ""

    sample = {
        "scene_name": args.scene_name,
        "frame_indices": frames,
        "video_path": str(stage / "rgb"),
        "control_path": str(stage / "depth"),
        "mask_path": str(stage / "mask"),
        "caption_path": caption_path,
        "prompt": "",
    }

    pickle_path = stage / args.pickle_name
    with pickle_path.open("wb") as handle:
        pickle.dump({"samples": [sample]}, handle)
    print("OK:", pickle_path)


if __name__ == "__main__":
    main()
