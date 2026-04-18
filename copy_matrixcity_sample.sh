#!/usr/bin/env bash
set -euo pipefail

DATA_MATRIXCITY=/bj-mlp-buaa-prod/user_data/yinli/datasets/Dynamic/MatrixCity
OUT_ROOT=/vePFS-buaa/linming/Dataset/Matrixcity_sample

SPLIT=train_dense_half
STREET=small_city_road_down_dense
NUM_FRAMES=401

RGB_SRC="${DATA_MATRIXCITY}/small_city/street/${SPLIT}/${STREET}/${STREET}"
DEPTH_SRC="${DATA_MATRIXCITY}/small_city_depth/street/${SPLIT}/${STREET}_depth/${STREET}_depth"
NORMAL_SRC="${DATA_MATRIXCITY}/small_city_normal/street/${SPLIT}/${STREET}_normal/${STREET}_normal"

RGB_DST="${OUT_ROOT}/${STREET}/rgb"
DEPTH_DST="${OUT_ROOT}/${STREET}/depth"
NORMAL_DST="${OUT_ROOT}/${STREET}/normal"

mkdir -p "$RGB_DST" "$DEPTH_DST" "$NORMAL_DST"

export NUM_FRAMES RGB_SRC DEPTH_SRC NORMAL_SRC RGB_DST DEPTH_DST NORMAL_DST

python - <<'PY'
from pathlib import Path
import shutil
import re
import os
import sys

num_frames = int(os.environ["NUM_FRAMES"])

pairs = [
    ("rgb", Path(os.environ["RGB_SRC"]), Path(os.environ["RGB_DST"]), {".png", ".jpg", ".jpeg", ".webp", ".bmp"}),
    ("depth", Path(os.environ["DEPTH_SRC"]), Path(os.environ["DEPTH_DST"]), {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr"}),
    ("normal", Path(os.environ["NORMAL_SRC"]), Path(os.environ["NORMAL_DST"]), {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".exr"}),
]

def frame_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    return (int(nums[-1]), p.name) if nums else (10**18, p.name)

for name, src, dst, exts in pairs:
    if not src.is_dir():
        print(f"[error] missing {name} dir: {src}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files, key=frame_key)

    if len(files) < num_frames:
        print(f"[error] {name} only has {len(files)} files, need {num_frames}: {src}", file=sys.stderr)
        sys.exit(1)

    for f in files[:num_frames]:
        shutil.copy2(f, dst / f.name)

    print(f"[done] {name}: copied {num_frames} files")
PY
