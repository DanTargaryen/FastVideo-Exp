from pathlib import Path
import os, pickle

src = Path("/vePFS-MLP/buaa/wangyuzhen/Dataset/verify/0001")
stage = Path("/vePFS-buaa/linming/workspace/worldrender/remote_data/verify_0001_0_80_stage")
frames = list(range(0, 81))  # 0-80

def resolve_frame(d, i):
    for name in (f"{i:06d}.png", f"{i}.png"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"missing frame {i} in {d}")

for sub in ("depth", "mask", "maskrgb", "rgb"):
    (stage / sub).mkdir(parents=True, exist_ok=True)
    for i in frames:
        src_p = resolve_frame(src / sub, i)
        dst_p = stage / sub / f"{i:06d}.png"
        if dst_p.exists():
            dst_p.unlink()
        os.symlink(src_p, dst_p)

# build warp_out so preprocess uses maskrgb directly
warp_root = stage / "warp_out" / "0001"
(warp_root / "warped_masked_rgb").mkdir(parents=True, exist_ok=True)
(warp_root / "warped_mask").mkdir(parents=True, exist_ok=True)
# link directories (contain 6-digit symlinks already)
for i in frames:
    os.symlink(stage / "maskrgb" / f"{i:06d}.png", warp_root / "warped_masked_rgb" / f"{i:06d}.png")
    os.symlink(stage / "mask" / f"{i:06d}.png", warp_root / "warped_mask" / f"{i:06d}.png")

# caption file: use 1-81.*
text_dir = src / "text"
caps = list(text_dir.glob("1-81.*"))
caption_path = str(caps[0]) if caps else ""

sample = {
    "scene_name": "0001",
    "frame_indices": frames,
    "video_path": str(stage / "rgb"),
    "control_path": str(stage / "depth"),
    "mask_path": str(stage / "mask"),
    "caption_path": caption_path,
    "prompt": "",
}

pickle_path = stage / "verify_0001_0_80.pickle"
with open(pickle_path, "wb") as f:
    pickle.dump({"samples": [sample]}, f)

print("OK:", pickle_path)