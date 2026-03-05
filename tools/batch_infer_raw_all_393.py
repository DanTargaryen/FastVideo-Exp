#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def _is_sample_dir(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "rgb").is_dir()
        and (p / "depth").is_dir()
        and (p / "normal").is_dir()
    )


def _build_cmd(args, sid: str, train_dir: Path, mask_dir: Path, out_dir: Path) -> list[str]:
    infer_py = Path(args.infer_script).resolve()
    return [
        "python",
        str(infer_py),
        "--input_mode",
        "raw",
        "--raw_sample_root",
        str(train_dir),
        "--raw_rgb_dir",
        str(train_dir / "rgb"),
        "--raw_depth_dir",
        str(train_dir / "depth"),
        "--raw_normal_dir",
        str(train_dir / "normal"),
        "--raw_mask_dir",
        str(mask_dir / "mask"),
        "--raw_masked_rgb_dir",
        str(mask_dir / "rgb"),
        "--base_model",
        args.base_model,
        "--transformer_dir",
        args.transformer_dir,
        "--controlnet_dir",
        args.controlnet_dir,
        "--attention_mode",
        "causal",
        "--scheduler",
        "flowmatch_euler",
        "--dmd_steps",
        args.dmd_steps,
        "--update_rule",
        args.update_rule,
        "--warp_denoising_step",
        "--local_attn_size",
        str(args.local_attn_size),
        "--sink_size",
        str(args.sink_size),
        "--guidance_scale",
        str(args.guidance_scale),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--index",
        "0",
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--dtype",
        args.dtype,
        "--save_frames",
        "--out_dir",
        str(out_dir / sid),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch raw causal inference for all scenes under train root."
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="/vePFS-buaa/wangyuzhen/Dataset/train",
        help="Root containing scene dirs like 0000/0001/...",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="/vePFS-buaa/yinli/datasets/test_dataset",
        help="Root containing scene dirs with mask/ and rgb/ (masked_rgb).",
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--transformer_dir", type=str, required=True)
    parser.add_argument("--controlnet_dir", type=str, required=True)
    parser.add_argument(
        "--infer_script",
        type=str,
        default="tools/infer_wan_controlnet_ti2v.py",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="outputs/raw_all_causal_f393",
    )
    parser.add_argument("--num_frames", type=int, default=393)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--local_attn_size", type=int, default=21)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument("--dmd_steps", type=str, default="1000,750,500,250")
    parser.add_argument("--update_rule", type=str, default="renoise_x0", choices=["renoise_x0", "euler_dt"])
    parser.add_argument("--cuda_visible_devices", type=str, default="7")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29631")
    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive start scene index in sorted list.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Inclusive end scene index in sorted list. -1 means last.")
    parser.add_argument("--resume", action="store_true", help="Skip scene if mp4 already exists.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not train_root.is_dir():
        raise FileNotFoundError(f"train_root not found: {train_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")

    scenes = sorted([p for p in train_root.iterdir() if _is_sample_dir(p)], key=lambda p: p.name)
    if not scenes:
        raise RuntimeError(f"No valid scenes found under: {train_root}")
    start_idx = max(0, int(args.start_idx))
    end_idx = len(scenes) - 1 if int(args.end_idx) < 0 else min(int(args.end_idx), len(scenes) - 1)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, total={len(scenes)}")
    scenes = scenes[start_idx:end_idx + 1]

    print(f"[INFO] selected {len(scenes)} scenes under {train_root} (range {start_idx}..{end_idx})")
    ok = 0
    skip = 0
    fail = 0

    for idx, scene in enumerate(scenes):
        sid = scene.name
        scene_mask = mask_root / sid
        if not (scene_mask / "mask").is_dir() or not (scene_mask / "rgb").is_dir():
            print(f"[SKIP {idx+1}/{len(scenes)}] {sid}: missing mask/rgb under {scene_mask}")
            skip += 1
            continue

        scene_out = out_root / sid
        scene_out.mkdir(parents=True, exist_ok=True)
        expected_mp4 = list(scene_out.glob("*.mp4"))
        if args.resume and expected_mp4:
            print(f"[SKIP {idx+1}/{len(scenes)}] {sid}: mp4 exists")
            skip += 1
            continue

        cmd = _build_cmd(args, sid, scene, scene_mask, out_root)
        env = os.environ.copy()
        env["MASTER_ADDR"] = args.master_addr
        env["MASTER_PORT"] = args.master_port
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        print(f"[RUN  {idx+1}/{len(scenes)}] {sid}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, env=env)
        if proc.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f"[FAIL {idx+1}/{len(scenes)}] {sid} returncode={proc.returncode}")

    print(f"[DONE] ok={ok} skip={skip} fail={fail} total={len(scenes)}")


if __name__ == "__main__":
    main()
