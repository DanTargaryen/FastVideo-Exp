#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump the real Diff-Factory example pipeline inputs and compare to rollout.

This script intentionally drives
examples/wan/run_wan_controlnet_union_long_w_selection_backwarp.py, because the
existing 0032 rerun artifacts come from that script variant.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

import tools.debug_wan_controlnet_forward_align as forward_align
import tools.infer_wan_controlnet_ti2v as base


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
    "fingers, still picture, messy background, three legs, many people in the "
    "background, walking backwards"
)


def _load_diff_factory_example(diff_factory_root: Path):
    src = diff_factory_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    script = (
        diff_factory_root / "examples" / "wan" /
        "run_wan_controlnet_union_long_w_selection_backwarp.py"
    )
    spec = importlib.util.spec_from_file_location("df_backwarp_example", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import Diff-Factory example: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sorted_files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )


def _stats(x: torch.Tensor) -> dict[str, Any]:
    xf = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "l2": float(torch.linalg.vector_norm(xf).item()),
    }


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return forward_align._compare_tensor(a.detach().cpu(), b.detach().cpu())


def _jsonable_compare(a: torch.Tensor | None,
                      b: torch.Tensor | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    return _compare(a, b)


def _load_reference_latents(path: Path, key: str) -> torch.Tensor | None:
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "diff_final_blended_latents" in data:
            return data["diff_final_blended_latents"]
    if torch.is_tensor(data):
        return data
    raise KeyError(f"Could not find reference latent key {key!r} in {path}")


def _load_chunk_arrays(preprocessed_dir: Path, num_frames: int):
    rgb_dir = preprocessed_dir / "merged_rgb"
    mask_dir = preprocessed_dir / "merged_mask"
    rgb_paths = _sorted_files(rgb_dir, (".png", ".jpg", ".jpeg"))
    mask_paths = _sorted_files(mask_dir, (".png", ".jpg", ".jpeg"))
    if len(rgb_paths) < num_frames:
        raise ValueError(f"Need {num_frames} merged RGB frames, got {len(rgb_paths)}")
    if len(mask_paths) < num_frames:
        raise ValueError(f"Need {num_frames} merged mask frames, got {len(mask_paths)}")
    rgbs = [imageio.imread(p)[:, :, :3] for p in rgb_paths[:num_frames]]
    masks = [imageio.imread(p) for p in mask_paths[:num_frames]]
    return rgbs, masks


def _build_pipe_inputs(args: argparse.Namespace, df_example, device: torch.device):
    sample_root = Path(args.raw_sample_root).expanduser().resolve()
    depth_paths = _sorted_files(sample_root / "depth",
                                (".png", ".jpg", ".jpeg", ".exr"))
    normal_paths = _sorted_files(sample_root / "normal",
                                 (".png", ".jpg", ".jpeg", ".exr"))
    image_path = Path(args.image_path)
    if not image_path.is_file():
        image_path = sample_root / "rgb" / "rgb_0000.png"

    ref_img = imageio.imread(image_path)
    src_h, src_w = ref_img.shape[:2]
    crop_params = df_example.get_crop_params(
        src_w, src_h, int(args.width), int(args.height)
    )

    depth_paths = depth_paths[:int(args.num_frames)]
    normal_paths = normal_paths[:int(args.num_frames)]
    merged_rgbs, merged_masks = _load_chunk_arrays(
        Path(args.preprocessed_dir), int(args.num_frames)
    )

    union_input = df_example.WanControlNetUnionInput()
    if depth_paths:
        depth_processed = df_example.process_depth(
            [str(p) for p in depth_paths],
            int(args.width),
            int(args.height),
            crop_params,
        )
        union_input.depth = df_example.prepare_controlnet_frames(
            depth_processed, int(args.height), int(args.width)
        ).to(device)
    if normal_paths:
        normal_processed = df_example.process_normal(
            [str(p) for p in normal_paths],
            int(args.width),
            int(args.height),
            crop_params,
        )
        union_input.normal = df_example.prepare_controlnet_frames(
            normal_processed, int(args.height), int(args.width)
        ).to(device)

    masks_list = []
    for mask_arr in merged_masks:
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0) / 255.0
        masks_list.append((mask_t > 0.5).float())
    masks_tensor = torch.stack(masks_list, dim=0).unsqueeze(0).to(device)

    masked_frames_list = []
    for rgb_arr in merged_rgbs:
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0
        masked_frames_list.append(rgb_t)
    masked_frames_tensor = torch.stack(
        masked_frames_list, dim=0
    ).unsqueeze(0).to(device)

    image_tensor = masked_frames_tensor[:, 0, :, :, :].squeeze(0).cpu()
    image_tensor_normalized = (image_tensor * 2.0 - 1.0).unsqueeze(0)
    masks_tensor[:, 0] = 1.0
    masked_frames_tensor[:, 0] = image_tensor.to(device)

    return {
        "crop_params": crop_params,
        "controlnet_cond": union_input,
        "masked_video_frames": masked_frames_tensor * 2.0 - 1.0,
        "mask_frames": masks_tensor.repeat(1, 1, 3, 1, 1),
        "image": image_tensor_normalized.to(device),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    base._ensure_single_process_dist_env()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    device = torch.device(args.device)
    dtype = forward_align._dtype(args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fastvideo_args = forward_align._make_fastvideo_args(args)
    shared = forward_align._prepare_shared_inputs(
        args, fastvideo_args, device, dtype
    )

    df_example = _load_diff_factory_example(Path(args.diff_factory_root))
    pipe_inputs = _build_pipe_inputs(args, df_example, device)
    pipe = df_example.prepare_pipeline(
        base_model=str(args.base_model),
        controlnet_model=str(args.controlnet_dir),
        device=str(device),
        dtype=dtype,
    )

    dump: dict[str, torch.Tensor] = {}
    step_dump: dict[str, torch.Tensor] = {}

    orig_encode_prompt = pipe.encode_prompt

    def encode_prompt_wrapper(*a, **kw):
        prompt_embeds, negative_prompt_embeds = orig_encode_prompt(*a, **kw)
        dump["prompt_embeds"] = prompt_embeds.detach().cpu()
        if negative_prompt_embeds is not None:
            dump["negative_prompt_embeds"] = negative_prompt_embeds.detach().cpu()
        return prompt_embeds, negative_prompt_embeds

    pipe.encode_prompt = encode_prompt_wrapper

    orig_prepare_latents = pipe.prepare_latents

    def prepare_latents_wrapper(*a, **kw):
        latents, image_latents = orig_prepare_latents(*a, **kw)
        dump["initial_latents"] = latents.detach().cpu()
        if image_latents is not None:
            dump["image_latents"] = image_latents.detach().cpu()
        return latents, image_latents

    pipe.prepare_latents = prepare_latents_wrapper

    orig_prepare_control = pipe.prepare_controlnet_union_input

    def prepare_control_wrapper(*a, **kw):
        control = orig_prepare_control(*a, **kw)
        if control.depth is not None:
            dump["depth_lat"] = control.depth.detach().cpu()
        if control.normal is not None:
            dump["normal_lat"] = control.normal.detach().cpu()
        return control

    pipe.prepare_controlnet_union_input = prepare_control_wrapper

    orig_prepare_inpaint = pipe.prepare_inpainting_input

    def prepare_inpaint_wrapper(*a, **kw):
        mask_lat, masked_lat = orig_prepare_inpaint(*a, **kw)
        dump["mask_lat"] = mask_lat.detach().cpu()
        dump["masked_lat"] = masked_lat.detach().cpu()
        return mask_lat, masked_lat

    pipe.prepare_inpainting_input = prepare_inpaint_wrapper

    def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if latents is not None:
            step_dump[f"latents_after_step_{int(step)}"] = latents.detach().cpu()
        stop_after = int(args.stop_after_steps)
        if stop_after > 0 and int(step) + 1 >= stop_after:
            pipeline._interrupt = True
        return {}

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    out = pipe(
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        controlnet_cond=pipe_inputs["controlnet_cond"],
        masked_video_frames=pipe_inputs["masked_video_frames"],
        mask_frames=pipe_inputs["mask_frames"],
        image=pipe_inputs["image"],
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        generator=generator,
        output_type="latent",
        controlnet_guidance_start=float(args.controlnet_guidance_start),
        controlnet_guidance_end=float(args.controlnet_guidance_end),
        controlnet_weight=float(args.controlnet_weight),
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    ).frames
    final_latents = out.detach().cpu() if torch.is_tensor(out) else out[0].detach().cpu()
    dump["final_blended_latents"] = final_latents

    latent_t = int(dump["initial_latents"].shape[2])
    first_frame_mask = torch.ones(
        (1, 1, latent_t, int(dump["initial_latents"].shape[3]),
         int(dump["initial_latents"].shape[4])),
        dtype=torch.float32,
    )
    first_frame_mask[:, :, 0] = 0.0
    actual_latent_model_input = (
        (1.0 - first_frame_mask) * dump["image_latents"] +
        first_frame_mask * dump["initial_latents"]
    )

    reference_latents = _load_reference_latents(
        Path(args.reference_latents), str(args.reference_latent_key)
    )

    comparisons = {
        "prompt_embeds": _jsonable_compare(shared["prompt_embeds"],
                                           dump.get("prompt_embeds")),
        "negative_prompt_embeds": _jsonable_compare(
            shared["negative_prompt_embeds"],
            dump.get("negative_prompt_embeds"),
        ),
        "initial_latents": _jsonable_compare(shared["initial_latents"],
                                             dump.get("initial_latents")),
        "latent_model_input": _compare(shared["latent_model_input"],
                                       actual_latent_model_input),
        "depth_lat": _jsonable_compare(shared["depth_lat"], dump.get("depth_lat")),
        "normal_lat": _jsonable_compare(shared["normal_lat"],
                                        dump.get("normal_lat")),
        "masked_lat": _jsonable_compare(shared["masked_lat"],
                                        dump.get("masked_lat")),
        "mask_lat": _jsonable_compare(shared["mask_lat"], dump.get("mask_lat")),
        "final_blended_latents_vs_reference": (
            _compare(reference_latents, final_latents)
            if reference_latents is not None else None
        ),
    }

    report = {
        "settings": {
            "base_model": str(args.base_model),
            "controlnet_dir": str(args.controlnet_dir),
            "raw_sample_root": str(args.raw_sample_root),
            "preprocessed_dir": str(args.preprocessed_dir),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "num_inference_steps": int(args.num_inference_steps),
            "stop_after_steps": int(args.stop_after_steps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "crop_params_from_shared": list(shared["crop_params"]),
            "crop_params_from_diff_factory_script": list(pipe_inputs["crop_params"]),
            "reference_latents": str(args.reference_latents),
        },
        "comparisons": comparisons,
        "dump_stats": {name: _stats(t) for name, t in dump.items()},
        "step_dump_stats": {name: _stats(t) for name, t in step_dump.items()},
    }
    if hasattr(pipe.scheduler, "timesteps"):
        report["scheduler_timesteps"] = [
            float(x) for x in pipe.scheduler.timesteps.detach().cpu().flatten()
        ]

    report_path = out_dir / "diff_factory_pipeline_dump_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    torch.save(
        {
            "final_blended_latents": final_latents,
            "step_dump": step_dump,
        },
        out_dir / "diff_factory_pipeline_dump_tensors.pt",
    )

    print("[df-pipeline-dump] wrote", report_path)
    for name, stat in comparisons.items():
        if stat is None:
            continue
        print(
            "[df-pipeline-dump]",
            name,
            "mae=",
            f"{float(stat['mae']):.6g}",
            "rel_l2=",
            f"{float(stat.get('rel_l2_to_a', stat.get('rel_l2', 0.0))):.6g}",
            "max_abs=",
            f"{float(stat['max_abs']):.6g}",
        )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Dump actual Diff-Factory backwarp pipeline inputs")
    p.add_argument("--base_model",
                   default="/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--transformer_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer")
    p.add_argument("--controlnet_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/world-renderer-controlnet-union")
    p.add_argument("--diff_factory_root",
                   default="/vePFS-buaa/yinli/workspace/Diff-Factory")
    p.add_argument("--raw_sample_root",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032")
    p.add_argument("--preprocessed_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/diff_factory_0032_backwarp_480x832_81f_current_rerun/output_chunks/chunk_0_warped_inputs")
    p.add_argument("--reference_latents",
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/rollout_align_0032_fp32_1step_decode_dualvae_smoke/final_blended_latents.pt")
    p.add_argument("--reference_latent_key",
                   default="diff_final_blended_latents")
    p.add_argument("--image_path",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032/rgb/rgb_0000.png")
    p.add_argument("--cam_k",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032/camera/camera_K.txt")
    p.add_argument("--cam_rt_dir",
                   default="/vePFS-buaa/wangyuzhen/Dataset/test/0032/camera")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timestep", type=float, default=999.0)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--stop_after_steps", type=int, default=1)
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    p.add_argument("--controlnet_guidance_end", type=float, default=1.0)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument(
        "--prompt",
        default="",
        help=(
            "Optional prompt override. Leave empty to resolve the sample prompt "
            "from --raw_sample_root/text.txt or caption*.json, matching raw inference."
        ),
    )
    p.add_argument("--negative_prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/diff_factory_pipeline_dump_0032_1step")
    return p.parse_args()


if __name__ == "__main__":
    main()
