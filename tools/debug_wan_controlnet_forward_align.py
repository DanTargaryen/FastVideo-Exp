#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Single-step forward alignment check between Diff-Factory and FastVideo Wan
ControlNet Union inference.

The script builds one shared input state:
- same random latent from one seed
- same first-frame/image latent
- same VAE-encoded depth/normal/mask/masked_rgb control latents
- same prompt/negative prompt embeddings
- same framewise timestep tokens

It then runs Diff-Factory and FastVideo sequentially, compares `control_res`
and guided `noise_pred`, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

import tools.infer_wan_controlnet_ti2v as base
import tools.infer_wan_controlnet_ti2v_long_firstframe_warp as longwarp


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[str(name)]


def _make_fastvideo_args(args: argparse.Namespace) -> FastVideoArgs:
    fastvideo_args = FastVideoArgs.from_kwargs(
        model_path=str(args.base_model),
        mode="inference",
        workload_type="i2v",
        inference_mode=True,
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=True,
    )
    fastvideo_args.pipeline_config.dit_precision = str(args.dtype)
    fastvideo_args.override_transformer_cls_name = "WanTransformer3DModel"
    fastvideo_args.override_controlnet_cls_name = "WanControlnetUnion3DModel"
    fastvideo_args.pipeline_config.dit_config.boundary_ratio = None
    if hasattr(fastvideo_args.pipeline_config, "expand_timesteps"):
        fastvideo_args.pipeline_config.expand_timesteps = True
    if hasattr(fastvideo_args.pipeline_config, "dit_config"):
        fastvideo_args.pipeline_config.dit_config.expand_timesteps = True
    return fastvideo_args


def _load_text_components(args: argparse.Namespace, fastvideo_args: FastVideoArgs):
    tokenizer = PipelineComponentLoader.load_module(
        "tokenizer",
        str(Path(args.base_model) / "tokenizer"),
        "transformers",
        fastvideo_args,
    )
    text_encoder = PipelineComponentLoader.load_module(
        "text_encoder",
        str(Path(args.base_model) / "text_encoder"),
        "transformers",
        fastvideo_args,
    )
    return tokenizer, text_encoder


def _make_raw_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        raw_rgb_dir="",
        raw_depth_dir="",
        raw_normal_dir="",
        raw_mask_dir="",
        raw_masked_rgb_dir="",
        raw_require_normal=True,
        raw_prompt=str(args.prompt),
        raw_caption_path="",
        raw_caption_key="Video_Caption",
        raw_fps=int(args.fps),
        cam_k=str(args.cam_k),
        cam_rt_dir=str(args.cam_rt_dir),
        num_frames=int(args.num_frames),
        height=int(args.height),
        width=int(args.width),
    )


@torch.no_grad()
def _prepare_shared_inputs(args: argparse.Namespace,
                           fastvideo_args: FastVideoArgs,
                           device: torch.device,
                           dtype: torch.dtype) -> dict[str, Any]:
    tokenizer, text_encoder = _load_text_components(args, fastvideo_args)
    vae = PipelineComponentLoader.load_module(
        "vae",
        str(Path(args.base_model) / "vae"),
        "diffusers",
        fastvideo_args,
    ).to(device)

    cfg_path = Path(args.base_model) / "transformer" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    latent_c = int(cfg.get("in_channels", getattr(vae.config, "z_dim", 48)))
    max_text_len = int(getattr(fastvideo_args.pipeline_config.dit_config,
                               "text_len", 512))

    raw_args = _make_raw_args(args)
    sequence = longwarp._load_raw_long_sequence_nomask(
        sample_root=Path(args.raw_sample_root).expanduser().resolve(),
        args=raw_args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=SimpleNamespace(text_len=max_text_len),
        inference_device=device,
        dtype=dtype,
    )
    prompt_embeds = sequence.text_embedding_bld.to(device=device, dtype=dtype)
    negative_prompt_embeds = base._compute_negative_prompt_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        negative_prompt=str(args.negative_prompt),
        max_sequence_length=max_text_len,
        dtype=dtype,
        target_device=device,
    )

    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    h, w = int(args.height), int(args.width)
    first_rgb = base._load_rgb_frame(sequence.rgb_paths[0], h, w)
    first_frame_latent = base._encode_first_frame_latent(
        vae=vae,
        first_rgb_chw=first_rgb,
        target_c=latent_c,
        inference_device=device,
        compute_dtype=dtype,
    ).to(device=device, dtype=dtype)

    depth_path_by_id = {
        int(fid): p
        for fid, p in zip(sequence.frame_ids, sequence.depth_paths)
    }
    first_rgb_u8 = longwarp._chw_float_to_u8(first_rgb)
    target_ids = sequence.frame_ids[1:int(args.num_frames)]
    warped_masked_rgb_valid, warped_mask_valid = (
        longwarp._warp_maskrgb_from_keyframes_md_aligned(
            keyframe_rgbs_u8=[first_rgb_u8],
            keyframe_frame_ids=[int(sequence.frame_ids[0])],
            target_frame_ids=target_ids,
            depth_path_by_frame_id=depth_path_by_id,
            camera_k_aligned=sequence.camera_k_aligned,
            camera_rt_dir=sequence.camera_rt_dir,
            crop_params=sequence.crop_params,
            target_height=h,
            target_width=w,
        ))
    warped_masked_rgb = longwarp._pad_tchw(
        warped_masked_rgb_valid, max(int(args.num_frames) - 1, 1))
    warped_mask = longwarp._pad_tchw(
        warped_mask_valid, max(int(args.num_frames) - 1, 1))
    mask_tchw = torch.cat(
        [
            torch.ones((1, 1, h, w), dtype=torch.float32),
            warped_mask[:max(int(args.num_frames) - 1, 0)],
        ],
        dim=0,
    )
    masked_rgb_tchw = torch.cat(
        [
            first_rgb.unsqueeze(0),
            warped_masked_rgb[:max(int(args.num_frames) - 1, 0)],
        ],
        dim=0,
    )

    depth_tchw = base._load_depth_sequence(
        sequence.depth_paths[:int(args.num_frames)],
        h,
        w,
        pmin=2.0,
        pmax=98.0,
        invert_depth=False,
        normalization_mode="md_align",
        crop_params=sequence.crop_params,
    )
    normal_tchw = torch.stack(
        [
            base._load_normal_frame(p, h, w)
            for p in sequence.normal_paths[:int(args.num_frames)]
        ],
        dim=0,
    )
    control_latent = base._encode_control_latent_from_tchw(
        vae=vae,
        depth_tchw=depth_tchw,
        normal_tchw=normal_tchw,
        masked_rgb_tchw=masked_rgb_tchw,
        mask_tchw=mask_tchw,
        target_c=latent_c,
        inference_device=device,
        compute_dtype=dtype,
    ).to(device=device, dtype=dtype)

    latent_t = int(control_latent.shape[2])
    latent_h = int(control_latent.shape[3])
    latent_w = int(control_latent.shape[4])
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    latents = torch.randn(
        (1, latent_c, latent_t, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    image_latents = base._build_image_latent_from_first_frame_latent(
        first_frame_latent_bcfhw=first_frame_latent,
        target_frames=latent_t,
    ).to(device=device, dtype=dtype)

    first_frame_mask = torch.ones((1, 1, latent_t, latent_h, latent_w),
                                  device=device,
                                  dtype=torch.float32)
    first_frame_mask[:, :, 0] = 0.0
    latent_model_input = ((1.0 - first_frame_mask).to(dtype=dtype) *
                          image_latents +
                          first_frame_mask.to(dtype=dtype) * latents)

    patch_size = tuple(cfg.get("patch_size", [1, 2, 2]))
    patch_h = int(patch_size[-2])
    patch_w = int(patch_size[-1])
    t_scalar = torch.tensor(float(args.timestep), device=device,
                            dtype=torch.float32)
    timestep = (first_frame_mask[0, 0] * t_scalar)[:, ::patch_h, ::
                                                   patch_w].flatten()
    timestep = timestep.unsqueeze(0).expand(latent_model_input.shape[0], -1)

    depth_lat, normal_lat, masked_lat, mask_lat = base._split_union_control_latent(
        control_latent, latent_c)

    shared = {
        "initial_latents": latents.detach(),
        "image_latents": image_latents.detach(),
        "first_frame_mask": first_frame_mask.detach(),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "latent_c": int(latent_c),
        "latent_model_input": latent_model_input.detach(),
        "timestep": timestep.detach(),
        "prompt_embeds": prompt_embeds.detach(),
        "negative_prompt_embeds": negative_prompt_embeds.detach(),
        "depth_lat": depth_lat.detach(),
        "normal_lat": normal_lat.detach(),
        "masked_lat": masked_lat.detach(),
        "mask_lat": mask_lat.detach(),
        "latent_shape": tuple(latents.shape),
        "control_latent_shape": tuple(control_latent.shape),
        "prompt": sequence.prompt,
        "crop_params": tuple(sequence.crop_params),
    }

    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return shared


def _tensor_l2(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu().item())


def _tensor_stat(x: torch.Tensor) -> dict[str, Any]:
    x_f = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "l2": float(torch.linalg.vector_norm(x_f).cpu().item()),
        "mean": float(x_f.mean().cpu().item()),
        "std": float(x_f.std(unbiased=False).cpu().item()),
        "min": float(x_f.min().cpu().item()),
        "max": float(x_f.max().cpu().item()),
    }


def _compare_tensor(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_f = a.detach().float().cpu()
    b_f = b.detach().float().cpu()
    if tuple(a_f.shape) != tuple(b_f.shape):
        return {
            "shape_a": list(a_f.shape),
            "shape_b": list(b_f.shape),
            "shape_mismatch": True,
        }
    diff = a_f - b_f
    sq_sum = float((diff * diff).sum().item())
    abs_sum = float(diff.abs().sum().item())
    n = int(diff.numel())
    l2_a = _tensor_l2(a_f)
    l2_b = _tensor_l2(b_f)
    return {
        "shape": list(a_f.shape),
        "numel": n,
        "mae": abs_sum / max(n, 1),
        "rmse": (sq_sum / max(n, 1))**0.5,
        "l2_diff": sq_sum**0.5,
        "l2_a": l2_a,
        "l2_b": l2_b,
        "rel_l2_to_a": (sq_sum**0.5) / max(l2_a, 1e-12),
        "max_abs": float(diff.abs().max().item()),
        "mean_diff": float(diff.mean().item()),
    }


def _compare_control_res(diff_res: list[torch.Tensor],
                         fast_res: list[torch.Tensor]) -> dict[str, Any]:
    blocks = []
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0
    total_l2_diff_ref_sq = 0.0
    for i, (a, b) in enumerate(zip(diff_res, fast_res, strict=True)):
        stat = _compare_tensor(a, b)
        stat["index"] = i
        blocks.append(stat)
        if not stat.get("shape_mismatch"):
            total_abs += float(stat["mae"]) * int(stat["numel"])
            total_sq += float(stat["rmse"])**2 * int(stat["numel"])
            total_n += int(stat["numel"])
            total_l2_diff_ref_sq += float(stat["l2_a"])**2
    aggregate = {
        "num_blocks": len(blocks),
        "numel": total_n,
        "mae": total_abs / max(total_n, 1),
        "rmse": (total_sq / max(total_n, 1))**0.5,
        "l2_diff": total_sq**0.5,
        "rel_l2_to_diff": (total_sq**0.5) /
        max(total_l2_diff_ref_sq**0.5, 1e-12),
    }
    return {"aggregate": aggregate, "blocks": blocks}


@torch.no_grad()
def _run_diff_factory(args: argparse.Namespace, shared: dict[str, Any],
                      device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    diff_src = str(Path(args.diff_factory_root) / "src")
    if diff_src not in sys.path:
        sys.path.insert(0, diff_src)
    from diffactory.models.controlnets.controlnet_wan_union import (  # noqa: PLC0415
        WanControlNetUnionInput,
        WanControlnetUnion,
    )
    from diffactory.models.transformers.transformer_controlnet_wan import (  # noqa: PLC0415
        WanTransformerControlnet3DModel,
    )

    controlnet = WanControlnetUnion.from_pretrained(
        str(args.controlnet_dir)).to(device=device, dtype=dtype).eval()
    transformer = WanTransformerControlnet3DModel.from_pretrained(
        str(args.base_model), subfolder="transformer").to(device=device,
                                                           dtype=dtype).eval()

    controlnet_cond = WanControlNetUnionInput(
        depth=shared["depth_lat"].to(device=device, dtype=dtype),
        normal=shared["normal_lat"].to(device=device, dtype=dtype),
    )
    latent_model_input = shared["latent_model_input"].to(device=device,
                                                          dtype=dtype)
    timestep = shared["timestep"].to(device=device, dtype=torch.float32)
    prompt_embeds = shared["prompt_embeds"].to(device=device, dtype=dtype)
    negative_prompt_embeds = shared["negative_prompt_embeds"].to(device=device,
                                                                  dtype=dtype)
    mask = shared["mask_lat"].to(device=device, dtype=dtype)
    masked_latent = shared["masked_lat"].to(device=device, dtype=dtype)

    control_res = controlnet(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=controlnet_cond,
        mask=mask,
        masked_latent=masked_latent,
        return_dict=False,
    )[0]
    noise_cond = transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        block_controlnet_hidden_states=control_res,
        return_dict=False,
    )[0]
    noise_uncond = transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=negative_prompt_embeds,
        block_controlnet_hidden_states=control_res,
        return_dict=False,
    )[0]
    noise_pred = noise_uncond + float(args.guidance_scale) * (noise_cond -
                                                              noise_uncond)

    out = {
        "control_res": [x.detach().cpu() for x in control_res],
        "noise_cond": noise_cond.detach().cpu(),
        "noise_uncond": noise_uncond.detach().cpu(),
        "noise_pred": noise_pred.detach().cpu(),
        "side_stats": {
            "control_res_l2": [_tensor_l2(x) for x in control_res],
            "noise_cond": _tensor_stat(noise_cond),
            "noise_uncond": _tensor_stat(noise_uncond),
            "noise_pred": _tensor_stat(noise_pred),
        },
    }
    del transformer, controlnet, control_res, noise_cond, noise_uncond, noise_pred
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _run_fastvideo(args: argparse.Namespace, fastvideo_args: FastVideoArgs,
                   shared: dict[str, Any], device: torch.device,
                   dtype: torch.dtype) -> dict[str, Any]:
    transformer = PipelineComponentLoader.load_module(
        "transformer",
        str(args.transformer_dir),
        "diffusers",
        fastvideo_args,
    ).to(device).eval()
    controlnet = PipelineComponentLoader.load_module(
        "controlnet",
        str(args.controlnet_dir),
        "diffusers",
        fastvideo_args,
    ).to(device).eval()

    latent_model_input = shared["latent_model_input"].to(device=device,
                                                          dtype=dtype)
    timestep = shared["timestep"].to(device=device, dtype=torch.float32)
    prompt_embeds = shared["prompt_embeds"].to(device=device, dtype=dtype)
    negative_prompt_embeds = shared["negative_prompt_embeds"].to(device=device,
                                                                  dtype=dtype)
    latent_c = int(latent_model_input.shape[1])
    control_latent = torch.cat(
        [
            shared["depth_lat"],
            shared["normal_lat"],
            shared["masked_lat"],
            shared["mask_lat"],
        ],
        dim=1,
    ).to(device=device, dtype=dtype)
    control_kwargs = base._build_controlnet_kwargs(controlnet, control_latent,
                                                   latent_c)

    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = [prompt_embeds]
    batch.height = int(args.height)
    batch.width = int(args.width)
    batch.num_frames = int(args.num_frames)

    with set_forward_context(current_timestep=0,
                             attn_metadata=None,
                             forward_batch=batch):
        control_res = controlnet(
            hidden_states=latent_model_input,
            encoder_hidden_states=[prompt_embeds],
            timestep=timestep,
            **control_kwargs,
        )
        noise_cond = transformer(
            latent_model_input,
            [prompt_embeds],
            timestep,
            block_controlnet_hidden_states=control_res,
        )
        noise_uncond = transformer(
            latent_model_input,
            [negative_prompt_embeds],
            timestep,
            block_controlnet_hidden_states=control_res,
        )
    noise_pred = noise_uncond + float(args.guidance_scale) * (noise_cond -
                                                              noise_uncond)

    out = {
        "control_res": [x.detach().cpu() for x in control_res],
        "noise_cond": noise_cond.detach().cpu(),
        "noise_uncond": noise_uncond.detach().cpu(),
        "noise_pred": noise_pred.detach().cpu(),
        "side_stats": {
            "control_res_l2": [_tensor_l2(x) for x in control_res],
            "noise_cond": _tensor_stat(noise_cond),
            "noise_uncond": _tensor_stat(noise_uncond),
            "noise_pred": _tensor_stat(noise_pred),
        },
    }
    del transformer, controlnet, control_res, noise_cond, noise_uncond, noise_pred
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Wan ControlNet single-step forward align")
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
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument(
        "--prompt",
        default="",
        help=(
            "Optional prompt override. Leave empty to resolve the sample prompt "
            "from --raw_sample_root/text.txt or caption*.json, matching raw inference."
        ),
    )
    p.add_argument(
        "--negative_prompt",
        default=("Bright tones, overexposed, static, blurred details, subtitles, "
                 "style, works, paintings, images, static, overall gray, worst "
                 "quality, low quality, JPEG compression residue, ugly, incomplete, "
                 "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                 "disfigured, misshapen limbs, fused fingers, still picture, messy "
                 "background, three legs, many people in the background, walking "
                 "backwards"),
    )
    p.add_argument("--out_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/forward_align_0032_step999_fp32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    base._ensure_single_process_dist_env()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = _dtype(args.dtype)
    fastvideo_args = _make_fastvideo_args(args)

    print("[align] preparing shared inputs")
    shared = _prepare_shared_inputs(args, fastvideo_args, device, dtype)
    print("[align] shared latent shape:", shared["latent_shape"])
    print("[align] shared control latent shape:", shared["control_latent_shape"])

    print("[align] running Diff-Factory forward")
    diff_out = _run_diff_factory(args, shared, device, dtype)
    print("[align] running FastVideo forward")
    fast_out = _run_fastvideo(args, fastvideo_args, shared, device, dtype)

    print("[align] comparing outputs")
    report = {
        "settings": {
            "base_model": str(args.base_model),
            "controlnet_dir": str(args.controlnet_dir),
            "raw_sample_root": str(args.raw_sample_root),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "seed": int(args.seed),
            "timestep": float(args.timestep),
            "guidance_scale": float(args.guidance_scale),
            "dtype": str(args.dtype),
            "prompt": str(shared["prompt"]),
            "crop_params": list(shared["crop_params"]),
        },
        "diff_factory": diff_out["side_stats"],
        "fastvideo": fast_out["side_stats"],
        "compare": {
            "control_res": _compare_control_res(diff_out["control_res"],
                                                fast_out["control_res"]),
            "noise_cond": _compare_tensor(diff_out["noise_cond"],
                                          fast_out["noise_cond"]),
            "noise_uncond": _compare_tensor(diff_out["noise_uncond"],
                                            fast_out["noise_uncond"]),
            "noise_pred": _compare_tensor(diff_out["noise_pred"],
                                          fast_out["noise_pred"]),
        },
    }
    report_path = Path(args.out_dir) / "forward_align_report.json"
    _write_report(report_path, report)

    c = report["compare"]["control_res"]["aggregate"]
    nc = report["compare"]["noise_cond"]
    nu = report["compare"]["noise_uncond"]
    n = report["compare"]["noise_pred"]
    print("[align] control_res aggregate:",
          f"mae={c['mae']:.6g}",
          f"rmse={c['rmse']:.6g}",
          f"rel_l2={c['rel_l2_to_diff']:.6g}")
    print("[align] noise_cond:",
          f"mae={nc['mae']:.6g}",
          f"rmse={nc['rmse']:.6g}",
          f"rel_l2={nc['rel_l2_to_a']:.6g}",
          f"max_abs={nc['max_abs']:.6g}")
    print("[align] noise_uncond:",
          f"mae={nu['mae']:.6g}",
          f"rmse={nu['rmse']:.6g}",
          f"rel_l2={nu['rel_l2_to_a']:.6g}",
          f"max_abs={nu['max_abs']:.6g}")
    print("[align] noise_pred:",
          f"mae={n['mae']:.6g}",
          f"rmse={n['rmse']:.6g}",
          f"rel_l2={n['rel_l2_to_a']:.6g}",
          f"max_abs={n['max_abs']:.6g}")
    print("[align] wrote", report_path)


if __name__ == "__main__":
    main()
