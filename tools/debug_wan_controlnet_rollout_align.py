#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-step rollout alignment check for Diff-Factory vs FastVideo.

This reuses the shared raw input builder from debug_wan_controlnet_forward_align.py
and compares the full denoising trajectory with the same initial latent, control
latents, prompt embeddings, scheduler config, and timestep tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from diffusers import UniPCMultistepScheduler

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage

import tools.infer_wan_controlnet_ti2v as base
import tools.debug_wan_controlnet_forward_align as forward_align


def _scheduler(args: argparse.Namespace, device: torch.device):
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(args.base_model), subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(
        scheduler.config, flow_shift=float(args.flow_shift))
    scheduler.set_timesteps(int(args.num_inference_steps), device=device)
    return scheduler


def _latent_model_input(shared: dict[str, Any], latents: torch.Tensor,
                        dtype: torch.dtype) -> torch.Tensor:
    image_latents = shared["image_latents"].to(device=latents.device, dtype=dtype)
    first_frame_mask = shared["first_frame_mask"].to(device=latents.device,
                                                      dtype=torch.float32)
    return ((1.0 - first_frame_mask).to(dtype=dtype) * image_latents +
            first_frame_mask.to(dtype=dtype) * latents).to(dtype=dtype)


def _timestep_tokens(shared: dict[str, Any], latents: torch.Tensor,
                     t_cur: torch.Tensor) -> torch.Tensor:
    first_frame_mask = shared["first_frame_mask"].to(device=latents.device,
                                                      dtype=torch.float32)
    patch_h = int(shared["patch_h"])
    patch_w = int(shared["patch_w"])
    temp_ts = first_frame_mask[0, 0] * t_cur.to(dtype=torch.float32)
    temp_ts = temp_ts[:, ::patch_h, ::patch_w].flatten()
    return temp_ts.unsqueeze(0).expand(latents.shape[0], -1)


def _control_latent(shared: dict[str, Any], device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    return torch.cat(
        [
            shared["depth_lat"],
            shared["normal_lat"],
            shared["masked_lat"],
            shared["mask_lat"],
        ],
        dim=1,
    ).to(device=device, dtype=dtype)


def _extract_step_sample(step_out):
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        return step_out[0]
    if torch.is_tensor(step_out):
        return step_out
    raise RuntimeError(f"Unsupported scheduler.step output type: {type(step_out)}")


def _cache_context(model, name: str):
    if hasattr(model, "cache_context"):
        return model.cache_context(name)
    return nullcontext()


def _parse_step_set(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    steps: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        steps.add(int(part))
    return steps


def _brief_compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    stat = forward_align._compare_tensor(a, b)
    return {
        "mae": float(stat["mae"]),
        "rmse": float(stat["rmse"]),
        "rel_l2": float(stat["rel_l2_to_a"]),
        "max_abs": float(stat["max_abs"]),
        "l2_a": float(stat["l2_a"]),
        "l2_b": float(stat["l2_b"]),
    }


@torch.no_grad()
def _run_diff_factory(args: argparse.Namespace, shared: dict[str, Any],
                      device: torch.device,
                      dtype: torch.dtype) -> dict[str, Any]:
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
    scheduler = _scheduler(args, device)

    prompt_embeds = shared["prompt_embeds"].to(device=device, dtype=dtype)
    negative_prompt_embeds = shared["negative_prompt_embeds"].to(device=device,
                                                                  dtype=dtype)
    latents = shared["initial_latents"].to(device=device, dtype=dtype).clone()
    controlnet_cond = WanControlNetUnionInput(
        depth=shared["depth_lat"].to(device=device, dtype=dtype),
        normal=shared["normal_lat"].to(device=device, dtype=dtype),
    )
    mask = shared["mask_lat"].to(device=device, dtype=dtype)
    masked_latent = shared["masked_lat"].to(device=device, dtype=dtype)

    traces: list[dict[str, Any]] = []
    latents_before: list[torch.Tensor] = []
    noise_preds: list[torch.Tensor] = []
    latents_after: list[torch.Tensor] = []
    control_res_by_step: dict[int, list[torch.Tensor]] = {}
    control_compare_steps = _parse_step_set(str(args.compare_control_steps))

    for step_i, t_cur in enumerate(scheduler.timesteps[:int(args.max_steps)]):
        latents_before.append(latents.detach().cpu())
        latent_model_input = _latent_model_input(shared, latents, dtype)
        timestep = _timestep_tokens(shared, latents, t_cur)
        control_res = controlnet(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            mask=mask,
            masked_latent=masked_latent,
            return_dict=False,
        )[0]
        control_res = [x.to(dtype=latents.dtype) for x in control_res]
        if step_i in control_compare_steps:
            control_res_by_step[int(step_i)] = [
                x.detach().cpu() for x in control_res
            ]

        with _cache_context(transformer, "cond"):
            noise_cond = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                block_controlnet_hidden_states=control_res,
                return_dict=False,
            )[0]
        with _cache_context(transformer, "uncond"):
            noise_uncond = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                block_controlnet_hidden_states=control_res,
                return_dict=False,
            )[0]
        noise_pred = noise_uncond + float(args.guidance_scale) * (noise_cond -
                                                                  noise_uncond)
        latents = scheduler.step(noise_pred, t_cur, latents,
                                 return_dict=False)[0]

        traces.append({
            "step_index": int(step_i),
            "timestep": float(t_cur.detach().float().cpu().item()),
            "latent_l2_after": forward_align._tensor_l2(latents),
            "noise_pred_l2": forward_align._tensor_l2(noise_pred),
            "control_l2": float(sum(forward_align._tensor_l2(x)
                                    for x in control_res)),
        })
        noise_preds.append(noise_pred.detach().cpu())
        latents_after.append(latents.detach().cpu())
        del control_res, noise_cond, noise_uncond, noise_pred

    final_latents = latents.detach().cpu()
    del transformer, controlnet, scheduler, latents
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "traces": traces,
        "latents_before": latents_before,
        "noise_preds": noise_preds,
        "latents_after": latents_after,
        "control_res_by_step": control_res_by_step,
        "final_latents": final_latents,
    }


@torch.no_grad()
def _run_fastvideo_and_compare(args: argparse.Namespace, fastvideo_args,
                               shared: dict[str, Any],
                               diff_out: dict[str, Any],
                               device: torch.device,
                               dtype: torch.dtype) -> dict[str, Any]:
    transformer = PipelineComponentLoader.load_module(
        "transformer", str(args.transformer_dir), "diffusers",
        fastvideo_args).to(device).eval()
    controlnet = PipelineComponentLoader.load_module(
        "controlnet", str(args.controlnet_dir), "diffusers",
        fastvideo_args).to(device).eval()
    scheduler = _scheduler(args, device)

    prompt_embeds = shared["prompt_embeds"].to(device=device, dtype=dtype)
    negative_prompt_embeds = shared["negative_prompt_embeds"].to(device=device,
                                                                  dtype=dtype)
    latents = shared["initial_latents"].to(device=device, dtype=dtype).clone()
    control_latent = _control_latent(shared, device, dtype)
    latent_c = int(shared["latent_c"])
    control_kwargs = base._build_controlnet_kwargs(controlnet, control_latent,
                                                   latent_c)

    batch = ForwardBatch(data_type="ti2v_controlnet")
    batch.prompt_embeds = [prompt_embeds]
    batch.height = int(args.height)
    batch.width = int(args.width)
    batch.num_frames = int(args.num_frames)

    steps: list[dict[str, Any]] = []
    first_noise_break = None
    first_latent_break = None

    for step_i, t_cur in enumerate(scheduler.timesteps[:int(args.max_steps)]):
        if step_i == 0:
            ref_before = shared["initial_latents"]
        else:
            ref_before = diff_out["latents_after"][step_i - 1]
        if bool(args.use_ref_latents_before):
            latents = ref_before.to(device=device, dtype=dtype).clone()
        before_cmp = _brief_compare(ref_before, latents.detach().cpu())

        latent_model_input = _latent_model_input(shared, latents, dtype)
        timestep = _timestep_tokens(shared, latents, t_cur)
        with set_forward_context(current_timestep=int(step_i),
                                 attn_metadata=None,
                                 forward_batch=batch):
            control_res = controlnet(
                hidden_states=latent_model_input,
                encoder_hidden_states=[prompt_embeds],
                timestep=timestep,
                **control_kwargs,
            )
            control_cmp = None
            if step_i in diff_out["control_res_by_step"]:
                control_cmp = forward_align._compare_control_res(
                    diff_out["control_res_by_step"][step_i],
                    [x.detach().cpu() for x in control_res],
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
        step_out = scheduler.step(noise_pred, t_cur, latents)
        latents = _extract_step_sample(step_out)

        noise_cmp = _brief_compare(diff_out["noise_preds"][step_i],
                                   noise_pred.detach().cpu())
        after_cmp = _brief_compare(diff_out["latents_after"][step_i],
                                   latents.detach().cpu())
        if first_noise_break is None and noise_cmp["rel_l2"] > float(args.break_rel_l2):
            first_noise_break = int(step_i)
        if first_latent_break is None and after_cmp["rel_l2"] > float(args.break_rel_l2):
            first_latent_break = int(step_i)

        steps.append({
            "step_index": int(step_i),
            "timestep": float(t_cur.detach().float().cpu().item()),
            "latents_before": before_cmp,
            "noise_pred": noise_cmp,
            "latents_after": after_cmp,
            "fast_noise_pred_l2": forward_align._tensor_l2(noise_pred),
            "fast_latent_l2_after": forward_align._tensor_l2(latents),
        })
        if control_cmp is not None:
            steps[-1]["control_res"] = control_cmp
        del control_res, noise_cond, noise_uncond, noise_pred

    final_latents = latents.detach().cpu()
    final_cmp = _brief_compare(diff_out["final_latents"], final_latents)
    del transformer, controlnet, scheduler, latents
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "steps": steps,
        "first_noise_break": first_noise_break,
        "first_latent_break": first_latent_break,
        "final_latents": final_cmp,
        "final_latents_tensor": final_latents,
    }


def _summarize_steps(steps: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not steps:
        return {}
    maes = [float(s[key]["mae"]) for s in steps]
    rels = [float(s[key]["rel_l2"]) for s in steps]
    maxes = [float(s[key]["max_abs"]) for s in steps]
    max_rel_idx = max(range(len(steps)), key=lambda i: rels[i])
    return {
        "mean_mae": float(sum(maes) / len(maes)),
        "max_mae": float(max(maes)),
        "mean_rel_l2": float(sum(rels) / len(rels)),
        "max_rel_l2": float(max(rels)),
        "max_abs": float(max(maxes)),
        "max_rel_step": int(steps[max_rel_idx]["step_index"]),
        "max_rel_timestep": float(steps[max_rel_idx]["timestep"]),
    }


def _summarize_control_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [s for s in steps if "control_res" in s]
    if not selected:
        return {}
    maes = [float(s["control_res"]["aggregate"]["mae"]) for s in selected]
    rels = [
        float(s["control_res"]["aggregate"]["rel_l2_to_diff"])
        for s in selected
    ]
    max_rel_idx = max(range(len(selected)), key=lambda i: rels[i])
    return {
        "steps": [int(s["step_index"]) for s in selected],
        "mean_mae": float(sum(maes) / len(maes)),
        "max_mae": float(max(maes)),
        "mean_rel_l2": float(sum(rels) / len(rels)),
        "max_rel_l2": float(max(rels)),
        "max_rel_step": int(selected[max_rel_idx]["step_index"]),
        "max_rel_timestep": float(selected[max_rel_idx]["timestep"]),
    }


def _blend_first_frame(shared: dict[str, Any],
                       latents: torch.Tensor) -> torch.Tensor:
    first_frame_mask = shared["first_frame_mask"].to(device=latents.device,
                                                      dtype=torch.float32)
    image_latents = shared["image_latents"].to(device=latents.device,
                                                dtype=latents.dtype)
    return ((1.0 - first_frame_mask).to(dtype=latents.dtype) * image_latents +
            first_frame_mask.to(dtype=latents.dtype) * latents)


@torch.no_grad()
def _decode_final_videos(args: argparse.Namespace, fastvideo_args,
                         shared: dict[str, Any], diff_out: dict[str, Any],
                         compare: dict[str, Any], device: torch.device,
                         dtype: torch.dtype) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_latents = _blend_first_frame(
        shared, diff_out["final_latents"].to(device=device, dtype=dtype))
    fast_latents = _blend_first_frame(
        shared,
        compare["final_latents_tensor"].to(device=device, dtype=dtype),
    )
    latent_cmp = _brief_compare(diff_latents.detach().cpu(),
                                fast_latents.detach().cpu())

    if bool(args.save_final_latents):
        torch.save(
            {
                "diff_final_blended_latents": diff_latents.detach().cpu(),
                "fast_final_blended_latents": fast_latents.detach().cpu(),
            },
            out_dir / "final_blended_latents.pt",
        )

    fastvideo_args.pipeline_config.vae_precision = str(args.dtype)
    fastvideo_vae = PipelineComponentLoader.load_module(
        "vae",
        str(Path(args.base_model) / "vae"),
        "diffusers",
        fastvideo_args,
    ).to(device)
    fastvideo_decoding = DecodingStage(vae=fastvideo_vae)

    diff_video_fastvae = fastvideo_decoding.decode(
        diff_latents, fastvideo_args).detach().cpu()
    fast_video_fastvae = fastvideo_decoding.decode(
        fast_latents, fastvideo_args).detach().cpu()
    fastvae_video_cmp = _brief_compare(diff_video_fastvae,
                                       fast_video_fastvae)

    del fastvideo_vae, fastvideo_decoding
    gc.collect()
    torch.cuda.empty_cache()

    from diffusers import AutoencoderKLWan  # noqa: PLC0415

    diffusers_vae = AutoencoderKLWan.from_pretrained(
        str(args.base_model), subfolder="vae").to(device=device,
                                                   dtype=dtype).eval()

    def _decode_diffusers_vae(latents: torch.Tensor) -> torch.Tensor:
        lat = latents.to(device=device, dtype=dtype)
        mean = torch.tensor(diffusers_vae.config.latents_mean,
                            device=device,
                            dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(diffusers_vae.config.latents_std,
                           device=device,
                           dtype=dtype).view(1, -1, 1, 1, 1)
        lat = lat * std + mean
        video = diffusers_vae.decode(lat, return_dict=False)[0]
        return (video / 2.0 + 0.5).clamp(0, 1).detach().cpu()

    diff_video_diffvae = _decode_diffusers_vae(diff_latents)
    fast_video_diffvae = _decode_diffusers_vae(fast_latents)
    diffvae_video_cmp = _brief_compare(diff_video_diffvae,
                                       fast_video_diffvae)
    diff_latent_decoder_cmp = _brief_compare(diff_video_diffvae,
                                             diff_video_fastvae)
    fast_latent_decoder_cmp = _brief_compare(fast_video_diffvae,
                                             fast_video_fastvae)

    diff_fastvae_path = out_dir / "diff_factory_rollout_fastvideo_vae.mp4"
    fast_fastvae_path = out_dir / "fastvideo_rollout_fastvideo_vae.mp4"
    diff_diffvae_path = out_dir / "diff_factory_rollout_diffusers_vae.mp4"
    fast_diffvae_path = out_dir / "fastvideo_rollout_diffusers_vae.mp4"
    base._save_mp4(diff_video_fastvae, str(diff_fastvae_path), fps=int(args.fps))
    base._save_mp4(fast_video_fastvae, str(fast_fastvae_path), fps=int(args.fps))
    base._save_mp4(diff_video_diffvae, str(diff_diffvae_path), fps=int(args.fps))
    base._save_mp4(fast_video_diffvae, str(fast_diffvae_path), fps=int(args.fps))

    del (diffusers_vae, diff_latents, fast_latents, diff_video_fastvae,
         fast_video_fastvae, diff_video_diffvae, fast_video_diffvae)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "final_blended_latents": latent_cmp,
        "fastvideo_vae_video": fastvae_video_cmp,
        "fastvideo_vae_video_mae_0_255": float(fastvae_video_cmp["mae"] *
                                                255.0),
        "diffusers_vae_video": diffvae_video_cmp,
        "diffusers_vae_video_mae_0_255": float(diffvae_video_cmp["mae"] *
                                               255.0),
        "diff_latent_decoder_cmp": diff_latent_decoder_cmp,
        "diff_latent_decoder_mae_0_255": float(
            diff_latent_decoder_cmp["mae"] * 255.0),
        "fast_latent_decoder_cmp": fast_latent_decoder_cmp,
        "fast_latent_decoder_mae_0_255": float(
            fast_latent_decoder_cmp["mae"] * 255.0),
        "diff_fastvideo_vae_video_path": str(diff_fastvae_path),
        "fast_fastvideo_vae_video_path": str(fast_fastvae_path),
        "diff_diffusers_vae_video_path": str(diff_diffvae_path),
        "fast_diffusers_vae_video_path": str(fast_diffvae_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Wan ControlNet rollout align")
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
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--timestep", type=float, default=999.0)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--break_rel_l2", type=float, default=1e-4)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--compare_control_steps",
                   default="0,10,40,43,48,49",
                   help="Comma-separated denoise step indices for control_res "
                   "L2/MAE comparison. Empty string disables it.")
    p.add_argument("--use_ref_latents_before",
                   action="store_true",
                   help="Force FastVideo to use Diff-Factory latents_before at "
                   "each step, isolating per-step forward differences from "
                   "scheduler accumulation.")
    p.add_argument("--decode_final",
                   action="store_true",
                   help="Decode and save final Diff-Factory/FastVideo rollout "
                   "latents with the same VAE, then compare decoded frames.")
    p.add_argument("--save_final_latents",
                   action="store_true",
                   help="Save final blended Diff-Factory/FastVideo latents to "
                   "final_blended_latents.pt when --decode_final is set.")
    p.add_argument("--diff_only",
                   action="store_true",
                   help="Run only the manual Diff-Factory rollout and optionally "
                   "save its blended final latent. This is useful when comparing "
                   "against the full Diff-Factory pipeline output.")
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
                   default="/vePFS-buaa/linming/workspace/worldrender/FastVideo-Exp/outputs/rollout_align_0032_fp32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    base._ensure_single_process_dist_env()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = forward_align._dtype(args.dtype)
    fastvideo_args = forward_align._make_fastvideo_args(args)

    print("[rollout-align] preparing shared inputs")
    shared = forward_align._prepare_shared_inputs(args, fastvideo_args, device,
                                                  dtype)
    print("[rollout-align] running Diff-Factory rollout")
    diff_out = _run_diff_factory(args, shared, device, dtype)
    if bool(args.diff_only):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        blended = _blend_first_frame(
            shared, diff_out["final_latents"].to(device=device, dtype=dtype)
        ).detach().cpu()
        if bool(args.save_final_latents):
            torch.save({"diff_final_blended_latents": blended},
                       out_dir / "final_blended_latents.pt")
        report = {
            "settings": {
                "base_model": str(args.base_model),
                "controlnet_dir": str(args.controlnet_dir),
                "raw_sample_root": str(args.raw_sample_root),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "num_inference_steps": int(args.num_inference_steps),
                "max_steps": int(args.max_steps),
                "flow_shift": float(args.flow_shift),
                "seed": int(args.seed),
                "guidance_scale": float(args.guidance_scale),
                "dtype": str(args.dtype),
                "prompt": str(shared["prompt"]),
                "crop_params": list(shared["crop_params"]),
                "diff_only": True,
            },
            "diff_factory_trace": diff_out["traces"],
            "final_blended_latents_stats": forward_align._tensor_stat(blended),
        }
        out_path = out_dir / "rollout_align_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print("[rollout-align] diff_only wrote", out_path)
        return
    print("[rollout-align] running FastVideo rollout and comparing")
    compare = _run_fastvideo_and_compare(args, fastvideo_args, shared, diff_out,
                                         device, dtype)
    decoded_compare = None
    if bool(args.decode_final):
        print("[rollout-align] decoding final latents")
        decoded_compare = _decode_final_videos(args, fastvideo_args, shared,
                                               diff_out, compare, device,
                                               dtype)

    report = {
        "settings": {
            "base_model": str(args.base_model),
            "controlnet_dir": str(args.controlnet_dir),
            "raw_sample_root": str(args.raw_sample_root),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "num_inference_steps": int(args.num_inference_steps),
            "max_steps": int(args.max_steps),
            "flow_shift": float(args.flow_shift),
            "seed": int(args.seed),
            "guidance_scale": float(args.guidance_scale),
            "dtype": str(args.dtype),
            "prompt": str(shared["prompt"]),
            "crop_params": list(shared["crop_params"]),
            "compare_control_steps": str(args.compare_control_steps),
            "use_ref_latents_before": bool(args.use_ref_latents_before),
        },
        "diff_factory_trace": diff_out["traces"],
        "compare_summary": {
            "latents_before": _summarize_steps(compare["steps"],
                                               "latents_before"),
            "noise_pred": _summarize_steps(compare["steps"], "noise_pred"),
            "latents_after": _summarize_steps(compare["steps"],
                                              "latents_after"),
            "control_res": _summarize_control_steps(compare["steps"]),
            "first_noise_break": compare["first_noise_break"],
            "first_latent_break": compare["first_latent_break"],
            "final_latents": compare["final_latents"],
        },
        "steps": compare["steps"],
    }
    if decoded_compare is not None:
        report["decoded_compare"] = decoded_compare
    out_path = Path(args.out_dir) / "rollout_align_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    n = report["compare_summary"]["noise_pred"]
    l = report["compare_summary"]["latents_after"]
    print("[rollout-align] noise_pred:",
          f"mean_mae={n['mean_mae']:.6g}",
          f"max_rel_l2={n['max_rel_l2']:.6g}",
          f"max_rel_step={n['max_rel_step']}")
    print("[rollout-align] latents_after:",
          f"mean_mae={l['mean_mae']:.6g}",
          f"max_rel_l2={l['max_rel_l2']:.6g}",
          f"max_rel_step={l['max_rel_step']}")
    print("[rollout-align] first breaks:",
          f"noise={compare['first_noise_break']}",
          f"latent={compare['first_latent_break']}")
    print("[rollout-align] wrote", out_path)


if __name__ == "__main__":
    main()
