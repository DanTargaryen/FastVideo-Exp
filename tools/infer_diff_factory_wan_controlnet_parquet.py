#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Diff-Factory Wan ControlNet Union inference from a FastVideo parquet row."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from transformers import T5TokenizerFast, UMT5EncoderModel

import tools.infer_wan_controlnet_ti2v as base


def _load_diff_factory_modules(diff_factory_root: Path):
    diff_src = diff_factory_root / "src"
    if str(diff_src) not in sys.path:
        sys.path.insert(0, str(diff_src))
    from diffactory.models.controlnets.controlnet_wan_union import (  # noqa: PLC0415
        WanControlNetUnionInput,
        WanControlnetUnion,
    )
    from diffactory.models.transformers.transformer_controlnet_wan import (  # noqa: PLC0415
        WanTransformerControlnet3DModel,
    )

    return WanControlNetUnionInput, WanControlnetUnion, WanTransformerControlnet3DModel


def _decode_latents_diffusers_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(device=next(vae.parameters()).device, dtype=vae.dtype)
    mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    decoded = vae.decode(latents * std + mean, return_dict=False)[0]
    return (decoded / 2.0 + 0.5).clamp(0, 1).detach().cpu()


def _write_manifest(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    sample_id: str,
    prompt: str,
    output_path: Path,
) -> None:
    manifest = {
        "sample_id": str(sample_id),
        "prompt": str(prompt),
        "output_path": str(output_path),
        "argv": [str(x) for x in sys.argv],
        "paths": {
            "diff_factory_root": str(args.diff_factory_root),
            "base_model": str(args.base_model),
            "controlnet_dir": str(args.controlnet_dir),
            "data_path": str(args.data_path),
        },
        "sampling": {
            "scheduler": "unipc",
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "negative_prompt": str(args.negative_prompt),
            "flow_shift": float(args.flow_shift),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
        },
        "shape": {
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": int(args.fps),
        },
    }
    manifest_path = out_dir / f"{sample_id}_diff_factory_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[str(args.dtype)]

    (
        WanControlNetUnionInput,
        WanControlnetUnion,
        WanTransformerControlnet3DModel,
    ) = _load_diff_factory_modules(Path(args.diff_factory_root).expanduser().resolve())

    sample = base._load_sample(str(args.data_path), int(args.index))
    prompt_embeds = sample.text_embedding_bld.to(device=device, dtype=dtype)
    max_text_len = int(prompt_embeds.shape[1])

    tokenizer = T5TokenizerFast.from_pretrained(str(Path(args.base_model) / "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(Path(args.base_model) / "text_encoder"),
        torch_dtype=dtype,
    ).to(device=device).eval()
    negative_prompt_embeds = base._compute_negative_prompt_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        negative_prompt=str(args.negative_prompt),
        max_sequence_length=max_text_len,
        dtype=dtype,
        target_device=device,
    )
    del tokenizer, text_encoder
    torch.cuda.empty_cache()

    transformer = WanTransformerControlnet3DModel.from_pretrained(
        str(args.base_model),
        subfolder="transformer",
    ).to(device=device, dtype=dtype).eval()
    controlnet = WanControlnetUnion.from_pretrained(
        str(args.controlnet_dir),
    ).to(device=device, dtype=dtype).eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(args.base_model),
        subfolder="scheduler",
    )
    scheduler = UniPCMultistepScheduler.from_config(
        scheduler.config,
        flow_shift=float(args.flow_shift),
    )
    scheduler.set_timesteps(int(args.num_inference_steps), device=device)

    control_latent = sample.control_latent_bcfhw.to(device=device, dtype=dtype)
    latent_t = int(control_latent.shape[2])
    latent_h = int(control_latent.shape[3])
    latent_w = int(control_latent.shape[4])
    latent_c = int(sample.first_frame_latent_bcfhw.shape[1])
    if int(args.num_frames) > (latent_t - 1) * 4 + 1:
        raise ValueError(
            f"Requested num_frames={int(args.num_frames)} exceeds latent coverage "
            f"{(latent_t - 1) * 4 + 1}")

    depth_lat, normal_lat, masked_lat, mask_lat = base._split_union_control_latent(
        control_latent,
        latent_c,
    )
    controlnet_cond = WanControlNetUnionInput(depth=depth_lat, normal=normal_lat)

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    latents = torch.randn(
        (1, latent_c, latent_t, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    image_latents = base._build_image_latent_from_first_frame_latent(
        first_frame_latent_bcfhw=sample.first_frame_latent_bcfhw.to(device=device,
                                                                    dtype=dtype),
        target_frames=latent_t,
    ).to(device=device, dtype=dtype)
    first_frame_mask = torch.ones((1, 1, latent_t, latent_h, latent_w),
                                  device=device,
                                  dtype=torch.float32)
    first_frame_mask[:, :, 0] = 0.0

    patch_size = tuple(getattr(transformer.config, "patch_size", (1, 2, 2)))
    patch_h = int(patch_size[-2])
    patch_w = int(patch_size[-1])

    print(
        "[diff-factory-parquet]",
        f"sample={sample.sample_id}",
        f"caption={sample.caption}",
        f"latent_shape={tuple(latents.shape)}",
        f"steps={int(args.num_inference_steps)}",
    )

    for step_i, t_cur in enumerate(scheduler.timesteps):
        latent_model_input = (
            (1.0 - first_frame_mask).to(dtype=dtype) * image_latents +
            first_frame_mask.to(dtype=dtype) * latents
        )
        timestep = (first_frame_mask[0, 0] *
                    t_cur.to(dtype=torch.float32))[:, ::patch_h, ::patch_w].flatten()
        timestep = timestep.unsqueeze(0).expand(latents.shape[0], -1)

        control_res = controlnet(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            mask=mask_lat,
            masked_latent=masked_lat,
            return_dict=False,
        )[0]
        control_res = [x.to(dtype=latents.dtype) for x in control_res]

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
        latents = scheduler.step(noise_pred, t_cur, latents, return_dict=False)[0]
        if step_i == 0 or step_i == len(scheduler.timesteps) - 1:
            print(
                "[diff-factory-parquet]",
                f"step={step_i}",
                f"t={float(t_cur.detach().float().cpu().item()):.1f}",
                f"latent_l2={float(torch.linalg.vector_norm(latents.detach().float()).cpu().item()):.4f}",
            )

    latents = ((1.0 - first_frame_mask).to(dtype=dtype) * image_latents +
               first_frame_mask.to(dtype=dtype) * latents)

    vae = AutoencoderKLWan.from_pretrained(
        str(args.base_model),
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device=device).eval()
    decoded = _decode_latents_diffusers_vae(vae, latents)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = str(sample.sample_id).replace("/", "__")
    out_path = out_dir / f"{safe_id}_diff_factory_unipc{int(args.num_inference_steps)}_fps{int(args.fps)}.mp4"
    base._save_mp4(decoded, str(out_path), fps=int(args.fps))
    _write_manifest(
        out_dir=out_dir,
        args=args,
        sample_id=safe_id,
        prompt=sample.caption,
        output_path=out_path,
    )
    print("[diff-factory-parquet] saved", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Diff-Factory Wan ControlNet Union parquet inference")
    p.add_argument("--diff_factory_root",
                   default="/vePFS-buaa/linming/workspace/worldrender/Diff-Factory")
    p.add_argument("--base_model",
                   default="/vePFS-buaa/linming/workspace/worldrender/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--controlnet_dir",
                   default="/vePFS-buaa/linming/workspace/worldrender/world-renderer-controlnet-union")
    p.add_argument("--data_path", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--negative_prompt", type=str, default="bad quality, worst quality")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
