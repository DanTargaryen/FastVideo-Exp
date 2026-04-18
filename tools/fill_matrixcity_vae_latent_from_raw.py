#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.configs.pipelines.base import PipelineConfig

logger = init_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill empty vae_latent fields for an existing MatrixCity parquet dataset from raw RGB frames."
    )
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--raw_root", type=str, required=True)
    p.add_argument("--street_split", type=str, default="train_dense")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _to_torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _parse_matrixcity_record_id(record_id: str) -> tuple[str, int, int, int]:
    parts = [p for p in str(record_id).split("/") if p]
    if len(parts) < 3:
        raise ValueError(f"Invalid MatrixCity record id: {record_id}")
    scene_name = parts[-3]
    window_name = parts[-2]
    clip_name = parts[-1]
    if "_" not in window_name or not clip_name.startswith("clip_start_"):
        raise ValueError(f"Invalid MatrixCity record id layout: {record_id}")
    window_start_str, window_end_str = window_name.split("_", 1)
    clip_start = int(clip_name.split("_")[-1])
    return scene_name, int(window_start_str), int(window_end_str), int(clip_start)


def _resolve_matrixcity_scene_dir(raw_root: Path, street_split: str,
                                  scene_name: str) -> Path:
    candidates = [
        raw_root / "small_city" / "street" / street_split / scene_name,
        raw_root / "street" / street_split / scene_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve MatrixCity scene dir for scene={scene_name} under {raw_root}"
    )


def _get_rgb_map(scene_cache: dict[str, dict[int, Path]], *,
                 raw_root: Path, street_split: str,
                 scene_name: str) -> dict[int, Path]:
    cached = scene_cache.get(scene_name)
    if cached is not None:
        return cached

    from tools import preprocess_matrixcity_ti2v_controlnet_parquet as mcprep

    scene_dir = _resolve_matrixcity_scene_dir(raw_root, street_split, scene_name)
    rgb_dir = scene_dir / scene_name
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB dir not found for scene={scene_name}: {rgb_dir}")
    rgb_files = mcprep._sorted_pngs(rgb_dir)
    rgb_map = mcprep._build_numeric_file_map(rgb_files)
    if not rgb_map:
        raise FileNotFoundError(f"No RGB frames found for scene={scene_name}: {rgb_dir}")
    scene_cache[scene_name] = rgb_map
    return rgb_map


def _load_clip_rgb_tchw(*, rgb_map: dict[int, Path], frame_ids: list[int],
                        height: int, width: int) -> torch.Tensor:
    from tools import preprocess_matrixcity_ti2v_controlnet_parquet as mcprep

    rgb_paths = mcprep._pick_by_target_ids(rgb_map, frame_ids)
    return torch.stack(
        [mcprep._load_rgb_frame(p, int(height), int(width)) for p in rgb_paths],
        dim=0,
    )


@torch.no_grad()
def _encode_rgb_clip_to_vae_latent(*, vae, infer_base, rgb_tchw: torch.Tensor,
                                   device: torch.device,
                                   compute_dtype: torch.dtype) -> torch.Tensor:
    rgb_bcthw = infer_base._to_vae_input(rgb_tchw, normalize=True).to(
        device=device,
        dtype=compute_dtype,
    )
    lat = infer_base._encode_video_latents(
        vae,
        rgb_bcthw,
        sample_mode="mode",
        compute_dtype=compute_dtype,
    )[0]
    return lat.to("cpu", dtype=torch.float32).contiguous()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONPATH", f"{Path(__file__).resolve().parents[1]}:{os.environ.get('PYTHONPATH','')}")

    input_dir = Path(os.path.expanduser(os.path.expandvars(args.input_dir)))
    output_dir = Path(os.path.expanduser(os.path.expandvars(args.output_dir)))
    model_path = os.path.expanduser(os.path.expandvars(args.model_path))
    raw_root = Path(os.path.expanduser(os.path.expandvars(args.raw_root)))
    street_split = str(args.street_split)
    compute_dtype = _to_torch_dtype(args.dtype)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{env_local_rank}")
        torch.cuda.set_device(env_local_rank)
        dist_init_method = "env://" if env_world_size > 1 else "tcp://127.0.0.1:29691"
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=1,
            distributed_init_method=dist_init_method,
        )
    else:
        device = torch.device("cpu")

    pipeline_config = PipelineConfig.from_pretrained(model_path)
    fastvideo_args = FastVideoArgs(
        model_path=model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pipeline_config=pipeline_config,
    )

    vae = PipelineComponentLoader.load_module(
        module_name="vae",
        component_model_path=os.path.join(model_path, "vae"),
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    ).to(device)
    vae.eval()

    from tools import infer_wan_controlnet_ti2v as infer_base

    parquet_files = sorted(input_dir.rglob("*.parquet"))
    parquet_files = [
        p for i, p in enumerate(parquet_files) if (i % env_world_size) == env_rank
    ]
    scene_cache: dict[str, dict[int, Path]] = {}

    rank_desc = f"fill_vae[r{env_rank}/{env_world_size}]"
    for parquet_path in tqdm(parquet_files, desc=rank_desc):
        rel = parquet_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not bool(args.overwrite):
            continue

        table = pq.read_table(parquet_path)
        data = table.to_pydict()
        num_rows = table.num_rows

        vae_latent_bytes: list[bytes] = []
        vae_latent_shape: list[list[int]] = []
        vae_latent_dtype: list[str] = []

        for row_idx in range(num_rows):
            existing_shape = data["vae_latent_shape"][row_idx]
            existing_bytes = data["vae_latent_bytes"][row_idx]
            existing_dtype = data["vae_latent_dtype"][row_idx]
            if existing_shape and existing_bytes:
                vae_latent_bytes.append(existing_bytes)
                vae_latent_shape.append(list(existing_shape))
                vae_latent_dtype.append(existing_dtype)
                continue

            record_id = str(data["id"][row_idx])
            scene_name, _window_start, _window_end, _clip_start = _parse_matrixcity_record_id(
                record_id)
            clip_start_global = int(data["clip_start_global_id"][row_idx])
            num_frames = int(data["num_frames"][row_idx])
            height = int(data["height"][row_idx])
            width = int(data["width"][row_idx])
            frame_ids = [clip_start_global + i for i in range(num_frames)]

            rgb_map = _get_rgb_map(
                scene_cache,
                raw_root=raw_root,
                street_split=street_split,
                scene_name=scene_name,
            )
            rgb_tchw = _load_clip_rgb_tchw(
                rgb_map=rgb_map,
                frame_ids=frame_ids,
                height=height,
                width=width,
            )
            vae_lat = _encode_rgb_clip_to_vae_latent(
                vae=vae,
                infer_base=infer_base,
                rgb_tchw=rgb_tchw,
                device=device,
                compute_dtype=compute_dtype,
            )
            vae_latent_bytes.append(vae_lat.numpy().tobytes())
            vae_latent_shape.append(list(vae_lat.shape))
            vae_latent_dtype.append("float32")

        data["vae_latent_bytes"] = vae_latent_bytes
        data["vae_latent_shape"] = vae_latent_shape
        data["vae_latent_dtype"] = vae_latent_dtype

        out_table = pa.Table.from_pydict(data, schema=table.schema)
        pq.write_table(out_table, out_path)
        logger.info("Wrote %s", out_path)

    logger.info("Completed VAE latent fill for %s -> %s", input_dir, output_dir)


if __name__ == "__main__":
    main()
