#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Merge a FastVideo distributed checkpoint (torch.distributed.checkpoint) into a
consolidated, single-file inference checkpoint (safetensors).

This is for cases where training produced:
  checkpoint-XXX/distributed_checkpoint/generator/...
and you want to create:
  checkpoint-XXX/generator_inference_transformer/diffusion_pytorch_model.safetensors
  checkpoint-XXX/generator_inference_controlnet/diffusion_pytorch_model.safetensors

Usage (8 GPUs, recommended - matches how DCP shards were saved):
  cd FastVideo-Exp
  export PYTHONPATH=$PWD
  torchrun --standalone --nproc_per_node 8 \
    tools/merge_distributed_checkpoint_to_inference.py \
    --config configs/train_wan_controlnet_self_forcing_phase2.yaml \
    --checkpoint_dir /path/to/checkpoint-800
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml

from fastvideo.logger import init_logger
from fastvideo.training.checkpointing_utils import ModelWrapper
from fastvideo.training.training_utils import save_distillation_checkpoint
from fastvideo.training.wan_controlnet_self_forcing_distillation_pipeline import (
    WanControlnetSelfForcingDistillationPipeline,
)

logger = init_logger(__name__)


@dataclass
class _CfgArgs:
    namespace: argparse.Namespace


def _load_yaml_as_namespace(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML dict, got: {type(cfg)}")
    ns = argparse.Namespace(**cfg)
    # Make TrainingArgs.from_cli_args treat these as "explicitly provided".
    ns._provided = set(cfg.keys())  # type: ignore[attr-defined]
    return ns


class _CheckpointContainer(torch.nn.Module):
    def __init__(self, **modules: torch.nn.Module | None):
        super().__init__()
        for name, module in modules.items():
            if module is not None:
                setattr(self, name, module)


def _extract_step_from_checkpoint_dir(checkpoint_dir: str) -> int:
    base = os.path.basename(os.path.normpath(checkpoint_dir))
    try:
        return int(base.split("-")[-1])
    except Exception as e:
        raise ValueError(
            f"Invalid checkpoint_dir basename '{base}' (expected 'checkpoint-<step>')"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge FastVideo distributed checkpoint into consolidated inference safetensors."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the FastVideo training config YAML used to build the model.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint-XXX directory (must contain distributed_checkpoint/generator).",
    )
    args = parser.parse_args()

    cfg_ns = _load_yaml_as_namespace(args.config)

    # Build the pipeline (initializes distributed environment and creates modules).
    model_path = getattr(cfg_ns, "pretrained_model_name_or_path", None) or getattr(
        cfg_ns, "model_path", None
    )
    if not model_path:
        raise ValueError("Config must define pretrained_model_name_or_path or model_path.")

    pipe = WanControlnetSelfForcingDistillationPipeline.from_pretrained(
        model_path, args=cfg_ns
    )

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized; run with torchrun.")
    rank = dist.get_rank()

    checkpoint_dir = os.path.normpath(args.checkpoint_dir)
    step = _extract_step_from_checkpoint_dir(checkpoint_dir)
    output_parent = os.path.dirname(checkpoint_dir)

    generator_dcp_dir = os.path.join(checkpoint_dir, "distributed_checkpoint", "generator")
    if not os.path.isdir(generator_dcp_dir):
        raise FileNotFoundError(f"Missing generator DCP dir: {generator_dcp_dir}")

    generator_transformer: torch.nn.Module = pipe.get_module("transformer")
    generator_controlnet: torch.nn.Module | None = getattr(pipe, "controlnet", None)

    # Load generator weights from distributed checkpoint.
    generator_ckpt_model: torch.nn.Module = generator_transformer
    if generator_controlnet is not None:
        generator_ckpt_model = _CheckpointContainer(
            transformer=generator_transformer, controlnet=generator_controlnet
        )
    states: dict[str, Any] = {"model": ModelWrapper(generator_ckpt_model)}

    logger.info(
        "rank: %s, loading generator distributed checkpoint from %s",
        rank,
        generator_dcp_dir,
        local_main_process_only=False,
    )
    dcp.load(states, checkpoint_id=generator_dcp_dir)
    dist.barrier()

    # Save consolidated inference weights into the SAME checkpoint-XXX directory.
    # This will write under:
    #   <output_parent>/checkpoint-<step>/generator_inference_transformer/
    #   <output_parent>/checkpoint-<step>/generator_inference_controlnet/
    save_distillation_checkpoint(
        generator_transformer=generator_transformer,
        fake_score_transformer=generator_transformer,  # unused when only_save_generator_weight=True
        rank=rank,
        output_dir=output_parent,
        step=step,
        only_save_generator_weight=True,
        save_consolidated_inference_checkpoint=True,
        generator_controlnet=generator_controlnet,
        fake_score_controlnet=None,
    )
    dist.barrier()

    if rank == 0:
        logger.info(
            "Done. Wrote consolidated inference weights under: %s",
            os.path.join(output_parent, f"checkpoint-{step}"),
        )


if __name__ == "__main__":
    main()

