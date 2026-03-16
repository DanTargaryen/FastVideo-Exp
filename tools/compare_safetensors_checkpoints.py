#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compare two FastVideo checkpoints or safetensors files.

Supports:
- direct `.safetensors` files
- component dirs containing `diffusion_pytorch_model.safetensors`
- checkpoint dirs containing `generator_inference_transformer/` and
  `generator_inference_controlnet/`

Reports:
- key overlap
- exact-identical tensor count
- global abs/rms/max diff
- cosine similarity over all shared params
- top changed tensors
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


def _resolve_component(root: str, component: str) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(root)))
    if p.is_file():
        if p.suffix != ".safetensors":
            raise ValueError(f"Expected .safetensors file, got: {p}")
        return p

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    direct = p / "diffusion_pytorch_model.safetensors"
    if direct.is_file():
        return direct

    component_dir = p / f"generator_inference_{component}"
    component_file = component_dir / "diffusion_pytorch_model.safetensors"
    if component_file.is_file():
        return component_file

    alt_component_dir = p / component
    alt_component_file = alt_component_dir / "diffusion_pytorch_model.safetensors"
    if alt_component_file.is_file():
        return alt_component_file

    candidates = sorted(p.rglob("*.safetensors"))
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve {component} safetensors under: {p}"
    )


@dataclass
class TensorStat:
    key: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    numel: int
    mean_abs: float
    rms: float
    max_abs: float
    cosine: float
    identical: bool


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.double().flatten()
    b64 = b.double().flatten()
    na = torch.linalg.vector_norm(a64)
    nb = torch.linalg.vector_norm(b64)
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float(torch.dot(a64, b64) / (na * nb))


def _compare_state_dict(path_a: Path, path_b: Path) -> tuple[dict, list[TensorStat]]:
    sd_a = load_file(str(path_a), device="cpu")
    sd_b = load_file(str(path_b), device="cpu")

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    shared = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    stats: list[TensorStat] = []
    total_numel = 0
    total_abs_sum = 0.0
    total_sq_sum = 0.0
    global_max_abs = 0.0
    exact_count = 0

    flat_a_parts = []
    flat_b_parts = []

    for key in shared:
        a = sd_a[key].detach().cpu()
        b = sd_b[key].detach().cpu()
        if tuple(a.shape) != tuple(b.shape):
            continue
        if a.dtype != torch.float32 and not a.dtype.is_floating_point:
            a = a.float()
        else:
            a = a.float()
        if b.dtype != torch.float32 and not b.dtype.is_floating_point:
            b = b.float()
        else:
            b = b.float()

        diff = (a - b).abs()
        numel = diff.numel()
        mean_abs = float(diff.mean())
        rms = float(torch.sqrt(torch.mean((a - b) ** 2)))
        max_abs = float(diff.max())
        identical = bool(torch.equal(a, b))
        cosine = _cosine(a, b)

        stats.append(
            TensorStat(
                key=key,
                shape_a=tuple(a.shape),
                shape_b=tuple(b.shape),
                numel=numel,
                mean_abs=mean_abs,
                rms=rms,
                max_abs=max_abs,
                cosine=cosine,
                identical=identical,
            )
        )

        total_numel += numel
        total_abs_sum += float(diff.sum())
        total_sq_sum += float(((a - b) ** 2).sum())
        global_max_abs = max(global_max_abs, max_abs)
        exact_count += int(identical)

        flat_a_parts.append(a.flatten().double())
        flat_b_parts.append(b.flatten().double())

    if flat_a_parts:
        flat_a = torch.cat(flat_a_parts)
        flat_b = torch.cat(flat_b_parts)
        global_cos = _cosine(flat_a, flat_b)
    else:
        global_cos = float("nan")

    summary = {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "keys_a": len(keys_a),
        "keys_b": len(keys_b),
        "shared_keys": len(shared),
        "only_a": only_a,
        "only_b": only_b,
        "compared_tensors": len(stats),
        "exact_identical_tensors": exact_count,
        "global_mean_abs": (total_abs_sum / total_numel) if total_numel else float("nan"),
        "global_rms": math.sqrt(total_sq_sum / total_numel) if total_numel else float("nan"),
        "global_max_abs": global_max_abs,
        "global_cosine": global_cos,
        "total_numel": total_numel,
    }
    return summary, stats


def _print_report(title: str, summary: dict, stats: list[TensorStat], topk: int) -> None:
    print(f"\n=== {title} ===")
    print(f"path_a: {summary['path_a']}")
    print(f"path_b: {summary['path_b']}")
    print(
        "keys: "
        f"a={summary['keys_a']} "
        f"b={summary['keys_b']} "
        f"shared={summary['shared_keys']} "
        f"compared={summary['compared_tensors']}"
    )
    print(
        "global: "
        f"identical_tensors={summary['exact_identical_tensors']} "
        f"mean_abs={summary['global_mean_abs']:.8f} "
        f"rms={summary['global_rms']:.8f} "
        f"max_abs={summary['global_max_abs']:.8f} "
        f"cosine={summary['global_cosine']:.8f}"
    )
    if summary["only_a"]:
        print(f"only_a_keys_sample: {summary['only_a'][:10]}")
    if summary["only_b"]:
        print(f"only_b_keys_sample: {summary['only_b'][:10]}")

    ranked = sorted(stats, key=lambda x: (x.mean_abs, x.max_abs), reverse=True)
    print(f"top_{topk}_changed_by_mean_abs:")
    for item in ranked[:topk]:
        print(
            f"- {item.key}: shape={item.shape_a} "
            f"mean_abs={item.mean_abs:.8f} rms={item.rms:.8f} "
            f"max_abs={item.max_abs:.8f} cosine={item.cosine:.8f} "
            f"identical={item.identical}"
        )


def main() -> None:
    p = argparse.ArgumentParser("Compare FastVideo safetensors checkpoints")
    p.add_argument("--ckpt_a", required=True, help="Path A: checkpoint dir, component dir, or safetensors file")
    p.add_argument("--ckpt_b", required=True, help="Path B: checkpoint dir, component dir, or safetensors file")
    p.add_argument(
        "--component",
        choices=["transformer", "controlnet", "both"],
        default="both",
        help="Which component to compare",
    )
    p.add_argument("--topk", type=int, default=20, help="Show top-k changed tensors")
    args = p.parse_args()

    components = ["transformer", "controlnet"] if args.component == "both" else [args.component]
    for component in components:
        path_a = _resolve_component(args.ckpt_a, component)
        path_b = _resolve_component(args.ckpt_b, component)
        summary, stats = _compare_state_dict(path_a, path_b)
        _print_report(component, summary, stats, args.topk)


if __name__ == "__main__":
    main()
