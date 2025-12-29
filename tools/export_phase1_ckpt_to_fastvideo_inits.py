#!/usr/bin/env python3
"""
Export Diff-Factory phase-1 (Lightning/Deepspeed) checkpoint into FastVideo init safetensors.

This script generates:
  - transformer_init.safetensors  (for `--init-weights-from-safetensors`)
  - controlnet_init.safetensors   (for `--init-controlnet-weights-from-safetensors`)

It expects the input checkpoint to contain both transformer and controlnet weights,
typically under key prefixes like:
  - transformer.*
  - controlnet.*

If your checkpoint uses different prefixes, pass `--transformer_prefix` and/or
`--controlnet_prefix` explicitly.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export phase-1 ckpt to FastVideo init safetensors")
    p.add_argument("--in_pt", type=str, required=True, help="Input .pt/.ckpt file (state_dict or lightning ckpt)")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder for init safetensors")
    p.add_argument("--transformer_prefix", type=str, default="", help="Prefix for transformer keys (e.g. 'transformer.')")
    p.add_argument("--controlnet_prefix", type=str, default="", help="Prefix for controlnet keys (e.g. 'controlnet.')")
    p.add_argument("--min_keys", type=int, default=1000, help="Sanity threshold for auto-detected prefixes")
    return p.parse_args()


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in sd.values()):
                return sd
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise TypeError("Unsupported checkpoint format; expected state_dict or lightning ckpt with 'state_dict'.")


def _count_prefix(keys: list[str], prefix: str) -> int:
    if not prefix:
        return 0
    return sum(1 for k in keys if k.startswith(prefix))


def _auto_pick_prefix(keys: list[str], candidates: list[str], min_keys: int) -> str:
    scored = [(c, _count_prefix(keys, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, best_n = scored[0]
    if best_n < min_keys:
        top_levels = Counter(k.split(".", 1)[0] for k in keys).most_common(30)
        hint = ", ".join([f"{k}:{v}" for k, v in top_levels])
        raise ValueError(
            f"Failed to auto-detect prefix (best={best!r}, matched={best_n}, min_keys={min_keys}). "
            f"Top-level key counts: {hint}. Please pass --transformer_prefix / --controlnet_prefix."
        )
    return best


def _filter_and_strip(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        nk = k[len(prefix):]
        if nk.startswith("."):
            nk = nk[1:]
        out[nk] = v
    return out


def _save_safetensors(path: Path, sd: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sd2 = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    save_file(sd2, str(path))


def main() -> None:
    args = parse_args()
    in_path = Path(os.path.expandvars(os.path.expanduser(args.in_pt)))
    out_dir = Path(os.path.expandvars(os.path.expanduser(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(str(in_path), map_location="cpu")
    sd = _extract_state_dict(obj)
    keys = list(sd.keys())

    transformer_prefix = args.transformer_prefix
    controlnet_prefix = args.controlnet_prefix

    if not transformer_prefix:
        transformer_prefix = _auto_pick_prefix(
            keys,
            candidates=[
                "transformer.",
                "model.transformer.",
                "module.transformer.",
                "net.transformer.",
                "generator.transformer.",
                "student.transformer.",
            ],
            min_keys=int(args.min_keys),
        )
    if not controlnet_prefix:
        controlnet_prefix = _auto_pick_prefix(
            keys,
            candidates=[
                "controlnet.",
                "control_net.",
                "model.controlnet.",
                "module.controlnet.",
                "net.controlnet.",
                "generator.controlnet.",
                "student.controlnet.",
            ],
            min_keys=max(100, int(args.min_keys) // 10),
        )

    sd_tr = _filter_and_strip(sd, transformer_prefix)
    sd_cn = _filter_and_strip(sd, controlnet_prefix)

    if len(sd_tr) == 0:
        raise ValueError(f"No transformer keys matched prefix={transformer_prefix!r}")
    if len(sd_cn) == 0:
        raise ValueError(f"No controlnet keys matched prefix={controlnet_prefix!r}")

    tr_path = out_dir / "transformer_init.safetensors"
    cn_path = out_dir / "controlnet_init.safetensors"
    _save_safetensors(tr_path, sd_tr)
    _save_safetensors(cn_path, sd_cn)

    print("saved:", str(tr_path), "num_tensors:", len(sd_tr), "prefix:", transformer_prefix)
    print("saved:", str(cn_path), "num_tensors:", len(sd_cn), "prefix:", controlnet_prefix)


if __name__ == "__main__":
    main()

