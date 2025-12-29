#!/usr/bin/env python3
"""
Convert a PyTorch state dict checkpoint (*.pt / *.pth / lightning *.ckpt) to a single safetensors file.

This is useful for FastVideo training where you want to initialize the STUDENT transformer
from a custom weight file via `--init_weights_from_safetensors`.

Examples:
  # If input is a plain state_dict produced by deepspeed -> fp32 conversion:
  python tools/convert_pt_state_dict_to_safetensors.py \
    --in_pt /path/to/epoch=99-step=800.ckpt_fp32_state_dict.pt \
    --out_safetensors /path/to/student_init.safetensors

  # If the input dict uses a prefix (e.g. "transformer."), filter and strip it:
  python tools/convert_pt_state_dict_to_safetensors.py \
    --in_pt /path/to/epoch=99-step=800.ckpt_fp32_state_dict.pt \
    --out_safetensors /path/to/student_transformer_init.safetensors \
    --select_prefix transformer. \
    --strip_prefix transformer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert torch state_dict to safetensors")
    p.add_argument("--in_pt", type=str, required=True)
    p.add_argument("--out_safetensors", type=str, required=True)
    p.add_argument(
        "--select_prefix",
        type=str,
        default="",
        help="If set, only keep keys starting with this prefix.",
    )
    p.add_argument(
        "--strip_prefix",
        type=str,
        default="",
        help="If set, strip this prefix from kept keys.",
    )
    return p.parse_args()


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        # Lightning-style: {"state_dict": {...}, ...}
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in sd.values()):
                return sd
        # Plain state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise TypeError(
        "Unsupported checkpoint format. Expected a plain state_dict or a dict with key 'state_dict'."
    )


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_pt)
    out_path = Path(args.out_safetensors)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = torch.load(str(in_path), map_location="cpu")
    state_dict = _extract_state_dict(obj)

    select_prefix = str(args.select_prefix or "")
    strip_prefix = str(args.strip_prefix or "")

    if select_prefix:
        state_dict = {k: v for k, v in state_dict.items() if k.startswith(select_prefix)}
        if not state_dict:
            raise ValueError(f"No keys matched select_prefix={select_prefix!r}")

    if strip_prefix:
        bad = [k for k in state_dict.keys() if not k.startswith(strip_prefix)]
        if bad:
            raise ValueError(
                f"strip_prefix={strip_prefix!r} requested, but {len(bad)} keys do not start with it (e.g. {bad[0]!r})"
            )
        state_dict = {k[len(strip_prefix):]: v for k, v in state_dict.items()}

    # safetensors requires tensors on CPU and contiguous.
    state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}

    save_file(state_dict, str(out_path))
    print("saved:", str(out_path), "num_tensors:", len(state_dict))


if __name__ == "__main__":
    main()
