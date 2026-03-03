#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two rollout trace JSONL files emitted by infer_wan_controlnet_ti2v.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda x: (str(x.get("sample_id", "")), int(x.get("step_index", -1))))
    return rows


def main() -> None:
    p = argparse.ArgumentParser("Compare rollout trace JSONL files")
    p.add_argument("--trace_a", type=str, required=True)
    p.add_argument("--trace_b", type=str, required=True)
    args = p.parse_args()

    a = _load(args.trace_a)
    b = _load(args.trace_b)
    n = min(len(a), len(b))
    if n == 0:
        raise SystemExit("No rows to compare.")

    keys = [
        "timestep",
        "control_scale",
        "latent_l2_before",
        "control_cond_l2",
        "control_uncond_l2",
        "noise_cond_l2",
        "noise_uncond_l2",
        "noise_final_l2",
        "latent_l2_after",
    ]

    print(f"rows_a={len(a)} rows_b={len(b)} compared={n}")
    for k in keys:
        diffs = []
        for i in range(n):
            va = float(a[i].get(k, 0.0))
            vb = float(b[i].get(k, 0.0))
            diffs.append(abs(va - vb))
        mean_abs = sum(diffs) / len(diffs)
        max_abs = max(diffs)
        print(f"{k:>18}: mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}")

    print("\nTop-10 latent_l2_after diffs:")
    pairs = []
    for i in range(n):
        da = float(a[i].get("latent_l2_after", 0.0))
        db = float(b[i].get("latent_l2_after", 0.0))
        pairs.append((abs(da - db), i))
    pairs.sort(reverse=True)
    for d, i in pairs[:10]:
        ra = a[i]
        rb = b[i]
        print(
            f"diff={d:.6e} idx={i} sample_a={ra.get('sample_id')} sample_b={rb.get('sample_id')} "
            f"step_a={ra.get('step_index')} step_b={rb.get('step_index')}"
        )


if __name__ == "__main__":
    main()

