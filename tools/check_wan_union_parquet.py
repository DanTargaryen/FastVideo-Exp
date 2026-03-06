#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _summarize_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, list):
        return {"type": "list", "len": len(value), "value": value}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__, "value": str(value)}


def _find_parquet_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    return sorted(root.rglob("*.parquet"))


def _infer_ti2v_capabilities(schema_names: list[str], row: dict) -> dict:
    has_first_frame_latent = "first_frame_latent_bytes" in schema_names
    has_control_latent = "control_latent_bytes" in schema_names
    has_image_latent = "image_latent_bytes" in schema_names
    has_pil_image = "pil_image_bytes" in schema_names

    control_shape = row.get("control_latent_shape")
    first_shape = row.get("first_frame_latent_shape")
    image_shape = row.get("image_latent_shape")
    pil_shape = row.get("pil_image_shape")

    can_run_union_ti2v = bool(has_first_frame_latent and has_control_latent)
    has_exact_image_condition = bool(has_image_latent or has_pil_image)
    needs_external_rgb = bool(
        can_run_union_ti2v and (not has_exact_image_condition)
    )

    return {
        "has_first_frame_latent": has_first_frame_latent,
        "has_control_latent": has_control_latent,
        "has_image_latent": has_image_latent,
        "has_pil_image": has_pil_image,
        "first_frame_latent_shape": first_shape,
        "control_latent_shape": control_shape,
        "image_latent_shape": image_shape,
        "pil_image_shape": pil_shape,
        "can_run_union_ti2v": can_run_union_ti2v,
        "has_exact_image_condition": has_exact_image_condition,
        "needs_external_rgb_for_exact_bidir": needs_external_rgb,
        "recommendation": (
            "Parquet alone is enough for approximate TI2V+Union inference."
            if can_run_union_ti2v and not needs_external_rgb else
            "Parquet contains exact image conditioning; no extra first-frame RGB is required."
            if can_run_union_ti2v and has_exact_image_condition else
            "Parquet is missing required TI2V+Union fields."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WAN Union parquet contents for TI2V/ControlNet inference."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--show_row", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_path).expanduser()
    files = _find_parquet_files(root)
    if not files:
        raise SystemExit(f"No parquet files found under: {root}")

    target_file = files[0]
    pf = pq.ParquetFile(target_file)
    schema_names = list(pf.schema_arrow.names)

    table = pq.read_table(target_file)
    num_rows = table.num_rows
    if num_rows <= 0:
        raise SystemExit(f"Parquet file has no rows: {target_file}")
    row_idx = max(0, min(int(args.sample_index), num_rows - 1))
    row = table.slice(row_idx, 1).to_pylist()[0]

    tensor_fields = {}
    for prefix in [
        "text_embedding",
        "first_frame_latent",
        "control_latent",
        "image_latent",
        "pil_image",
        "vae_latent",
        "trajectory_latents",
        "trajectory_timesteps",
    ]:
        shape = row.get(f"{prefix}_shape")
        dtype = row.get(f"{prefix}_dtype")
        blob = row.get(f"{prefix}_bytes")
        if shape is None and blob is None:
            continue
        tensor_fields[prefix] = {
            "shape": shape,
            "dtype": dtype,
            "bytes_len": len(blob) if blob is not None else None,
        }

    summary = {
        "data_path": str(root),
        "num_parquet_files": len(files),
        "first_parquet_file": str(target_file),
        "num_rows_in_first_file": num_rows,
        "sample_index_used": row_idx,
        "schema_names": schema_names,
        "tensor_fields": tensor_fields,
        "ti2v_union_capabilities": _infer_ti2v_capabilities(schema_names, row),
    }

    if args.show_row:
        summary["sample_row_summary"] = {
            k: _summarize_value(v) for k, v in row.items()
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
