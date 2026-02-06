#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _extract_first_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Failed to parse JSON from model output")


@dataclass(frozen=True)
class CaptionSpec:
    short_caption: str
    pc_caption: str
    background_caption: str
    camera_caption: str
    video_caption: str
    key_tags: str

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "captions": {
                "Short_Caption": self.short_caption,
                "PC_Caption": self.pc_caption,
                "Background_Caption": self.background_caption,
                "Camera_Caption": self.camera_caption,
                "Video_Caption": self.video_caption,
                "Key_Tags": self.key_tags,
            }
        }


def _coerce_caption_spec(obj: dict[str, Any]) -> CaptionSpec:
    caps = obj.get("captions", obj)
    if not isinstance(caps, dict):
        raise TypeError("Expected dict with key 'captions' (or a captions dict)")

    def _get(key: str) -> str:
        v = caps.get(key, "")
        return str(v).strip()

    return CaptionSpec(
        short_caption=_get("Short_Caption"),
        pc_caption=_get("PC_Caption"),
        background_caption=_get("Background_Caption"),
        camera_caption=_get("Camera_Caption"),
        video_caption=_get("Video_Caption"),
        key_tags=_get("Key_Tags"),
    )


def _instruction() -> str:
    return (
        "You will be given ordered video frames. "
        "Return ONLY a valid JSON object with exactly this schema:\n"
        "{\n"
        '  \"captions\": {\n'
        '    \"Short_Caption\": \"...\",\n'
        '    \"PC_Caption\": \"...\",\n'
        '    \"Background_Caption\": \"...\",\n'
        '    \"Camera_Caption\": \"...\",\n'
        '    \"Video_Caption\": \"...\",\n'
        '    \"Key_Tags\": \"tag1, tag2, tag3\"\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Output MUST be valid JSON (no markdown, no code fences).\n"
        "- Use English.\n"
        "- Short_Caption: 1 concise sentence.\n"
        "- PC_Caption: describe the main subject's appearance + action progression.\n"
        "- Background_Caption: describe the environment.\n"
        "- Camera_Caption: describe camera POV/motion.\n"
        "- Video_Caption: 5-8 sentences summarizing the whole clip.\n"
        "- Key_Tags: 5-12 comma-separated keywords.\n"
    )


def _load_qwen(model_id_or_path: str, dtype: str, device_map: str, attn_impl: str | None):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForVision2Seq  # type: ignore
        model_cls = AutoModelForVision2Seq

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported --dtype {dtype} (use bf16/fp16/fp32/auto)")

    processor = AutoProcessor.from_pretrained(model_id_or_path)
    kwargs: dict[str, Any] = {"device_map": device_map}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    model = model_cls.from_pretrained(model_id_or_path, **kwargs)
    model.eval()
    return model, processor


def _caption_from_image_paths(
    model,
    processor,
    image_paths: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    conversation = [
        {
            "role": "user",
            "content": ([{"type": "image", "path": p} for p in image_paths] +
                        [{"type": "text", "text": _instruction()}]),
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        from PIL import Image

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(text=[prompt], images=images, return_tensors="pt")

    inputs = inputs.to(model.device)

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
        })

    output_ids = model.generate(**inputs, **gen_kwargs)
    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None:
        input_ids = inputs["input_ids"]
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output_ids)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    obj = _extract_first_json(text)
    spec = _coerce_caption_spec(obj)
    return spec.to_json_obj()


def _iter_clip_dirs(mask_scene_dir: Path) -> Iterable[Path]:
    # mask_scene_dir layout: <scene_key>/<window_start>_<window_end>/clip_start_xxx/
    for window_dir in sorted(mask_scene_dir.iterdir()):
        if not window_dir.is_dir():
            continue
        if "_" not in window_dir.name:
            continue
        for clip_dir in sorted(window_dir.iterdir()):
            if clip_dir.is_dir() and clip_dir.name.startswith("clip_start_"):
                yield clip_dir


def _parse_scene_key(scene_key: str) -> tuple[str, str, str]:
    # "HD1_3FO4JXIK2PXE_original_1_1" -> ("HD1", "3FO4JXIK2PXE", "original_1_1")
    parts = scene_key.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid scene key: {scene_key}")
    return parts[0], parts[1], parts[2]


def _sorted_images(dir_path: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Caption InteriorNet clips by aligning MASK_INTERIORNET clips to DATA_INTERIORNET cam0/data frames.",
    )
    parser.add_argument("--mask_root", type=str, required=True, help="MASK_INTERIORNET root.")
    parser.add_argument("--data_root", type=str, required=True, help="DATA_INTERIORNET root.")
    parser.add_argument(
        "--scene_key",
        type=str,
        default="",
        help="Optional scene key like 'HD1_3FO4JXIK2PXE_original_1_1'. If empty, process all scenes under mask_root.",
    )
    parser.add_argument(
        "--use_mask_frame_indices",
        action="store_true",
        help="Use indices present in clip_dir/mask/*.png (e.g., 00000.png, 00007.png, ...) as local indices.",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=12,
        help="If not using mask indices, uniformly sample N frames from the 81-frame clip (local indices 0..80).",
    )
    parser.add_argument(
        "--use_masked_rgb",
        action="store_true",
        help="Caption using clip_dir/masked_rgb images instead of original cam0/data frames (fallback for webp decode).",
    )
    parser.add_argument("--model", type=str, required=True, help="Local path or HF id for Qwen2.5-VL Instruct.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_impl", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--out_name",
        type=str,
        default="caption_g{global_start:06d}_{global_end:06d}.json",
        help=(
            "Output JSON filename under each clip_dir. If the value contains '{...}', it is treated as a "
            "Python format string with available fields: global_start, global_end, local_start, local_end, "
            "window_start, clip_start."
        ),
    )
    args = parser.parse_args()

    mask_root = Path(args.mask_root)
    data_root = Path(args.data_root)
    if not mask_root.is_dir():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    model, processor = _load_qwen(
        model_id_or_path=str(args.model),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        attn_impl=str(args.attn_impl).strip() or None,
    )

    scene_dirs: list[Path]
    if args.scene_key:
        scene_dirs = [mask_root / args.scene_key]
    else:
        scene_dirs = [p for p in sorted(mask_root.iterdir()) if p.is_dir()]

    for scene_dir in scene_dirs:
        scene_key = scene_dir.name
        hd, scene_id, sequence_id = _parse_scene_key(scene_key)
        cam_dir = data_root / hd / scene_id / sequence_id / "cam0" / "data"
        if not cam_dir.is_dir() and not args.use_masked_rgb:
            print(f"[WARN] Missing cam dir for {scene_key}: {cam_dir}. Use --use_masked_rgb to caption from masked_rgb.")

        cam_files = _sorted_images(cam_dir) if cam_dir.is_dir() else []

        for clip_dir in _iter_clip_dirs(scene_dir):
            # parse window start from parent dir name "<start>_<end>"
            window_dir = clip_dir.parent
            try:
                window_start = int(window_dir.name.split("_", 1)[0])
            except Exception:
                print(f"[WARN] Skip invalid window dir: {window_dir}")
                continue
            try:
                clip_local_start = int(clip_dir.name.split("_")[-1])
            except Exception:
                print(f"[WARN] Skip invalid clip dir: {clip_dir}")
                continue

            if args.use_mask_frame_indices:
                mask_dir = clip_dir / "mask"
                if not mask_dir.is_dir():
                    print(f"[WARN] Missing mask dir: {mask_dir}")
                    continue
                local_indices_all = []
                for p in _sorted_images(mask_dir):
                    stem = p.stem
                    if stem.isdigit():
                        local_indices_all.append(int(stem))
                if not local_indices_all:
                    print(f"[WARN] No mask frames in: {mask_dir}")
                    continue
            else:
                # Uniform sample over 81-frame clip local indices [0..80]
                n = int(args.sample_frames)
                if n <= 0:
                    local_indices_all = [0, 40, 80]
                else:
                    local_indices_all = []
                    if n == 1:
                        local_indices_all = [0]
                    else:
                        for k in range(n):
                            t = k * 80 / (n - 1)
                            local_indices_all.append(int(round(t)))
                    # de-dup
                    local_indices_all = sorted(set(local_indices_all))

            # Compute global range for naming/debugging.
            # Mapping follows UniDataset InteriorNet loader convention:
            # global_idx = window_start + clip_start + local_idx
            local_start = min(local_indices_all)
            local_end = max(local_indices_all)
            global_start = window_start + clip_local_start + local_start
            global_end = window_start + clip_local_start + local_end

            # Choose which local indices to actually caption with.
            # If --use_mask_frame_indices is set and mask has 81 frames, you likely don't want to feed all frames.
            # Reuse --sample_frames as "how many frames to feed" in both modes.
            local_indices_used = local_indices_all
            if args.use_mask_frame_indices:
                n = int(args.sample_frames)
                if n > 0 and n < len(local_indices_all):
                    if n == 1:
                        local_indices_used = [local_indices_all[0]]
                    else:
                        local_indices_used = []
                        last = len(local_indices_all) - 1
                        for k in range(n):
                            pos = k * last / (n - 1)
                            local_indices_used.append(local_indices_all[int(round(pos))])
                        local_indices_used = sorted(set(local_indices_used))

            out_name_fields = {
                "global_start": global_start,
                "global_end": global_end,
                "local_start": local_start,
                "local_end": local_end,
                "window_start": window_start,
                "clip_start": clip_local_start,
            }

            out_name = args.out_name
            if "{" in out_name and "}" in out_name:
                try:
                    out_name = out_name.format(**out_name_fields)
                except Exception as e:
                    raise ValueError(
                        f"Failed to format --out_name template: {args.out_name!r} with {out_name_fields}"
                    ) from e

            if args.use_masked_rgb:
                src_dir = clip_dir / "masked_rgb"
                if not src_dir.is_dir():
                    print(f"[WARN] Missing masked_rgb dir: {src_dir}")
                    continue
                # Use file list directly; if local_indices exceed, just clamp.
                files = _sorted_images(src_dir)
                if not files:
                    print(f"[WARN] No images in {src_dir}")
                    continue
                picked = []
                for li in local_indices_used:
                    if li < 0:
                        continue
                    if li >= len(files):
                        continue
                    picked.append(str(files[li]))
                if not picked:
                    print(f"[WARN] No picked frames in {src_dir} for indices={local_indices_used}")
                    continue
                out_path = clip_dir / out_name
            else:
                clip_global_start = window_start + clip_local_start
                global_indices = [clip_global_start + li for li in local_indices_used]
                if not cam_files:
                    print(f"[WARN] No cam files for {scene_key}; skip clip {clip_dir}")
                    continue
                picked = []
                for gi in global_indices:
                    if gi < 0 or gi >= len(cam_files):
                        continue
                    picked.append(str(cam_files[gi]))
                if not picked:
                    print(f"[WARN] No picked frames for {clip_dir} global_indices={global_indices}")
                    continue
                out_path = clip_dir / out_name

            try:
                obj = _caption_from_image_paths(
                    model=model,
                    processor=processor,
                    image_paths=picked,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            except Exception as e:
                print(f"[ERROR] caption failed for {clip_dir}: {e}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
            print(f"[OK] {clip_dir} -> {out_path}")


if __name__ == "__main__":
    main()
