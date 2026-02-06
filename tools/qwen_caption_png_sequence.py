#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_int_list(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _uniform_indices(start: int, end: int, n: int) -> list[int]:
    if n <= 0:
        return []
    if end < start:
        raise ValueError(f"end must be >= start (got start={start}, end={end})")
    if n == 1:
        return [start]
    length = end - start + 1
    if n >= length:
        return list(range(start, end + 1))
    # include endpoints
    # i_k = round(k*(length-1)/(n-1))
    idxs = []
    for k in range(n):
        t = k * (length - 1) / (n - 1)
        idxs.append(start + int(round(t)))
    # de-dup while preserving order
    seen = set()
    dedup = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    return dedup


def _extract_first_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    # Common failure: model wraps JSON in markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        obj = json.loads(candidate)
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


def build_instruction() -> str:
    # Keep it strict: downstream uses json.loads.
    return (
        "You will be given ordered video frames. "
        "Return ONLY a valid JSON object with exactly this schema:\n"
        "{\n"
        '  "captions": {\n'
        '    "Short_Caption": "...",\n'
        '    "PC_Caption": "...",\n'
        '    "Background_Caption": "...",\n'
        '    "Camera_Caption": "...",\n'
        '    "Video_Caption": "...",\n'
        '    "Key_Tags": "tag1, tag2, tag3"\n'
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
    except Exception:  # pragma: no cover
        # Fallback: older transformers versions
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


def generate_caption_json(
    model,
    processor,
    frame_paths: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    # Qwen2.5-VL supports multi-image conversations.
    conversation = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "path": p} for p in frame_paths]
                + [{"type": "text", "text": build_instruction()}]
            ),
        }
    ]

    # Newer transformers: processor.apply_chat_template can directly return tensor inputs.
    # Older versions may only support `tokenize=False`; fall back to manual processing.
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
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )

    output_ids = model.generate(**inputs, **gen_kwargs)
    # Trim the prompt part (same trick as HF docs).
    input_ids = getattr(inputs, "input_ids", None)
    if input_ids is None:
        input_ids = inputs["input_ids"]
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    obj = _extract_first_json(text)
    spec = _coerce_caption_spec(obj)
    return spec.to_json_obj()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Generate structured captions JSON for an 81-frame PNG sequence using Qwen2.5-VL.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing PNG frames (e.g. 000001.png).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=80,
        help="End frame index (inclusive).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="{idx:06d}.png",
        help="Filename pattern. Use {idx} (e.g. '{idx:06d}.png').",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=8,
        help="Number of frames to uniformly sample from [start,end] for captioning.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional explicit indices list (comma-separated). If set, overrides --sample_frames.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model id or local path (Qwen2.5-VL Instruct).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "auto"],
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map (e.g. 'auto' or 'cuda:0').",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="",
        help="Optional attention implementation (e.g. flash_attention_2, sdpa).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"--frames_dir not found: {frames_dir}")

    indices = _parse_int_list(args.indices)
    if indices:
        selected = indices
    else:
        selected = _uniform_indices(int(args.start), int(args.end), int(args.sample_frames))

    frame_paths: list[str] = []
    for idx in selected:
        fp = frames_dir / args.pattern.format(idx=idx)
        if not fp.is_file():
            raise FileNotFoundError(f"Missing frame: {fp}")
        frame_paths.append(str(fp))

    model, processor = _load_qwen(
        model_id_or_path=str(args.model),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        attn_impl=str(args.attn_impl).strip() or None,
    )
    out_obj = generate_caption_json(
        model=model,
        processor=processor,
        frame_paths=frame_paths,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
