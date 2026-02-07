#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _is_int_str(s: str) -> bool:
    return bool(s) and s.isdigit()


def _extract_first_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = dec.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
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
            "content": ([{"type": "image", "path": p} for p in image_paths] + [{"type": "text", "text": _instruction()}]),
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

        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                # best-effort: let processor load by path if supported
                images.append(p)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=images, return_tensors="pt")

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if float(temperature) > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        gen_kwargs.update({"do_sample": False})

    output_ids = model.generate(**inputs, **gen_kwargs)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    obj = _extract_first_json(text)
    spec = _coerce_caption_spec(obj)
    return spec.to_json_obj()


def _sorted_images(dir_path: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def _iter_clip_dirs(mask_scene_dir: Path) -> Iterable[tuple[int, int, int, Path]]:
    # mask_scene_dir layout: <street_dir>/<window_start>_<window_end>/clip_start_xxx/
    for window_dir in sorted(mask_scene_dir.iterdir(), key=lambda p: p.name):
        if not window_dir.is_dir() or "_" not in window_dir.name:
            continue
        try:
            window_start = int(window_dir.name.split("_", 1)[0])
            window_end = int(window_dir.name.split("_", 1)[1])
        except Exception:
            continue

        for clip_dir in sorted(window_dir.iterdir()):
            if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_start_"):
                continue
            try:
                clip_start = int(clip_dir.name.split("_")[-1])
            except Exception:
                continue
            yield window_start, window_end, clip_start, clip_dir


def _build_rgb_index(rgb_base_dir: Path) -> tuple[list[int], list[Path], set[int]]:
    # MatrixCity frames are named like "0000.png" (but may exceed 4 digits). Use integer stem sorting.
    ids: list[int] = []
    paths: list[Path] = []
    for p in _sorted_images(rgb_base_dir):
        if _is_int_str(p.stem):
            ids.append(int(p.stem))
            paths.append(p)
    if not ids:
        return [], [], set()
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids_sorted = [ids[i] for i in order]
    paths_sorted = [paths[i] for i in order]
    return ids_sorted, paths_sorted, set(ids_sorted)


def _slice_window_files(
    *,
    window_start: int,
    window_end: int,
    clip_dir: Path,
    rgb_ids: list[int],
    rgb_paths: list[Path],
    rgb_id_set: set[int],
) -> tuple[list[Path], list[int], str]:
    """
    Returns (window_files, window_ids, mode).

    mode:
      - "id_range": window_start/window_end are interpreted as actual frame ids (stems).
      - "index_range": window_start/window_end are interpreted as 0-based indices into rgb_paths.
    """
    if not rgb_ids or not rgb_paths:
        return [], [], "empty"

    # Prefer id-range if both ends exist as real ids.
    if window_start in rgb_id_set and window_end in rgb_id_set:
        import bisect

        i0 = bisect.bisect_left(rgb_ids, window_start)
        i1 = bisect.bisect_right(rgb_ids, window_end)
        return rgb_paths[i0:i1], rgb_ids[i0:i1], "id_range"

    # Otherwise, treat as indices into the sorted list if plausible.
    if 0 <= window_start <= window_end < len(rgb_paths):
        return rgb_paths[window_start : window_end + 1], rgb_ids[window_start : window_end + 1], "index_range"

    print(
        f"[WARN] Cannot interpret window dir as id-range or index-range: {window_start}_{window_end} for {clip_dir}. "
        f"rgb_ids_min={rgb_ids[0]} rgb_ids_max={rgb_ids[-1]} rgb_len={len(rgb_paths)}"
    )
    return [], [], "unsupported"


def _find_scene_root(data_root: Path, street_dir: str, street_split: str | None) -> Path | None:
    # MatrixCity split files contain paths like:
    # small_city/street/train_dense_half/small_city_road_down_dense
    candidates: list[Path] = []
    street_base = data_root / "small_city" / "street"
    if street_split:
        for p in street_base.glob(f"{street_split}/{street_dir}"):
            if (p / "transforms.json").is_file():
                candidates.append(p)
    else:
        for p in street_base.glob(f"*/{street_dir}"):
            if (p / "transforms.json").is_file():
                candidates.append(p)
        if (p / "transforms.json").is_file():
            candidates.append(p)
    for p in (data_root / "small_city" / "aerial").glob(f"*/{street_dir}"):
        if (p / "transforms.json").is_file():
            candidates.append(p)
    if not candidates:
        return None
    # Prefer street (most common for our masks) by shorter path depth ordering.
    candidates = sorted(candidates, key=lambda x: str(x))
    return candidates[0]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(
        description="Caption MatrixCity mask clips by aligning <window_start>_<window_end>/clip_start_xxx to RGB frames.",
    )
    parser.add_argument("--mask_root", type=str, required=True, help="MASK_MATRIXCITY root.")
    parser.add_argument("--data_root", type=str, default="", help="MatrixCity root (needed unless --use_masked_rgb).")
    parser.add_argument(
        "--rgb_root",
        type=str,
        default="",
        help="Alias of --data_root (MatrixCity root). Kept for compatibility with other tools.",
    )
    parser.add_argument("--street_dir", type=str, default="", help="Optional single street_dir (e.g. small_city_road_down_dense).")
    parser.add_argument(
        "--street_split",
        type=str,
        default="",
        help="Optional street split to search under data_root/small_city/street (e.g. train_dense, train_dense_half, test).",
    )
    parser.add_argument("--rank", type=int, default=0, help="Shard rank (0-based). Only process clips where (idx mod world_size) == rank.")
    parser.add_argument("--world_size", type=int, default=1, help="Number of shards for clip-level parallelism.")
    parser.add_argument("--window_len", type=int, default=243, help="Expected window length for padding (MatrixCity commonly uses 243).")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if the output caption file already exists.")
    parser.add_argument(
        "--use_mask_frame_indices",
        action="store_true",
        help="Use indices present in clip_dir/mask/*.png (usually 0..80) as clip-local indices.",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=12,
        help="How many frames to feed into Qwen (uniformly sampled over the clip).",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=81,
        help="Clip length in frames (used when not using mask frame indices).",
    )
    parser.add_argument(
        "--use_masked_rgb",
        action="store_true",
        help="Caption using clip_dir/masked_rgb images instead of original MatrixCity frames.",
    )
    parser.add_argument(
        "--clip_list",
        type=str,
        default="",
        help=(
            "Optional text file containing one clip path per line (a directory named clip_start_XXX, "
            "or a file inside it). When set, only listed clips will be captioned. "
            "Relative paths are resolved under --mask_root."
        ),
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
        default="caption_f{frame_start:06d}_{frame_end:06d}.json",
        help=(
            "Output JSON filename under each clip_dir. If the value contains '{...}', it is treated as a "
            "Python format string with available fields: window_start, window_end, clip_start, local_start, "
            "local_end, frame_start, frame_end."
        ),
    )

    args = parser.parse_args()
    if int(args.world_size) <= 0:
        raise ValueError("--world_size must be >= 1")
    if int(args.rank) < 0 or int(args.rank) >= int(args.world_size):
        raise ValueError(f"--rank must be in [0, {int(args.world_size) - 1}]")

    mask_root = Path(args.mask_root)
    data_root_str = str(args.data_root).strip() or str(args.rgb_root).strip()
    data_root = Path(data_root_str) if data_root_str else Path()
    if not mask_root.is_dir():
        raise FileNotFoundError(f"mask_root not found: {mask_root}")
    if not args.use_masked_rgb and (not data_root_str or not data_root.is_dir()):
        raise FileNotFoundError("data_root is required unless --use_masked_rgb is set")

    model, processor = _load_qwen(
        model_id_or_path=str(args.model),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        attn_impl=str(args.attn_impl).strip() or None,
    )

    def _resolve_clip_dir_from_path(path: Path) -> Path | None:
        if path.is_dir() and path.name.startswith("clip_start_"):
            return path
        for parent in path.parents:
            if parent.name.startswith("clip_start_"):
                return parent
        return None

    def _read_clip_list(clip_list_path: str) -> list[Path]:
        p = Path(os.path.expanduser(os.path.expandvars(clip_list_path)))
        if not p.is_file():
            raise FileNotFoundError(f"--clip_list not found: {p}")
        clips: list[Path] = []
        seen: set[str] = set()
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(os.path.expanduser(os.path.expandvars(line)))
            if not candidate.is_absolute():
                candidate = mask_root / candidate
            clip_dir = _resolve_clip_dir_from_path(candidate)
            if clip_dir is None:
                print(f"[WARN] Invalid clip path in list (no clip_start_* found): {line}")
                continue
            if not clip_dir.is_dir():
                print(f"[WARN] Clip dir not found: {clip_dir}")
                continue
            key = str(clip_dir)
            if key in seen:
                continue
            seen.add(key)
            clips.append(clip_dir)
        return clips

    # Cache per street_dir to avoid repeated scanning.
    rgb_cache: dict[str, tuple[Path | None, list[int], list[Path], set[int]]] = {}

    def _get_rgb_index(street_dir: str) -> tuple[Path | None, list[int], list[Path], set[int]]:
        cached = rgb_cache.get(street_dir)
        if cached is not None:
            return cached
        if args.use_masked_rgb:
            rgb_cache[street_dir] = (None, [], [], set())
            return rgb_cache[street_dir]
        scene_root = _find_scene_root(data_root, street_dir, str(args.street_split).strip() or None)
        if scene_root is None:
            print(f"[WARN] Cannot find MatrixCity scene root for {street_dir} under {data_root}; skip")
            rgb_cache[street_dir] = (None, [], [], set())
            return rgb_cache[street_dir]
        rgb_base_dir = scene_root / street_dir
        if not rgb_base_dir.is_dir():
            print(f"[WARN] Missing rgb dir for {street_dir}: {rgb_base_dir}; skip")
            rgb_cache[street_dir] = (rgb_base_dir, [], [], set())
            return rgb_cache[street_dir]
        rgb_ids, rgb_paths, rgb_id_set = _build_rgb_index(rgb_base_dir)
        rgb_cache[street_dir] = (rgb_base_dir, rgb_ids, rgb_paths, rgb_id_set)
        return rgb_cache[street_dir]

    def _process_clip(street_dir: str, window_start: int, window_end: int, clip_start: int, clip_dir: Path) -> None:
        # Decide clip-local indices.
        if args.use_mask_frame_indices:
            mask_dir = clip_dir / "mask"
            if not mask_dir.is_dir():
                print(f"[WARN] Missing mask dir: {mask_dir}")
                return
            local_indices_all: list[int] = []
            for p in _sorted_images(mask_dir):
                if p.stem.isdigit():
                    local_indices_all.append(int(p.stem))
            if not local_indices_all:
                print(f"[WARN] No mask frames in: {mask_dir}")
                return
            local_indices_all = sorted(set(local_indices_all))
        else:
            n_total = int(args.clip_length)
            local_indices_all = list(range(max(n_total, 1)))

        local_start = min(local_indices_all)
        local_end = max(local_indices_all)

        # Choose which indices to feed to Qwen (uniform on the local_indices_all list positions).
        n = int(args.sample_frames)
        if n <= 0 or n >= len(local_indices_all):
            local_indices_used = local_indices_all
        elif n == 1:
            local_indices_used = [local_indices_all[0]]
        else:
            last = len(local_indices_all) - 1
            picked: list[int] = []
            for k in range(n):
                pos = k * last / (n - 1)
                picked.append(local_indices_all[int(round(pos))])
            local_indices_used = sorted(set(picked))

        # Map clip-local idx -> window idx: (clip_start + idx)
        window_indices = [clip_start + li for li in local_indices_used]

        if args.use_masked_rgb:
            src_dir = clip_dir / "masked_rgb"
            if not src_dir.is_dir():
                print(f"[WARN] Missing masked_rgb dir: {src_dir}")
                return
            files = _sorted_images(src_dir)
            if not files:
                print(f"[WARN] No images in {src_dir}")
                return
            picked_paths: list[str] = []
            for wi in window_indices:
                li = wi - clip_start
                if li < 0 or li >= len(files):
                    continue
                picked_paths.append(str(files[li]))
            if not picked_paths:
                print(f"[WARN] No picked frames in {src_dir} for window_indices={window_indices}")
                return

            frame_start = window_start + clip_start + local_start
            frame_end = window_start + clip_start + local_end
        else:
            rgb_base_dir, rgb_ids, rgb_paths, rgb_id_set = _get_rgb_index(street_dir)
            if not rgb_ids or not rgb_paths or not rgb_id_set:
                print(f"[WARN] No rgb index for {street_dir}; skip clip {clip_dir}")
                return

            window_files, window_ids, _ = _slice_window_files(
                window_start=window_start,
                window_end=window_end,
                clip_dir=clip_dir,
                rgb_ids=rgb_ids,
                rgb_paths=rgb_paths,
                rgb_id_set=rgb_id_set,
            )
            if not window_files:
                print(f"[WARN] No rgb files in {rgb_base_dir} within [{window_start},{window_end}] for {clip_dir}")
                return

            needed_len = int(clip_start) + int(local_end) + 1
            target_len = max(int(args.window_len), needed_len)
            if len(window_files) < target_len:
                pad_n = target_len - len(window_files)
                window_files = window_files + [window_files[-1]] * pad_n
                if window_ids:
                    window_ids = window_ids + [window_ids[-1]] * pad_n

            picked_paths = []
            for wi in window_indices:
                if wi < 0 or wi >= len(window_files):
                    continue
                picked_paths.append(str(window_files[wi]))
            if not picked_paths:
                print(f"[WARN] No picked rgb frames for {clip_dir} window_indices={window_indices}")
                return

            si = clip_start + local_start
            ei = clip_start + local_end
            if window_ids and 0 <= si < len(window_ids) and 0 <= ei < len(window_ids):
                frame_start = int(window_ids[si])
                frame_end = int(window_ids[ei])
            else:
                frame_start = int(window_files[si].stem) if 0 <= si < len(window_files) and _is_int_str(window_files[si].stem) else window_start
                frame_end = int(window_files[ei].stem) if 0 <= ei < len(window_files) and _is_int_str(window_files[ei].stem) else window_end

        out_name_fields = {
            "window_start": window_start,
            "window_end": window_end,
            "clip_start": clip_start,
            "local_start": local_start,
            "local_end": local_end,
            "frame_start": frame_start,
            "frame_end": frame_end,
        }
        out_name = str(args.out_name)
        if "{" in out_name and "}" in out_name:
            out_name = out_name.format(**out_name_fields)
        out_path = clip_dir / out_name
        if args.skip_existing and out_path.is_file():
            print(f"[SKIP] {clip_dir} -> {out_path} (exists)")
            return

        try:
            obj = _caption_from_image_paths(
                model=model,
                processor=processor,
                image_paths=picked_paths,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
        except Exception as e:
            print(f"[ERROR] caption failed for {clip_dir}: {e}")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        print(f"[OK] {clip_dir} -> {out_path}")

    clip_list = str(args.clip_list).strip()
    if clip_list:
        clips_all = _read_clip_list(clip_list)
        if args.street_dir:
            clips_all = [c for c in clips_all if c.parent.parent.name == str(args.street_dir)]
        clips = [p for i, p in enumerate(clips_all) if (i % int(args.world_size)) == int(args.rank)]
        print(
            f"[INFO] clip_list mode: total_clips={len(clips_all)} "
            f"rank={int(args.rank)}/{int(args.world_size)} assigned={len(clips)}"
        )
        for clip_dir in clips:
            window_dir = clip_dir.parent
            scene_dir = window_dir.parent
            street_dir = scene_dir.name
            try:
                window_start_str, window_end_str = window_dir.name.split("_", 1)
                window_start = int(window_start_str)
                window_end = int(window_end_str)
            except Exception:
                print(f"[WARN] Skip invalid window dir: {window_dir}")
                continue
            try:
                clip_start = int(clip_dir.name.split("_")[-1])
            except Exception:
                print(f"[WARN] Skip invalid clip dir: {clip_dir}")
                continue
            _process_clip(street_dir, window_start, window_end, clip_start, clip_dir)
        return

    street_dirs: list[Path]
    if args.street_dir:
        street_dirs = [mask_root / args.street_dir]
    else:
        street_dirs = [p for p in sorted(mask_root.iterdir()) if p.is_dir()]

    for scene_dir in street_dirs:
        street_dir = scene_dir.name
        clip_iter_index = -1
        for window_start, window_end, clip_start, clip_dir in _iter_clip_dirs(scene_dir):
            clip_iter_index += 1
            if (clip_iter_index % int(args.world_size)) != int(args.rank):
                continue
            _process_clip(street_dir, window_start, window_end, clip_start, clip_dir)


if __name__ == "__main__":
    main()
