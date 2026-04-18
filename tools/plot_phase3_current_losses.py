#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Plot current phase-3 fake/generator loss curves from a training log.

Outputs:
- fake_score_loss_current.csv / .png
- generator_loss_every5_current.csv / .png
- loss_plot_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOSS_PATTERN = re.compile(
    r"Steps:\s+\d+%\|[^\r\n]*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*"
    r"\[[^\]]*?total_loss=(?P<total_loss>[0-9.]+),\s*"
    r"generator_loss=(?P<generator_loss>[0-9.]+),\s*"
    r"fake_score_loss=(?P<fake_score_loss>[0-9.]+),"
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def parse_losses(log_path: Path) -> tuple[list[dict[str, float]], int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_step: OrderedDict[int, dict[str, float]] = OrderedDict()
    total_steps = 0
    for match in LOSS_PATTERN.finditer(text):
        step = int(match.group("step"))
        total_steps = int(match.group("total"))
        total_loss = float(match.group("total_loss"))
        generator_loss = float(match.group("generator_loss"))
        fake_score_loss = float(match.group("fake_score_loss"))
        if step in by_step:
            prev = by_step[step]
            by_step[step] = {
                "step": float(step),
                "total_loss": total_loss,
                "generator_loss": max(float(prev["generator_loss"]),
                                       generator_loss),
                "fake_score_loss": fake_score_loss,
            }
        else:
            by_step[step] = {
                "step": float(step),
                "total_loss": total_loss,
                "generator_loss": generator_loss,
                "fake_score_loss": fake_score_loss,
            }
    rows = [by_step[k] for k in sorted(by_step)]
    return rows, total_steps


def write_csv(path: Path, rows: list[dict[str, float]], value_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", value_key])
        for row in rows:
            writer.writerow([int(row["step"]), f"{row[value_key]:.6f}"])


def _format_float(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def plot_png(
    *,
    rows: list[dict[str, float]],
    value_key: str,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    out_path: Path,
) -> None:
    width, height = 1600, 900
    margin_left, margin_right = 110, 40
    margin_top, margin_bottom = 110, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), (251, 250, 247))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(30)
    subtitle_font = _load_font(16)
    label_font = _load_font(18)
    tick_font = _load_font(14)

    draw.text((70, 34), title, fill=(29, 35, 47), font=title_font)
    draw.text((70, 72), subtitle, fill=(92, 103, 125), font=subtitle_font)

    x0 = margin_left
    y0 = margin_top
    x1 = margin_left + plot_w
    y1 = margin_top + plot_h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(255, 255, 255), outline=(216, 221, 232), width=2)

    if not rows:
        draw.text((x0 + 20, y0 + 20), "No data parsed from log.", fill=(200, 40, 40), font=label_font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return

    steps = [row["step"] for row in rows]
    values = [row[value_key] for row in rows]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-3)
        y_min -= pad
        y_max += pad
    else:
        pad = max((y_max - y_min) * 0.08, 1e-3)
        y_min -= pad
        y_max += pad

    def x_to_px(step: float) -> float:
        if math.isclose(x_min, x_max):
            return x0 + plot_w / 2
        return x0 + (step - x_min) / (x_max - x_min) * plot_w

    def y_to_px(val: float) -> float:
        return y1 - (val - y_min) / (y_max - y_min) * plot_h

    # Grid
    y_ticks = 6
    for i in range(y_ticks):
        frac = i / (y_ticks - 1)
        py = y0 + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        draw.line((x0, py, x1, py), fill=(224, 228, 235), width=1)
        txt = _format_float(val)
        bbox = draw.textbbox((0, 0), txt, font=tick_font)
        draw.text((x0 - 12 - (bbox[2] - bbox[0]), py - 8), txt, fill=(107, 114, 128), font=tick_font)

    x_ticks = min(10, max(2, len(rows)))
    for i in range(x_ticks):
        frac = i / (x_ticks - 1) if x_ticks > 1 else 0.0
        step = x_min + frac * (x_max - x_min)
        px = x_to_px(step)
        draw.line((px, y0, px, y1), fill=(224, 228, 235), width=1)
        txt = str(int(round(step)))
        bbox = draw.textbbox((0, 0), txt, font=tick_font)
        draw.text((px - (bbox[2] - bbox[0]) / 2, y1 + 12), txt, fill=(107, 114, 128), font=tick_font)

    # Series
    pts = [(x_to_px(step), y_to_px(val)) for step, val in zip(steps, values)]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=4, joint="curve")
    for px, py in pts:
        r = 3
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=color)

    latest_step = int(steps[-1])
    latest_val = values[-1]
    legend = f"{value_key} latest: step={latest_step} value={latest_val:.6f}"
    draw.text((x0 + 20, y0 + 18), legend, fill=color, font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot current fake/generator loss curves from a phase-3 training log.")
    parser.add_argument("--log", required=True, type=str, help="Path to training latest.log")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory for CSV/PNG outputs")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, total_steps = parse_losses(log_path)
    if not rows:
        raise SystemExit(f"No loss rows parsed from log: {log_path}")

    fake_rows = [{"step": row["step"], "fake_score_loss": row["fake_score_loss"]} for row in rows]
    generator_rows_nonzero = [
        {"step": row["step"], "generator_loss": row["generator_loss"]}
        for row in rows
        if row["generator_loss"] > 1e-9
    ]
    generator_rows = (generator_rows_nonzero if generator_rows_nonzero else [
        {"step": row["step"], "generator_loss": row["generator_loss"]}
        for row in rows
        if int(row["step"]) % 5 == 0
    ])

    fake_csv = out_dir / "fake_score_loss_current.csv"
    gen_csv = out_dir / "generator_loss_every5_current.csv"
    fake_png = out_dir / "fake_score_loss_current.png"
    gen_png = out_dir / "generator_loss_every5_current.png"
    summary_txt = out_dir / "loss_plot_summary.txt"

    write_csv(fake_csv, fake_rows, "fake_score_loss")
    write_csv(gen_csv, generator_rows, "generator_loss")

    plot_png(
        rows=fake_rows,
        value_key="fake_score_loss",
        title="Current Fake Score Loss",
        subtitle=f"Source log: {log_path} | parsed steps: {len(rows)} / total target: {total_steps}",
        color=(37, 99, 235),
        out_path=fake_png,
    )
    plot_png(
        rows=generator_rows,
        value_key="generator_loss",
        title="Current Generator Loss (Sampled Every 5 Steps)",
        subtitle=f"Source log: {log_path} | sampled points: {len(generator_rows)}",
        color=(220, 38, 38),
        out_path=gen_png,
    )

    latest = rows[-1]
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"log={log_path}\n")
        f.write(f"parsed_steps={len(rows)}\n")
        f.write(f"step_range={int(rows[0]['step'])}-{int(rows[-1]['step'])}\n")
        f.write(f"latest_step={int(latest['step'])}\n")
        f.write(f"latest_fake_score_loss={latest['fake_score_loss']:.6f}\n")
        f.write(f"latest_generator_loss_raw={latest['generator_loss']:.6f}\n")
        f.write(f"generator_nonzero_points={len(generator_rows_nonzero)}\n")
        if generator_rows:
            f.write(f"latest_generator_sampled_step={int(generator_rows[-1]['step'])}\n")
            f.write(f"latest_generator_sampled_value={generator_rows[-1]['generator_loss']:.6f}\n")
        f.write(f"fake_csv={fake_csv}\n")
        f.write(f"generator_csv={gen_csv}\n")
        f.write(f"fake_png={fake_png}\n")
        f.write(f"generator_png={gen_png}\n")

    print(f"Saved: {fake_csv}")
    print(f"Saved: {gen_csv}")
    print(f"Saved: {fake_png}")
    print(f"Saved: {gen_png}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
