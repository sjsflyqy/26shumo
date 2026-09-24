from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "text": (52, 101, 164),
    "audio": (230, 120, 35),
    "vision": (63, 145, 85),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def video_metadata(path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0, 0.0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return duration, fps, frames


def extract_keyframe(video_path: Path, time_seconds: float, output_path: Path) -> Path | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def _draw_bars(
    draw: ImageDraw.ImageDraw,
    contributions: dict[str, float],
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    width = right - left
    row = (bottom - top) // 3
    label_font = _font(23)
    for index, name in enumerate(("text", "audio", "vision")):
        y = top + index * row
        value = float(contributions.get(name, 0.0))
        draw.text((left, y), name.title(), fill=(20, 20, 20), font=label_font)
        bar_left = left + 100
        bar_right = bar_left + int(max(0.0, min(1.0, value)) * (width - 190))
        draw.rounded_rectangle(
            (bar_left, y + 2, right - 70, y + 24), radius=8, fill=(225, 225, 225)
        )
        draw.rounded_rectangle(
            (bar_left, y + 2, bar_right, y + 24), radius=8, fill=COLORS[name]
        )
        draw.text((right - 62, y), f"{value:.3f}", fill=(20, 20, 20), font=label_font)


def _draw_curves(
    draw: ImageDraw.ImageDraw,
    curves: dict[str, np.ndarray],
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(175, 175, 175), width=2)
    label_font = _font(19)
    for curve_index, name in enumerate(("text", "audio", "vision")):
        values = np.asarray(curves.get(name, []), dtype=np.float64)
        if values.size == 0:
            continue
        maximum = float(np.max(np.abs(values)))
        normalized = values / maximum if maximum > 1e-12 else values
        band_top = top + curve_index * (bottom - top) / 3
        band_bottom = top + (curve_index + 1) * (bottom - top) / 3
        points = []
        for index, value in enumerate(normalized):
            x = left + 85 + index / max(1, len(values) - 1) * (right - left - 100)
            y = band_bottom - 12 - float(value) * (band_bottom - band_top - 28)
            points.append((int(x), int(y)))
        draw.text((left + 8, int(band_top + 8)), name.title(), fill=COLORS[name], font=label_font)
        if len(points) > 1:
            draw.line(points, fill=COLORS[name], width=3)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=COLORS[name])


def create_explanation_card(
    output_path: Path,
    *,
    sample_id: str,
    predicted_annotation: str,
    predicted_intensity: float,
    probabilities: list[float],
    main_modality: str,
    contributions: dict[str, float],
    curves: dict[str, np.ndarray],
    raw_text: str,
    evidence_summary: list[str],
    keyframe_path: Path | None,
) -> None:
    canvas = Image.new("RGB", (1400, 1040), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38)
    heading_font = _font(27)
    body_font = _font(21)
    draw.text((45, 30), f"Sample {sample_id} - Multimodal Explanation", fill=(15, 15, 15), font=title_font)
    prediction = (
        f"Prediction: {predicted_annotation}   Intensity: {predicted_intensity:.3f}   "
        f"P(Neg/Neu/Pos): {probabilities[0]:.3f}/{probabilities[1]:.3f}/{probabilities[2]:.3f}"
    )
    draw.text((45, 85), prediction, fill=(30, 30, 30), font=body_font)
    draw.text((45, 125), f"Main modality: {main_modality.title()}", fill=COLORS[main_modality], font=heading_font)
    draw.text((45, 180), "Counterfactual modality contribution", fill=(20, 20, 20), font=heading_font)
    _draw_bars(draw, contributions, (45, 225, 720, 345))
    draw.text((45, 375), "Internal local-importance curves", fill=(20, 20, 20), font=heading_font)
    _draw_curves(draw, curves, (45, 420, 1350, 710))
    draw.text((45, 745), "Top counterfactual evidence", fill=(20, 20, 20), font=heading_font)
    for index, line in enumerate(evidence_summary[:5]):
        draw.text((55, 790 + index * 31), line[:105], fill=(35, 35, 35), font=body_font)
    draw.text((760, 180), "Visual keyframe", fill=(20, 20, 20), font=heading_font)
    if keyframe_path is not None and keyframe_path.exists():
        frame = Image.open(keyframe_path).convert("RGB")
        frame.thumbnail((570, 180), Image.Resampling.LANCZOS)
        canvas.paste(frame, (760, 225))
    text = " ".join(str(raw_text).split())
    draw.text((45, 965), f"Text: {text[:118]}", fill=(55, 55, 55), font=body_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
