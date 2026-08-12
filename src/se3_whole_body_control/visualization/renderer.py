"""Headless MuJoCo renderer for demo frames."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_overlay(image, metadata: Mapping[str, object] | None) -> None:
    """Draw a compact, renderer-independent status overlay on a frame."""
    if not metadata:
        return
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    scale = max(1.0, image.width / 960.0)
    margin = int(18 * scale)
    font = _font(max(16, int(20 * scale)))
    small = _font(max(14, int(16 * scale)))
    lines = [
        f"t = {float(metadata.get('time_s', 0.0)):.2f} s",
        f"controller: {metadata.get('controller', 'unknown')}",
        f"push: {float(metadata.get('push_magnitude_N', 0.0)):.0f} N @ {float(metadata.get('push_direction_deg', 0.0)):.0f} deg",
        f"status: {metadata.get('status', 'unknown')}",
    ]
    line_height = int(28 * scale)
    panel_width = int(430 * scale)
    panel_height = margin * 2 + line_height * len(lines)
    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=int(10 * scale), fill=(0, 0, 0, 165), outline=(255, 255, 255, 180), width=max(1, int(scale)),
    )
    for i, line in enumerate(lines):
        draw.text((margin + int(14 * scale), margin + int(8 * scale) + i * line_height), line, font=font, fill=(255, 255, 255, 255))

    # A small top-down diagnostic inset gives the CoM and horizontal push a
    # visible marker without relying on renderer-specific world projection APIs.
    inset = int(170 * scale)
    x0 = image.width - inset - margin
    y0 = margin
    draw.rounded_rectangle(
        (x0, y0, x0 + inset, y0 + inset),
        radius=int(10 * scale), fill=(0, 0, 0, 165), outline=(255, 255, 255, 180), width=max(1, int(scale)),
    )
    center = (x0 + inset // 2, y0 + inset // 2)
    draw.ellipse((center[0] - int(7 * scale), center[1] - int(7 * scale), center[0] + int(7 * scale), center[1] + int(7 * scale)), fill=(50, 220, 100, 255))
    draw.text((x0 + int(10 * scale), y0 + int(10 * scale)), "CoM", font=small, fill=(50, 255, 100, 255))
    force = np.asarray(metadata.get("push_force", [0.0, 0.0]), dtype=float).reshape(-1)
    horizontal = force[:2] if force.size >= 2 else np.zeros(2)
    norm = float(np.linalg.norm(horizontal))
    if norm > 1e-9:
        direction = horizontal / norm
        length = int(48 * scale)
        end = (int(center[0] + direction[0] * length), int(center[1] - direction[1] * length))
        draw.line((center[0], center[1], end[0], end[1]), fill=(255, 170, 40, 255), width=max(2, int(5 * scale)))
        draw.ellipse((end[0] - int(5 * scale), end[1] - int(5 * scale), end[0] + int(5 * scale), end[1] + int(5 * scale)), fill=(255, 170, 40, 255))
    contacts = f"contacts: L={'OK' if metadata.get('contact_left') else 'LOST'}  R={'OK' if metadata.get('contact_right') else 'LOST'}"
    draw.text((x0 + int(10 * scale), y0 + inset - int(30 * scale)), contacts, font=small, fill=(255, 255, 255, 255))


def render_trial_frames(
    model,
    qpos_history,
    output_dir: str | Path,
    width: int = 960,
    height: int = 540,
    max_frames: int | None = None,
    stride: int = 1,
    overlay_data: Sequence[Mapping[str, object]] | None = None,
) -> list[Path]:
    import mujoco

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model.model, height=height, width=width)
    paths = []
    history = qpos_history[::max(1, stride)] if max_frames is None else qpos_history[::max(1, stride)][:max_frames]
    for i, qpos in enumerate(history):
        model.data.qpos[:] = qpos
        mujoco.mj_forward(model.model, model.data)
        renderer.update_scene(model.data, camera="track")
        path = out / f"frame_{i:06d}.png"
        pixels = renderer.render()
        from PIL import Image
        image = Image.fromarray(pixels)
        if overlay_data is not None:
            overlay_index = i * max(1, stride)
            if overlay_index < len(overlay_data):
                _draw_overlay(image, overlay_data[overlay_index])
        image.save(path)
        paths.append(path)
    renderer.close()
    return paths
