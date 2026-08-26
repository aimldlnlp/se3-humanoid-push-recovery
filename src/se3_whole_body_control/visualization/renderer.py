"""Headless MuJoCo renderer with restrained research overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fonts import pil_font


PUSH_RGBA = np.array([0.84, 0.37, 0.08, 1.0], dtype=float)
COM_RGBA = np.array([0.0, 0.45, 0.70, 1.0], dtype=float)
CONTACT_RGBA = np.array([0.0, 0.62, 0.45, 1.0], dtype=float)
CONTACT_LOST_RGBA = np.array([0.75, 0.30, 0.25, 1.0], dtype=float)
TARGET_RGBA = np.array([0.0, 0.45, 0.70, 1.0], dtype=float)
EVENT_RGBA = np.array([0.80, 0.20, 0.50, 1.0], dtype=float)


def _font(size: int):
    return pil_font(size)


def _draw_overlay(image, metadata: Mapping[str, object] | None) -> None:
    """Draw compact essential telemetry without covering the robot."""
    if not metadata:
        return
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    scale = float(np.clip(image.width / 1920.0, 0.72, 1.0))
    margin = int(28 * scale)
    compact = bool(metadata.get("compact_overlay", False))
    title_font = pil_font(max(18, int((25 if compact else 30) * scale)), weight="bold")
    body_font = _font(max(15, int((18 if compact else 20) * scale)))
    controller = str(metadata.get("controller", "unknown"))
    qp_status = str(metadata.get("status", "unknown"))
    force = np.asarray(metadata.get("push_force", [0.0, 0.0]), dtype=float).reshape(-1)
    push_active = force.size >= 2 and float(np.linalg.norm(force[:2])) > 1e-9
    push_label = (
        f"Push  {float(metadata.get('push_magnitude_N', 0.0)):.0f} N @ "
        f"{float(metadata.get('push_direction_deg', 0.0)):.0f}°"
        if push_active else "Push  inactive"
    )
    lines = ([
        f"t = {float(metadata.get('time_s', 0.0)):.2f} s    QP: {qp_status}",
        push_label,
    ] if compact else [
        controller,
        f"t = {float(metadata.get('time_s', 0.0)):.2f} s    QP: {qp_status}",
        push_label,
    ])
    line_widths = [draw.textbbox((0, 0), line, font=title_font if index == 0 and not compact else body_font)[2] for index, line in enumerate(lines)]
    panel_width = max(int(240 * scale), max(line_widths, default=0) + int(36 * scale))
    panel_height = int((82 if compact else 124) * scale)
    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=int(10 * scale), fill=(255, 255, 255, 224), outline=(0, 0, 0, 180), width=max(1, int(scale)),
    )
    if compact:
        for index, line in enumerate(lines):
            draw.text((margin + int(18 * scale), margin + int(14 * scale) + index * int(26 * scale)), line, font=body_font, fill=(0, 0, 0, 255))
    else:
        draw.text((margin + int(20 * scale), margin + int(12 * scale)), lines[0], font=title_font, fill=(0, 0, 0, 255))
        for index, line in enumerate(lines[1:]):
            draw.text((margin + int(21 * scale), margin + int(50 * scale) + index * int(26 * scale)), line, font=body_font, fill=(0, 0, 0, 255))

    contacts = f"L  {'CONTACT' if metadata.get('contact_left') else 'LOST'}     R  {'CONTACT' if metadata.get('contact_right') else 'LOST'}"
    mode = str(metadata.get("control_mode", "double_support")).replace("_", " ").upper()
    phase = str(metadata.get("step_phase", "stance")).replace("_", " ")
    support_margin = metadata.get("support_margin_m")
    margin_text = f"Support margin  {float(support_margin):+.3f} m" if support_margin is not None and np.isfinite(float(support_margin)) else "Support margin  n/a"
    event = str(metadata.get("event_label") or "")
    event_text = f"Event  {event.replace('_', ' ')}" if event else ""
    status_lines = [contacts, f"Mode  {mode}   |   {phase}", margin_text]
    if event_text:
        status_lines.append(event_text)
    status_width = max(
        int(230 * scale),
        max(draw.textbbox((0, 0), line, font=body_font)[2] for line in status_lines) + int(30 * scale),
    )
    status_height = int((38 + 26 * len(status_lines)) * scale)
    y0 = image.height - margin - status_height
    draw.rounded_rectangle(
        (margin, y0, margin + status_width, y0 + status_height),
        radius=int(8 * scale), fill=(255, 255, 255, 218), outline=(0, 0, 0, 150), width=max(1, int(scale)),
    )
    for index, line in enumerate(status_lines):
        color = (128, 32, 80, 255) if line.startswith("Event") else (0, 0, 0, 255)
        draw.text((margin + int(16 * scale), y0 + int(9 * scale) + index * int(25 * scale)), line, font=body_font, fill=color)


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    z = np.asarray(direction, dtype=float)
    z /= max(float(np.linalg.norm(z)), 1e-12)
    reference = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(reference, z); x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z))


def _add_scene_geom(renderer, mujoco, geom_type, size, position, rotation, rgba) -> None:
    scene = renderer.scene
    if int(scene.ngeom) >= int(scene.maxgeom):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(position, dtype=float),
        np.asarray(rotation, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def _add_scene_annotations(renderer, mujoco, metadata: Mapping[str, object] | None) -> None:
    if not metadata:
        return
    com = np.asarray(metadata.get("com_world", []), dtype=float).reshape(-1)
    if com.size >= 3 and np.all(np.isfinite(com[:3])):
        # The projection is deliberately visible on the ground even when the
        # physical CoM lies inside the torso.  The vertical marker remains at
        # the measured 3-D CoM and is useful in close three-quarter views.
        _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.038, 0.0, 0.0], com[:3], np.eye(3), COM_RGBA)
        _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.028, 0.0, 0.0], [com[0], com[1], 0.035], np.eye(3), COM_RGBA)

    feet_xy = np.asarray(metadata.get("feet_xy", []), dtype=float).reshape(-1, 2) if metadata.get("feet_xy") is not None else np.empty((0, 2))
    for index, foot in enumerate(feet_xy[:2]):
        color = CONTACT_RGBA if bool(metadata.get("contact_left" if index == 0 else "contact_right")) else CONTACT_LOST_RGBA
        _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.026, 0.0, 0.0], [foot[0], foot[1], 0.055], np.eye(3), color)

    target = np.asarray(metadata.get("planned_foot_target_world", []), dtype=float).reshape(-1)
    if target.size >= 3 and np.all(np.isfinite(target[:3])):
        _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0.0, 0.0], target[:3], np.eye(3), TARGET_RGBA)
        _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.022, 0.0, 0.0], [target[0], target[1], 0.035], np.eye(3), EVENT_RGBA)

    force = np.asarray(metadata.get("push_force", []), dtype=float).reshape(-1)
    point = np.asarray(metadata.get("push_point_world", []), dtype=float).reshape(-1)
    if force.size >= 3 and point.size >= 3:
        horizontal = force[:3]
        magnitude = float(np.linalg.norm(horizontal))
        if magnitude > 1e-9:
            direction = horizontal / magnitude
            length = float(np.clip(0.0032 * magnitude, 0.20, 0.48))
            origin = point[:3] + 0.20 * direction
            center = origin + 0.5 * length * direction
            rotation = _rotation_from_z(direction)
            _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.022, 0.5 * length, 0.0], center, rotation, PUSH_RGBA)
            _add_scene_geom(renderer, mujoco, mujoco.mjtGeom.mjGEOM_SPHERE, [0.034, 0.0, 0.0], origin + length * direction, np.eye(3), PUSH_RGBA)


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
    step = max(1, stride)
    history = qpos_history[::step] if max_frames is None else qpos_history[::step][:max_frames]
    for i, qpos in enumerate(history):
        model.data.qpos[:] = qpos
        mujoco.mj_forward(model.model, model.data)
        renderer.update_scene(model.data, camera="track")
        overlay_index = i * step
        metadata = overlay_data[overlay_index] if overlay_data is not None and overlay_index < len(overlay_data) else None
        _add_scene_annotations(renderer, mujoco, metadata)
        pixels = renderer.render()
        from PIL import Image
        image = Image.fromarray(pixels)
        _draw_overlay(image, metadata)
        path = out / f"frame_{i:06d}.png"
        image.save(path)
        paths.append(path)
    renderer.close()
    return paths
