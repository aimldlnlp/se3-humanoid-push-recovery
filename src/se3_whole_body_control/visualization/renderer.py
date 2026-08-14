"""Headless MuJoCo renderer with restrained research overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PUSH_RGBA = np.array([0.84, 0.37, 0.08, 1.0], dtype=float)
COM_RGBA = np.array([0.0, 0.45, 0.70, 1.0], dtype=float)
CONTACT_RGBA = np.array([0.0, 0.62, 0.45, 1.0], dtype=float)
CONTACT_LOST_RGBA = np.array([0.75, 0.30, 0.25, 1.0], dtype=float)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_overlay(image, metadata: Mapping[str, object] | None) -> None:
    """Draw compact essential telemetry without covering the robot."""
    if not metadata:
        return
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    scale = float(np.clip(image.width / 1920.0, 0.72, 1.0))
    margin = int(28 * scale)
    title_font = _font(max(16, int(25 * scale)))
    body_font = _font(max(13, int(16 * scale)))
    controller = str(metadata.get("controller", "unknown"))
    qp_status = str(metadata.get("status", "unknown"))
    force = np.asarray(metadata.get("push_force", [0.0, 0.0]), dtype=float).reshape(-1)
    push_active = force.size >= 2 and float(np.linalg.norm(force[:2])) > 1e-9
    push_label = (
        f"Push  {float(metadata.get('push_magnitude_N', 0.0)):.0f} N @ "
        f"{float(metadata.get('push_direction_deg', 0.0)):.0f}°"
        if push_active else "Push  inactive"
    )
    lines = [
        controller,
        f"t = {float(metadata.get('time_s', 0.0)):.2f} s    QP: {qp_status}",
        push_label,
    ]
    panel_width = int(392 * scale)
    panel_height = int(104 * scale)
    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=int(10 * scale), fill=(255, 255, 255, 224), outline=(31, 41, 51, 180), width=max(1, int(scale)),
    )
    draw.text((margin + int(16 * scale), margin + int(10 * scale)), lines[0], font=title_font, fill=(31, 41, 51, 255))
    for index, line in enumerate(lines[1:]):
        draw.text((margin + int(17 * scale), margin + int(46 * scale) + index * int(23 * scale)), line, font=body_font, fill=(31, 41, 51, 255))

    contacts = f"L  {'CONTACT' if metadata.get('contact_left') else 'LOST'}     R  {'CONTACT' if metadata.get('contact_right') else 'LOST'}"
    status_width = int(250 * scale)
    status_height = int(32 * scale)
    y0 = image.height - margin - status_height
    draw.rounded_rectangle(
        (margin, y0, margin + status_width, y0 + status_height),
        radius=int(8 * scale), fill=(255, 255, 255, 218), outline=(31, 41, 51, 150), width=max(1, int(scale)),
    )
    draw.text((margin + int(12 * scale), y0 + int(7 * scale)), contacts, font=body_font, fill=(31, 41, 51, 255))


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
