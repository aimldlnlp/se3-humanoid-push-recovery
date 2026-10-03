"""Video and GIF encoding via imageio/FFmpeg."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np


def _ffmpeg_executable() -> str:
    """Return a system ffmpeg or the bundled imageio-ffmpeg binary."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def encode_video(frame_dir: str | Path, output_path: str | Path, fps: int = 30) -> Path:
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        _ffmpeg_executable(), "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(Path(frame_dir) / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ], check=True)
    return output


def make_gif(frame_dir: str | Path, output_path: str | Path, fps: int = 12, max_width: int = 960) -> Path:
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fps <= 0 or max_width < 0:
        raise ValueError("GIF fps must be positive and width nonnegative")
    # FFmpeg preserves timestamps at GIF's centisecond resolution. Pillow
    # plugins disagree on duration units and can silently write zero delays.
    scale = f"scale='min({max_width},iw)':-1:flags=lanczos," if max_width else ""
    subprocess.run([
        _ffmpeg_executable(), "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(Path(frame_dir) / "frame_%06d.png"),
        "-filter_complex", scale + "split[a][b];[a]palettegen[p];[b][p]paletteuse",
        "-loop", "0", str(output),
    ], check=True)
    return output
