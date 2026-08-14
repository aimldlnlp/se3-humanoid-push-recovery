"""Video and GIF encoding via imageio/FFmpeg."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np


def encode_video(frame_dir: str | Path, output_path: str | Path, fps: int = 30) -> Path:
    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(Path(frame_dir) / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ], check=True)
    return output


def make_gif(frame_dir: str | Path, output_path: str | Path, fps: int = 12, max_width: int = 960) -> Path:
    import imageio.v2 as imageio

    frames = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    images = []
    for frame in frames:
        image = Image.fromarray(imageio.imread(frame)).convert("RGB")
        if max_width and image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        images.append(np.asarray(image))
    imageio.mimsave(output, images, duration=1.0 / fps, loop=0, subrectangles=True)
    return output
