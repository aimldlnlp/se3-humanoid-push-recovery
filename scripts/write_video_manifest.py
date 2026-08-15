"""Record provenance and media properties for rendered experiment videos."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def media_info(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {"file": path.name, "ffprobe": "unavailable"}
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
    except Exception as exc:
        return {"file": path.name, "ffprobe_error": f"{type(exc).__name__}: {exc}"}
    stream = (payload.get("streams") or [{}])[0]
    return {
        "file": path.name,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "frame_rate": stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
        "duration_s": float((payload.get("format") or {}).get("duration", 0.0)),
    }


def main() -> None:
    source_sha = os.environ.get("SE3_SOURCE_VERSION", "")
    if len(source_sha) != 40:
        raise SystemExit("SE3_SOURCE_VERSION must be the full frozen commit SHA")
    run_ids = {
        "hero": "final-a0d5055-hero",
        "comparison": "final-a0d5055-comparison",
        "perturbed": "final-a0d5055-perturbed-video",
        "com_support": "final-a0d5055-com-support",
        "se3_geometry": "final-a0d5055-geometry",
    }
    files = sorted((ROOT / "results" / "videos").glob("*.mp4"))
    files += sorted((ROOT / "results" / "videos").glob("*.gif"))
    payload = {
        "run_id": "final-a0d5055-videos",
        "source_version": source_sha,
        "seed": 0,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": None,
        "video_run_ids": run_ids,
        "media": [media_info(path) for path in files],
    }
    sweep_manifest = ROOT / "results" / "logs" / "push_sweep_manifest.json"
    if sweep_manifest.exists():
        payload["config_sha256"] = json.loads(sweep_manifest.read_text(encoding="utf-8")).get("config_sha256")
    out = ROOT / "results" / "logs" / "video_manifest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
