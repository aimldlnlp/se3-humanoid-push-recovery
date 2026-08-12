"""Collect local environment facts without modifying source files."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> str:
    try:
        output = subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        return "\n".join(line.rstrip() for line in output.splitlines())
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    names = ["numpy", "scipy", "mujoco", "osqp", "cvxpy", "jax", "matplotlib", "PIL", "yaml", "imageio"]
    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"executable_name={Path(sys.executable).name}",
        f"git={command(['git', '--version'])}",
        f"ffmpeg={command(['ffmpeg', '-version']).splitlines()[0] if shutil.which('ffmpeg') else 'unavailable'}",
        f"nvidia_smi={command(['nvidia-smi']) if shutil.which('nvidia-smi') else 'unavailable'}",
    ]
    for name in names:
        spec = importlib.util.find_spec(name)
        version = ""
        if spec:
            try:
                module = __import__(name)
                version = getattr(module, "__version__", "")
            except Exception as exc:
                version = f"import_error={type(exc).__name__}: {exc}"
        lines.append(f"package.{name}={'yes' if spec else 'no'};version={version}")
    path = ROOT / "results" / "logs" / "environment.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
