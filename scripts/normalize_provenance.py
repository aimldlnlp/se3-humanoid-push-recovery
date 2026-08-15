"""Normalize experiment metadata to the frozen full source commit.

This utility changes provenance fields only. Numeric CSV and NPZ measurement
arrays are loaded and written back unchanged; only ``manifest.source_version``
and the serialized metadata mirror are updated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _replace_manifest(payload: dict, old_prefix: str, source_sha: str) -> bool:
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict) and "source_version" in payload:
        manifest = payload
    if not isinstance(manifest, dict):
        return False
    if manifest.get("source_version") != old_prefix:
        return False
    manifest["source_version"] = source_sha
    return True


def _update_json(path: Path, old_prefix: str, source_sha: str) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = _replace_manifest(payload, old_prefix, source_sha)
    if changed:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return changed


def _update_npz(path: Path, old_prefix: str, source_sha: str) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "metadata_json" not in arrays:
        return False
    payload = json.loads(str(arrays["metadata_json"].item()))
    changed = _replace_manifest(payload, old_prefix, source_sha)
    if changed:
        arrays["metadata_json"] = np.asarray(json.dumps(payload, sort_keys=True))
        np.savez_compressed(path, **arrays)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="artifact root containing data/ and logs/")
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in args.source_sha):
        parser.error("--source-sha must be a 40-character hexadecimal commit SHA")

    root = args.root
    changed: list[str] = []
    for path in sorted((root / "data").glob("*.json")):
        if _update_json(path, args.old_prefix, args.source_sha):
            changed.append(path.as_posix())
    for path in sorted((root / "data").glob("*.npz")):
        if _update_npz(path, args.old_prefix, args.source_sha):
            changed.append(path.as_posix())
    for path in sorted((root / "logs").glob("*manifest.json")):
        if _update_json(path, args.old_prefix, args.source_sha):
            changed.append(path.as_posix())

    log_path = root / "logs" / "provenance_normalization.txt"
    log_path.write_text(
        "timestamp_utc=" + datetime.now(timezone.utc).isoformat() + "\n"
        f"old_source_prefix={args.old_prefix}\n"
        f"source_sha={args.source_sha}\n"
        "measurement_arrays_changed=false\n"
        "changed_metadata_files=" + str(len(changed)) + "\n"
        + "\n".join(changed)
        + "\n",
        encoding="utf-8",
    )
    print(f"normalized={len(changed)}")
    print(f"audit_log={log_path}")


if __name__ == "__main__":
    main()
