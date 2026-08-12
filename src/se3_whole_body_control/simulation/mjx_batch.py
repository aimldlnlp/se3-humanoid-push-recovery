"""Optional MJX capability probe.

The CPU MuJoCo controller remains the reference. This module deliberately does
not silently replace contact-constrained WBC with a different model when MJX
is unavailable or when its feature set is insufficient.
"""

from __future__ import annotations


def probe_mjx() -> dict:
    try:
        import jax
        devices = jax.devices()
        import mujoco.mjx  # noqa: F401
        return {
            "status": "available",
            "devices": [str(device) for device in devices],
            "gpu_devices": [str(device) for device in devices if device.platform == "gpu"],
            "validated_batched_wbc": False,
            "reason": "MJX import succeeded; batched contact-constrained WBC still requires validation",
        }
    except Exception as exc:
        return {"status": "unavailable", "devices": [], "gpu_devices": [], "validated_batched_wbc": False, "reason": f"{type(exc).__name__}: {exc}"}
