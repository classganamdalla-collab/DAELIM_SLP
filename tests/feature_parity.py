from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from features import FACE_BASE_POINTS, V2_FEATURE_DIM, extract_v2_frame  # noqa: E402


def compare(name: str, py: np.ndarray, js: np.ndarray) -> None:
    if py.shape != (V2_FEATURE_DIM,) or js.shape != (V2_FEATURE_DIM,):
        raise AssertionError(f"{name}: shape mismatch python={py.shape}, js={js.shape}")
    max_abs = float(np.max(np.abs(py - js)))
    mean_abs = float(np.mean(np.abs(py - js)))
    if not np.allclose(py, js, atol=2e-6, rtol=2e-6):
        bad = np.where(~np.isclose(py, js, atol=2e-6, rtol=2e-6))[0][:10].tolist()
        raise AssertionError(
            f"{name}: JS/Python mismatch max_abs={max_abs}, mean_abs={mean_abs}, indices={bad}"
        )
    print(f"{name} parity OK: dim={V2_FEATURE_DIM}, max_abs={max_abs:.3g}")


def main() -> None:
    fixture = json.loads((ROOT / "tests" / "feature_fixture.json").read_text(encoding="utf-8"))
    legacy = {
        "hands": fixture["hands"],
        "facePresent": True,
        "face": [fixture["faceExtended"][str(idx)] for idx in FACE_BASE_POINTS],
    }

    proc = subprocess.run(
        ["node", str(ROOT / "tests" / "feature_parity.mjs")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    js_payload = json.loads(proc.stdout)

    compare(
        "full",
        extract_v2_frame(fixture, profile="full"),
        np.asarray(js_payload["full"], dtype=np.float32),
    )
    compare(
        "base8",
        extract_v2_frame(legacy, profile="base8"),
        np.asarray(js_payload["base8"], dtype=np.float32),
    )

    # Legacy profile must keep extended/blendshape-only slots disabled.
    base8 = extract_v2_frame(legacy, profile="base8")
    # Face block starts after 138 hand + 3 inter-hand = 141.
    derived_start = 141 + 24
    # First six derived features require extended points; blendshapes are all disabled.
    assert np.allclose(base8[derived_start:derived_start + 6], 0.0)
    blend_start = derived_start + 8
    assert np.allclose(base8[blend_start:blend_start + 14], 0.0)
    print("legacy base8 unsupported face features correctly zeroed")


if __name__ == "__main__":
    main()
