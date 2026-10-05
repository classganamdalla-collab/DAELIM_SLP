from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from features import V2_FEATURE_DIM, extract_v2_frame  # noqa: E402


def main() -> None:
    fixture = json.loads((ROOT / "tests" / "feature_fixture.json").read_text(encoding="utf-8"))
    py = extract_v2_frame(fixture)

    proc = subprocess.run(
        ["node", str(ROOT / "tests" / "feature_parity.mjs")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    js = np.asarray(json.loads(proc.stdout), dtype=np.float32)

    if py.shape != (V2_FEATURE_DIM,) or js.shape != (V2_FEATURE_DIM,):
        raise AssertionError(f"shape mismatch: python={py.shape}, js={js.shape}")

    max_abs = float(np.max(np.abs(py - js)))
    mean_abs = float(np.mean(np.abs(py - js)))
    if not np.allclose(py, js, atol=2e-6, rtol=2e-6):
        bad = np.where(~np.isclose(py, js, atol=2e-6, rtol=2e-6))[0][:10].tolist()
        raise AssertionError(
            f"JS/Python feature mismatch: max_abs={max_abs}, mean_abs={mean_abs}, indices={bad}"
        )

    print(f"JS/Python v2 feature parity OK: dim={V2_FEATURE_DIM}, max_abs={max_abs:.3g}")


if __name__ == "__main__":
    main()
