#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TFJS_MODEL = ROOT / "model" / "tfjs_model" / "model.json"
OUT_DIR = ROOT / "model" / "onnx"
FP32_MODEL = OUT_DIR / "sign_language_fp32.onnx"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TFJS_MODEL.exists():
    raise SystemExit(f"Missing TFJS model: {TFJS_MODEL}")

cmd = [
    sys.executable, "-m", "tf2onnx.convert",
    "--tfjs", str(TFJS_MODEL),
    "--opset", "18",
    "--output", str(FP32_MODEL),
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Created: {FP32_MODEL} ({FP32_MODEL.stat().st_size:,} bytes)")
