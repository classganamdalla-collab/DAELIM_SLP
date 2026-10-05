#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
IN_MODEL = ROOT / "model" / "onnx" / "sign_language_fp32.onnx"
OUT_MODEL = ROOT / "model" / "onnx" / "sign_language_int8.onnx"

if not IN_MODEL.exists():
    raise SystemExit(f"Missing FP32 ONNX model: {IN_MODEL}")

# Dynamic INT8 is the first-line PTQ method recommended for RNN/LSTM-style models.
# Try per-channel weights first, then fall back to ordinary dynamic quantization
# if a converted recurrent subgraph is not compatible with per-channel quantization.
try:
    quantize_dynamic(
        model_input=str(IN_MODEL),
        model_output=str(OUT_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    mode = "dynamic-int8-per-channel"
except Exception as exc:
    print("Per-channel quantization failed, retrying without per-channel:")
    print(exc)
    quantize_dynamic(
        model_input=str(IN_MODEL),
        model_output=str(OUT_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    mode = "dynamic-int8"

print(f"Created: {OUT_MODEL} ({OUT_MODEL.stat().st_size:,} bytes), mode={mode}")
