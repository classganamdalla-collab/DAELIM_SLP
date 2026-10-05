#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
IN_MODEL = ROOT / "model" / "onnx" / "sign_language_fp32.onnx"
OUT_MODEL = ROOT / "model" / "onnx" / "sign_language_int8.onnx"

if not IN_MODEL.exists():
    raise SystemExit(f"Missing FP32 ONNX model: {IN_MODEL}")

model = onnx.load(str(IN_MODEL))
op_types = sorted({node.op_type for node in model.graph.node})
print("FP32 ONNX op types:", ", ".join(op_types))

# ONNX Runtime recommends dynamic quantization as the first PTQ choice for
# RNN/LSTM-style models. Explicitly include LSTM plus dense MatMul/Gemm ops.
# If the exported graph has decomposed recurrent cells instead of an LSTM op,
# MatMul/Gemm quantization still applies.
candidate_ops = [op for op in ("LSTM", "MatMul", "Gemm") if op in op_types]
if not candidate_ops:
    raise SystemExit(
        "No quantizable LSTM/MatMul/Gemm operations found. "
        f"Observed ops: {op_types}"
    )

print("Quantizing ops:", candidate_ops)

try:
    quantize_dynamic(
        model_input=str(IN_MODEL),
        model_output=str(OUT_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=candidate_ops,
    )
    mode = "dynamic-int8-per-channel"
except Exception as exc:
    print("Per-channel dynamic quantization failed; retrying per-tensor:")
    print(repr(exc))
    quantize_dynamic(
        model_input=str(IN_MODEL),
        model_output=str(OUT_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=False,
        op_types_to_quantize=candidate_ops,
    )
    mode = "dynamic-int8"

print(f"Created: {OUT_MODEL} ({OUT_MODEL.stat().st_size:,} bytes), mode={mode}")
