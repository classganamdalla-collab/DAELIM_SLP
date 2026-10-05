#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import tempfile

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parents[1]
IN_MODEL = ROOT / "model" / "onnx" / "sign_language_fp32.onnx"
OUT_MODEL = ROOT / "model" / "onnx" / "sign_language_int8.onnx"

if not IN_MODEL.exists():
    raise SystemExit(f"Missing FP32 ONNX model: {IN_MODEL}")

model = onnx.load(str(IN_MODEL))
op_types = sorted({node.op_type for node in model.graph.node})
print("FP32 ONNX top-level op types:", ", ".join(op_types))

# This LSTM export contains recurrent computation inside Loop subgraphs.
# EnableSubgraph=True allows dynamic MatMul quantization to reach those cells.
candidate_ops = ["LSTM", "GRU", "MatMul", "Gemm"]

with tempfile.TemporaryDirectory(prefix="ieum_quant_") as tmp:
    preprocessed = Path(tmp) / "preprocessed.onnx"
    quant_pre_process(
        input_model=str(IN_MODEL),
        output_model_path=str(preprocessed),
        skip_symbolic_shape=True,
        skip_optimization=False,
        skip_onnx_shape=False,
    )

    pre_model = onnx.load(str(preprocessed))
    pre_ops = sorted({node.op_type for node in pre_model.graph.node})
    print("Preprocessed top-level op types:", ", ".join(pre_ops))
    print("Quantizing ops:", candidate_ops, "with subgraph quantization enabled")

    try:
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(OUT_MODEL),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=candidate_ops,
            extra_options={"EnableSubgraph": True},
        )
        mode = "dynamic-int8-per-channel-preprocessed-subgraphs"
    except Exception as exc:
        print("Per-channel dynamic quantization failed; retrying per-tensor:")
        print(repr(exc))
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(OUT_MODEL),
            weight_type=QuantType.QInt8,
            per_channel=False,
            op_types_to_quantize=candidate_ops,
            extra_options={"EnableSubgraph": True},
        )
        mode = "dynamic-int8-preprocessed-subgraphs"

print(f"Created: {OUT_MODEL} ({OUT_MODEL.stat().st_size:,} bytes), mode={mode}")
