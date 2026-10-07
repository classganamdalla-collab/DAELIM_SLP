#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic INT8 quantization for Ieum v2 ONNX.")
    ap.add_argument("--input", default="model/v2/best_model_fp32.onnx")
    ap.add_argument("--output", default="model/v2/best_model_int8.onnx")
    args = ap.parse_args()

    inp = ROOT / args.input
    out = ROOT / args.output
    if not inp.exists():
        raise SystemExit(f"Missing ONNX model: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(inp))
    op_types = sorted({node.op_type for node in model.graph.node})
    candidates = [op for op in ("LSTM", "GRU", "MatMul", "Gemm") if op in op_types]
    if not candidates:
        raise SystemExit(f"No recurrent/dense quantizable ops found. Observed={op_types}")

    print("Observed ops:", ", ".join(op_types))
    print("Quantizing:", candidates)

    try:
        quantize_dynamic(
            model_input=str(inp),
            model_output=str(out),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=candidates,
        )
        mode = "dynamic-int8-per-channel"
    except Exception as exc:
        print("Per-channel quantization failed; retrying per-tensor:", repr(exc))
        quantize_dynamic(
            model_input=str(inp),
            model_output=str(out),
            weight_type=QuantType.QInt8,
            per_channel=False,
            op_types_to_quantize=candidates,
        )
        mode = "dynamic-int8"

    print(f"Created: {out} ({out.stat().st_size:,} bytes), mode={mode}")


if __name__ == "__main__":
    main()
