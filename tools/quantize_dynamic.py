#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic INT8 quantization for Ieum sequence models.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--no-per-channel", action="store_false", dest="per_channel")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise SystemExit(f"Missing ONNX model: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ieum_quant_") as tmp:
        prep = Path(tmp) / "preprocessed.onnx"
        quant_pre_process(
            input_model=str(src),
            output_model_path=str(prep),
            skip_symbolic_shape=True,
            skip_optimization=False,
            skip_onnx_shape=False,
        )

        model = onnx.load(str(prep))
        top_ops = sorted({node.op_type for node in model.graph.node})
        print("Top-level ops:", top_ops)

        quantize_dynamic(
            model_input=str(prep),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            op_types_to_quantize=["LSTM", "GRU", "MatMul", "Gemm"],
            extra_options={"EnableSubgraph": True},
        )

    print(f"Created {dst}: {dst.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
