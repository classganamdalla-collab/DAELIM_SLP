#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/v2/best_model.keras")
    ap.add_argument("--output", default="model/v2/best_model_fp32.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    model_path = ROOT / args.model
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    import tensorflow as tf
    import keras
    import tf2onnx

    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    input_shape = tuple(model.input_shape)
    if len(input_shape) != 3:
        raise SystemExit(f"Expected [batch, time, features], got {input_shape}")

    signature = [
        tf.TensorSpec(
            shape=(None, int(input_shape[1]), int(input_shape[2])),
            dtype=tf.float32,
            name="input",
        )
    ]

    @tf.function(input_signature=signature)
    def serving(x):
        return model(x, training=False)

    print(
        "Converting directly with tf2onnx.from_function "
        f"(shape=[None,{input_shape[1]},{input_shape[2]}], opset={args.opset})"
    )

    tf2onnx.convert.from_function(
        serving,
        input_signature=signature,
        opset=args.opset,
        output_path=str(output_path),
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise SystemExit("ONNX conversion did not create a valid output file.")

    print(f"Created: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
