#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
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
    import keras

    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    input_shape = tuple(model.input_shape)
    if len(input_shape) != 3:
        raise SystemExit(f"Expected [batch, time, features], got {input_shape}")

    with tempfile.TemporaryDirectory(prefix="ieum_v2_savedmodel_") as tmp:
        saved = Path(tmp) / "saved_model"
        spec = [
            keras.InputSpec(
                shape=(None, int(input_shape[1]), int(input_shape[2])),
                dtype="float32",
                name="input",
            )
        ]
        try:
            model.export(saved, format="tf_saved_model", input_signature=spec)
        except TypeError:
            model.export(saved, format="tf_saved_model")

        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", str(saved),
            "--opset", str(args.opset),
            "--output", str(output_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Created: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
