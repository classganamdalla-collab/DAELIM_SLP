#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H5_MODEL = ROOT / "model" / "sign_language_model.h5"
OUT_DIR = ROOT / "model" / "onnx"
FP32_MODEL = OUT_DIR / "sign_language_fp32.onnx"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not H5_MODEL.exists():
    raise SystemExit(f"Missing Keras model: {H5_MODEL}")

# The repository's TFJS model is a Keras "layers-model". tf2onnx explicitly
# supports TFJS graph-model conversion only, so converting model.json directly
# is not reliable for this project. Load the original .h5 with Keras 3,
# export a TensorFlow SavedModel, then convert that SavedModel to ONNX.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402

print(f"Keras: {keras.__version__}")
print(f"Loading: {H5_MODEL}")

model = keras.models.load_model(H5_MODEL, compile=False, safe_mode=False)

print("Loaded model:")
model.summary()

with tempfile.TemporaryDirectory(prefix="ieum_savedmodel_") as tmp:
    saved_model = Path(tmp) / "saved_model"

    # Keras 3 export path. The explicit input signature keeps the web model
    # contract stable: [batch, 100 frames, 90 features].
    input_signature = [
        keras.InputSpec(shape=(None, 100, 90), dtype="float32", name="input")
    ]

    try:
        model.export(
            saved_model,
            format="tf_saved_model",
            input_signature=input_signature,
        )
    except TypeError:
        # Compatibility fallback for Keras versions whose export() does not
        # accept input_signature.
        model.export(saved_model, format="tf_saved_model")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model),
        "--opset", "18",
        "--output", str(FP32_MODEL),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print(f"Created: {FP32_MODEL} ({FP32_MODEL.stat().st_size:,} bytes)")
