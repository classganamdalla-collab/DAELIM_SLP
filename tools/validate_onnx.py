#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# CI smoke-test entrypoint; this file change triggers the ONNX workflow.
ROOT = Path(__file__).resolve().parents[1]
SOURCE_H5 = ROOT / "model" / "sign_language_model.h5"
FP32 = ROOT / "model" / "onnx" / "sign_language_fp32.onnx"
INT8 = ROOT / "model" / "onnx" / "sign_language_int8.onnx"

def make_inputs(count: int, seq_len: int, feat_dim: int, seed: int = 20261005):
    rng = np.random.default_rng(seed)
    xs = []
    for _ in range(count):
        valid = int(rng.integers(15, seq_len + 1))
        x = np.zeros((1, seq_len, feat_dim), dtype=np.float32)
        # Relative MediaPipe coordinates are typically small; this is only a
        # numerical parity/stress test, not a clinical/model-accuracy test.
        x[:, :valid, :] = rng.normal(0.0, 0.18, size=(1, valid, feat_dim)).astype(np.float32)
        xs.append(x)
    return xs

def timed_run(session: ort.InferenceSession, input_name: str, x: np.ndarray, warmup: int = 2):
    for _ in range(warmup):
        session.run(None, {input_name: x})
    t0 = time.perf_counter()
    y = session.run(None, {input_name: x})[0]
    return y, (time.perf_counter() - t0) * 1000.0

def pct(values, q):
    if not values:
        return None
    arr = np.sort(np.asarray(values, dtype=np.float64))
    idx = min(len(arr) - 1, max(0, int(math.ceil(q * len(arr))) - 1))
    return float(arr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--report", default=str(ROOT / "model" / "onnx" / "validation-report.json"))
    args = ap.parse_args()

    for p in (SOURCE_H5, FP32, INT8):
        if not p.exists():
            raise SystemExit(f"Missing model: {p}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    fp = ort.InferenceSession(str(FP32), sess_options=so, providers=["CPUExecutionProvider"])
    iq = ort.InferenceSession(str(INT8), sess_options=so, providers=["CPUExecutionProvider"])

    fp_in = fp.get_inputs()[0]
    iq_in = iq.get_inputs()[0]
    if fp_in.name != iq_in.name:
        print(f"Input names differ: fp32={fp_in.name}, int8={iq_in.name}")

    shape = fp_in.shape
    seq_len = 100 if len(shape) < 2 or not isinstance(shape[1], int) else shape[1]
    feat_dim = 90 if len(shape) < 3 or not isinstance(shape[2], int) else shape[2]

    xs = make_inputs(args.samples, seq_len, feat_dim)

    fp_times, iq_times = [], []
    fp_outputs = []
    abs_errs, max_errs = [], []
    argmax_match = 0

    # Warm up both sessions.
    for x in xs[: min(10, len(xs))]:
        fp.run(None, {fp_in.name: x})
        iq.run(None, {iq_in.name: x})

    for sample_idx, x in enumerate(xs):
        t0 = time.perf_counter()
        y_fp = fp.run(None, {fp_in.name: x})[0]
        fp_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        y_iq = iq.run(None, {iq_in.name: x})[0]
        iq_times.append((time.perf_counter() - t0) * 1000.0)

        fp_outputs.append(y_fp.astype(np.float32).copy())

        delta = np.abs(y_fp.astype(np.float32) - y_iq.astype(np.float32))
        abs_errs.append(float(delta.mean()))
        max_errs.append(float(delta.max()))
        argmax_match += int(np.argmax(y_fp) == np.argmax(y_iq))

    # Load TensorFlow/Keras only after ORT latency measurements. TensorFlow creates
    # its own CPU thread pools, which would otherwise distort the small-model timing.
    import os
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras

    source = keras.models.load_model(SOURCE_H5, compile=False, safe_mode=False)
    source_batch = np.concatenate(xs, axis=0)
    source_outputs = source.predict(source_batch, batch_size=32, verbose=0).astype(np.float32)

    src_abs_errs, src_max_errs = [], []
    source_argmax_match = 0
    for sample_idx, y_fp in enumerate(fp_outputs):
        y_src = source_outputs[sample_idx:sample_idx + 1]
        src_delta = np.abs(y_src - y_fp)
        src_abs_errs.append(float(src_delta.mean()))
        src_max_errs.append(float(src_delta.max()))
        source_argmax_match += int(np.argmax(y_src) == np.argmax(y_fp))

    report = {
        "purpose": "Numerical parity and runtime smoke test only; not recognition accuracy.",
        "samples": args.samples,
        "input_shape": [1, int(seq_len), int(feat_dim)],
        "fp32": {
            "bytes": FP32.stat().st_size,
            "latency_ms_mean": statistics.fmean(fp_times),
            "latency_ms_median": statistics.median(fp_times),
            "latency_ms_p95": pct(fp_times, 0.95),
        },
        "int8": {
            "bytes": INT8.stat().st_size,
            "latency_ms_mean": statistics.fmean(iq_times),
            "latency_ms_median": statistics.median(iq_times),
            "latency_ms_p95": pct(iq_times, 0.95),
        },
        "source_keras_vs_onnx_fp32": {
            "mean_abs_output_error": statistics.fmean(src_abs_errs),
            "max_abs_output_error": max(src_max_errs),
            "argmax_agreement_percent": 100.0 * source_argmax_match / args.samples,
        },
        "int8_vs_fp32": {
            "size_reduction_percent": 100.0 * (1.0 - INT8.stat().st_size / FP32.stat().st_size),
            "mean_abs_output_error": statistics.fmean(abs_errs),
            "max_abs_output_error": max(max_errs),
            "argmax_agreement_percent": 100.0 * argmax_match / args.samples,
        },
        "onnxruntime_version": ort.__version__,
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
