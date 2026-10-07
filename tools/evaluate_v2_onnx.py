#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score

import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from dataset import load_dataset  # noqa: E402


def infer(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: X.astype(np.float32)})[0], dtype=np.float32)


def latency(session: ort.InferenceSession, X: np.ndarray, limit: int = 100) -> dict:
    name = session.get_inputs()[0].name
    xs = X[: min(limit, len(X))]
    if not len(xs):
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "samples": 0}
    for x in xs[: min(10, len(xs))]:
        session.run(None, {name: x[None].astype(np.float32)})
    times = []
    for x in xs:
        t0 = time.perf_counter()
        session.run(None, {name: x[None].astype(np.float32)})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "samples": len(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Ieum v2 ONNX FP32 vs INT8 on held-out test IDs.")
    ap.add_argument("--data", nargs="+", default=["data/raw/**/*.json"])
    ap.add_argument("--model-dir", default="model/v2")
    ap.add_argument("--out", default="model/v2/onnx_evaluation.json")
    args = ap.parse_args()

    model_dir = ROOT / args.model_dir
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((model_dir / "split_manifest.json").read_text(encoding="utf-8"))
    labels = [str(x) for x in meta["labels"]]
    test_ids = set(manifest["test"])

    bundle = load_dataset(
        args.data,
        max_len=int(meta.get("max_sequence_length", 100)),
        min_frames=15,
        allowed_labels=labels,
        feature_profile=str(meta.get("feature_profile") or "full"),
    )

    selected = [i for i, sid in enumerate(bundle.sample_ids) if sid in test_ids]
    missing = sorted(test_ids - {bundle.sample_ids[i] for i in selected})
    if missing:
        raise SystemExit(
            f"{len(missing)} test sample IDs are missing from the supplied raw dataset; "
            f"first missing={missing[:5]}"
        )
    if not selected:
        raise SystemExit("No held-out test samples were recovered from split_manifest.json.")

    X = bundle.X[selected]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_to_idx[bundle.y_labels[i]] for i in selected], dtype=np.int64)
    label_ids = np.arange(len(labels))

    keras_path = model_dir / "best_model.keras"
    fp_path = model_dir / "best_model_fp32.onnx"
    iq_path = model_dir / "best_model_int8.onnx"
    for path in (keras_path, fp_path, iq_path):
        if not path.exists():
            raise SystemExit(f"Missing model: {path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fp = ort.InferenceSession(str(fp_path), sess_options=opts, providers=["CPUExecutionProvider"])
    iq = ort.InferenceSession(str(iq_path), sess_options=opts, providers=["CPUExecutionProvider"])

    keras_model = keras.models.load_model(keras_path, compile=False)
    p_keras = np.asarray(keras_model.predict(X, verbose=0), dtype=np.float32)
    p_fp = infer(fp, X)
    p_iq = infer(iq, X)
    y_keras = p_keras.argmax(axis=1)
    y_fp = p_fp.argmax(axis=1)
    y_iq = p_iq.argmax(axis=1)

    def metrics(pred: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(
                f1_score(y, pred, labels=label_ids, average="macro", zero_division=0)
            ),
        }

    m_keras = metrics(y_keras)
    m_fp = metrics(y_fp)
    m_iq = metrics(y_iq)
    convert_diff = np.abs(p_keras - p_fp)
    diff = np.abs(p_fp - p_iq)

    accuracy_drop = m_fp["accuracy"] - m_iq["accuracy"]
    f1_drop = m_fp["macro_f1"] - m_iq["macro_f1"]
    agreement = float((y_fp == y_iq).mean())

    recommended = (
        accuracy_drop <= 0.01
        and f1_drop <= 0.01
        and agreement >= 0.99
    )

    report = {
        "purpose": "Held-out v2 test-set comparison of ONNX FP32 and Dynamic INT8.",
        "test_samples": int(len(y)),
        "labels": labels,
        "split_strategy": meta.get("split_strategy"),
        "dataset_fingerprint_sha256": bundle.fingerprint,
        "keras_reference": {
            **m_keras,
        },
        "keras_vs_onnx_fp32": {
            "accuracy_drop": float(m_keras["accuracy"] - m_fp["accuracy"]),
            "macro_f1_drop": float(m_keras["macro_f1"] - m_fp["macro_f1"]),
            "argmax_agreement": float((y_keras == y_fp).mean()),
            "mean_abs_output_error": float(convert_diff.mean()),
            "max_abs_output_error": float(convert_diff.max()),
        },
        "fp32": {
            **m_fp,
            "bytes": fp_path.stat().st_size,
            "latency_cpu": latency(fp, X),
        },
        "int8": {
            **m_iq,
            "bytes": iq_path.stat().st_size,
            "latency_cpu": latency(iq, X),
        },
        "int8_vs_fp32": {
            "accuracy_drop": float(accuracy_drop),
            "macro_f1_drop": float(f1_drop),
            "argmax_agreement": agreement,
            "mean_abs_output_error": float(diff.mean()),
            "max_abs_output_error": float(diff.max()),
            "size_reduction_percent": float(100.0 * (1.0 - iq_path.stat().st_size / fp_path.stat().st_size)),
        },
        "fp32_conversion_gate_pass": bool(
            (m_keras["accuracy"] - m_fp["accuracy"]) <= 0.005
            and (m_keras["macro_f1"] - m_fp["macro_f1"]) <= 0.005
            and float((y_keras == y_fp).mean()) >= 0.995
        ),
        "int8_accuracy_gate_pass": bool(recommended),
        "decision_note": (
            "Keras-to-ONNX conversion parity is reported separately. "
            "Passing the INT8 accuracy gate is necessary but not sufficient. "
            "Use browser benchmark results before selecting INT8 as the default web runtime."
        ),
        "onnxruntime_version": ort.__version__,
    }

    out = ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
