#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("\n>>>", " ".join(map(str, args)))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command Ieum v2 audit/train/export/quantization pipeline.")
    ap.add_argument("--data", nargs="+", default=["data/raw/**/*.json"])
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-class-samples", type=int, default=10)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    patterns = args.data

    run([py, "tools/audit_dataset.py", "--data", *patterns])
    train = [
        py, "training/train_v2.py",
        "--data", *patterns,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--min-class-samples", str(args.min_class_samples),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--seed", str(args.seed),
    ]
    if args.no_augment:
        train.append("--no-augment")
    run(train)

    run([py, "tools/export_v2_onnx.py"])
    run([py, "tools/quantize_v2_onnx.py"])
    run([py, "tools/evaluate_v2_onnx.py", "--data", *patterns])

    print("\nIeum v2 pipeline complete.")
    print("Review model/v2/metrics.json and model/v2/onnx_evaluation.json before deployment.")


if __name__ == "__main__":
    main()
