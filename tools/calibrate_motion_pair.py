#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from features import extract_v2_frame  # noqa: E402


def separation_gain(sample: dict) -> float | None:
    values: list[float] = []
    for frame in sample.get("frames") or []:
        row = extract_v2_frame(frame, profile="base8")
        if row[187] < 0.5 or row[188] < 0.5:
            continue
        d = float(np.linalg.norm(row[138:141]))
        if math.isfinite(d):
            values.append(d)

    if len(values) < 6:
        return None

    window = max(3, min(10, int(len(values) * 0.22)))
    early = float(np.median(values[:window]))
    late = float(np.median(values[-window:]))
    return late - early


def balanced_accuracy(a: np.ndarray, b: np.ndarray, threshold: float, greater_is_b: bool) -> tuple[float, float, float]:
    if greater_is_b:
        a_ok = float(np.mean(a < threshold))
        b_ok = float(np.mean(b >= threshold))
    else:
        a_ok = float(np.mean(a > threshold))
        b_ok = float(np.mean(b <= threshold))
    return (a_ok + b_ok) / 2.0, a_ok, b_ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate a simple motion rule to separate two legacy base8 signs."
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--source-label", default="아메리카노")
    ap.add_argument("--target-label", default="영수증")
    ap.add_argument("--output", default="model/v2/motion_rules.json")
    args = ap.parse_args()

    payload = json.loads(Path(args.data).expanduser().read_text(encoding="utf-8"))
    groups = {args.source_label: [], args.target_label: []}

    for sample in payload.get("samples") or []:
        label = str(sample.get("label") or "")
        if label not in groups:
            continue
        g = separation_gain(sample)
        if g is not None:
            groups[label].append(g)

    a = np.asarray(groups[args.source_label], dtype=np.float64)
    b = np.asarray(groups[args.target_label], dtype=np.float64)
    if len(a) < 5 or len(b) < 5:
        raise SystemExit(
            f"Not enough usable two-hand samples: {args.source_label}={len(a)}, {args.target_label}={len(b)}"
        )

    candidates = sorted(set(a.tolist() + b.tolist()))
    mids = [(x + y) / 2.0 for x, y in zip(candidates, candidates[1:])]
    thresholds = [candidates[0] - 1e-6, *mids, candidates[-1] + 1e-6]

    best = None
    for greater_is_b in (True, False):
        for threshold in thresholds:
            bal, source_specificity, target_recall = balanced_accuracy(a, b, threshold, greater_is_b)
            # Tie-break toward preserving the source class (Americano) because
            # this rule is only meant to repair strong motion evidence.
            score = (bal, source_specificity, target_recall)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "threshold": float(threshold),
                    "greater_is_b": greater_is_b,
                    "balanced_accuracy": bal,
                    "source_specificity": source_specificity,
                    "target_recall": target_recall,
                }

    enabled = bool(best["balanced_accuracy"] >= 0.85 and best["source_specificity"] >= 0.85)

    report = {
        "format": "ieum-motion-pair-rule-v1",
        "enabled": enabled,
        "source_label": args.source_label,
        "target_label": args.target_label,
        "metric": "interhand_separation_gain",
        "direction": "greater_is_target" if best["greater_is_b"] else "less_is_target",
        "threshold": best["threshold"],
        "training_balanced_accuracy": best["balanced_accuracy"],
        "training_source_specificity": best["source_specificity"],
        "training_target_recall": best["target_recall"],
        "samples": {
            args.source_label: int(len(a)),
            args.target_label: int(len(b)),
        },
        "distribution": {
            args.source_label: {
                "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)),
            },
            args.target_label: {
                "median": float(np.median(b)),
                "p10": float(np.percentile(b, 10)),
                "p90": float(np.percentile(b, 90)),
            },
        },
        "note": (
            "Deterministic pair repair calibrated from existing training sequences. "
            "It only overrides source_label -> target_label when the motion threshold is met."
        ),
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCreated: {out}")


if __name__ == "__main__":
    main()
