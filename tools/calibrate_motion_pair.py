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

METRICS = (
    "interhand_separation_gain",
    "wrist_downward_gain",
    "hand_openness_early",
    "hand_openness_mean",
    "interhand_separation_early",
)


def _edge_stats(values: list[float]) -> dict | None:
    if len(values) < 6:
        return None
    window = max(3, min(10, int(len(values) * 0.22)))
    early = float(np.median(values[:window]))
    late = float(np.median(values[-window:]))
    return {
        "early": early,
        "late": late,
        "gain": late - early,
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def _hand_openness(row: np.ndarray, offset: int) -> float:
    ds = []
    for lm in (4, 8, 12, 16, 20):
        i = offset + lm * 3
        ds.append(float(np.linalg.norm(row[i:i+3])))
    return float(np.mean(ds))


def metrics_for_sample(sample: dict) -> dict[str, float]:
    sep: list[float] = []
    wrist_y: list[float] = []
    openness: list[float] = []

    for frame in sample.get("frames") or []:
        row = np.asarray(extract_v2_frame(frame, profile="base8"), dtype=np.float32)
        lp, rp = row[187] >= 0.5, row[188] >= 0.5

        if lp and rp:
            d = float(np.linalg.norm(row[138:141]))
            if math.isfinite(d):
                sep.append(d)

        ys = []
        ops = []
        if lp:
            ys.append(float(row[67]))
            ops.append(_hand_openness(row, 0))
        if rp:
            ys.append(float(row[136]))
            ops.append(_hand_openness(row, 69))

        if ys and all(math.isfinite(v) for v in ys):
            wrist_y.append(float(np.mean(ys)))
        if ops and all(math.isfinite(v) for v in ops):
            openness.append(float(np.mean(ops)))

    out: dict[str, float] = {}
    s = _edge_stats(sep)
    if s:
        out["interhand_separation_gain"] = s["gain"]
        out["interhand_separation_early"] = s["early"]

    w = _edge_stats(wrist_y)
    if w:
        out["wrist_downward_gain"] = w["gain"]

    o = _edge_stats(openness)
    if o:
        out["hand_openness_early"] = o["early"]
        out["hand_openness_mean"] = o["mean"]

    return out


def balanced_accuracy(a: np.ndarray, b: np.ndarray, threshold: float, greater_is_b: bool) -> tuple[float, float, float]:
    if greater_is_b:
        a_ok = float(np.mean(a < threshold))
        b_ok = float(np.mean(b >= threshold))
    else:
        a_ok = float(np.mean(a > threshold))
        b_ok = float(np.mean(b <= threshold))
    return (a_ok + b_ok) / 2.0, a_ok, b_ok


def fit_metric(a: np.ndarray, b: np.ndarray) -> dict:
    candidates = sorted(set(a.tolist() + b.tolist()))
    mids = [(x + y) / 2.0 for x, y in zip(candidates, candidates[1:])]
    thresholds = [candidates[0] - 1e-6, *mids, candidates[-1] + 1e-6]

    best = None
    for greater_is_b in (True, False):
        for threshold in thresholds:
            bal, source_specificity, target_recall = balanced_accuracy(a, b, threshold, greater_is_b)
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
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate a simple scalar motion/hand-shape rule between two base8 signs."
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--source-label", default="아메리카노")
    ap.add_argument("--target-label", default="영수증")
    ap.add_argument("--metric", choices=("auto", *METRICS), default="auto")
    ap.add_argument("--output", default="model/v2/motion_rules.json")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append the calibrated rule to an existing motion_rules.json instead of replacing it.",
    )
    args = ap.parse_args()

    payload = json.loads(Path(args.data).expanduser().read_text(encoding="utf-8"))
    samples = {args.source_label: [], args.target_label: []}

    for sample in payload.get("samples") or []:
        label = str(sample.get("label") or "")
        if label not in samples:
            continue
        samples[label].append(metrics_for_sample(sample))

    metric_names = METRICS if args.metric == "auto" else (args.metric,)
    ranked = []

    for metric in metric_names:
        a = np.asarray([x[metric] for x in samples[args.source_label] if metric in x], dtype=np.float64)
        b = np.asarray([x[metric] for x in samples[args.target_label] if metric in x], dtype=np.float64)
        if len(a) < 5 or len(b) < 5:
            continue

        best = fit_metric(a, b)
        ranked.append({
            "metric": metric,
            "a": a,
            "b": b,
            **best,
        })

    if not ranked:
        raise SystemExit("No metric had enough usable samples.")

    ranked.sort(
        key=lambda x: (x["balanced_accuracy"], x["source_specificity"], x["target_recall"]),
        reverse=True,
    )
    best = ranked[0]
    a, b = best["a"], best["b"]

    enabled = bool(best["balanced_accuracy"] >= 0.85 and best["source_specificity"] >= 0.85)

    rule = {
        "enabled": enabled,
        "source_label": args.source_label,
        "target_label": args.target_label,
        "metric": best["metric"],
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
    }

    report = {
        "format": "ieum-motion-pair-rule-v2",
        "rule": rule,
        "candidates": [
            {
                "metric": x["metric"],
                "balanced_accuracy": x["balanced_accuracy"],
                "source_specificity": x["source_specificity"],
                "target_recall": x["target_recall"],
            }
            for x in ranked
        ],
        "note": (
            "Deterministic repair calibrated from existing base8 sequences. "
            "Auto mode compares simple trajectory/hand-shape metrics and chooses the best one."
        ),
    }

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        old_rules = existing.get("rules") if isinstance(existing, dict) else None
        if not isinstance(old_rules, list):
            if isinstance(existing, dict) and "source_label" in existing:
                old_rules = [existing]
            elif isinstance(existing, dict) and isinstance(existing.get("rule"), dict):
                old_rules = [existing["rule"]]
            else:
                old_rules = []

        # Replace an existing rule for the same pair, otherwise append.
        filtered = [
            r for r in old_rules
            if not (
                r.get("source_label") == args.source_label
                and r.get("target_label") == args.target_label
            )
        ]
        filtered.append(rule)
        payload_out = {
            "format": "ieum-motion-rules-v2",
            "rules": filtered,
        }
    else:
        payload_out = report

    out.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCreated: {out}")


if __name__ == "__main__":
    main()
