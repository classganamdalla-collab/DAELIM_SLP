#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def paths_from(patterns):
    out = set()
    for pattern in patterns:
        for name in glob.glob(pattern, recursive=True):
            p = Path(name)
            if p.is_file() and p.suffix.lower() == ".json":
                out.add(p.resolve())
    return sorted(out)


def ratio(n, d):
    return 0.0 if d == 0 else n / d


def main():
    ap = argparse.ArgumentParser(description="Audit Ieum collector JSON before v2 training.")
    ap.add_argument("--data", nargs="+", default=["data/raw/**/*.json"])
    ap.add_argument("--out", default="data/dataset-audit.json")
    args = ap.parse_args()

    paths = paths_from(args.data)
    if not paths:
        raise SystemExit("No JSON files matched. Put local exports under data/raw/ or pass --data.")

    labels = Counter()
    participants = Counter()
    sessions = Counter()
    versions = Counter()
    lengths = []
    total_frames = 0
    two_hand_frames = 0
    face_frames = 0
    extended_face_frames = 0
    blendshape_frames = 0
    malformed = []
    per_label_quality = defaultdict(lambda: {"samples": 0, "frames": 0, "two": 0, "face": 0})

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            malformed.append({"file": path.name, "error": repr(exc)})
            continue

        versions[str(payload.get("version") or "unknown")] += 1
        top_p = str(payload.get("participantCode") or "").strip()
        top_s = str(payload.get("sessionCode") or "").strip()

        for i, sample in enumerate(payload.get("samples") or []):
            label = str(sample.get("label") or "").strip() or "(missing)"
            frames = sample.get("frames") or []
            p = str(sample.get("participantCode") or top_p or f"unknown:{path.name}").strip()
            s = str(sample.get("sessionCode") or top_s or f"unknown:{path.name}").strip()

            labels[label] += 1
            participants[p] += 1
            sessions[s] += 1
            lengths.append(len(frames))
            q = per_label_quality[label]
            q["samples"] += 1
            q["frames"] += len(frames)

            for frame in frames:
                total_frames += 1
                hands = frame.get("hands") or []
                if len(hands) >= 2:
                    two_hand_frames += 1
                    q["two"] += 1
                if frame.get("facePresent") or frame.get("face"):
                    face_frames += 1
                    q["face"] += 1
                if frame.get("faceExtended"):
                    extended_face_frames += 1
                if frame.get("faceBlendshapes"):
                    blendshape_frames += 1

    sample_count = sum(labels.values())
    class_counts = list(labels.values())
    report = {
        "files": len(paths),
        "malformed_files": malformed,
        "samples": sample_count,
        "labels": dict(labels),
        "participant_groups": len(participants),
        "participants": dict(participants),
        "session_groups": len(sessions),
        "sessions": dict(sessions),
        "collector_versions": dict(versions),
        "frame_length": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": statistics.fmean(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "class_balance": {
            "min_samples_per_label": min(class_counts) if class_counts else 0,
            "max_samples_per_label": max(class_counts) if class_counts else 0,
            "max_min_ratio": (
                max(class_counts) / max(1, min(class_counts)) if class_counts else 0
            ),
        },
        "frame_coverage": {
            "two_hand_ratio": ratio(two_hand_frames, total_frames),
            "face_ratio": ratio(face_frames, total_frames),
            "extended_face_ratio": ratio(extended_face_frames, total_frames),
            "blendshape_ratio": ratio(blendshape_frames, total_frames),
        },
        "per_label_quality": {
            label: {
                "samples": q["samples"],
                "two_hand_frame_ratio": ratio(q["two"], q["frames"]),
                "face_frame_ratio": ratio(q["face"], q["frames"]),
            }
            for label, q in sorted(per_label_quality.items())
        },
        "training_readiness": {
            "participant_group_split_possible": len(participants) >= 4 and not all(k.startswith("unknown:") for k in participants),
            "session_group_split_possible": len(sessions) >= 4,
            "all_labels_have_30_samples": bool(labels) and min(class_counts) >= 30,
            "all_labels_have_50_samples": bool(labels) and min(class_counts) >= 50,
        },
        "notes": [
            "Old collector JSON can still contribute hand/base-face landmarks, but extended face/blendshape features will be zero when absent.",
            "Participant-group evaluation is preferred; unknown/file-based groups are weaker evidence.",
            "This audit does not calculate recognition accuracy.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
