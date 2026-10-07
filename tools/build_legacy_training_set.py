#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_UNKNOWN_LABELS = ["나", "사랑해", "만나서 반갑습니다"]


def load_json(path: Path) -> dict:
    print(f"Loading {path} ...")
    return json.loads(path.read_text(encoding="utf-8"))


def sample_id(sample: dict, fallback: str) -> str:
    return str(sample.get("id") or fallback)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a legacy-compatible Ieum cafe training JSON from existing 2.0-facemesh exports."
    )
    ap.add_argument("--cafe", required=True, help="Final cafe-order samples.json (recommended: 930-sample file).")
    ap.add_argument("--unknown", nargs="*", default=[], help="Older out-of-domain 2.0-facemesh JSON files.")
    ap.add_argument(
        "--unknown-labels",
        nargs="*",
        default=DEFAULT_UNKNOWN_LABELS,
        help="Labels to remap to 기타 for open-set examples.",
    )
    ap.add_argument("--unknown-per-label", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default="data/raw/legacy_cafe_base8.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cafe_path = Path(args.cafe).expanduser().resolve()
    cafe = load_json(cafe_path)

    if cafe.get("version") != "2.0-facemesh":
        print(f"Warning: expected version 2.0-facemesh, got {cafe.get('version')!r}")

    cafe_samples = [copy.deepcopy(x) for x in (cafe.get("samples") or [])]
    ids = set()
    deduped = []
    duplicate_ids = 0

    for i, sample in enumerate(cafe_samples):
        sid = sample_id(sample, f"cafe:{i}")
        if sid in ids:
            duplicate_ids += 1
            continue
        ids.add(sid)
        sample["id"] = sid
        sample.setdefault("sourceFile", cafe_path.name)
        deduped.append(sample)

    unknown_pool: dict[str, list[dict]] = defaultdict(list)
    allowed_unknown = set(args.unknown_labels)

    for source_name in args.unknown:
        path = Path(source_name).expanduser().resolve()
        payload = load_json(path)
        for i, raw in enumerate(payload.get("samples") or []):
            label = str(raw.get("label") or "").strip()
            if label not in allowed_unknown:
                continue
            sid = sample_id(raw, f"{path.name}:{i}")
            # Prefix source file to prevent accidental UUID collision with cafe data.
            merged_id = f"unknown:{path.stem}:{sid}"
            if merged_id in ids:
                continue
            sample = copy.deepcopy(raw)
            sample["id"] = merged_id
            sample["sourceLabel"] = label
            sample["label"] = "기타"
            sample["sourceFile"] = path.name
            unknown_pool[label].append(sample)

    selected_unknown = []
    unknown_summary = {}
    for label in args.unknown_labels:
        pool = unknown_pool.get(label, [])
        rng.shuffle(pool)
        chosen = pool[: max(0, args.unknown_per_label)]
        selected_unknown.extend(chosen)
        unknown_summary[label] = {
            "available": len(pool),
            "selected": len(chosen),
        }

    merged = deduped + selected_unknown
    counts = Counter(str(x.get("label") or "UNKNOWN") for x in merged)

    output = {
        "exportedAt": cafe.get("exportedAt"),
        "version": "2.1-legacy-base8-merged",
        "sourceVersion": cafe.get("version"),
        "featureProfile": "base8",
        "targetFeatureSchema": "ieum_v2_base8_190",
        "situation": cafe.get("situation") or "카페 주문",
        "totalSamples": len(merged),
        "labelsSummary": dict(counts),
        "legacyPreparation": {
            "cafeSource": str(cafe_path),
            "cafeOriginalSamples": len(cafe_samples),
            "duplicateCafeIdsRemoved": duplicate_ids,
            "unknownSources": [str(Path(x).expanduser().resolve()) for x in args.unknown],
            "unknownLabels": args.unknown_labels,
            "unknownPerLabelCap": args.unknown_per_label,
            "unknownSelection": unknown_summary,
            "seed": args.seed,
            "note": (
                "Out-of-domain signs are remapped to 기타 only for target-vocabulary rejection. "
                "They remain valid Korean Sign Language signs, not erroneous signs."
            ),
        },
        "samples": merged,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    print("\nCreated:", out)
    print("Total:", len(merged))
    print("Labels:")
    for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {count}")
    print("\nUse:")
    print(
        f'  python tools/run_v2_pipeline.py --data "{out}" --feature-profile base8'
    )


if __name__ == "__main__":
    main()
