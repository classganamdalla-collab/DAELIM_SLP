from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from dataset import load_dataset, split_dataset  # noqa: E402


FACE_IDS = [4,10,13,14,33,61,105,133,145,152,159,234,263,291,334,362,374,386,454]


def frame(seed: int) -> dict:
    base = seed * 0.0001
    hands = []
    for handedness, offset in (("Left", 0.15), ("Right", 0.65)):
        hands.append({
            "handedness": handedness,
            "landmarks": [
                {"x": offset + i * 0.004 + base, "y": 0.45 - i * 0.003, "z": -i * 0.006}
                for i in range(21)
            ],
        })
    return {
        "hands": hands,
        "facePresent": True,
        "faceExtended": {
            str(idx): {"x": 0.38 + j * 0.011, "y": 0.26 + j * 0.006 + base, "z": -0.02}
            for j, idx in enumerate(FACE_IDS)
        },
        "faceBlendshapes": {"jawOpen": 0.12 + base, "browInnerUp": 0.08},
    }


def main() -> None:
    labels = ["안녕하세요", "아메리카노", "카드", "기타"]
    with tempfile.TemporaryDirectory(prefix="ieum_dataset_smoke_") as td:
        root = Path(td)
        for p_idx in range(5):
            participant = f"P{p_idx + 1:02d}"
            payload = {
                "version": "3.0-raw-landmarks",
                "participantCode": participant,
                "sessionCode": "S01",
                "samples": [],
            }
            for label_idx, label in enumerate(labels):
                for rep in range(4):
                    seed = 1000 * p_idx + 100 * label_idx + rep
                    payload["samples"].append({
                        "id": f"{participant}_{label}_{rep}",
                        "label": label,
                        "participantCode": participant,
                        "sessionCode": "S01",
                        "frames": [frame(seed + k) for k in range(20)],
                    })
            (root / f"{participant}_S01.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

        bundle = load_dataset([str(root / "*.json")], min_frames=15, allowed_labels=labels)
        assert bundle.X.shape == (5 * len(labels) * 4, 100, 190)
        splits, strategy = split_dataset(bundle, seed=20261005)
        assert strategy == "participant-group"

        group_sets = {}
        for name, idx in splits.items():
            assert len(idx) > 0
            group_sets[name] = {bundle.participant_groups[i] for i in idx}
        assert group_sets["train"].isdisjoint(group_sets["val"])
        assert group_sets["train"].isdisjoint(group_sets["test"])
        assert group_sets["val"].isdisjoint(group_sets["test"])

        train_labels = {bundle.y_labels[i] for i in splits["train"]}
        assert train_labels == set(labels)

        assert np.isfinite(bundle.X).all()
        print("Dataset grouped-split smoke test OK", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
