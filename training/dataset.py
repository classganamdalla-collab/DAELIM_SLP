from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from features import V2_FEATURE_DIM, V2_SCHEMA_ID, pad_or_trim


@dataclass
class DatasetBundle:
    X: np.ndarray
    y_labels: list[str]
    participant_groups: list[str]
    session_groups: list[str]
    sample_ids: list[str]
    source_files: list[str]
    fingerprint: str
    skipped: list[dict]


def _expand_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for p in glob.glob(pattern, recursive=True):
            pp = Path(p)
            if pp.is_file() and pp.suffix.lower() == ".json":
                paths.add(pp.resolve())
    return sorted(paths)


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.name).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def load_dataset(
    patterns: Iterable[str],
    *,
    max_len: int = 100,
    min_frames: int = 15,
    allowed_labels: list[str] | None = None,
    feature_profile: str = "full",
) -> DatasetBundle:
    paths = _expand_patterns(patterns)
    if not paths:
        raise FileNotFoundError("No collector JSON files matched the supplied patterns.")

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    participants: list[str] = []
    sessions: list[str] = []
    ids: list[str] = []
    sources: list[str] = []
    skipped: list[dict] = []

    allow = set(allowed_labels) if allowed_labels else None

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        top_participant = str(payload.get("participantCode") or "").strip()
        top_session = str(payload.get("sessionCode") or "").strip()
        for idx, sample in enumerate(payload.get("samples") or []):
            label = str(sample.get("label") or "").strip()
            frames = sample.get("frames") or []
            sid = str(sample.get("id") or f"{path.name}:{idx}")

            if not label:
                skipped.append({"id": sid, "reason": "missing-label", "source": path.name})
                continue
            if allow is not None and label not in allow:
                skipped.append({"id": sid, "reason": "label-not-allowed", "label": label, "source": path.name})
                continue
            if len(frames) < min_frames:
                skipped.append({"id": sid, "reason": "too-short", "frames": len(frames), "source": path.name})
                continue

            seq, _ = pad_or_trim(frames, max_len=max_len, profile=feature_profile)
            sequences.append(seq)
            labels.append(label)

            participant = str(sample.get("participantCode") or top_participant or "").strip()
            session = str(sample.get("sessionCode") or top_session or "").strip()
            participants.append(participant or f"unknown-participant:{path.name}")
            sessions.append(session or f"unknown-session:{path.name}")
            ids.append(sid)
            sources.append(path.name)

    if not sequences:
        raise ValueError("No usable samples remained after validation.")

    X = np.stack(sequences).astype(np.float32)
    if X.shape[2] != V2_FEATURE_DIM:
        raise ValueError(f"Expected {V2_FEATURE_DIM} features ({V2_SCHEMA_ID}), got {X.shape}")

    return DatasetBundle(
        X=X,
        y_labels=labels,
        participant_groups=participants,
        session_groups=sessions,
        sample_ids=ids,
        source_files=sources,
        fingerprint=_fingerprint(paths),
        skipped=skipped,
    )


def _coverage(labels: np.ndarray, idx: np.ndarray, universe: set[str]) -> float:
    if len(idx) == 0:
        return 0.0
    return len(set(labels[idx]) & universe) / max(1, len(universe))


def _group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    universe = set(labels.tolist())
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        return None

    best = None
    best_score = -1.0
    indices = np.arange(len(labels))

    for offset in range(40):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + offset)
        train_idx, temp_idx = next(gss.split(indices, labels, groups))

        temp_groups = groups[temp_idx]
        if len(set(temp_groups.tolist())) < 2:
            continue

        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 100 + offset)
        rel_val, rel_test = next(inner.split(temp_idx, labels[temp_idx], temp_groups))
        val_idx, test_idx = temp_idx[rel_val], temp_idx[rel_test]

        train_coverage = _coverage(labels, train_idx, universe)
        if train_coverage < 1.0:
            # Never choose a grouped split that leaves a target class unseen in training.
            continue

        score = (
            2.0 * train_coverage
            + _coverage(labels, val_idx, universe)
            + _coverage(labels, test_idx, universe)
        )
        if score > best_score:
            best = (train_idx, val_idx, test_idx)
            best_score = score
        if score >= 4.0:
            break

    return best


def split_dataset(
    bundle: DatasetBundle,
    *,
    seed: int = 20261005,
) -> tuple[dict[str, np.ndarray], str]:
    labels = np.asarray(bundle.y_labels)
    participants = np.asarray(bundle.participant_groups)
    sessions = np.asarray(bundle.session_groups)

    for name, groups in (("participant-group", participants), ("session-group", sessions)):
        split = _group_split(labels, groups, seed=seed)
        if split is not None:
            train_idx, val_idx, test_idx = split
            return {"train": train_idx, "val": val_idx, "test": test_idx}, name

    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed + 1,
        stratify=labels[temp_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}, "stratified-random-fallback"
