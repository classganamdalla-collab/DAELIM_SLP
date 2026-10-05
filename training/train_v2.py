from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

from dataset import DatasetBundle, load_dataset, split_dataset
from features import V2_FEATURE_DIM, V2_SCHEMA_ID

ROOT = Path(__file__).resolve().parents[1]


def read_label_order(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(x["id"]) for x in payload.get("labels", [])]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def augment_sequences(X: np.ndarray, seed: int) -> np.ndarray:
    """Conservative feature-space augmentation for training data only."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    coordinate_end = 173  # hands + pair geometry + face geometry/derived metrics
    noise = rng.normal(0.0, 0.004, size=out[:, :, :coordinate_end].shape).astype(np.float32)

    # Do not add noise to padding-only frames.
    active = np.any(np.abs(out[:, :, :138]) > 1e-8, axis=2, keepdims=True)
    out[:, :, :coordinate_end] += noise * active

    # Small frame replacement inside non-padded regions improves tolerance to
    # occasional missed MediaPipe updates without changing sequence length.
    for i in range(len(out)):
        valid = np.where(np.any(np.abs(out[i]) > 1e-8, axis=1))[0]
        if len(valid) < 20:
            continue
        drop_n = max(1, int(len(valid) * 0.03))
        for j in rng.choice(valid, size=drop_n, replace=False):
            if j > 0:
                out[i, j] = out[i, j - 1]
    return out


def build_lstm(num_classes: int) -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(100, V2_FEATURE_DIM), name="input"),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.30),
        layers.LSTM(64),
        layers.Dropout(0.30),
        layers.Dense(48, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="ieum_v2_lstm")


def build_gru(num_classes: int) -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(100, V2_FEATURE_DIM), name="input"),
        layers.Masking(mask_value=0.0),
        layers.GRU(112, return_sequences=True),
        layers.Dropout(0.28),
        layers.GRU(56),
        layers.Dropout(0.28),
        layers.Dense(48, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="ieum_v2_gru")


def probability_summary(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    rows = np.arange(len(probs))
    scores = probs[rows, top1]
    if probs.shape[1] >= 2:
        top2 = order[:, -2]
        margins = scores - probs[rows, top2]
    else:
        margins = scores.copy()
    return top1, scores, margins


def evaluate(model: keras.Model, X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    probs = model.predict(X, verbose=0)
    pred, scores, margins = probability_summary(probs)
    label_ids = np.arange(len(labels))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=label_ids, average="weighted", zero_division=0)),
        "pred": pred,
        "probs": probs,
        "scores": scores,
        "margins": margins,
        "report": classification_report(
            y,
            pred,
            labels=label_ids,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_matrix(y, pred, labels=label_ids),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    samples: int = 1000,
) -> dict:
    if len(y_true) < 2 or samples <= 0:
        return {"estimate": float(metric(y_true, y_pred)), "low": None, "high": None, "samples": 0}

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        values.append(float(metric(y_true[idx], y_pred[idx])))

    return {
        "estimate": float(metric(y_true, y_pred)),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
        "samples": int(samples),
    }


def calibrate_rejection(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    *,
    target_accuracy: float = 0.95,
    target_unknown_far: float = 0.05,
) -> dict:
    top1, scores, margins = probability_summary(probs)
    correct = top1 == y_true

    unknown_idx = labels.index("기타") if "기타" in labels else None
    known_mask = np.ones(len(y_true), dtype=bool) if unknown_idx is None else y_true != unknown_idx
    unknown_mask = np.zeros(len(y_true), dtype=bool) if unknown_idx is None else y_true == unknown_idx

    feasible = []
    fallback = []

    for score_threshold in np.arange(0.50, 0.951, 0.025):
        for margin_threshold in np.arange(0.05, 0.401, 0.025):
            accepted = (scores >= score_threshold) & (margins >= margin_threshold)
            if unknown_idx is not None:
                accepted &= top1 != unknown_idx

            accepted_known = accepted & known_mask
            known_count = int(known_mask.sum())
            known_coverage = float(accepted_known.sum() / max(1, known_count))
            selective_accuracy = (
                float(correct[accepted_known].mean()) if accepted_known.any() else 0.0
            )

            unknown_far = (
                float((accepted & unknown_mask).sum() / max(1, int(unknown_mask.sum())))
                if unknown_mask.any() else None
            )

            item = {
                "min_score": round(float(score_threshold), 3),
                "min_margin": round(float(margin_threshold), 3),
                "known_coverage": known_coverage,
                "selective_accuracy_known": selective_accuracy,
                "accepted_known_samples": int(accepted_known.sum()),
                "known_validation_samples": known_count,
                "unknown_validation_samples": int(unknown_mask.sum()),
                "unknown_false_accept_rate": unknown_far,
            }

            safety_ok = unknown_far is None or unknown_far <= target_unknown_far
            utility = selective_accuracy * max(known_coverage, 1e-6)
            if unknown_far is not None:
                utility *= max(0.0, 1.0 - unknown_far)

            fallback.append((utility, item))
            if selective_accuracy >= target_accuracy and safety_ok and accepted_known.any():
                feasible.append((known_coverage, selective_accuracy, -(unknown_far or 0.0), item))

    if feasible:
        *_, best = max(feasible, key=lambda x: (x[0], x[1], x[2]))
        best["selection_rule"] = (
            f"maximum known coverage with selective accuracy >= {target_accuracy:.2f}"
            + (
                f" and unknown FAR <= {target_unknown_far:.2f}"
                if unknown_idx is not None else ""
            )
        )
    elif fallback:
        _, best = max(fallback, key=lambda x: x[0])
        best["selection_rule"] = "fallback: maximize accuracy × coverage × unknown-safety utility"
    else:
        best = {
            "min_score": 0.70,
            "min_margin": 0.20,
            "known_coverage": 0.0,
            "selective_accuracy_known": 0.0,
            "accepted_known_samples": 0,
            "known_validation_samples": int(known_mask.sum()),
            "unknown_validation_samples": int(unknown_mask.sum()),
            "unknown_false_accept_rate": None,
            "selection_rule": "fallback defaults; no accepted validation samples",
        }

    best["target_selective_accuracy"] = target_accuracy
    best["target_unknown_false_accept_rate"] = target_unknown_far if unknown_idx is not None else None

    # Compatibility aliases consumed by the browser/UI and older analysis code.
    best["coverage"] = best["known_coverage"]
    best["selective_accuracy"] = best["selective_accuracy_known"]
    best["accepted_samples"] = best["accepted_known_samples"]
    best["validation_samples"] = best["known_validation_samples"]
    return best


def rejection_test_metrics(
    evaluation: dict,
    y_true: np.ndarray,
    labels: list[str],
    calibration: dict,
) -> dict:
    pred = evaluation["pred"]
    scores = evaluation["scores"]
    margins = evaluation["margins"]
    threshold_score = float(calibration["min_score"])
    threshold_margin = float(calibration["min_margin"])

    accepted = (scores >= threshold_score) & (margins >= threshold_margin)
    unknown_idx = labels.index("기타") if "기타" in labels else None
    if unknown_idx is not None:
        accepted &= pred != unknown_idx

    correct = pred == y_true
    known_mask = np.ones(len(y_true), dtype=bool) if unknown_idx is None else y_true != unknown_idx
    unknown_mask = np.zeros(len(y_true), dtype=bool) if unknown_idx is None else y_true == unknown_idx
    accepted_known = accepted & known_mask

    result = {
        "known_coverage": float(accepted_known.sum() / max(1, int(known_mask.sum()))),
        "selective_accuracy_known": (
            float(correct[accepted_known].mean()) if accepted_known.any() else 0.0
        ),
        "accepted_known_samples": int(accepted_known.sum()),
        "known_test_samples": int(known_mask.sum()),
        "unknown_test_samples": int(unknown_mask.sum()),
    }
    if unknown_mask.any():
        far = float((accepted & unknown_mask).sum() / int(unknown_mask.sum()))
        result["unknown_false_accept_rate"] = far
        result["unknown_rejection_rate"] = 1.0 - far
    else:
        result["unknown_false_accept_rate"] = None
        result["unknown_rejection_rate"] = None
    return result


def write_confusion(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, cm):
            writer.writerow([label, *map(int, row)])


def write_predictions(
    path: Path,
    bundle: DatasetBundle,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    evaluation: dict,
    labels: list[str],
    calibration: dict,
) -> None:
    pred = evaluation["pred"]
    scores = evaluation["scores"]
    margins = evaluation["margins"]
    accepted = (scores >= float(calibration["min_score"])) & (
        margins >= float(calibration["min_margin"])
    )
    if "기타" in labels:
        accepted &= pred != labels.index("기타")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "source_file", "participant", "session", "true_label",
            "pred_label", "confidence", "margin", "correct", "accepted_message",
        ])
        for local_i, global_i in enumerate(test_idx):
            writer.writerow([
                bundle.sample_ids[global_i],
                bundle.source_files[global_i],
                bundle.participant_groups[global_i],
                bundle.session_groups[global_i],
                labels[int(y_true[local_i])],
                labels[int(pred[local_i])],
                f"{float(scores[local_i]):.8f}",
                f"{float(margins[local_i]):.8f}",
                int(pred[local_i] == y_true[local_i]),
                int(accepted[local_i]),
            ])


def group_metrics(
    bundle: DatasetBundle,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    evaluation: dict,
    labels: list[str],
) -> dict:
    pred = evaluation["pred"]
    out = {}
    test_groups = [bundle.participant_groups[i] for i in test_idx]
    label_ids = np.arange(len(labels))
    for group in sorted(set(test_groups)):
        mask = np.asarray([g == group for g in test_groups], dtype=bool)
        if not mask.any():
            continue
        out[group] = {
            "samples": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], pred[mask])),
            "macro_f1": float(
                f1_score(
                    y_true[mask],
                    pred[mask],
                    labels=label_ids,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Ieum v2 two-hand/non-manual sequence model.")
    ap.add_argument("--data", nargs="+", default=["data/raw/**/*.json"])
    ap.add_argument("--out", default="model/v2")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--min-class-samples", type=int, default=10)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    label_order = read_label_order(ROOT / "labels.json")
    bundle = load_dataset(
        args.data,
        max_len=100,
        min_frames=args.min_frames,
        allowed_labels=label_order,
    )

    counts = Counter(bundle.y_labels)
    kept_labels = [label for label in label_order if counts[label] >= args.min_class_samples]
    dropped = {label: counts[label] for label in label_order if 0 < counts[label] < args.min_class_samples}
    if len(kept_labels) < 2:
        raise SystemExit(
            f"Need at least two labels with >= {args.min_class_samples} samples. Counts={dict(counts)}"
        )

    keep_set = set(kept_labels)
    keep_idx = np.where(np.asarray([label in keep_set for label in bundle.y_labels]))[0]

    # Compact the bundle after rare-class filtering.
    bundle.X = bundle.X[keep_idx]
    bundle.y_labels = [bundle.y_labels[i] for i in keep_idx]
    bundle.participant_groups = [bundle.participant_groups[i] for i in keep_idx]
    bundle.session_groups = [bundle.session_groups[i] for i in keep_idx]
    bundle.sample_ids = [bundle.sample_ids[i] for i in keep_idx]
    bundle.source_files = [bundle.source_files[i] for i in keep_idx]

    splits, split_strategy = split_dataset(bundle, seed=args.seed)
    label_to_index = {label: i for i, label in enumerate(kept_labels)}
    y = np.asarray([label_to_index[x] for x in bundle.y_labels], dtype=np.int64)

    X_train, y_train = bundle.X[splits["train"]], y[splits["train"]]
    X_val, y_val = bundle.X[splits["val"]], y[splits["val"]]
    X_test, y_test = bundle.X[splits["test"]], y[splits["test"]]

    if not args.no_augment:
        X_aug = augment_sequences(X_train, args.seed + 9)
        original_y = y_train.copy()
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, original_y], axis=0)

    present_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(present_classes, weights)}

    builders = {"lstm": build_lstm, "gru": build_gru}
    results = {}
    best_name = None
    best_score = (-1.0, -1.0)

    for name, builder in builders.items():
        seed_everything(args.seed)
        model = builder(len(kept_labels))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, min_delta=1e-4
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        val = evaluate(model, X_val, y_val, kept_labels)
        candidate_path = candidates_dir / f"{name}.keras"
        model.save(candidate_path)

        (candidates_dir / f"{name}_history.json").write_text(
            json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2),
            encoding="utf-8",
        )

        results[name] = {
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
            "epochs_ran": len(history.history.get("loss", [])),
            "params": int(model.count_params()),
        }
        score = (val["macro_f1"], val["accuracy"])
        if score > best_score:
            best_score = score
            best_name = name

    assert best_name is not None
    best_model = keras.models.load_model(candidates_dir / f"{best_name}.keras", compile=False)
    best_val = evaluate(best_model, X_val, y_val, kept_labels)
    calibration = calibrate_rejection(
        best_val["probs"],
        y_val,
        kept_labels,
        target_accuracy=0.95,
        target_unknown_far=0.05,
    )
    test = evaluate(best_model, X_test, y_test, kept_labels)
    reject_test = rejection_test_metrics(test, y_test, kept_labels, calibration)

    best_model.save(out_dir / "best_model.keras")
    try:
        best_model.save(out_dir / "best_model.h5")
    except Exception as exc:
        print("Legacy H5 export skipped:", repr(exc))

    write_confusion(out_dir / "confusion_matrix.csv", test["confusion"], kept_labels)
    write_predictions(
        out_dir / "test_predictions.csv",
        bundle,
        splits["test"],
        y_test,
        test,
        kept_labels,
        calibration,
    )

    (out_dir / "classification_report.json").write_text(
        json.dumps(test["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "participant_test_metrics.json").write_text(
        json.dumps(
            group_metrics(bundle, splits["test"], y_test, test, kept_labels),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    split_manifest = {
        name: [bundle.sample_ids[i] for i in idxs] for name, idxs in splits.items()
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    label_ids = np.arange(len(kept_labels))
    ci = {
        "accuracy_95ci": bootstrap_ci(
            y_test,
            test["pred"],
            labels=label_ids,
            metric=lambda a, b: float(accuracy_score(a, b)),
            seed=args.seed + 501,
            samples=args.bootstrap_samples,
        ),
        "macro_f1_95ci": bootstrap_ci(
            y_test,
            test["pred"],
            labels=label_ids,
            metric=lambda a, b: float(
                f1_score(a, b, labels=label_ids, average="macro", zero_division=0)
            ),
            seed=args.seed + 502,
            samples=args.bootstrap_samples,
        ),
    }

    metrics = {
        "feature_schema": V2_SCHEMA_ID,
        "feature_dim": V2_FEATURE_DIM,
        "sequence_length": 100,
        "selected_model": best_name,
        "candidate_validation": results,
        "test": {
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "weighted_f1": test["weighted_f1"],
            "confidence_intervals": ci,
            "rejection": reject_test,
        },
        "rejection_calibration": calibration,
        "split_strategy": split_strategy,
        "labels": kept_labels,
        "class_counts_before_filter": dict(counts),
        "dropped_rare_classes": dropped,
        "dataset_samples": len(bundle.y_labels),
        "dataset_fingerprint_sha256": bundle.fingerprint,
        "skipped_samples": bundle.skipped,
        "seed": args.seed,
        "warning": (
            "If split_strategy is stratified-random-fallback, performance can be optimistic when samples "
            "from the same signer/session appear in multiple splits. Prefer participant-group evaluation."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 2,
        "feature_schema": V2_SCHEMA_ID,
        "feature_dim": V2_FEATURE_DIM,
        "max_sequence_length": 100,
        "labels": kept_labels,
        "num_classes": len(kept_labels),
        "selected_model": best_name,
        "split_strategy": split_strategy,
        "dataset_fingerprint_sha256": bundle.fingerprint,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_accuracy_95ci": ci["accuracy_95ci"],
        "test_macro_f1_95ci": ci["macro_f1_95ci"],
        "rejection": calibration,
        "created_with": {"keras": keras.__version__},
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
