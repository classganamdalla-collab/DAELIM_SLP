from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

from dataset import load_dataset, split_dataset
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
    rng = np.random.default_rng(seed)
    out = X.copy()
    coordinate_end = 173  # hand/geometry + derived face metrics; blendshape/presence remain untouched.
    noise = rng.normal(0.0, 0.004, size=out[:, :, :coordinate_end].shape).astype(np.float32)

    active = np.any(np.abs(out[:, :, :138]) > 1e-8, axis=2, keepdims=True)
    out[:, :, :coordinate_end] += noise * active

    # Small frame dropout inside non-padded regions to improve tolerance to missed MediaPipe frames.
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


def evaluate(model: keras.Model, X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "pred": pred,
        "probs": probs,
        "report": classification_report(
            y,
            pred,
            labels=np.arange(len(labels)),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_matrix(y, pred, labels=np.arange(len(labels))),
    }


def write_confusion(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, cm):
            writer.writerow([label, *map(int, row)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Ieum v2 two-hand/non-manual sequence model.")
    ap.add_argument("--data", nargs="+", default=["data/raw/**/*.json"])
    ap.add_argument("--out", default="model/v2")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--min-class-samples", type=int, default=10)
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

    keep_mask = np.asarray([label in set(kept_labels) for label in bundle.y_labels])
    keep_idx = np.where(keep_mask)[0]

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
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)

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

        hist_path = candidates_dir / f"{name}_history.json"
        hist_path.write_text(
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
    test = evaluate(best_model, X_test, y_test, kept_labels)

    best_model.save(out_dir / "best_model.keras")
    try:
        best_model.save(out_dir / "best_model.h5")
    except Exception as exc:
        print("Legacy H5 export skipped:", repr(exc))

    write_confusion(out_dir / "confusion_matrix.csv", test["confusion"], kept_labels)
    (out_dir / "classification_report.json").write_text(
        json.dumps(test["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_manifest = {
        name: [bundle.sample_ids[i] for i in idxs] for name, idxs in splits.items()
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        },
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
        "created_with": {
            "keras": keras.__version__,
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
