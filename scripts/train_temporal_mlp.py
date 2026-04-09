"""
Sequence classifier (GRU) on sliding windows of landmark features — prototype vs single-frame MLP.

Requires CSV from extract_landmarks.py with `group` and `path` columns. Windows use rows sorted by
path within each group; groups with mixed labels are skipped.

Usage:
  .venv\\Scripts\\python scripts/train_temporal_mlp.py --csv dataset/processed/landmarks.csv \\
      --out_dir models/saved_model/asl_temporal --label_map models/label_map_temporal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometric_features import concat_with_geometry


def build_gru(seq_len: int, input_dim: int, num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, input_dim), name="sequence")
    x = tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.0)(inp)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    model = tf.keras.Model(inp, out, name="asl_temporal_gru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True, help="models/saved_model/asl_temporal")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_geometric",
        action="store_true",
        help="Use 63-D frames only (default: 83-D with wrist distances).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "group" not in df.columns or "path" not in df.columns:
        raise SystemExit(
            "CSV must include 'group' and 'path' columns — re-run extract_landmarks.py with the updated script."
        )

    feat_cols = sorted([c for c in df.columns if c.startswith("f")], key=lambda x: int(x[1:]))
    if len(feat_cols) != 63:
        raise ValueError(f"Expected 63 feature columns, got {len(feat_cols)}")

    use_geo = not args.no_geometric
    seq_len = args.seq_len

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].astype(str).values)

    df = df.copy()
    df["_y"] = y_all

    for g, sub in df.groupby("group", sort=False):
        sub = sub.sort_values("path")
        if len(sub) < seq_len:
            continue
        if sub["label"].nunique() > 1:
            continue
        raw63 = sub[feat_cols].values.astype(np.float32)
        if use_geo:
            feats = concat_with_geometry(raw63)
        else:
            feats = raw63
        label_idx = int(sub["_y"].iloc[0])
        for i in range(len(sub) - seq_len + 1):
            X_list.append(feats[i : i + seq_len])
            y_list.append(label_idx)

    if len(X_list) < 50:
        raise SystemExit(
            f"Too few sequences ({len(X_list)}). Need more images per group or lower --seq_len."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )

    mean = np.mean(X_train, axis=(0, 1)).astype(np.float32)
    std = np.std(X_train, axis=(0, 1)).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    input_dim = X_train.shape[2]

    label_map_path = Path(args.label_map)
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({int(i): str(lbl) for i, lbl in enumerate(le.classes_)}, f, indent=2)

    tf.keras.utils.set_random_seed(args.seed)
    model = build_gru(seq_len, input_dim, num_classes=len(le.classes_))

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Temporal model test accuracy: {test_acc:.4f} (loss={test_loss:.4f})")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    meta_path = Path(args.out_dir).parent / "temporal_training_artifacts"
    meta_path.mkdir(parents=True, exist_ok=True)
    with open(meta_path / "temporal_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seq_len": seq_len,
                "input_dim": int(input_dim),
                "num_classes": int(len(le.classes_)),
                "geometric_features": use_geo,
                "test_accuracy": float(test_acc),
                "mean_shape": list(mean.shape),
            },
            f,
            indent=2,
        )

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(out_dir))
    print(f"Exported temporal SavedModel to: {out_dir}")
    print("Note: Android MVP still uses single-frame MLP; integrate GRU + state buffer separately.")


if __name__ == "__main__":
    main()
