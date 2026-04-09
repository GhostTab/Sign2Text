import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmark_preprocessing import batch_augment
from geometric_features import concat_with_geometry
from calibration_utils import fit_temperature


def build_mlp(input_dim: int, num_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), name="landmarks")

    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = tf.keras.Model(inp, out, name="asl_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset/processed/landmarks.csv")
    ap.add_argument("--out_dir", required=True, help="models/saved_model/asl_mlp")
    ap.add_argument("--label_map", required=True, help="models/label_map.json")
    ap.add_argument(
        "--feature_norm",
        default="models/feature_norm.json",
        help="Where to save z-score mean/std; copy to Android assets for inference parity.",
    )
    ap.add_argument("--calibration_out", default="models/calibration.json", help="Temperature scaling output.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable training-time feature augmentation (noise/flip/scale).",
    )
    ap.add_argument(
        "--no_class_weight",
        action="store_true",
        help="Disable class-weighted loss (use if classes are balanced).",
    )
    ap.add_argument(
        "--no_geometric",
        action="store_true",
        help="Use 63-D only (no wrist-distance geometry). Default appends 20 distances -> 83-D.",
    )
    ap.add_argument(
        "--use_group_split",
        action="store_true",
        help="Use CSV column 'group' for GroupShuffleSplit (requires extract_landmarks with groups).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feature_cols = sorted([c for c in df.columns if c.startswith("f")], key=lambda x: int(x[1:]))
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 feature columns f0..f62, got {len(feature_cols)}")

    X63 = df[feature_cols].values.astype(np.float32)
    y_labels = df["label"].astype(str).values

    use_geo = not args.no_geometric
    if use_geo:
        X = concat_with_geometry(X63)
        input_dim = 83
    else:
        X = X63
        input_dim = 63

    le = LabelEncoder()
    y = le.fit_transform(y_labels).astype(np.int32)

    label_map_path = Path(args.label_map)
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({int(i): str(lbl) for i, lbl in enumerate(le.classes_)}, f, indent=2)

    groups = None
    if args.use_group_split and "group" in df.columns:
        groups = df["group"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, tmp_idx = next(gss.split(X, y, groups=groups))
        X_train, y_train = X[train_idx], y[train_idx]
        X_tmp, y_tmp = X[tmp_idx], y[tmp_idx]
        print(f"GroupShuffleSplit: train={len(X_train)} val+test={len(X_tmp)} groups.")
    else:
        if args.use_group_split:
            print("WARNING: --use_group_split but no 'group' column — using stratified split.")
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp
    )

    mean = np.mean(X_train, axis=0).astype(np.float32)
    std = np.std(X_train, axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))

    def zscore(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    X_train = zscore(X_train)
    X_val = zscore(X_val)
    X_test = zscore(X_test)

    feature_norm_path = Path(args.feature_norm)
    feature_norm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_norm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved feature z-score stats to: {feature_norm_path} (input_dim={input_dim})")

    if not args.no_augment:
        if use_geo:
            X_aug63_z = batch_augment(
                X_train[:, :63],
                args.seed + 1,
                flip_x_prob=0.5,
                noise_std=0.025,
                scale_range=(0.90, 1.10),
            )
            f63_raw = X_aug63_z * std[:63] + mean[:63]
            X_aug = concat_with_geometry(f63_raw)
            X_aug = zscore(X_aug)
        else:
            X_aug = batch_augment(
                X_train,
                args.seed + 1,
                flip_x_prob=0.5,
                noise_std=0.025,
                scale_range=(0.90, 1.10),
            )
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
        p = np.random.default_rng(args.seed + 2).permutation(len(X_train))
        X_train, y_train = X_train[p], y_train[p]

    num_classes = len(le.classes_)
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    tf.keras.utils.set_random_seed(args.seed)
    model = build_mlp(input_dim=input_dim, num_classes=num_classes)

    class_weight = None
    if not args.no_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    fit_kwargs = dict(
        x=X_train,
        y=y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(**fit_kwargs)

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} (loss={test_loss:.4f})")

    logits_test = model.predict(X_test, verbose=0)
    y_pred = np.argmax(logits_test, axis=1)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    logits_val = model.predict(X_val, verbose=0)
    T = fit_temperature(logits_val, y_val)
    print(f"Calibration temperature T={T:.4f} (val set)")

    cal_path = Path(args.calibration_out)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature": T,
                "output_is_logits": True,
                "input_dim": input_dim,
            },
            f,
            indent=2,
        )
    print(f"Wrote calibration to: {cal_path}")

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(out_dir))
    print(f"Exported SavedModel to: {out_dir}")
    print(f"Saved label map to: {label_map_path}")

    artifacts_dir = out_dir.parent / "training_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    meta = {
        "input_dim": input_dim,
        "num_classes": int(len(le.classes_)),
        "classes": [str(c) for c in le.classes_],
        "augment": not args.no_augment,
        "class_weight": not args.no_class_weight,
        "geometric_features": use_geo,
        "group_split": groups is not None,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "temperature": float(T),
        "zscore_feature_norm": str(feature_norm_path),
        "calibration": str(cal_path),
    }
    with open(artifacts_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
