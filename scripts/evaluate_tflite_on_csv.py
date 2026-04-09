"""
Per-class accuracy + confusion matrix using the exported TFLite model (matches Android runtime).

Supports 63- or 83-D features (from feature_norm.json input_dim), logits + optional calibration.json.

Usage:
  .venv\\Scripts\\python scripts/evaluate_tflite_on_csv.py --csv dataset/processed/landmarks.csv \\
      --tflite models/tflite/asl_mlp_fp32.tflite --label_map models/label_map.json \\
      --feature_norm models/feature_norm.json --calibration models/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tensorflow as tf
except ImportError as e:
    raise SystemExit("pip install tensorflow") from e

from calibration_utils import softmax
from geometric_features import concat_with_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--label_map", required=True, help="label_map.json from training")
    ap.add_argument(
        "--feature_norm",
        default="",
        help="feature_norm.json from training (required if model was trained with z-score).",
    )
    ap.add_argument(
        "--calibration",
        default="",
        help="calibration.json (temperature for logits); optional.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_fraction", type=float, default=0.1, help="Hold-out for eval (stratified)")
    args = ap.parse_args()

    with open(args.label_map, encoding="utf-8") as f:
        lm = json.load(f)
    idx_to_char = {int(k): v for k, v in lm.items()}
    num_classes = len(idx_to_char)

    df = pd.read_csv(args.csv)
    feat_cols = sorted([c for c in df.columns if c.startswith("f")], key=lambda x: int(x[1:]))
    if len(feat_cols) != 63:
        raise ValueError(f"Expected 63 landmark feature columns f0..f62, got {len(feat_cols)}")
    X63 = df[feat_cols].values.astype(np.float32)
    y_labels = df["label"].astype(str).values

    char_to_idx = {v: k for k, v in idx_to_char.items()}
    y = np.array([char_to_idx[c] for c in y_labels], dtype=np.int32)

    _, X_test, _, y_test = train_test_split(
        X63, y, test_size=args.test_fraction, random_state=args.seed, stratify=y
    )

    input_dim = 63
    temperature = 1.0
    if args.feature_norm:
        with open(args.feature_norm, encoding="utf-8") as f:
            fn = json.load(f)
        mean = np.array(fn["mean"], dtype=np.float32)
        std = np.maximum(np.array(fn["std"], dtype=np.float32), 1e-6)
        input_dim = int(fn.get("input_dim", len(mean)))
        if input_dim == 83:
            X_test = concat_with_geometry(X_test)
        elif input_dim != 63:
            raise ValueError(f"Unsupported input_dim {input_dim}")
        X_test = (X_test - mean) / std

    if args.calibration:
        with open(args.calibration, encoding="utf-8") as f:
            cal = json.load(f)
        temperature = float(cal.get("temperature", 1.0))

    interp = tf.lite.Interpreter(model_path=args.tflite)
    interp.allocate_tensors()
    tin = interp.get_input_details()[0]
    tout = interp.get_output_details()[0]
    expected_in = int(np.prod(tin["shape"][1:])) if len(tin["shape"]) > 1 else tin["shape"][-1]
    if expected_in != X_test.shape[1]:
        raise ValueError(
            f"TFLite expects input size {expected_in}, got feature vector width {X_test.shape[1]} "
            "(check feature_norm.json input_dim and --feature_norm)."
        )

    y_pred = []
    for i in range(len(X_test)):
        x = X_test[i : i + 1].astype(np.float32)
        interp.set_tensor(tin["index"], x)
        interp.invoke()
        logits = interp.get_tensor(tout["index"])[0].astype(np.float64)
        probs = softmax(logits / temperature)
        y_pred.append(int(np.argmax(probs)))
    y_pred = np.array(y_pred)

    labels_all = list(range(num_classes))
    print(
        classification_report(
            y_test, y_pred, labels=labels_all, digits=4, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=labels_all)
    print(cm)

    per_class = []
    for c in range(num_classes):
        mask = y_test == c
        if mask.sum() == 0:
            continue
        acc_c = (y_pred[mask] == c).mean()
        per_class.append((idx_to_char[c], float(acc_c), int(mask.sum())))
    per_class.sort(key=lambda t: t[1])
    print("\nWorst 5 classes by accuracy:")
    for name, acc, n in per_class[:5]:
        print(f"  {name}: {acc:.3f} (n={n})")


if __name__ == "__main__":
    main()
