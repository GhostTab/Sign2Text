"""
Verify preprocessing self-consistency and (optional) TF SavedModel vs TFLite output parity.

Usage (from repo root):
  .venv\\Scripts\\python scripts/verify_pipeline_parity.py
  .venv\\Scripts\\python scripts/verify_pipeline_parity.py --csv dataset/processed/landmarks.csv --row 0
  .venv\\Scripts\\python scripts/verify_pipeline_parity.py --compare-tflite \\
      --saved_model models/saved_model/asl_mlp --tflite models/tflite/asl_mlp_fp32.tflite
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from landmark_preprocessing import normalize_landmarks_xyz, flip_x_components63
from geometric_features import wrist_distances_from_flat63


def _feat_checksum(x: np.ndarray) -> str:
    b = x.astype(np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def reconstruct_xyz_from_flat63(f: np.ndarray) -> np.ndarray:
    """Inverse of flatten for debugging (only valid if no handedness flip was applied)."""
    return f.reshape(21, 3).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Optional: landmarks.csv to dump row checksum")
    ap.add_argument("--row", type=int, default=0, help="Row index in CSV (excluding header)")
    ap.add_argument("--saved_model", default="", help="SavedModel dir for TF inference")
    ap.add_argument("--tflite", default="", help=".tflite path for TFLite inference")
    ap.add_argument(
        "--input_dim",
        type=int,
        default=83,
        help="Tensor size for TF vs TFLite compare (must match exported model).",
    )
    args = ap.parse_args()

    # Synthetic sanity: normalize random hand-like points
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(21, 3)).astype(np.float32)
    raw[0] = 0.5, 0.5, 0.0
    f = normalize_landmarks_xyz(raw)
    f2 = normalize_landmarks_xyz(raw)
    assert np.allclose(f, f2), "normalize must be deterministic"
    print("normalize_landmarks_xyz: deterministic OK")
    print("  checksum(synthetic):", _feat_checksum(f))

    # Flip twice = identity on x structure
    g = flip_x_components63(f)
    g2 = flip_x_components63(g)
    assert np.allclose(f, g2), "double flip_x should restore"
    print("flip_x_components63: involution OK")

    f_geo = wrist_distances_from_flat63(f)
    assert f_geo.shape == (20,), f_geo.shape
    print("wrist_distances_from_flat63: shape OK (20,)")

    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        cols = [c for c in df.columns if c.startswith("f")]
        row = df.iloc[args.row][cols].values.astype(np.float32)
        label = df.iloc[args.row]["label"]
        print(f"\nCSV row {args.row} label={label}")
        print("  checksum:", _feat_checksum(row))
        print("  f0..f5:", row[:6])

    if args.saved_model and args.tflite:
        import tensorflow as tf

        d = args.input_dim
        m = tf.saved_model.load(args.saved_model)
        infer = m.signatures["serving_default"]
        out_keys = list(infer.structured_outputs.keys())
        x = np.random.uniform(-1, 1, size=(1, d)).astype(np.float32)
        raw = infer(landmarks=tf.constant(x))
        out_tf = raw[out_keys[0]].numpy()

        interp = tf.lite.Interpreter(model_path=args.tflite)
        interp.allocate_tensors()
        tin, tout = interp.get_input_details()[0], interp.get_output_details()[0]
        interp.set_tensor(tin["index"], x)
        interp.invoke()
        out_tflite = interp.get_tensor(tout["index"])

        diff = np.max(np.abs(out_tf - out_tflite))
        print(f"\nTF vs TFLite max abs diff: {diff:.6f}")
        if diff > 1e-3:
            print("WARNING: large diff — check input/output dtypes and tensor names.")
        else:
            print("TF vs TFLite: OK (close)")


if __name__ == "__main__":
    main()
