"""
Print per-feature and global stats for landmarks.csv (compare with live Logcat feature logs).

Usage:
  .venv\\Scripts\\python scripts/dataset_feature_stats.py --csv dataset/processed/landmarks.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = [c for c in df.columns if c.startswith("f")]
    X = df[cols].values.astype(np.float32)

    print(f"rows={len(X)}  features={X.shape[1]}")
    print(f"global min={X.min():.4f} max={X.max():.4f} mean={X.mean():.4f} std={X.std():.4f}")
    print("per-feature mean (first 12):", np.mean(X, axis=0)[:12])
    print("per-feature std  (first 12):", np.std(X, axis=0)[:12])


if __name__ == "__main__":
    main()
