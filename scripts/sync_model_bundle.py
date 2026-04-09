"""
Copy a single training/export bundle into Android assets so tflite, label_map, feature_norm,
and optional calibration always match.

Usage (from repo root):
  .venv\\Scripts\\python scripts/sync_model_bundle.py
  .venv\\Scripts\\python scripts/sync_model_bundle.py --android_dir mobile/android/app/src/main/assets
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy model bundle into Android assets.")
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    ap.add_argument(
        "--android_dir",
        type=Path,
        default=None,
        help="Target assets folder (default: <repo>/mobile/android/app/src/main/assets).",
    )
    ap.add_argument(
        "--tflite",
        type=Path,
        default=None,
        help="Source FP32 model (default: models/tflite/asl_mlp_fp32.tflite).",
    )
    ap.add_argument(
        "--label_map",
        type=Path,
        default=None,
        help="Source label_map.json (default: models/label_map.json).",
    )
    ap.add_argument(
        "--feature_norm",
        type=Path,
        default=None,
        help="Source feature_norm.json (default: models/feature_norm.json).",
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Source calibration.json (default: models/calibration.json if present).",
    )
    args = ap.parse_args()

    root = args.repo_root
    android = args.android_dir or (root / "mobile" / "android" / "app" / "src" / "main" / "assets")
    android.mkdir(parents=True, exist_ok=True)

    tflite_src = args.tflite or (root / "models" / "tflite" / "asl_mlp_fp32.tflite")
    lm_src = args.label_map or (root / "models" / "label_map.json")
    fn_src = args.feature_norm or (root / "models" / "feature_norm.json")

    bundle = [
        (tflite_src, android / "asl_mlp_fp32.tflite"),
        (lm_src, android / "label_map.json"),
        (fn_src, android / "feature_norm.json"),
    ]

    cal_src = args.calibration
    if cal_src is None:
        default_cal = root / "models" / "calibration.json"
        if default_cal.exists():
            cal_src = default_cal
    if cal_src is not None and Path(cal_src).exists():
        bundle.append((Path(cal_src), android / "calibration.json"))

    missing = [str(s) for s, _ in bundle if not Path(s).exists()]
    if missing:
        print("ERROR: Missing source file(s):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    for src, dst in bundle:
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    print(f"\nBundle synced to {android}")


if __name__ == "__main__":
    main()
