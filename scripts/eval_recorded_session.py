"""
Evaluate TFLite on a CSV from a recorded session. Same schema as landmarks.csv (label, group, path, f0..f62).

Delegates to evaluate_tflite_on_csv.py — use the same CLI flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_tflite_on_csv import main

if __name__ == "__main__":
    main()
