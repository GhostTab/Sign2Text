# Sign2Text (ASL A–Z) — MediaPipe Hands + MLP + TFLite

Real-time Sign Language (A–Z) to text pipeline:

Camera → MediaPipe Hand Landmarker → landmarks → **63-D (+ optional 20 geometry) → z-score** → MLP (logits) → **temperature softmax** → Letter UI

## Repository layout

```
sign2text/
  dataset/
    raw/asl_alphabet/      # A/ … Z/ (optionally A/subject1/ … for grouped splits)
    processed/landmarks.csv
  scripts/
    extract_landmarks.py   # images -> CSV (label, group, path, f0..f62)
    train_mlp.py           # SavedModel + label_map + feature_norm + calibration
    train_temporal_mlp.py  # optional GRU on sliding windows (same CSV with group/path)
    export_tflite.py       # SavedModel -> .tflite (fp32 + int8)
    sync_model_bundle.py   # copy one consistent bundle into Android assets
    evaluate_tflite_on_csv.py
    eval_recorded_session.py
    geometric_features.py
  models/
    saved_model/asl_mlp/
    tflite/
    label_map.json
    feature_norm.json      # includes input_dim (63 or 83)
    calibration.json       # temperature for logits (optional)
  mobile/android/
```

## 1) Setup (Python)

From `sign2text/`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Dataset

Place an ASL alphabet dataset (folders `A`…`Z`) under e.g. `dataset/raw/asl_alphabet/`.

For **subject-level splits** when training, use nested folders so `group` differs within a class, e.g. `dataset/raw/asl_alphabet/A/person01/img.jpg`.

## 3) Extract landmarks (CSV)

```bash
python scripts/extract_landmarks.py ^
  --input_dir dataset/raw/asl_alphabet ^
  --output_csv dataset/processed/landmarks.csv ^
  --max_per_class 500 ^
  --hand_task_path path/to/hand_landmarker.task
```

CSV columns: `label`, `group`, `path`, `f0`…`f62`. Use `train_mlp.py --use_group_split` to split by `group` (reduces leakage).

## 4) Train MLP

Default **83-D** input (63 landmarks + 20 wrist–joint distances). Use `--no_geometric` for legacy **63-D**.

```bash
python scripts/train_mlp.py ^
  --csv dataset/processed/landmarks.csv ^
  --out_dir models/saved_model/asl_mlp ^
  --label_map models/label_map.json ^
  --epochs 40
```

Writes `models/feature_norm.json` (with `input_dim`), `models/calibration.json` (temperature on logits), and `models/training_artifacts/meta.json`.

## 5) Export TFLite

`--input_dim` must match training (83 default, or 63 with `--no_geometric`):

```bash
python scripts/export_tflite.py ^
  --saved_model_dir models/saved_model/asl_mlp ^
  --out_fp32 models/tflite/asl_mlp_fp32.tflite ^
  --out_int8 models/tflite/asl_mlp_int8.tflite ^
  --input_dim 83
```

## 6) Sync Android assets (one bundle)

Always ship **the same run**: `asl_mlp_fp32.tflite`, `label_map.json`, `feature_norm.json`, and `calibration.json` (if present).

```bash
python scripts/sync_model_bundle.py
```

Optional: `--android_dir`, `--tflite`, `--label_map`, `--feature_norm`, `--calibration`.

## 7) Optional: temporal GRU prototype

Requires `group` and `path` in the CSV (see extract script). Trains a sequence model for comparison; the Android MVP still uses the **single-frame** MLP.

```bash
python scripts/train_temporal_mlp.py ^
  --csv dataset/processed/landmarks.csv ^
  --out_dir models/saved_model/asl_temporal ^
  --label_map models/label_map_temporal.json ^
  --seq_len 8
```

## 8) Offline evaluation (TFLite + calibration)

```bash
python scripts/evaluate_tflite_on_csv.py ^
  --csv dataset/processed/landmarks.csv ^
  --tflite models/tflite/asl_mlp_fp32.tflite ^
  --label_map models/label_map.json ^
  --feature_norm models/feature_norm.json ^
  --calibration models/calibration.json
```

Recorded sessions: same CSV schema — use `scripts/eval_recorded_session.py` (same CLI).

## 9) Android

Open `mobile/android/` in Android Studio. Assets should include:

- `asl_mlp_fp32.tflite`
- `label_map.json`
- `feature_norm.json` (required for 83-D models)
- `calibration.json` (optional; defaults to temperature=1)
- `hand_landmarker.task`

Preprocessing parity: `scripts/verify_pipeline_parity.py`; geometric checks included.

Notes:

- Static-letter MVP; dynamic letters (J, Z) need motion or a temporal model.
- Model outputs **logits**; the app applies **softmax(logits / T)** using `calibration.json`.
# Sign2Text
