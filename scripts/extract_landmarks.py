import csv
import json
import argparse
import sys
import time
from pathlib import Path

# Allow `python scripts/extract_landmarks.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

from landmark_preprocessing import normalize_landmarks_xyz


LETTERS = [chr(ord("A") + i) for i in range(26)]


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dataset/raw/asl_alphabet (contains A..Z folders)")
    ap.add_argument("--output_csv", required=True, help="dataset/processed/landmarks.csv")
    ap.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Cap images per letter (e.g. 300). Use 0 to try every image (more rows if hands are detected).",
    )
    # Lower defaults → more images pass hand detection (dataset photos are often tight crops / odd poses).
    ap.add_argument("--min_det_conf", type=float, default=0.35)
    ap.add_argument("--min_track_conf", type=float, default=0.35)
    ap.add_argument("--write_label_map", default="", help="models/label_map.json (optional)")
    ap.add_argument(
        "--hand_task_path",
        default="",
        help="Path to hand_landmarker.task (required when mediapipe.tasks API is used).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N images (0 = only per-class lines).",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.write_label_map:
        label_map_path = Path(args.write_label_map)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({i: LETTERS[i] for i in range(26)}, f, indent=2)

    rng = np.random.default_rng(args.seed)

    # group = parent path under input_dir (e.g. A or A/subject1); path = relative file path for ordering
    header = ["label", "group", "path"] + [f"f{i}" for i in range(63)]

    per_class_found = {l: 0 for l in LETTERS}
    per_class_written = {l: 0 for l in LETTERS}
    skipped_no_hand = 0

    use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

    if use_legacy_solutions:
        mp_hands = mp.solutions.hands
        hands_ctx = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=args.min_det_conf,
            min_tracking_confidence=args.min_track_conf,
        )
    else:
        if not args.hand_task_path:
            raise ValueError(
                "This MediaPipe build does not expose mp.solutions. "
                "Use Tasks API by passing --hand_task_path path/to/hand_landmarker.task"
            )
        hand_task_path = Path(args.hand_task_path)
        if not hand_task_path.exists():
            raise FileNotFoundError(
                f"hand_landmarker.task not found at: {hand_task_path}\n"
                "Download from:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_task_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=args.min_det_conf,
            min_hand_presence_confidence=args.min_det_conf,
            min_tracking_confidence=args.min_track_conf,
        )
        hands_ctx = HandLandmarker.create_from_options(options)

    t0 = time.perf_counter()
    total_done = 0

    with hands_ctx as hands, open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for letter in LETTERS:
            class_dir = input_dir / letter
            if not class_dir.exists():
                continue

            imgs = list(iter_images(class_dir))
            per_class_found[letter] = len(imgs)

            if args.max_per_class and len(imgs) > args.max_per_class:
                idx = rng.choice(len(imgs), size=args.max_per_class, replace=False)
                imgs = [imgs[i] for i in idx]

            print(f"[{letter}] processing {len(imgs)} images…", flush=True)

            for img_path in imgs:
                total_done += 1
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if use_legacy_solutions:
                    result = hands.process(img_rgb)
                    if not result.multi_hand_landmarks:
                        skipped_no_hand += 1
                        continue
                    lm = result.multi_hand_landmarks[0].landmark
                    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    result = hands.detect(mp_image)
                    if not result.hand_landmarks:
                        skipped_no_hand += 1
                        continue
                    lm = result.hand_landmarks[0]
                    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                feats = normalize_landmarks_xyz(pts)
                rel = img_path.relative_to(input_dir)
                group_str = str(rel.parent).replace("\\", "/")
                path_str = str(rel).replace("\\", "/")

                writer.writerow([letter, group_str, path_str] + feats.tolist())
                per_class_written[letter] += 1
                pe = args.progress_every
                if pe > 0 and total_done % pe == 0:
                    dt = time.perf_counter() - t0
                    rate = total_done / dt if dt > 0 else 0.0
                    print(
                        f"  … {total_done} images done, ~{rate:.1f} img/s, {dt / 60:.1f} min elapsed",
                        flush=True,
                    )

    total_written = sum(per_class_written.values())
    dt_total = time.perf_counter() - t0
    print(f"Finished in {dt_total / 60:.1f} min ({total_done} images attempted).")
    print(f"Wrote {total_written} rows to {output_csv}")
    print(f"Skipped (no hand detected): {skipped_no_hand}")
    print("Per-class counts (found -> written):")
    for l in LETTERS:
        if per_class_found[l] > 0:
            print(f"  {l}: {per_class_found[l]} -> {per_class_written[l]}")


if __name__ == "__main__":
    main()

