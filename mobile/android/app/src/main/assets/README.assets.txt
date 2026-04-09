Place these files here (one training run — use scripts/sync_model_bundle.py from repo root):

1) Classifier
   - asl_mlp_fp32.tflite
     Export: python scripts/export_tflite.py --input_dim 83 (or 63 if trained with --no_geometric)

2) label_map.json
   - From the same train_mlp.py run as the .tflite.

3) feature_norm.json
   - Written by train_mlp.py; includes "input_dim" (63 or 83). Required for 83-D models.

4) calibration.json (optional)
   - Written by train_mlp.py; temperature for logits. If missing, the app uses T=1.

5) hand_landmarker.task
   - MediaPipe Tasks Hand Landmarker (required for live tracking).
