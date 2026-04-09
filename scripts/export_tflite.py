import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True, help="models/saved_model/asl_mlp")
    ap.add_argument("--out_fp32", required=True, help="models/tflite/asl_mlp_fp32.tflite")
    ap.add_argument("--out_int8", required=True, help="models/tflite/asl_mlp_int8.tflite")
    ap.add_argument(
        "--input_dim",
        type=int,
        default=83,
        help="Must match train_mlp input_dim (63 if --no_geometric, else 83).",
    )
    args = ap.parse_args()

    input_dim = args.input_dim

    def representative_dataset():
        for _ in range(200):
            yield [np.random.uniform(-1.0, 1.0, size=(1, input_dim)).astype(np.float32)]

    saved_model_dir = args.saved_model_dir

    # FP32
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_fp32 = converter.convert()
    out_fp32 = Path(args.out_fp32)
    out_fp32.parent.mkdir(parents=True, exist_ok=True)
    out_fp32.write_bytes(tflite_fp32)
    print(f"Wrote FP32 TFLite: {out_fp32}")

    # INT8
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    out_int8 = Path(args.out_int8)
    out_int8.parent.mkdir(parents=True, exist_ok=True)
    out_int8.write_bytes(tflite_int8)
    print(f"Wrote INT8 TFLite: {out_int8}")


if __name__ == "__main__":
    main()

