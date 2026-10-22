"""
Extra geometry from wrist-centered 63-D landmarks (21×3), parity with Kotlin HandPreprocessor.

- 20 distances from the wrist to landmarks 1..20
- 12 thumb-tip offsets to the index, middle, ring, and pinky PIP joints

The thumb offsets are what separate fist letters (M vs N vs S): which fingers sit over the thumb.
"""
from __future__ import annotations

import numpy as np

# MediaPipe: thumb tip, then PIP of index / middle / ring / pinky.
THUMB_TIP = 4
FINGER_PIPS = (6, 10, 14, 18)
WRIST_DISTANCE_DIM = 20
THUMB_RELATION_DIM = 12
MODEL_FEATURE_DIM = 63 + WRIST_DISTANCE_DIM + THUMB_RELATION_DIM  # 95


def wrist_distances_from_flat63(f: np.ndarray) -> np.ndarray:
    """
    f: (63,) or (N, 63) float32 — flattened normalized landmarks.
    Returns: (20,) or (N, 20) — L2 norm from wrist to each of landmarks 1..20.
    """
    if f.ndim == 1:
        x = f.reshape(1, 21, 3)
    else:
        x = f.reshape(len(f), 21, 3)
    pts = x[:, 1:, :]
    d = np.linalg.norm(pts, axis=-1).astype(np.float32)
    if f.ndim == 1:
        return d[0]
    return d


def thumb_relations_from_flat63(f: np.ndarray) -> np.ndarray:
    """
    f: (63,) or (N, 63).
    Returns: (12,) or (N, 12) — xyz of each finger PIP minus the thumb tip.
    Order: index, middle, ring, pinky. Must match HandPreprocessor.thumbRelations12.
    """
    single = f.ndim == 1
    x = f.reshape(1, 21, 3) if single else f.reshape(len(f), 21, 3)
    thumb = x[:, THUMB_TIP, :]
    rel = np.empty((x.shape[0], THUMB_RELATION_DIM), dtype=np.float32)
    for k, idx in enumerate(FINGER_PIPS):
        rel[:, k * 3 : k * 3 + 3] = x[:, idx, :] - thumb
    if single:
        return rel[0]
    return rel


def concat_with_geometry(f63: np.ndarray) -> np.ndarray:
    """(N, 63) + (N, 20) + (N, 12) -> (N, 95)."""
    d20 = wrist_distances_from_flat63(f63)
    t12 = thumb_relations_from_flat63(f63)
    return np.concatenate([f63, d20, t12], axis=1).astype(np.float32)
