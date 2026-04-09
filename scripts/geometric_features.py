"""
Extra geometry from wrist-centered 63-D landmarks (21×3), parity with Kotlin HandPreprocessor.

Distances from wrist (origin) to landmarks 1..20 in normalized space — nonlinear in raw coords
but cheap for the MLP.
"""
from __future__ import annotations

import numpy as np


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


def concat_with_geometry(f63: np.ndarray) -> np.ndarray:
    """(N, 63) + (N, 20) -> (N, 83)."""
    d20 = wrist_distances_from_flat63(f63)
    return np.concatenate([f63, d20], axis=1).astype(np.float32)
