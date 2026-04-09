"""
Shared landmark preprocessing for training and (documented parity with) mobile HandPreprocessor.

MediaPipe Hand: 21 landmarks, wrist index 0, middle finger MCP index 9.
"""
from __future__ import annotations

import numpy as np

# Middle finger MCP — used for left/right canonicalization
MID_MCP_IDX = 9


def normalize_landmarks_xyz(landmarks_xyz: np.ndarray) -> np.ndarray:
    """
    landmarks_xyz: (21, 3) float32, MediaPipe normalized image space.
    Returns: (63,) float32 — wrist-centered, scaled, optional handedness canon.
    """
    pts = landmarks_xyz.astype(np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist

    d = np.linalg.norm(pts[:, :2], axis=1)
    scale = float(np.max(d)) if float(np.max(d)) > 1e-6 else 1.0
    pts = pts / scale

    # Canonical "right-hand" frame: if middle MCP is left of wrist line, mirror X.
    if pts[MID_MCP_IDX, 0] < 0:
        pts[:, 0] *= -1.0

    return pts.reshape(-1).astype(np.float32)


def augment_features63(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    flip_x_prob: float = 0.5,
    noise_std: float = 0.02,
    scale_range: tuple[float, float] = (0.92, 1.08),
) -> np.ndarray:
    """
    Lightweight augmentation on the 63-D feature vector (matches mobile inference path
    after handedness canon: flip_x here simulates residual left/right ambiguity).
    """
    out = x.astype(np.float32).copy()

    if rng.random() < flip_x_prob:
        out = flip_x_components63(out)

    s = rng.uniform(scale_range[0], scale_range[1])
    out *= np.float32(s)

    out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return out


def flip_x_components63(x: np.ndarray) -> np.ndarray:
    """Flip X for each landmark (indices 0,3,...,60)."""
    out = x.reshape(21, 3).copy()
    out[:, 0] *= -1.0
    return out.reshape(-1)


def batch_augment(
    X: np.ndarray,
    seed: int,
    *,
    flip_x_prob: float = 0.5,
    noise_std: float = 0.02,
    scale_range: tuple[float, float] = (0.92, 1.08),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        out[i] = augment_features63(
            X[i],
            rng,
            flip_x_prob=flip_x_prob,
            noise_std=noise_std,
            scale_range=scale_range,
        )
    return out
