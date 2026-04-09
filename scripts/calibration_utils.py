"""Temperature scaling on logits (numpy-only, no scipy)."""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    if logits.ndim == 1:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    grid: tuple[float, float, int] = (0.2, 8.0, 200),
) -> float:
    """
    Minimize NLL of softmax(logits / T) w.r.t. one-hot targets (grid search over T).
    logits: (N, C), y_true_idx: (N,) int.
    """
    if len(logits) == 0:
        return 1.0
    n = logits.shape[0]
    lo, hi, steps = grid
    Ts = np.linspace(lo, hi, steps, dtype=np.float64)
    best_t = 1.0
    best_nll = float("inf")
    for T in Ts:
        p = softmax(logits.astype(np.float64) / float(T), axis=1)
        p = np.clip(p, 1e-12, 1.0)
        idx = (np.arange(n), y_true_idx.astype(np.int64))
        nll = -np.mean(np.log(p[idx]))
        if nll < best_nll:
            best_nll = nll
            best_t = float(T)
    return best_t
