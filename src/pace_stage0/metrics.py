from __future__ import annotations

from typing import Dict

import numpy as np


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("RMSE inputs must be finite")
    return a, b


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _paired(a, b)
    return float(np.sqrt(np.mean(np.square(a - b), dtype=np.float64)))


def per_joint_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _paired(a, b)
    if a.ndim != 2:
        raise ValueError(f"Expected [time,joint] arrays, got {a.shape}")
    return np.sqrt(np.mean(np.square(a - b), axis=0, dtype=np.float64))


def lag_rmse(ours: np.ndarray, reference: np.ndarray, lag: int) -> float:
    """Compare ours[k] with reference[k+lag] over the common valid interval."""
    ours, reference = _paired(ours, reference)
    if abs(lag) >= len(ours):
        raise ValueError("Absolute lag must be shorter than the trajectory")
    if lag > 0:
        return rmse(ours[:-lag], reference[lag:])
    if lag < 0:
        return rmse(ours[-lag:], reference[:lag])
    return rmse(ours, reference)


def lag_sweep(ours: np.ndarray, reference: np.ndarray, radius: int = 4) -> Dict[str, float]:
    return {str(lag): lag_rmse(ours, reference, lag) for lag in range(-radius, radius + 1)}

