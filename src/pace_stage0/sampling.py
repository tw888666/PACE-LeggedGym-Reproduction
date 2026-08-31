from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SampledReplay:
    q_true: np.ndarray
    qdot: np.ndarray
    interval_records: List[Any]


Transition = Callable[[int, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Any]]


def run_indexed_replay(
    targets: np.ndarray,
    q_true_0: np.ndarray,
    qdot_0: np.ndarray,
    transition: Transition,
) -> SampledReplay:
    """Record state[0], then apply target[t] over [t,t+1) to produce state[t+1]."""
    targets = np.asarray(targets)
    q_true_0 = np.asarray(q_true_0)
    qdot_0 = np.asarray(qdot_0)
    if targets.ndim != 2 or len(targets) < 2:
        raise ValueError("targets must be a [N,joint] array with N >= 2")
    if q_true_0.shape != targets.shape[1:] or qdot_0.shape != targets.shape[1:]:
        raise ValueError("Initial state shape must match one target row")

    q_true = np.empty_like(targets)
    qdot = np.empty_like(targets)
    q_true[0] = q_true_0
    qdot[0] = qdot_0
    records: List[Any] = []
    for t in range(len(targets) - 1):
        new_q, new_qdot, record = transition(t, targets[t], q_true[t], qdot[t])
        q_true[t + 1] = new_q
        qdot[t + 1] = new_qdot
        records.append(record)
    return SampledReplay(q_true=q_true, qdot=qdot, interval_records=records)

