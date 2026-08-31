from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .constants import DATA_PATH, EXPECTED_DATA_SHA256
from .decoder import sha256_file


@dataclass(frozen=True)
class ReplayData:
    time: np.ndarray
    real_des_dof_pos: np.ndarray
    real_dof_pos: np.ndarray
    real_dof_vel: np.ndarray
    real_dof_torques: np.ndarray
    sim_method_dof_pos: np.ndarray
    sim_method_dof_vel: np.ndarray
    sim_method_dof_torques: np.ndarray
    sim_nothing_dof_pos: np.ndarray
    sim_nothing_dof_vel: np.ndarray
    sim_nothing_dof_torques: np.ndarray
    sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.real_dof_pos.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.real_dof_pos.shape[1])


def _array(group: Dict[str, object], name: str) -> np.ndarray:
    if name not in group:
        raise KeyError(f"Missing replay field: {name}")
    value = np.asarray(group[name])
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"Replay field {name} is not numeric: {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"Replay field {name} contains non-finite values")
    return value.astype(np.float32, copy=True)


def load_replay_data(path: Path = DATA_PATH) -> ReplayData:
    path = path.resolve()
    if path != DATA_PATH.resolve():
        raise ValueError(f"Stage 0 replay path is frozen to {DATA_PATH}; got {path}")
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_DATA_SHA256:
        raise ValueError(
            f"data.npy SHA-256 mismatch: expected {EXPECTED_DATA_SHA256}, got {actual_sha}"
        )
    payload = np.load(path, allow_pickle=True)
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError("Expected data.npy to contain a scalar pickled dictionary")
    payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected replay dictionary, got {type(payload).__name__}")
    for group_name in ("real", "sim_method", "sim_nothing"):
        if group_name not in payload or not isinstance(payload[group_name], dict):
            raise KeyError(f"Missing replay group: {group_name}")

    real = payload["real"]
    sim_method = payload["sim_method"]
    sim_nothing = payload["sim_nothing"]
    result = ReplayData(
        time=_array(real, "time"),
        real_des_dof_pos=_array(real, "des_dof_pos"),
        real_dof_pos=_array(real, "dof_pos"),
        real_dof_vel=_array(real, "dof_vel"),
        real_dof_torques=_array(real, "dof_torques"),
        sim_method_dof_pos=_array(sim_method, "dof_pos"),
        sim_method_dof_vel=_array(sim_method, "dof_vel"),
        sim_method_dof_torques=_array(sim_method, "dof_torques"),
        sim_nothing_dof_pos=_array(sim_nothing, "dof_pos"),
        sim_nothing_dof_vel=_array(sim_nothing, "dof_vel"),
        sim_nothing_dof_torques=_array(sim_nothing, "dof_torques"),
        sha256=actual_sha,
    )
    expected_state_shape = (len(result.time), 12)
    for name, value in result.__dict__.items():
        if name in {"time", "sha256"}:
            continue
        if value.shape != expected_state_shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected_state_shape}")
    return result

