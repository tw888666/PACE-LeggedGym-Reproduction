from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .constants import (
    CANONICAL_JOINT_NAMES,
    EXPECTED_FIT_SHA256,
    FIT_PATH,
    RAW_JOINT_NAMES,
    RAW_TO_CANONICAL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _canonicalize(block: np.ndarray) -> np.ndarray:
    if block.shape != (12,):
        raise ValueError(f"Expected a 12-joint parameter block, got {block.shape}")
    return block[np.asarray(RAW_TO_CANONICAL, dtype=np.int64)].copy()


@dataclass(frozen=True)
class DecodedFit:
    friction: np.ndarray
    damping: np.ndarray
    armature: np.ndarray
    encoder_bias: np.ndarray
    delay_raw: float
    delay_steps: int
    params_raw: np.ndarray
    bounds_raw: np.ndarray
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "pace_stage0.decoded_fit.v1",
            "source": str(FIT_PATH),
            "sha256": self.sha256,
            "legacy_layout": {
                "0:12": "friction",
                "12:24": "damping",
                "24:36": "armature",
                "36:48": "encoder_bias",
                "48": "delay",
            },
            "raw_joint_order": list(RAW_JOINT_NAMES),
            "canonical_joint_order": list(CANONICAL_JOINT_NAMES),
            "raw_to_canonical_indices": list(RAW_TO_CANONICAL),
            "dtype": str(self.params_raw.dtype),
            "shape": list(self.params_raw.shape),
            "friction": self.friction.tolist(),
            "damping": self.damping.tolist(),
            "armature": self.armature.tolist(),
            "encoder_bias": self.encoder_bias.tolist(),
            "delay_raw": self.delay_raw,
            "delay_discretization": "int_truncation_toward_zero",
            "delay_steps": self.delay_steps,
            "params_raw": self.params_raw.tolist(),
            "bounds_raw": self.bounds_raw.tolist(),
        }


def decode_fit(path: Path = FIT_PATH) -> DecodedFit:
    path = path.resolve()
    if path != FIT_PATH.resolve():
        raise ValueError(f"Stage 0 fit path is frozen to {FIT_PATH}; got {path}")
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_FIT_SHA256:
        raise ValueError(
            f"fitting.npy SHA-256 mismatch: expected {EXPECTED_FIT_SHA256}, got {actual_sha}"
        )

    payload = np.load(path, allow_pickle=True)
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError("Expected fitting.npy to contain a scalar pickled dictionary")
    payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected fitting dictionary, got {type(payload).__name__}")
    for key in ("params", "bounds"):
        if key not in payload:
            raise KeyError(f"Missing fitting key: {key}")

    params = _as_numpy(payload["params"]).astype(np.float64, copy=True)
    bounds = _as_numpy(payload["bounds"]).astype(np.float64, copy=True)
    if params.shape != (49,):
        raise ValueError(f"Expected 49 fit parameters, got {params.shape}")
    if bounds.shape != (49, 2):
        raise ValueError(f"Expected (49,2) bounds, got {bounds.shape}")

    delay_raw = float(params[48])
    return DecodedFit(
        friction=_canonicalize(params[0:12]),
        damping=_canonicalize(params[12:24]),
        armature=_canonicalize(params[24:36]),
        encoder_bias=_canonicalize(params[36:48]),
        delay_raw=delay_raw,
        delay_steps=int(delay_raw),
        params_raw=params,
        bounds_raw=bounds,
        sha256=actual_sha,
    )


def write_decoded_fit(output: Path, decoded: DecodedFit) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decoded.to_dict(), indent=2) + "\n", encoding="utf-8")

