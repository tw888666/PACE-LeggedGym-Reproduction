"""Fail-closed training gate driven by the PACE v2 provenance matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_MATRIX_PATH = (
    PROJECT_ROOT / "provenance" / "gpt-pace-v2-protocol-matrix.json"
)


@dataclass(frozen=True)
class FormalTrainingStatus:
    allowed: bool
    blocking_items: Tuple[str, ...]
    declared_allowed: bool
    matrix_path: Path


def load_protocol_matrix(
    path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load protocol matrix: {path}") from error
    if payload.get("schema") != "pace_v2_protocol_provenance_matrix.v1":
        raise RuntimeError("unsupported or missing PACE v2 protocol matrix schema")
    if not isinstance(payload.get("matrix"), list):
        raise RuntimeError("protocol matrix entries must be a list")
    return payload


def formal_training_status(
    path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> FormalTrainingStatus:
    payload = load_protocol_matrix(path)
    entries = payload["matrix"]
    ids = [entry.get("id") for entry in entries]
    if any(not isinstance(entry_id, str) or not entry_id for entry_id in ids):
        raise RuntimeError("every protocol matrix entry must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise RuntimeError("protocol matrix contains duplicate ids")

    derived_blockers = tuple(
        sorted(
            entry["id"]
            for entry in entries
            if entry.get("blocks_formal_training") is True
        )
    )
    declared_blockers = tuple(sorted(payload.get("formal_freeze_blockers", ())))
    if derived_blockers != declared_blockers:
        raise RuntimeError(
            "protocol matrix blocker index disagrees with entry-level blockers"
        )

    declared_allowed = payload.get("baseline", {}).get("formal_training_allowed") is True
    allowed = declared_allowed and not derived_blockers
    return FormalTrainingStatus(
        allowed=allowed,
        blocking_items=derived_blockers,
        declared_allowed=declared_allowed,
        matrix_path=path,
    )


def require_formal_training_allowed(
    path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> FormalTrainingStatus:
    status = formal_training_status(path)
    if not status.allowed:
        blockers = ", ".join(status.blocking_items) or "matrix approval is not true"
        raise RuntimeError(
            "PACE_V2_FORMAL training is prohibited by the protocol matrix; "
            f"blocking_items=[{blockers}]"
        )
    return status
