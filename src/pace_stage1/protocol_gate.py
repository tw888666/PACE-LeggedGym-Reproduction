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
    validation_allowed: bool
    blocking_items: Tuple[str, ...]
    semantic_blockers: Tuple[str, ...]
    reproducibility_blockers: Tuple[str, ...]
    formal_reporting_blockers: Tuple[str, ...]
    declared_allowed: bool
    declared_validation_allowed: bool
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

    blocker_classes = ("SEMANTIC", "REPRODUCIBILITY", "FORMAL_REPORTING")
    formal_entries = [
        entry for entry in entries if entry.get("blocks_formal_training") is True
    ]
    for entry in formal_entries:
        if entry.get("blocker_class") not in blocker_classes:
            raise RuntimeError(
                f"formal blocker {entry['id']} has no valid blocker_class"
            )
        if entry.get("resolution_priority") not in (1, 2, 3, 4):
            raise RuntimeError(
                f"formal blocker {entry['id']} has no valid resolution_priority"
            )
        if entry.get("resolution_policy") not in payload.get(
            "resolution_policies", {}
        ):
            raise RuntimeError(
                f"formal blocker {entry['id']} has no valid resolution_policy"
            )

    derived_blockers = tuple(
        sorted(entry["id"] for entry in formal_entries)
    )
    declared_blockers = tuple(sorted(payload.get("formal_freeze_blockers", ())))
    if derived_blockers != declared_blockers:
        raise RuntimeError(
            "protocol matrix blocker index disagrees with entry-level blockers"
        )

    classified = {
        blocker_class: tuple(
            sorted(
                entry["id"]
                for entry in formal_entries
                if entry["blocker_class"] == blocker_class
            )
        )
        for blocker_class in blocker_classes
    }
    validation_blockers = tuple(
        sorted(
            entry["id"]
            for entry in formal_entries
            if entry.get("blocks_validation_training") is True
        )
    )
    if validation_blockers != classified["SEMANTIC"]:
        raise RuntimeError(
            "validation blockers must exactly match unresolved semantic blockers"
        )

    baseline = payload.get("baseline", {})
    declared_validation_allowed = (
        baseline.get("validation_training_allowed") is True
    )
    declared_allowed = baseline.get("formal_training_allowed") is True
    validation_allowed = declared_validation_allowed and not validation_blockers
    allowed = declared_allowed and not derived_blockers
    return FormalTrainingStatus(
        allowed=allowed,
        validation_allowed=validation_allowed,
        blocking_items=derived_blockers,
        semantic_blockers=classified["SEMANTIC"],
        reproducibility_blockers=classified["REPRODUCIBILITY"],
        formal_reporting_blockers=classified["FORMAL_REPORTING"],
        declared_allowed=declared_allowed,
        declared_validation_allowed=declared_validation_allowed,
        matrix_path=path,
    )


def require_validation_training_allowed(
    path: Path = DEFAULT_PROTOCOL_MATRIX_PATH,
) -> FormalTrainingStatus:
    status = formal_training_status(path)
    if not status.validation_allowed:
        blockers = ", ".join(status.semantic_blockers)
        reason = blockers or "matrix validation approval is not true"
        raise RuntimeError(
            "PACE_V2_VALIDATION training is prohibited by the protocol matrix; "
            f"semantic_blockers=[{reason}]"
        )
    return status


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
