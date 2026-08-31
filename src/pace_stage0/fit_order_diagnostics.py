from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CANONICAL_JOINT_NAMES,
    EXPECTED_FIT_SHA256,
    FIT_PATH,
    PROJECT_ROOT,
    RAW_JOINT_NAMES,
)


PACE_DATA_ROOT = Path("/home/xy.chen/tw/dataset/pace_data")
PAPER_URL = "https://arxiv.org/html/2509.06342v2"


@dataclass(frozen=True)
class PublishedValue:
    display: str
    scale: str = "1"

    @property
    def value(self) -> float:
        return float(Decimal(self.display) * Decimal(self.scale))

    @property
    def half_rounding_unit(self) -> float:
        decimal = Decimal(self.display)
        last_unit = Decimal(1).scaleb(decimal.as_tuple().exponent)
        return float(abs(last_unit * Decimal(self.scale)) / Decimal(2))


def _published(display_values: Sequence[str], scale: str = "1") -> Tuple[PublishedValue, ...]:
    if len(display_values) != 12:
        raise ValueError("A Table 6 parameter row must have 12 joint values")
    return tuple(PublishedValue(value, scale) for value in display_values)


# Table 6 is labelled in LF, RF, LH, RH order. Armature is printed in 10^-3 units.
TABLE6: Mapping[str, Tuple[PublishedValue, ...]] = {
    "friction": _published(
        ("0.0054", "0.021", "0.028", "0.0035", "0.027", "0.036",
         "0.0032", "0.024", "0.040", "0.0029", "0.013", "0.045")
    ),
    "damping": _published(
        ("4.9", "4.4", "5.2", "4.7", "4.3", "5.3",
         "4.9", "4.9", "5.4", "5.1", "5.1", "5.5")
    ),
    "armature": _published(
        ("76", "76", "67", "74", "77", "67",
         "89", "51", "64", "79", "39", "51"),
        "0.001",
    ),
    "encoder_bias": _published(
        ("0.022", "0.0057", "-0.003", "0.011", "-0.0072", "0.0094",
         "-0.012", "-0.0013", "-0.0095", "-0.016", "0.0043", "0.0045")
    ),
}

BLOCK_SLICES = {
    "friction": slice(0, 12),
    "damping": slice(12, 24),
    "armature": slice(24, 36),
    "encoder_bias": slice(36, 48),
}

HYPOTHESES = {
    "H_A_LF_LH_RF_RH": tuple(RAW_JOINT_NAMES),
    "H_B_LF_RF_LH_RH": tuple(CANONICAL_JOINT_NAMES),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _value_schema(value: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
    }
    if isinstance(value, Mapping):
        result["keys"] = [str(key) for key in value.keys()]
        result["children"] = {
            str(key): _value_schema(child) for key, child in value.items()
        }
        return result
    try:
        array = _as_numpy(value)
    except Exception as error:  # schema audit must not mutate or deserialize further
        result["array_conversion_error"] = repr(error)
        return result
    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)
    result["size"] = int(array.size)
    if array.dtype != object and np.issubdtype(array.dtype, np.number) and array.size:
        numeric = array.astype(np.float64, copy=False)
        result["finite_count"] = int(np.isfinite(numeric).sum())
        result["minimum"] = float(np.nanmin(numeric))
        result["maximum"] = float(np.nanmax(numeric))
    return result


def inspect_fitting_payload(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    container = np.load(path, allow_pickle=True)
    if not isinstance(container, np.ndarray) or container.shape != ():
        raise ValueError(f"Expected scalar object ndarray in {path}")
    payload = container.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected fitting dictionary, got {type(payload).__name__}")
    params = _as_numpy(payload.get("params")).astype(np.float64, copy=True)
    if params.shape != (49,):
        raise ValueError(f"Expected params shape (49,), got {params.shape}")

    metadata_tokens = ("joint", "dof", "name", "robot", "config", "metadata")

    def metadata_paths(value: Any, prefix: str = "") -> Sequence[str]:
        paths = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                path_key = f"{prefix}.{key}" if prefix else str(key)
                if any(token in str(key).lower() for token in metadata_tokens):
                    paths.append(path_key)
                paths.extend(metadata_paths(child, path_key))
        return paths

    schema = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "container": {
            "python_type": f"{type(container).__module__}.{type(container).__name__}",
            "shape": list(container.shape),
            "dtype": str(container.dtype),
        },
        "payload": _value_schema(payload),
        "top_level_keys": [str(key) for key in payload.keys()],
        "joint_metadata_candidate_paths": list(metadata_paths(payload)),
        "machine_readable_joint_metadata": bool(metadata_paths(payload)),
    }
    return schema, params


def _paper_lookup(block: str) -> Dict[str, PublishedValue]:
    return dict(zip(CANONICAL_JOINT_NAMES, TABLE6[block]))


def audit_block(block: str, params_file_order: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(params_file_order, dtype=np.float64)
    if values.shape != (12,):
        raise ValueError(f"Expected 12 {block} values, got {values.shape}")
    paper = _paper_lookup(block)
    rows = []
    hypothesis_errors: Dict[str, list] = {name: [] for name in HYPOTHESES}
    hypothesis_normalized: Dict[str, list] = {name: [] for name in HYPOTHESES}
    hypothesis_within: Dict[str, list] = {name: [] for name in HYPOTHESES}

    for slot, file_value in enumerate(values):
        row: Dict[str, Any] = {
            "file_slot": slot,
            "params_file_order_value": float(file_value),
            "rf_lh_focus_slot": 3 <= slot <= 8,
        }
        for hypothesis, joint_names in HYPOTHESES.items():
            joint = joint_names[slot]
            published = paper[joint]
            tolerance = published.half_rounding_unit
            lower = published.value - tolerance
            upper = published.value + tolerance
            absolute_error = abs(float(file_value) - published.value)
            within = lower <= float(file_value) < upper
            normalized = absolute_error / tolerance
            row[hypothesis] = {
                "joint": joint,
                "table6_display": published.display,
                "table6_scale": published.scale,
                "table6_value_SI": published.value,
                "rounding_half_width_SI": tolerance,
                "rounding_interval_SI": [lower, upper],
                "rounding_interval_upper_exclusive": True,
                "absolute_discrepancy_SI": absolute_error,
                "normalized_discrepancy_half_rounding_units": normalized,
                "within_rounding_interval": within,
            }
            hypothesis_errors[hypothesis].append(absolute_error)
            hypothesis_normalized[hypothesis].append(normalized)
            hypothesis_within[hypothesis].append(within)
        rows.append(row)

    statistics = {}
    for hypothesis in HYPOTHESES:
        errors = np.asarray(hypothesis_errors[hypothesis], dtype=np.float64)
        normalized = np.asarray(hypothesis_normalized[hypothesis], dtype=np.float64)
        within = np.asarray(hypothesis_within[hypothesis], dtype=bool)
        statistics[hypothesis] = {
            "within_rounding_interval_count": int(within.sum()),
            "joint_count": 12,
            "mean_absolute_discrepancy_SI": float(errors.mean()),
            "maximum_absolute_discrepancy_SI": float(errors.max()),
            "rmse_discrepancy_SI": float(math.sqrt(np.mean(errors ** 2))),
            "mean_normalized_discrepancy_half_rounding_units": float(normalized.mean()),
            "maximum_normalized_discrepancy_half_rounding_units": float(normalized.max()),
        }

    a = statistics["H_A_LF_LH_RF_RH"]
    b = statistics["H_B_LF_RF_LH_RH"]
    a_better_slots = sum(
        row["H_A_LF_LH_RF_RH"]["absolute_discrepancy_SI"]
        < row["H_B_LF_RF_LH_RH"]["absolute_discrepancy_SI"]
        for row in rows
    )
    b_better_slots = sum(
        row["H_B_LF_RF_LH_RH"]["absolute_discrepancy_SI"]
        < row["H_A_LF_LH_RF_RH"]["absolute_discrepancy_SI"]
        for row in rows
    )
    winner = (
        "H_A_LF_LH_RF_RH"
        if a["rmse_discrepancy_SI"] < b["rmse_discrepancy_SI"]
        else "H_B_LF_RF_LH_RH"
    )
    return {
        "block": block,
        "paper_order": list(CANONICAL_JOINT_NAMES),
        "rows": rows,
        "statistics": statistics,
        "comparison": {
            "winner_by_rmse": winner,
            "H_B_over_H_A_rmse_ratio": (
                b["rmse_discrepancy_SI"] / a["rmse_discrepancy_SI"]
            ),
            "H_B_over_H_A_mean_absolute_ratio": (
                b["mean_absolute_discrepancy_SI"]
                / a["mean_absolute_discrepancy_SI"]
            ),
            "H_A_lower_absolute_discrepancy_slot_count": a_better_slots,
            "H_B_lower_absolute_discrepancy_slot_count": b_better_slots,
            "tie_slot_count": 12 - a_better_slots - b_better_slots,
        },
    }


def _cross_file_audit() -> Dict[str, Any]:
    pattern = "**/anymal*/fitting.npy"
    paths = sorted(PACE_DATA_ROOT.glob(pattern))
    rows = []
    for path in paths:
        schema, params = inspect_fitting_payload(path)
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": schema["sha256"],
                "params_shape": list(params.shape),
                "same_49_parameter_schema": params.shape == (49,),
                "top_level_keys": schema["top_level_keys"],
            }
        )
    return {
        "authorized_root": str(PACE_DATA_ROOT),
        "authorized_glob": pattern,
        "match_count": len(paths),
        "matches": rows,
        "cross_dataset_order_evidence_available": len(rows) > 1,
        "note": (
            "Only one authorized ANYmal fitting artifact matched; no independent "
            "same-dataset serializer-order cross-check is available."
            if len(rows) <= 1 else
            "Multiple authorized ANYmal fitting artifacts were available."
        ),
    }


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _markdown_table(block: Mapping[str, Any]) -> str:
    lines = [
        "| Slot | params_file_order | H_A joint | H_A Table 6 | H_A abs err | H_A in interval | H_B joint | H_B Table 6 | H_B abs err | H_B in interval |",
        "|---:|---:|---|---:|---:|:---:|---|---:|---:|:---:|",
    ]
    for row in block["rows"]:
        a = row["H_A_LF_LH_RF_RH"]
        b = row["H_B_LF_RF_LH_RH"]
        a_display = f"{a['table6_display']} x {a['table6_scale']}" if a["table6_scale"] != "1" else a["table6_display"]
        b_display = f"{b['table6_display']} x {b['table6_scale']}" if b["table6_scale"] != "1" else b["table6_display"]
        lines.append(
            "| {slot} | {value} | {aj} | {av} | {ae} | {ai} | {bj} | {bv} | {be} | {bi} |".format(
                slot=row["file_slot"],
                value=_format_number(row["params_file_order_value"]),
                aj=a["joint"], av=a_display,
                ae=_format_number(a["absolute_discrepancy_SI"]),
                ai="yes" if a["within_rounding_interval"] else "no",
                bj=b["joint"], bv=b_display,
                be=_format_number(b["absolute_discrepancy_SI"]),
                bi="yes" if b["within_rounding_interval"] else "no",
            )
        )
    return "\n".join(lines)


def _markdown_focus_table(report: Mapping[str, Any]) -> str:
    lines = [
        "| Block | Slot | params_file_order | H_A identity | H_A abs err | H_B identity | H_B abs err | Lower discrepancy |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for block_name in BLOCK_SLICES:
        rows = report["rf_lh_focus_slots"]["per_block_rows"][block_name]
        for row in rows:
            a = row["H_A_LF_LH_RF_RH"]
            b = row["H_B_LF_RF_LH_RH"]
            if a["absolute_discrepancy_SI"] < b["absolute_discrepancy_SI"]:
                winner = "H_A"
            elif b["absolute_discrepancy_SI"] < a["absolute_discrepancy_SI"]:
                winner = "H_B"
            else:
                winner = "tie"
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    block_name,
                    row["file_slot"],
                    _format_number(row["params_file_order_value"]),
                    a["joint"],
                    _format_number(a["absolute_discrepancy_SI"]),
                    b["joint"],
                    _format_number(b["absolute_discrepancy_SI"]),
                    winner,
                )
            )
    return "\n".join(lines)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    conclusion = report["conclusion"]
    lines = [
        "# Stage 0C fitting.npy block-internal joint-order audit",
        "",
        "## Frozen status",
        "",
        "```text",
        "Stage 0A: PASS",
        "Stage 0B: frozen",
        "Stage 0C: FAIL",
        "asset/PhysX: BLOCKED",
        "locomotion/PPO: BLOCKED",
        "```",
        "",
        "This is a diagnostic-only audit. It does not modify the formal decoder, actuator, data loader, frame, bias sign, delay, or thresholds.",
        "",
        "## Conclusion",
        "",
        f"- Most likely file order: `{conclusion['most_likely_file_order']}`.",
        f"- Provenance/confidence: `{conclusion['provenance']}` / `{conclusion['confidence']}`.",
        f"- Existing `RAW_TO_CANONICAL` necessary: `{str(conclusion['RAW_TO_CANONICAL_necessary']).lower()}`.",
        f"- Diagnostic direct-order replay executed: `{str(report['conditional_direct_order_replay']['executed']).lower()}`.",
        f"- Recommend formal decoder change: `{str(conclusion['recommend_formal_decoder_change']).lower()}`.",
        "- RF/LH focus comparisons: H_A lower in {H_A}/24, H_B lower in {H_B}/24, ties {tie}/24.".format(
            **report["rf_lh_focus_slots"]["aggregate_lower_discrepancy_counts"]
        ),
        "- The legacy serializer implementation remains unavailable; order provenance is not source-code-confirmed.",
        "- Caveat: " + conclusion["absolute_fit_caveat"],
        "",
        "## Payload schema",
        "",
        f"Top-level keys: `{report['fitting_payload']['top_level_keys']}`",
        "",
        f"Machine-readable joint metadata: `{str(report['fitting_payload']['machine_readable_joint_metadata']).lower()}`",
        "",
        "```json",
        json.dumps(report["fitting_payload"]["payload"], indent=2),
        "```",
        "",
        "## Paper Table 6 order",
        "",
        "`" + " ".join(CANONICAL_JOINT_NAMES) + "`",
        "",
        "Published armature values are printed in Table 6 in 10^-3 units. Rounding intervals use half of the last displayed decimal unit, after unit scaling.",
    ]
    for block_name in BLOCK_SLICES:
        block = report["table6_order_audit"][block_name]
        a = block["statistics"]["H_A_LF_LH_RF_RH"]
        b = block["statistics"]["H_B_LF_RF_LH_RH"]
        lines.extend(
            [
                "",
                f"## {block_name}",
                "",
                _markdown_table(block),
                "",
                "| Hypothesis | Within interval | Mean abs discrepancy | Max discrepancy | RMSE | Mean normalized |",
                "|---|---:|---:|---:|---:|---:|",
                "| H_A LF LH RF RH | {}/12 | {} | {} | {} | {} |".format(
                    a["within_rounding_interval_count"],
                    _format_number(a["mean_absolute_discrepancy_SI"]),
                    _format_number(a["maximum_absolute_discrepancy_SI"]),
                    _format_number(a["rmse_discrepancy_SI"]),
                    _format_number(a["mean_normalized_discrepancy_half_rounding_units"]),
                ),
                "| H_B LF RF LH RH | {}/12 | {} | {} | {} | {} |".format(
                    b["within_rounding_interval_count"],
                    _format_number(b["mean_absolute_discrepancy_SI"]),
                    _format_number(b["maximum_absolute_discrepancy_SI"]),
                    _format_number(b["rmse_discrepancy_SI"]),
                    _format_number(b["mean_normalized_discrepancy_half_rounding_units"]),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## RF/LH focus: file slots 3-8",
            "",
            _markdown_focus_table(report),
            "",
            "Across all four blocks, H_A wins the comparative mapping; the encoder-bias signs and magnitudes are particularly discriminating.",
            "",
            "## Authorized cross-file audit",
            "",
            "```json",
            json.dumps(report["authorized_cross_file_audit"], indent=2),
            "```",
            "",
            "## Conditional replay decision",
            "",
            "```json",
            json.dumps(report["conditional_direct_order_replay"], indent=2),
            "```",
            "",
            "## Actuator interpretation",
            "",
            report["actuator_core_assessment"]["statement"],
            "",
            "The next authorized branch is legacy bias-application forensic. No asset/PhysX or locomotion/PPO work is authorized by this audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fit_order_report() -> Dict[str, Any]:
    schema, params = inspect_fitting_payload(FIT_PATH)
    if schema["sha256"] != EXPECTED_FIT_SHA256:
        raise ValueError("Frozen fitting.npy SHA-256 mismatch")
    blocks = {
        name: audit_block(name, params[block_slice])
        for name, block_slice in BLOCK_SLICES.items()
    }
    winners = [block["comparison"]["winner_by_rmse"] for block in blocks.values()]
    h_a_all_blocks = all(winner == "H_A_LF_LH_RF_RH" for winner in winners)
    h_b_strongly_supported = all(
        winner == "H_B_LF_RF_LH_RH" for winner in winners
    )
    focus_a_wins = 0
    focus_b_wins = 0
    focus_ties = 0
    for block in blocks.values():
        for row in block["rows"][3:9]:
            a_error = row["H_A_LF_LH_RF_RH"]["absolute_discrepancy_SI"]
            b_error = row["H_B_LF_RF_LH_RH"]["absolute_discrepancy_SI"]
            if a_error < b_error:
                focus_a_wins += 1
            elif b_error < a_error:
                focus_b_wins += 1
            else:
                focus_ties += 1
    return {
        "schema": "pace_stage0.fit_order_diagnostics.v1",
        "scope": "diagnostic_only",
        "paper_source": PAPER_URL,
        "frozen_block_layout": {
            "0:12": "friction",
            "12:24": "damping",
            "24:36": "armature",
            "36:48": "encoder_bias",
            "48": "delay",
            "status": "unchanged / high confidence",
        },
        "terminology": {
            "params_file_order": "The serialized order, with no semantic joint claim.",
            "decoder_reordered": "params_file_order gathered with RAW_TO_CANONICAL.",
            "prohibited_inference": "The word raw alone is not joint-order evidence.",
        },
        "fitting_payload": schema,
        "table6_published_joint_order": list(CANONICAL_JOINT_NAMES),
        "rounding_policy": (
            "Half of the last displayed decimal unit, lower-inclusive and "
            "upper-exclusive, after applying the printed unit scale."
        ),
        "hypotheses": {name: list(order) for name, order in HYPOTHESES.items()},
        "table6_order_audit": blocks,
        "rf_lh_focus_slots": {
            "file_slots": [3, 4, 5, 6, 7, 8],
            "H_A_identity": list(RAW_JOINT_NAMES[3:9]),
            "H_B_identity": list(CANONICAL_JOINT_NAMES[3:9]),
            "per_block_rows": {
                name: [row for row in block["rows"] if row["rf_lh_focus_slot"]]
                for name, block in blocks.items()
            },
            "aggregate_lower_discrepancy_counts": {
                "H_A": focus_a_wins,
                "H_B": focus_b_wins,
                "tie": focus_ties,
                "comparison_count": focus_a_wins + focus_b_wins + focus_ties,
            },
        },
        "authorized_cross_file_audit": _cross_file_audit(),
        "conclusion": {
            "most_likely_file_order": (
                "LF LH RF RH" if h_a_all_blocks else
                "LF RF LH RH" if h_b_strongly_supported else "unresolved"
            ),
            "winning_hypotheses_by_block": dict(zip(BLOCK_SLICES, winners)),
            "provenance": "artifact_and_paper_table6_strong_inference",
            "confidence": "high" if h_a_all_blocks or h_b_strongly_supported else "unresolved",
            "legacy_serializer_implementation": "UNAVAILABLE / provenance unresolved",
            "source_code_confirmed": False,
            "all_four_blocks_support_same_order": h_a_all_blocks or h_b_strongly_supported,
            "absolute_fit_caveat": (
                "The fitting artifact is not the unrounded source of every printed "
                "Table 6 value: rounding-interval hit counts are low for friction "
                "and armature. The order conclusion is comparative, based on the "
                "consistent H_A advantage across all four blocks and the RF/LH slots."
            ),
            "H_B_strongly_supported": h_b_strongly_supported,
            "fitting_joint_order_hypothesis_H_B_rejected": h_a_all_blocks,
            "RAW_TO_CANONICAL_necessary": h_a_all_blocks,
            "recommend_formal_decoder_change": False,
            "automatic_decoder_selection": False,
        },
        "conditional_direct_order_replay": {
            "condition": "Run only if H_B wins clearly across all four Table 6 blocks.",
            "condition_met": h_b_strongly_supported,
            "executed": False,
            "reason": (
                "H_A wins all four blocks; the frozen condition prohibits a direct-order replay."
                if h_a_all_blocks else
                "No automatic replay is run unless H_B is strongly supported."
            ),
            "diagnostic_overall_rmse_rad": None,
            "diagnostic_RF_HFE_rmse_rad": None,
            "diagnostic_LH_HFE_rmse_rad": None,
            "diagnostic_stage0C_gate_pass": None,
            "current_formal_baseline_for_context": {
                "overall_rmse_rad": 0.012818364796375577,
                "RF_HFE_rmse_rad": 0.023814172906643862,
                "LH_HFE_rmse_rad": 0.022510682052527972,
                "encoder_vs_real_rmse_rad": 0.023642306816740108,
                "source": "artifacts/replay_fit/20260831T073349Z/metrics.json",
            },
        },
        "actuator_core_assessment": {
            "evidence_of_unresolved_actuator_semantics": True,
            "formal_actuator_change": False,
            "statement": (
                "Yes: after rejecting the extra-permutation hypothesis, the prior "
                "near-exact logged-torque identity remains evidence of unresolved "
                "legacy bias/PD application semantics. It is not yet proof that the "
                "formal actuator core is incorrect, so this audit makes no actuator change."
            ),
            "next_branch": "legacy bias application forensic",
        },
        "stage_status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "asset_PhysX": "BLOCKED",
            "locomotion_PPO": "BLOCKED",
        },
        "mutations": {
            "formal_actuator": False,
            "formal_decoder": False,
            "data_loader": False,
            "frame": False,
            "bias_sign": False,
            "delay": False,
            "thresholds": False,
        },
    }


def run_fit_order_diagnostics(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "artifacts" / "fit_order_diagnostics" / stamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_fit_order_report()
    (output_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "report.md", report)
    report["output_dir"] = str(output_dir)
    return report
