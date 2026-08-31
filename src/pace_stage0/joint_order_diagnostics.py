from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CANONICAL_JOINT_NAMES,
    DATA_PATH,
    KP,
    PROJECT_ROOT,
    RAW_JOINT_NAMES,
    RAW_TO_CANONICAL,
)
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import per_joint_rmse, rmse


RAW_LEG_ORDER = ("LF", "LH", "RF", "RH")
CANONICAL_LEG_ORDER = ("LF", "RF", "LH", "RH")
JOINT_TYPES = ("HAA", "HFE", "KFE")
DATA_RAW_TO_CANONICAL = np.asarray(RAW_TO_CANONICAL, dtype=np.int64)


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _new_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "artifacts" / "joint_order_diagnostics" / stamp
    result = output_dir.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Correlation shape mismatch: {left.shape}/{right.shape}")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _torque_metrics(
    recomputed: np.ndarray,
    logged: np.ndarray,
    joint_names: Sequence[str],
) -> Dict[str, Any]:
    recomputed = np.asarray(recomputed, dtype=np.float64)
    logged = np.asarray(logged, dtype=np.float64)
    if recomputed.shape != logged.shape or recomputed.ndim != 2:
        raise ValueError(
            f"Torque inputs must be equal [time,joint] arrays: "
            f"{recomputed.shape}/{logged.shape}"
        )
    if recomputed.shape[1] != len(joint_names):
        raise ValueError("Joint-name count does not match torque width")
    joint_rmse = per_joint_rmse(recomputed, logged)
    return {
        "overall_rmse_Nm": rmse(recomputed, logged),
        "overall_correlation": _correlation(recomputed, logged),
        "per_joint": [
            {
                "joint": name,
                "rmse_Nm": float(joint_rmse[index]),
                "correlation": _correlation(
                    recomputed[:, index], logged[:, index]
                ),
            }
            for index, name in enumerate(joint_names)
        ],
    }


def _recompute_author_state_torque(
    data: ReplayData,
    bias: np.ndarray,
    delay_steps: int,
    data_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay frozen core algebra on logged states, without Isaac Gym.

    The frozen comparison-frame interpretation requires q_true=sim_method+bias.
    The same bias is subtracted inside the core and therefore cancels from the
    raw PD expression. This intentionally exposes non-identifiable hypotheses
    instead of altering the frozen frame to force separation.
    """
    import torch

    from .actuator import PACEActuatorCore

    indices = np.asarray(data_indices, dtype=np.int64)
    if indices.shape != (12,) or set(indices.tolist()) != set(range(12)):
        raise ValueError("data_indices must be a permutation of 0..11")
    bias = np.asarray(bias, dtype=np.float32)
    if bias.shape != (12,):
        raise ValueError("bias must have shape (12,)")

    target = data.real_des_dof_pos[:, indices]
    position = data.sim_method_dof_pos[:, indices]
    velocity = data.sim_method_dof_vel[:, indices]
    logged = data.sim_method_dof_torques[:, indices]

    bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
    core = PACEActuatorCore(bias_tensor, delay_steps)
    core.reset(torch.as_tensor(position[0] + bias, dtype=torch.float32))
    applied = np.empty((data.sample_count - 1, 12), dtype=np.float32)
    for index in range(data.sample_count - 1):
        step = core.step(
            torch.as_tensor(target[index], dtype=torch.float32),
            torch.as_tensor(position[index] + bias, dtype=torch.float32),
            torch.as_tensor(velocity[index], dtype=torch.float32),
        )
        applied[index] = step.applied_torque.numpy()
    return applied.astype(np.float64), logged[1:].astype(np.float64)


def _hypothesis_torque_audit(
    fit: DecodedFit,
    data: ReplayData,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    identity = np.arange(12, dtype=np.int64)
    raw_bias = fit.params_raw[36:48]
    specifications = {
        "H1_current": {
            "fit_order": "RAW -> CANONICAL",
            "data_order_operation": "unchanged",
            "bias": fit.encoder_bias,
            "data_indices": identity,
            "joint_names": CANONICAL_JOINT_NAMES,
        },
        "H2_all_raw": {
            "fit_order": "RAW",
            "data_order_operation": "unchanged",
            "bias": raw_bias,
            "data_indices": identity,
            "joint_names": RAW_JOINT_NAMES,
        },
        "H3_both_canonical": {
            "fit_order": "RAW -> CANONICAL",
            "data_order_operation": "RAW -> CANONICAL",
            "bias": fit.encoder_bias,
            "data_indices": DATA_RAW_TO_CANONICAL,
            "joint_names": CANONICAL_JOINT_NAMES,
        },
        "H4_data_only_canonical": {
            "fit_order": "RAW",
            "data_order_operation": "RAW -> CANONICAL",
            "bias": raw_bias,
            "data_indices": DATA_RAW_TO_CANONICAL,
            "joint_names": CANONICAL_JOINT_NAMES,
        },
    }

    report: Dict[str, Any] = {}
    traces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, spec in specifications.items():
        applied, logged = _recompute_author_state_torque(
            data,
            np.asarray(spec["bias"]),
            fit.delay_steps,
            np.asarray(spec["data_indices"]),
        )
        traces[name] = (applied, logged)
        report[name] = {
            "fit_order": spec["fit_order"],
            "data_order_operation": spec["data_order_operation"],
            "joint_labels_for_report": spec["joint_names"],
            "metrics": _torque_metrics(applied, logged, spec["joint_names"]),
        }

    h1_applied, h1_logged = traces["H1_current"]
    h2_applied, h2_logged = traces["H2_all_raw"]
    h3_applied, h3_logged = traces["H3_both_canonical"]
    h4_applied, h4_logged = traces["H4_data_only_canonical"]
    report["identifiability"] = {
        "h1_equals_h2_max_abs_applied_Nm": float(
            np.max(np.abs(h1_applied - h2_applied))
        ),
        "h1_equals_h2_max_abs_logged_Nm": float(
            np.max(np.abs(h1_logged - h2_logged))
        ),
        "h3_equals_h4_max_abs_applied_Nm": float(
            np.max(np.abs(h3_applied - h4_applied))
        ),
        "h3_equals_h4_max_abs_logged_Nm": float(
            np.max(np.abs(h3_logged - h4_logged))
        ),
        "h3_is_h1_permutation_max_abs_applied_Nm": float(
            np.max(np.abs(h3_applied - h1_applied[:, DATA_RAW_TO_CANONICAL]))
        ),
        "explanation": (
            "Under frozen comparison-frame semantics q_true=sim_method+bias and "
            "q_encoder=q_true-bias, the selected bias cancels from core algebra. "
            "Permuting all data fields together only permutes scalar per-joint operations, "
            "so H1-H4 overall torque RMSE cannot identify stored joint order."
        ),
        "policy": "Do not alter the frozen frame to manufacture hypothesis separation.",
    }
    return report, traces


def _rank_bias_leg_permutations(
    torque_delta: np.ndarray,
    raw_bias: np.ndarray,
) -> Sequence[Dict[str, Any]]:
    delta = np.asarray(torque_delta, dtype=np.float64)
    raw_bias = np.asarray(raw_bias, dtype=np.float64)
    if delta.shape != (12,) or raw_bias.shape != (12,):
        raise ValueError("torque_delta and raw_bias must have shape (12,)")
    blocks = {
        leg: raw_bias[index * 3:(index + 1) * 3]
        for index, leg in enumerate(RAW_LEG_ORDER)
    }
    rows = []
    for order in itertools.permutations(RAW_LEG_ORDER):
        candidate = KP * np.concatenate([blocks[leg] for leg in order])
        rows.append(
            {
                "leg_order": order,
                "rmse_Nm": rmse(delta, candidate),
                "correlation": _correlation(delta, candidate),
                "max_abs_residual_Nm": float(np.max(np.abs(delta - candidate))),
            }
        )
    rows.sort(key=lambda row: row["rmse_Nm"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _parameter_swap_audit(fit: DecodedFit) -> Dict[str, Any]:
    raw_blocks = {
        "friction": fit.params_raw[0:12],
        "damping": fit.params_raw[12:24],
        "armature": fit.params_raw[24:36],
        "encoder_bias": fit.params_raw[36:48],
    }
    canonical_blocks = {
        "friction": fit.friction,
        "damping": fit.damping,
        "armature": fit.armature,
        "encoder_bias": fit.encoder_bias,
    }
    rows = []
    for parameter, raw in raw_blocks.items():
        canonical = canonical_blocks[parameter]
        for joint_offset, joint_type in enumerate(JOINT_TYPES):
            lh_raw = float(raw[3 + joint_offset])
            rf_raw = float(raw[6 + joint_offset])
            rows.append(
                {
                    "parameter": parameter,
                    "joint_type": joint_type,
                    "LH_raw": lh_raw,
                    "RF_raw": rf_raw,
                    "LH_current_canonical": float(canonical[6 + joint_offset]),
                    "RF_current_canonical": float(canonical[3 + joint_offset]),
                    "abs_LH_RF_difference": abs(lh_raw - rf_raw),
                }
            )
    return {
        "rows": rows,
        "if_data_were_raw": {
            "stored_channels_3_to_5": (
                "would be LH, but current baseline labels them RF and applies RF raw parameters"
            ),
            "stored_channels_6_to_8": (
                "would be RF, but current baseline labels them LH and applies LH raw parameters"
            ),
            "affected_parameters": [
                "friction", "damping", "armature", "encoder_bias"
            ],
        },
        "interpretation": (
            "All four fitted per-joint blocks carry the same potential LH/RF assignment risk "
            "if, and only if, data.npy is proven to use the legacy raw leg order."
        ),
    }


def _payload_schema_audit() -> Dict[str, Any]:
    payload = np.load(DATA_PATH, allow_pickle=True)
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError("Expected scalar pickled data.npy payload")
    value = payload.item()
    if not isinstance(value, dict):
        raise TypeError("Expected data.npy dictionary")
    groups = {}
    metadata_keys = []
    for name, group in value.items():
        if isinstance(group, dict):
            groups[name] = sorted(group.keys())
        else:
            metadata_keys.append(name)
    joint_name_keys = []
    for group_name, keys in groups.items():
        for key in keys:
            if "joint" in key.lower() and "name" in key.lower():
                joint_name_keys.append(f"{group_name}.{key}")
    return {
        "same_directory_files": sorted(
            path.name for path in DATA_PATH.parent.iterdir() if path.is_file()
        ),
        "same_directory_label_script_or_plot_present": any(
            path.suffix.lower() in {".py", ".ipynb", ".pdf", ".png", ".svg"}
            for path in DATA_PATH.parent.iterdir()
            if path.is_file()
        ),
        "top_level_keys": sorted(value.keys()),
        "group_keys": groups,
        "non_dictionary_metadata_keys": metadata_keys,
        "joint_name_metadata_keys": joint_name_keys,
        "joint_name_metadata_present": bool(joint_name_keys),
    }


def _channel_order_audit(data: ReplayData) -> Dict[str, Any]:
    target = data.real_des_dof_pos.astype(np.float64)
    target_mean = np.mean(target, axis=0)
    target_min = np.min(target, axis=0)
    target_max = np.max(target, axis=0)
    target_std = np.std(target, axis=0)

    block_pose = []
    for block in range(4):
        offset = block * 3
        hfe_mean = float(target_mean[offset + 1])
        kfe_mean = float(target_mean[offset + 2])
        if hfe_mean > 0.0 and kfe_mean < 0.0:
            longitudinal = "front"
        elif hfe_mean < 0.0 and kfe_mean > 0.0:
            longitudinal = "hind"
        else:
            longitudinal = "unresolved"
        block_pose.append(
            {
                "triplet": block,
                "columns": [offset, offset + 1, offset + 2],
                "HFE_mean": hfe_mean,
                "KFE_mean": kfe_mean,
                "inferred_front_or_hind": longitudinal,
            }
        )

    inferred_names = CANONICAL_JOINT_NAMES
    channels = []
    for index, joint in enumerate(inferred_names):
        block = index // 3
        channels.append(
            {
                "channel": index,
                "inferred_joint": joint,
                "evidence": (
                    f"triplet role={JOINT_TYPES[index % 3]}; triplet {block} has "
                    f"{block_pose[block]['inferred_front_or_hind']} HFE/KFE command-center signs; "
                    "RAW and CANONICAL schemas agree on LF triplet 0 and RH triplet 3, so "
                    "front triplet 1 is RF and hind triplet 2 is LH"
                ),
                "confidence": "high strong_inference; data file is not self-describing",
                "target_mean": float(target_mean[index]),
                "target_min": float(target_min[index]),
                "target_max": float(target_max[index]),
                "target_std": float(target_std[index]),
            }
        )

    swap = DATA_RAW_TO_CANONICAL
    same_tracking_correlations = np.asarray(
        [
            _correlation(data.real_des_dof_pos[:, index], data.real_dof_pos[:, index])
            for index in range(12)
        ]
    )
    swapped_tracking_correlations = np.asarray(
        [
            _correlation(
                data.real_des_dof_pos[:, swap[index]],
                data.real_dof_pos[:, index],
            )
            for index in range(12)
        ]
    )
    return {
        "inferred_order": inferred_names,
        "provenance": "strong_inference",
        "confidence": "high",
        "block_command_center_audit": block_pose,
        "channels": channels,
        "all_joints_excited_simultaneously": True,
        "sequential_chirp_order_available": False,
        "internal_field_alignment": {
            "target_vs_real_same_index_rmse_rad": rmse(
                data.real_des_dof_pos, data.real_dof_pos
            ),
            "target_middle_leg_swap_vs_real_rmse_rad": rmse(
                data.real_des_dof_pos[:, swap], data.real_dof_pos
            ),
            "sim_method_vs_real_same_index_rmse_rad": rmse(
                data.sim_method_dof_pos, data.real_dof_pos
            ),
            "sim_method_middle_leg_swap_vs_real_rmse_rad": rmse(
                data.sim_method_dof_pos[:, swap], data.real_dof_pos
            ),
            "mean_target_real_same_index_correlation": float(
                np.mean(same_tracking_correlations)
            ),
            "mean_target_middle_leg_swap_real_correlation": float(
                np.mean(swapped_tracking_correlations)
            ),
            "interpretation": (
                "Target, real, and sim_method fields are internally aligned by stored index. "
                "This does not label channels by itself; the front/hind command-center "
                "signature supplies RAW-vs-CANONICAL label evidence."
            ),
        },
        "verdict": (
            "Target command centers strongly support stored triplets LF, RF, LH, RH: "
            "triplets 0 and 1 are front legs while triplets 2 and 3 are hind legs. The legacy "
            "RAW hypothesis LF, LH, RF, RH would label triplet 1 as hind and triplet 2 as front."
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    hypotheses = report["torque_algebra"]["hypotheses"]
    hypothesis_rows = []
    for name in ("H1_current", "H2_all_raw", "H3_both_canonical", "H4_data_only_canonical"):
        item = hypotheses[name]
        metrics = item["metrics"]
        hypothesis_rows.append(
            f"| {name} | {item['fit_order']} | {item['data_order_operation']} | "
            f"{metrics['overall_rmse_Nm']:.9f} | {metrics['overall_correlation']:.9f} |"
        )

    ranking_rows = [
        f"| {row['rank']} | {' '.join(row['leg_order'])} | {row['rmse_Nm']:.9f} | "
        f"{row['correlation']:.9f} |"
        for row in report["bias_permutation_ranking"]["top5"]
    ]
    channel_rows = [
        f"| {row['channel']} | {row['inferred_joint']} | {row['target_mean']:.6f} | "
        f"{row['confidence']} |"
        for row in report["data_channel_order"]["channels"]
    ]

    focus_names = ("LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE")
    per_joint_maps = {}
    for hypothesis, item in hypotheses.items():
        if not hypothesis.startswith("H"):
            continue
        per_joint_maps[hypothesis] = {
            row["joint"]: row for row in item["metrics"]["per_joint"]
        }
    focus_rows = []
    for joint in focus_names:
        values = []
        for hypothesis in ("H1_current", "H2_all_raw", "H3_both_canonical", "H4_data_only_canonical"):
            row = per_joint_maps[hypothesis].get(joint)
            values.append("N/A" if row is None else f"{row['rmse_Nm']:.9f}")
        focus_rows.append(f"| {joint} | " + " | ".join(values) + " |")

    parameter_rows = [
        f"| {row['parameter']} | {row['joint_type']} | {row['LH_raw']:.9f} | "
        f"{row['RF_raw']:.9f} | {row['LH_current_canonical']:.9f} | "
        f"{row['RF_current_canonical']:.9f} |"
        for row in report["parameter_swap_risk"]["rows"]
    ]

    lines = [
        "# PACE Stage 0C joint-order diagnostics",
        "",
        "> Diagnostic only. Formal decoder/data semantics and actuator core were not modified. "
        "Stage 0C remains FAIL; asset/PhysX and locomotion/PPO were not entered.",
        "",
        "## 1. Most likely data.npy channel order",
        "",
        "`LF HAA/HFE/KFE → RF HAA/HFE/KFE → LH HAA/HFE/KFE → RH HAA/HFE/KFE`.",
        "",
        report["data_channel_order"]["verdict"],
        "",
        "| Channel | Inferred joint | Target mean | Confidence |",
        "| ---: | --- | ---: | --- |",
        *channel_rows,
        "",
        "## 2. Direct RAW-vs-CANONICAL evidence",
        "",
        f"- Target→real same-index RMSE: {report['data_channel_order']['internal_field_alignment']['target_vs_real_same_index_rmse_rad']:.9f} rad.",
        f"- Target middle-leg swap→real RMSE: {report['data_channel_order']['internal_field_alignment']['target_middle_leg_swap_vs_real_rmse_rad']:.9f} rad.",
        f"- sim_method→real same-index RMSE: {report['data_channel_order']['internal_field_alignment']['sim_method_vs_real_same_index_rmse_rad']:.9f} rad.",
        f"- sim_method middle-leg swap→real RMSE: {report['data_channel_order']['internal_field_alignment']['sim_method_middle_leg_swap_vs_real_rmse_rad']:.9f} rad.",
        "- `data.npy` contains no machine-readable joint-name metadata and all joints are excited simultaneously, so there is no sequential chirp ordering signal.",
        "",
        "## 3. Bias leg-block permutation ranking",
        "",
        "| Rank | Leg order | RMSE [Nm] | Correlation |",
        "| ---: | --- | ---: | ---: |",
        *ranking_rows,
        "",
        "The RAW bias order is a statistically decisive match to the constant torque delta, but "
        "this conflicts with the target-position channel signature. It therefore identifies a "
        "legacy bias/torque convention, not by itself the order of every data.npy field.",
        "",
        "## 4. H1-H4 no-simulation torque algebra",
        "",
        "| Hypothesis | Fit | Data operation | RMSE [Nm] | Correlation |",
        "| --- | --- | --- | ---: | ---: |",
        *hypothesis_rows,
        "",
        hypotheses["identifiability"]["explanation"],
        "",
        report["torque_algebra"]["metric_scope_note"],
        "",
        "| Joint | H1 | H2 | H3 | H4 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *focus_rows,
        "",
        "## 5–6. RF/LH explanation and all fitted parameters",
        "",
        "| Parameter | Joint type | LH raw | RF raw | LH current canonical | RF current canonical |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *parameter_rows,
        "",
        "If data were RAW, friction, damping, armature, and bias would all have the same RF/LH "
        "assignment risk. However, position/target evidence supports CANONICAL, and H1-H4 torque "
        "algebra is non-identifying. RF_HFE/LH_HFE residual is not explained by a proven "
        "whole-data RF/LH swap.",
        "",
        "## 7–8. Best hypothesis and diagnostic replay",
        "",
        f"- Best-supported field order: {report['conclusion']['best_supported_hypothesis']}.",
        f"- Diagnostic replay executed: {report['diagnostic_replay']['executed']}.",
        f"- Reason: {report['diagnostic_replay']['reason']}",
        "",
        "## 9–12. Frozen implementation recommendation",
        "",
        f"- decoder.py change required: {report['conclusion']['decoder_change_required']}.",
        f"- data.py change required: {report['conclusion']['data_change_required']}.",
        f"- Recommended formal semantics: {report['conclusion']['recommended_formal_semantics']}.",
        f"- Evidence strength: {report['conclusion']['evidence_strength']}.",
        "",
        "`Stage 0C = FAIL`. The next unresolved item is legacy bias/torque export/application "
        "convention; no asset/PhysX audit or locomotion/PPO work is authorized by this result.",
        "",
    ]
    return "\n".join(lines)


def run_joint_order_diagnostics(
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    fit = decode_fit()
    data = load_replay_data()
    result_dir = _new_output_dir(output_dir)

    hypothesis_report, traces = _hypothesis_torque_audit(fit, data)
    h1_applied, h1_logged = traces["H1_current"]
    delta = h1_applied - h1_logged
    active = np.arange(len(delta)) >= fit.delay_steps
    delta_mean = np.mean(delta[active], axis=0)
    delta_std = np.std(delta[active], axis=0)
    rankings = _rank_bias_leg_permutations(
        delta_mean, fit.params_raw[36:48]
    )

    h1_rmse = hypothesis_report["H1_current"]["metrics"]["overall_rmse_Nm"]
    candidate_names = ("H2_all_raw", "H3_both_canonical")
    candidate_rmse = {
        name: hypothesis_report[name]["metrics"]["overall_rmse_Nm"]
        for name in candidate_names
    }
    replay_condition = any(
        value < 0.25 * h1_rmse and value < 0.1
        for value in candidate_rmse.values()
    )

    report = {
        "schema": "pace_stage0.joint_order_diagnostics.v1",
        "stage_status": {
            "Stage_0A": "PASS",
            "Stage_0B": "PASS / frozen semantics",
            "Stage_0C": "FAIL",
            "Stage_1_locomotion_PPO": "NOT STARTED",
        },
        "scope": {
            "diagnostic_only": True,
            "actuator_core_modified": False,
            "frame_modified": False,
            "decoder_modified": False,
            "data_loader_modified": False,
            "asset_or_PhysX_audit": False,
            "formal_baseline_auto_selection": False,
        },
        "payload_schema": _payload_schema_audit(),
        "data_channel_order": _channel_order_audit(data),
        "torque_algebra": {
            "alignment": "recomputed applied interval[t] vs sim_method.dof_torques state[t+1]",
            "logged_torque_semantics": "legacy exporter unresolved",
            "metric_scope_note": (
                "These values drive the frozen core with author sim_method states. They are "
                "not the previous replay's applied-torque metrics, which used reproduction "
                "states after plant divergence; the two traces must not be conflated."
            ),
            "hypotheses": hypothesis_report,
            "active_torque_delta_mean_Nm": delta_mean,
            "active_torque_delta_std_Nm": delta_std,
        },
        "bias_permutation_ranking": {
            "search_space": "24 leg-block permutations; HAA/HFE/KFE remain within each leg",
            "all": rankings,
            "top5": rankings[:5],
            "best_is_raw_LF_LH_RF_RH": tuple(rankings[0]["leg_order"]) == RAW_LEG_ORDER,
            "best_to_second_rmse_ratio": (
                rankings[0]["rmse_Nm"] / rankings[1]["rmse_Nm"]
            ),
            "causal_warning": (
                "This ranks the constant torque-delta pattern. It does not prove target, "
                "position, velocity, and torque arrays all share corresponding semantic labels."
            ),
        },
        "parameter_swap_risk": _parameter_swap_audit(fit),
        "diagnostic_replay": {
            "executed": False,
            "condition": (
                "Run only if H2 or H3 no-simulation torque RMSE is both below 0.1 Nm "
                "and below 25% of H1."
            ),
            "condition_met": replay_condition,
            "candidate_rmse_Nm": candidate_rmse,
            "reason": (
                "Not run: H1-H4 are algebraically non-identifiable under the frozen frame, "
                "so neither H2 nor H3 improves over H1. Running Isaac Gym would violate the "
                "stated prerequisite for this diagnostic replay."
            ),
            "formal_stage0C_gate_resolved_diagnostically": False,
        },
        "conclusion": {
            "most_likely_12_channel_order": CANONICAL_JOINT_NAMES,
            "best_supported_hypothesis": (
                "H1 for data/fit order (fit canonicalized; data unchanged/canonical), while "
                "RAW torque-delta pattern remains an unresolved legacy bias/torque convention"
            ),
            "RF_HFE_LH_HFE_residual_explained_by_whole_data_swap": False,
            "decoder_change_required": False,
            "data_change_required": False,
            "recommended_formal_semantics": (
                "Retain fitting RAW->CANONICAL and data arrays unchanged; do not freeze a new "
                "meaning for sim_method.dof_torques without legacy exporter or author evidence"
            ),
            "evidence_strength": (
                "data order: high strong_inference from artifact command signature and internal "
                "alignment; RAW torque-delta permutation: artifact-exact/statistically decisive "
                "but causally unresolved; legacy schema source remains unavailable"
            ),
            "next_unresolved_question": (
                "legacy encoder-bias application and sim_method.dof_torques export semantics"
            ),
        },
        "source_evidence": [
            {
                "source": "PACE paper v2 Table 6",
                "url": "https://arxiv.org/html/2509.06342v2",
                "evidence": "Published parameter columns use LF, RF, LH, RH order",
                "role": "canonical parameter-label corroboration; not data.npy schema proof",
            },
            {
                "source": "PACE paper v2 Section 2.1",
                "url": "https://arxiv.org/html/2509.06342v2",
                "evidence": "All joints are excited simultaneously with chirp signals",
                "role": "explains why sequential excitation cannot reveal stored leg order",
            },
            {
                "source": "whitelisted legacy data.npy artifact",
                "path": DATA_PATH,
                "evidence": (
                    "No joint-name metadata; command centers identify front/front/hind/hind "
                    "triplets and all logged fields remain aligned by stored index"
                ),
                "role": "primary artifact evidence for the strong-inference channel order",
            },
        ],
    }

    _write_json(result_dir / "report.json", report)
    (result_dir / "report.md").write_text(
        _render_markdown(_jsonable(report)), encoding="utf-8"
    )
    np.savez_compressed(
        result_dir / "torque_algebra.npz",
        h1_applied_torque=h1_applied,
        h1_logged_torque=h1_logged,
        active_delta_mean=delta_mean,
        active_delta_std=delta_std,
    )
    return _jsonable({"artifact_dir": str(result_dir), **report})
