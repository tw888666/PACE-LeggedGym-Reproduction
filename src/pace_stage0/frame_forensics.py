from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .constants import (
    CANONICAL_JOINT_NAMES,
    DATA_PATH,
    FIT_PATH,
    PACE_REFERENCE_COMMIT,
    PROJECT_ROOT,
    RAW_JOINT_NAMES,
    RAW_TO_CANONICAL,
)
from .data import load_replay_data
from .decoder import decode_fit


DEFAULT_JSON_PATH = PROJECT_ROOT / "artifacts" / "frame_forensics.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "artifacts" / "frame_forensics.md"

PAPER_ID = "arXiv:2509.06342v2"
OFFICIAL_REPOSITORY = "https://github.com/leggedrobotics/pace-sim2real"
DATASET_ITEM = "https://www.research-collection.ethz.ch/items/ea53e17a-76eb-460f-81ba-6e65f7078539"
DATASET_BITSTREAM = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "4939743e-8a89-40e9-ac78-8e98331c729c/content"
)


def _native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _relationship_stats(
    sim: np.ndarray,
    reference: np.ndarray,
    joint_names: Sequence[str],
) -> Dict[str, Any]:
    sim = np.asarray(sim, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if sim.shape != reference.shape or sim.ndim != 2:
        raise ValueError(
            "Frame-forensic inputs must be equal [time,joint] arrays; "
            f"got {sim.shape} and {reference.shape}"
        )
    if sim.shape[1] != len(joint_names):
        raise ValueError("Joint-name count does not match trajectory width")

    error = sim - reference
    squared = np.square(error)
    joint_squared_sum = np.sum(squared, axis=0, dtype=np.float64)
    total_squared_sum = float(np.sum(joint_squared_sum, dtype=np.float64))
    if total_squared_sum == 0.0:
        contribution = np.zeros_like(joint_squared_sum)
    else:
        contribution = joint_squared_sum / total_squared_sum

    return {
        "overall": {
            "rmse": float(np.sqrt(np.mean(squared, dtype=np.float64))),
            "state0_rmse": float(
                np.sqrt(np.mean(squared[0], dtype=np.float64))
            ),
            "mean_error": float(np.mean(error, dtype=np.float64)),
            "median_error": float(np.median(error)),
            "max_absolute_error": float(np.max(np.abs(error))),
        },
        "per_joint": [
            {
                "joint": name,
                "rmse": float(np.sqrt(np.mean(squared[:, index], dtype=np.float64))),
                "state0_error": float(error[0, index]),
                "state0_absolute_error": float(abs(error[0, index])),
                "mean_error": float(np.mean(error[:, index], dtype=np.float64)),
                "median_error": float(np.median(error[:, index])),
                "max_absolute_error": float(np.max(np.abs(error[:, index]))),
                "squared_error_contribution_ratio": float(contribution[index]),
                "squared_error_contribution_percent": float(100.0 * contribution[index]),
            }
            for index, name in enumerate(joint_names)
        ],
    }


def _per_joint_by_name(stats: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {row["joint"]: row for row in stats["per_joint"]}


def _format_number(value: float) -> str:
    return f"{value:.9f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header, rule] + body)


def _paper_parameter_audit(fit) -> Dict[str, Any]:
    # Values printed in paper Table 6, canonical LF RF LH RH order. They are
    # rounded publication values and therefore are used only as a semantic
    # block/order audit, never as replacement training parameters.
    paper = {
        "friction": np.asarray(
            [0.0054, 0.021, 0.028, 0.0035, 0.027, 0.036,
             0.0032, 0.024, 0.040, 0.0029, 0.013, 0.045],
            dtype=np.float64,
        ),
        "damping": np.asarray(
            [4.9, 4.4, 5.2, 4.7, 4.3, 5.3, 4.9, 4.9, 5.4, 5.1, 5.1, 5.5],
            dtype=np.float64,
        ),
        "armature": np.asarray(
            [0.076, 0.076, 0.067, 0.074, 0.077, 0.067,
             0.089, 0.051, 0.064, 0.079, 0.039, 0.051],
            dtype=np.float64,
        ),
        "encoder_bias": np.asarray(
            [0.022, 0.0057, -0.003, 0.011, -0.0072, 0.0094,
             -0.012, -0.0013, -0.0095, -0.016, 0.0043, 0.0045],
            dtype=np.float64,
        ),
    }
    decoded = {
        "friction": fit.friction,
        "damping": fit.damping,
        "armature": fit.armature,
        "encoder_bias": fit.encoder_bias,
    }
    comparisons = {}
    for name in paper:
        error = np.asarray(decoded[name], dtype=np.float64) - paper[name]
        comparisons[name] = {
            "paper_table6_rounded": paper[name],
            "decoded_canonical": decoded[name],
            "rmse_to_paper_rounded": float(
                np.sqrt(np.mean(np.square(error), dtype=np.float64))
            ),
            "max_abs_to_paper_rounded": float(np.max(np.abs(error))),
        }
    canonical_indices = np.asarray(RAW_TO_CANONICAL, dtype=np.int64)
    current_public_order_applied_to_legacy = {
        "armature": fit.params_raw[0:12][canonical_indices],
        "friction": fit.params_raw[24:36][canonical_indices],
    }
    swapped_order_errors = {}
    for name, values in current_public_order_applied_to_legacy.items():
        error = np.asarray(values, dtype=np.float64) - paper[name]
        swapped_order_errors[name] = {
            "decoded_if_current_public_order_were_used": values,
            "rmse_to_paper_rounded": float(
                np.sqrt(np.mean(np.square(error), dtype=np.float64))
            ),
            "max_abs_to_paper_rounded": float(np.max(np.abs(error))),
        }
    return {
        "paper_parameter_vector_equation_order": [
            "armature", "damping", "friction", "encoder_bias", "delay"
        ],
        "legacy_fitting_npy_observed_serialization": [
            "friction", "damping", "armature", "encoder_bias", "delay"
        ],
        "current_public_cmaes_serialization": [
            "armature", "damping", "friction", "encoder_bias", "delay"
        ],
        "comparisons": comparisons,
        "counterfactual_current_public_order_on_legacy_file": swapped_order_errors,
        "verdict": (
            "The reproduction's legacy friction/damping/armature/bias/delay decoder is "
            "strongly corroborated by block magnitudes and paper Table 6. The current "
            "public CMA-ES array order is not the legacy fitting.npy order. The actual "
            "legacy serializer source was not included in the dataset ZIP, so this is "
            "artifact-and-paper confirmation rather than legacy-code confirmation."
        ),
        "modification_required_now": False,
    }


def build_frame_forensics() -> Dict[str, Any]:
    """Build a read-only forensic report without importing Isaac Gym or replaying."""
    data = load_replay_data()
    fit = decode_fit()
    sim = data.sim_method_dof_pos.astype(np.float64)
    real = data.real_dof_pos.astype(np.float64)
    bias = fit.encoder_bias.astype(np.float64)

    references = {
        "H0_sim_approximately_real": real,
        "Hplus_sim_approximately_real_plus_bias": real + bias[None, :],
        "Hminus_sim_approximately_real_minus_bias": real - bias[None, :],
    }
    hypotheses = {
        name: _relationship_stats(sim, reference, CANONICAL_JOINT_NAMES)
        for name, reference in references.items()
    }
    by_hypothesis = {
        name: _per_joint_by_name(stats) for name, stats in hypotheses.items()
    }

    first_row: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    h0_rows = by_hypothesis["H0_sim_approximately_real"]
    hp_rows = by_hypothesis["Hplus_sim_approximately_real_plus_bias"]
    hm_rows = by_hypothesis["Hminus_sim_approximately_real_minus_bias"]
    for index, joint in enumerate(CANONICAL_JOINT_NAMES):
        sim_minus_real = sim[0, index] - real[0, index]
        first_row.append(
            {
                "joint": joint,
                "real_0": real[0, index],
                "sim_method_0": sim[0, index],
                "bias": bias[index],
                "sim_minus_real": sim_minus_real,
                "sim_minus_real_plus_bias": sim_minus_real - bias[index],
                "sim_minus_real_minus_bias": sim_minus_real + bias[index],
            }
        )
        mean_error = h0_rows[joint]["mean_error"]
        candidates = {
            "mean_sim_minus_real_near_zero": abs(mean_error),
            "mean_sim_minus_real_near_plus_bias": abs(mean_error - bias[index]),
            "mean_sim_minus_real_near_minus_bias": abs(mean_error + bias[index]),
        }
        trajectory_rows.append(
            {
                "joint": joint,
                "rmse_sim_real": h0_rows[joint]["rmse"],
                "rmse_sim_real_plus_bias": hp_rows[joint]["rmse"],
                "rmse_sim_real_minus_bias": hm_rows[joint]["rmse"],
                "mean_sim_minus_real": mean_error,
                "bias": bias[index],
                "closest_mean_relation": min(candidates, key=candidates.get),
                "closest_mean_relation_absolute_residual": min(candidates.values()),
                "h0_squared_error_contribution_percent": h0_rows[joint][
                    "squared_error_contribution_percent"
                ],
                "hplus_squared_error_contribution_percent": hp_rows[joint][
                    "squared_error_contribution_percent"
                ],
                "hminus_squared_error_contribution_percent": hm_rows[joint][
                    "squared_error_contribution_percent"
                ],
            }
        )

    target_mean = np.mean(data.real_des_dof_pos.astype(np.float64), axis=0)
    target_front_hfe_kfe = bool(
        np.all(target_mean[[1, 4]] > 0.0)
        and np.all(target_mean[[2, 5]] < 0.0)
    )
    target_hind_hfe_kfe = bool(
        np.all(target_mean[[7, 10]] < 0.0)
        and np.all(target_mean[[8, 11]] > 0.0)
    )

    paper_audit = _paper_parameter_audit(fit)
    raw_bias = fit.params_raw[36:48].astype(np.float64)
    canonical_from_raw = raw_bias[np.asarray(RAW_TO_CANONICAL, dtype=np.int64)]
    report = {
        "schema": "pace_stage0.frame_forensics.v2",
        "scope": {
            "mode": "read_only_legacy_frame_forensics",
            "formal_replay_run": False,
            "frame_acknowledgement_flag_used": False,
            "frame_assumption_frozen_for_stage0C": True,
            "actuator_core_modified": False,
            "formal_metric_frame": "q_encoder_ours vs sim_method.dof_pos at zero lag",
        },
        "stage0_status": {
            "Stage_0A": "PASS — runtime, parameter decoding, actuator implementation, unit tests",
            "Stage_0B": (
                "ACCEPTED FOR ENGINEERING REPLAY — exporter unavailable; frame is "
                "strong_inference/high"
            ),
            "Stage_0C": "AUTHORIZED — not run by this read-only forensic tool",
        },
        "inputs": {
            "data_path": DATA_PATH,
            "data_sha256": data.sha256,
            "fit_path": FIT_PATH,
            "fit_sha256": fit.sha256,
            "samples": data.sample_count,
            "joints": data.joint_count,
        },
        "field_frame_verdicts": {
            "real.dof_pos": {
                "best_supported_definition": "measured encoder-frame joint position",
                "confidence": "strong_inference_not_legacy_export_proof",
                "reason": (
                    "The paper describes encoder-only identification; current official fit code "
                    "uses data['dof_pos'] as measured_dof_pos and compares the simulated true "
                    "position after subtracting fitted bias. The legacy exporter is unavailable."
                ),
            },
            "real.des_dof_pos": {
                "best_supported_definition": (
                    "absolute joint-position command/setpoint in the robot command/encoder convention"
                ),
                "confidence": "strong_inference_not_legacy_export_proof",
                "reason": (
                    "The dataset plot labels it Target and the current official fit code applies it "
                    "directly as an absolute JointPositionAction target with scale 1 and no offset."
                ),
            },
            "sim_method.dof_pos": {
                "A_true_position": "not supported by the released array behavior alone",
                "B_true_minus_bias_encoder_comparison_position": "frozen engineering assumption",
                "C_other_conversion": "cannot be excluded",
                "formal_verdict": "strong_inference = bias-corrected comparison frame",
                "confidence_for_B": "high",
                "legacy_exporter_implementation": "UNAVAILABLE / provenance unresolved",
                "reason": (
                    "The first sample is almost identical to real[0], H0 beats H+ overall, and the "
                    "paper plotting script overlays sim_method and real without subtracting bias. "
                    "Current official tooling also explicitly converts simulated true trajectories "
                    "to encoder frame for plotting. None of these sources is the legacy data.npy "
                    "export statement, so they do not establish B as code-proven fact."
                ),
            },
        },
        "source_evidence": [
            {
                "source": PAPER_ID,
                "location": "Eq. 2 and Table 6",
                "commit": None,
                "evidence": (
                    "The conceptual vector is [armature,damping,friction,bias,delay]; Table 6 "
                    "publishes parameter values in canonical LF RF LH RH order."
                ),
                "role": "method and rounded-output corroboration; not legacy serialization code",
            },
            {
                "source": OFFICIAL_REPOSITORY,
                "location": "scripts/pace/data_collection.py:119-154",
                "commit": PACE_REFERENCE_COMMIT,
                "line_provenance": {
                    "initial_true_plus_bias_and_zero_velocity": "55769942c766d720a5844ccdf61a368180fc4265",
                    "saved_dof_pos_true_minus_bias": "2d1d85ffc9133d580149b7d29f65faa6fa994374",
                },
                "evidence": (
                    "Current migrated collector initializes q_true=trajectory+bias, qdot=0 and "
                    "stores joint_pos-bias as dof_pos."
                ),
                "role": "current-code frame corroboration only",
            },
            {
                "source": OFFICIAL_REPOSITORY,
                "location": (
                    "scripts/pace/fit.py:61-98; "
                    "source/pace_sim2real/pace_sim2real/optim/cma_es.py:58-74,111-134"
                ),
                "commit": PACE_REFERENCE_COMMIT,
                "line_provenance": "2d1d85ffc9133d580149b7d29f65faa6fa994374",
                "evidence": (
                    "des_dof_pos is sent as the absolute target; CMA-ES loss is "
                    "sim_dof_pos-real_dof_pos-bias; initialization is real[0]+bias with qdot=0; "
                    "delay converts via Tensor.to(torch.int)."
                ),
                "role": "current method semantics; repository history begins after dataset release",
            },
            {
                "source": OFFICIAL_REPOSITORY,
                "location": "scripts/pace/plot_trajectory.py:74-118",
                "commit": PACE_REFERENCE_COMMIT,
                "line_provenance": "d5208220a6f8eec86929cfc2d2eb6e8585af6d70",
                "evidence": "Plots best simulated true trajectory minus encoder_bias against real.",
                "role": "current-code plotting-frame corroboration only",
            },
            {
                "source": DATASET_ITEM,
                "location": "pace_data.zip/pace_data/1_in_air/anymal/10_plot_dof_pos.py",
                "commit": None,
                "bitstream_uuid": "4939743e-8a89-40e9-ac78-8e98331c729c",
                "zip_size_bytes": 2314171869,
                "zip_md5": "8558715c498862d8c480cc5b7ae0e919",
                "evidence": (
                    "The paper plotting script reads sim_method.dof_pos and real.dof_pos and "
                    "overlays them directly, with no bias operation."
                ),
                "role": "legacy released consumer, not producer",
            },
            {
                "source": DATASET_BITSTREAM,
                "location": "ZIP central directory (77 entries)",
                "commit": None,
                "evidence": (
                    "The official archive contains NumPy data, readmes, and plotting/analysis "
                    "scripts but no data.npy generator/exporter or legacy fitting implementation."
                ),
                "role": "documents the provenance gap",
            },
        ],
        "hypotheses": hypotheses,
        "first_row_joint_table": first_row,
        "trajectory_joint_table": trajectory_rows,
        "joint_order_audit": {
            "fitting_raw_joint_names": RAW_JOINT_NAMES,
            "raw_to_canonical_indices": RAW_TO_CANONICAL,
            "canonical_joint_names": CANONICAL_JOINT_NAMES,
            "raw_bias": raw_bias,
            "raw_bias_reordered_canonical": canonical_from_raw,
            "reordered_matches_decoder_bias": bool(
                np.array_equal(canonical_from_raw, fit.encoder_bias)
            ),
            "isaac_gym_asset_order_from_stage0_smoke": RAW_JOINT_NAMES,
            "data_target_mean": target_mean,
            "data_command_pattern_first_two_triplets_front": target_front_hfe_kfe,
            "data_command_pattern_last_two_triplets_hind": target_hind_hfe_kfe,
            "data_order_verdict": (
                "Strongly supports canonical LF RF LH RH: columns 0:6 have front-leg HFE/KFE "
                "command signs and columns 6:12 have hind-leg signs, matching the current official "
                "ANYmal joint_order and trajectory-direction convention. data.npy carries no "
                "joint-name metadata, so this remains strong inference rather than self-described "
                "schema."
            ),
            "wrong_order_dominance_check": (
                "H0 is independent of bias order and already wins. H+ degradation is distributed "
                "across multiple joints; the report's per-joint contribution columns expose the "
                "largest contributors. No single raw/canonical swap explains state0 sim≈real."
            ),
        },
        "fitting_layout_audit": paper_audit,
        "delay_audit": {
            "delay_raw": fit.delay_raw,
            "current_reproduction_steps": fit.delay_steps,
            "current_public_conversion": "Tensor.to(torch.int): truncation toward zero",
            "current_public_conversion_commit": "2d1d85ffc9133d580149b7d29f65faa6fa994374",
            "paper_reported_delay_ms": 7.5,
            "physics_dt_ms": 2.5,
            "paper_implied_steps": 3,
            "verdict": (
                "Three steps is strongly corroborated by int(3.240597...), current official code, "
                "and the paper's 7.5 ms at 400 Hz. The unavailable legacy implementation prevents "
                "claiming source-level proof of its discretization operator."
            ),
            "modification_required_now": False,
        },
        "frame_assumption_rationale": {
            "reason": (
                "The former preflight assumed sim_method.dof_pos was physics true position and "
                "therefore expected H+ to improve over H0. Released data instead has "
                "sim_method[0]≈real[0] and H0 lower than H+. Combined with the paper, later official "
                "implementation, and official dataset usage, this now supports the frozen "
                "strong_inference/high bias-corrected comparison-frame assumption."
            ),
            "current_reproduction_correct": [
                "PACE actuator bias equation q_encoder=q_true-bias",
                "absolute-target core/adapter boundary",
                "q_true_0=real[0]+bias and qdot_0=0 for current official sysID semantics",
                "legacy fitting.npy block interpretation is strongly supported",
                "raw-to-canonical fitting parameter permutation",
                "three-step delay is strongly supported",
                "no automatic frame selection",
            ],
            "approved_stage0C_changes": [
                "Treat released sim_method.dof_pos as bias-corrected comparison frame",
                "Use q_encoder_ours vs sim_method.dof_pos as the formal zero-lag metric",
            ],
            "not_changed_by_this_tool": True,
        },
        "unresolved": [
            "The exact legacy statement that writes data['sim_method']['dof_pos']",
            "Whether that writer subtracts fitted bias or performs another conversion",
            "The exact legacy statement that writes real.dof_pos and real.des_dof_pos",
            "The legacy fitting optimizer's serialized array layout and delay cast source code",
            "Machine-readable joint names for the columns of legacy data.npy",
        ],
        "recommendation": {
            "can_unblock_stage0B": True,
            "next_action": (
                "Proceed with Stage 0C using q_encoder_ours vs sim_method.dof_pos at zero lag. "
                "If it fails, diagnose lag, per-joint errors, and the first-10-frame torque/delay "
                "trace without changing the frozen frame. Upgrade provenance to author_confirmed "
                "only after an explicit author statement."
            ),
        },
    }
    return _native(report)


def render_markdown(report: Mapping[str, Any]) -> str:
    hypotheses = report["hypotheses"]
    summary_rows = []
    for name, stats in hypotheses.items():
        overall = stats["overall"]
        summary_rows.append(
            [
                name,
                _format_number(overall["rmse"]),
                _format_number(overall["state0_rmse"]),
                _format_number(overall["mean_error"]),
                _format_number(overall["median_error"]),
                _format_number(overall["max_absolute_error"]),
            ]
        )

    first_rows = [
        [
            row["joint"],
            _format_number(row["real_0"]),
            _format_number(row["sim_method_0"]),
            _format_number(row["bias"]),
            _format_number(row["sim_minus_real"]),
            _format_number(row["sim_minus_real_plus_bias"]),
            _format_number(row["sim_minus_real_minus_bias"]),
        ]
        for row in report["first_row_joint_table"]
    ]
    trajectory_rows = [
        [
            row["joint"],
            _format_number(row["rmse_sim_real"]),
            _format_number(row["rmse_sim_real_plus_bias"]),
            _format_number(row["rmse_sim_real_minus_bias"]),
            _format_number(row["mean_sim_minus_real"]),
            _format_number(row["bias"]),
            f"{row['h0_squared_error_contribution_percent']:.2f}%",
            f"{row['hplus_squared_error_contribution_percent']:.2f}%",
            row["closest_mean_relation"],
        ]
        for row in report["trajectory_joint_table"]
    ]
    evidence_rows = [
        [
            item["source"],
            item["location"],
            item.get("commit") or "N/A",
            item["evidence"],
            item["role"],
        ]
        for item in report["source_evidence"]
    ]

    lines = [
        "# PACE Stage 0 frame forensics",
        "",
        "> Read-only audit. Formal replay was not run by this tool; no acknowledgement flag "
        "was used, and actuator semantics were not modified.",
        "",
        "## Stage status",
        "",
        f"- Stage 0A: {report['stage0_status']['Stage_0A']}",
        f"- Stage 0B: {report['stage0_status']['Stage_0B']}",
        f"- Stage 0C: {report['stage0_status']['Stage_0C']}",
        "",
        "## Field verdicts",
        "",
        _markdown_table(
            ["Field", "Best supported definition", "Confidence / formal verdict"],
            [
                [
                    name,
                    value.get(
                        "best_supported_definition",
                        "likely encoder/comparison position (q_true-bias); not proven",
                    ),
                    value.get("confidence", value.get("formal_verdict")),
                ]
                for name, value in report["field_frame_verdicts"].items()
            ],
        ),
        "",
        "The released evidence supports freezing `sim_method.dof_pos` as an already "
        "bias-corrected comparison frame with `strong_inference/high` provenance. The legacy "
        "exporter remains unavailable, so this is not paper-exact or source-code-confirmed.",
        "",
        "## Evidence chain",
        "",
        _markdown_table(
            ["Source", "Path/location", "Commit", "Evidence", "Use"], evidence_rows
        ),
        "",
        "## Hypothesis summary",
        "",
        _markdown_table(
            ["Hypothesis", "Overall RMSE", "State0 RMSE", "Mean error", "Median error", "Max abs"],
            summary_rows,
        ),
        "",
        "## First sample by joint",
        "",
        _markdown_table(
            [
                "Joint", "real[0]", "sim_method[0]", "bias", "sim-real",
                "sim-(real+bias)", "sim-(real-bias)",
            ],
            first_rows,
        ),
        "",
        "## Full-trajectory joint statistics",
        "",
        _markdown_table(
            [
                "Joint", "RMSE H0", "RMSE H+", "RMSE H-", "mean(sim-real)",
                "bias", "H0 SE share", "H+ SE share", "closest mean relation",
            ],
            trajectory_rows,
        ),
        "",
        "## Parameter layout and delay",
        "",
        f"- Fitting layout: {report['fitting_layout_audit']['verdict']}",
        f"- Delay: {report['delay_audit']['verdict']}",
        f"- Joint order: {report['joint_order_audit']['data_order_verdict']}",
        "",
        "## Frozen Stage 0C frame rationale",
        "",
        report["frame_assumption_rationale"]["reason"],
        "",
        "## Unresolved provenance",
        "",
    ]
    lines.extend(f"- {item}" for item in report["unresolved"])
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["recommendation"]["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def write_frame_forensics(
    json_path: Path = DEFAULT_JSON_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
) -> Dict[str, Any]:
    report = build_frame_forensics()
    json_path = json_path.resolve()
    markdown_path = markdown_path.resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report
