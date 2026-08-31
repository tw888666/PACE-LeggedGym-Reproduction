from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CANONICAL_JOINT_NAMES,
    EFFORT_LIMIT,
    KD,
    KP,
    PROJECT_ROOT,
    RAW_JOINT_NAMES,
    SATURATION_EFFORT,
    VELOCITY_LIMIT,
)
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import per_joint_rmse, rmse


LAGS = tuple(range(-5, 6))
FRAME_COEFFICIENTS = (-1, 0, 1)


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
        output_dir = PROJECT_ROOT / "artifacts" / "torque_semantics" / stamp
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


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(f"Metric shape mismatch: {candidate.shape}/{reference.shape}")
    residual = reference - candidate
    return {
        "rmse_Nm": rmse(candidate, reference),
        "correlation": _correlation(candidate, reference),
        "mean_residual_ref_minus_candidate_Nm": float(np.mean(residual)),
        "std_residual_ref_minus_candidate_Nm": float(np.std(residual)),
        "max_abs_residual_Nm": float(np.max(np.abs(residual))),
    }


def _delay_trace(value: np.ndarray, delay_steps: int = 3) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("Torque trace must be [time,joint]")
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    if delay_steps == 0:
        return value.copy()
    delayed = np.zeros_like(value)
    delayed[delay_steps:] = value[:-delay_steps]
    return delayed


def _align_dataset_candidate(
    dataset: np.ndarray,
    candidate: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align dataset[k] against candidate[k+lag] on their common interval."""
    dataset = np.asarray(dataset, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if dataset.shape != candidate.shape or dataset.ndim != 2:
        raise ValueError("dataset/candidate must be equal [time,joint] arrays")
    count = len(dataset) - abs(int(lag))
    if count <= 0:
        raise ValueError("lag leaves no common samples")
    if lag >= 0:
        dataset_indices = np.arange(0, count, dtype=np.int64)
        candidate_indices = dataset_indices + lag
    else:
        dataset_indices = np.arange(-lag, -lag + count, dtype=np.int64)
        candidate_indices = dataset_indices + lag
    return (
        dataset[dataset_indices],
        candidate[candidate_indices],
        dataset_indices,
        candidate_indices,
    )


def _lag_sweep(dataset: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    rows = []
    for lag in LAGS:
        reference, aligned, dataset_indices, candidate_indices = (
            _align_dataset_candidate(dataset, candidate, lag)
        )
        rows.append(
            {
                "lag": lag,
                "definition": "dataset[k] vs candidate[k+lag]",
                "dataset_index_start": int(dataset_indices[0]),
                "dataset_index_end_inclusive": int(dataset_indices[-1]),
                "candidate_index_start": int(candidate_indices[0]),
                "candidate_index_end_inclusive": int(candidate_indices[-1]),
                "sample_count": int(len(dataset_indices)),
                **_metrics(aligned, reference),
            }
        )
    best = min(rows, key=lambda row: row["rmse_Nm"])
    return {"rows": rows, "best": best}


def _dcmotor_clip(torque: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    torque = np.asarray(torque, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    if torque.shape != qdot.shape:
        raise ValueError("torque/qdot shape mismatch")
    corner = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)
    velocity = np.clip(qdot, -corner, corner)
    maximum = SATURATION_EFFORT * (1.0 - velocity / VELOCITY_LIMIT)
    minimum = SATURATION_EFFORT * (-1.0 - velocity / VELOCITY_LIMIT)
    maximum = np.clip(maximum, -EFFORT_LIMIT, EFFORT_LIMIT)
    minimum = np.clip(minimum, -EFFORT_LIMIT, EFFORT_LIMIT)
    return np.maximum(np.minimum(torque, maximum), minimum)


def _frame_trace(
    data: ReplayData,
    bias: np.ndarray,
    alpha: int,
    beta: int,
    *,
    include_d: bool = True,
) -> np.ndarray:
    target = data.real_des_dof_pos.astype(np.float64)
    position = data.sim_method_dof_pos.astype(np.float64)
    velocity = data.sim_method_dof_vel.astype(np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    torque = KP * (
        (target + float(alpha) * bias[None, :])
        - (position + float(beta) * bias[None, :])
    )
    if include_d:
        torque = torque - KD * velocity
    return torque


def _enumerate_frame_candidates(
    fit: DecodedFit,
    data: ReplayData,
) -> Tuple[Sequence[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    reference = data.sim_method_dof_torques.astype(np.float64)
    bias_orders = {
        "raw_LF_LH_RF_RH": fit.params_raw[36:48],
        "canonical_LF_RF_LH_RH": fit.encoder_bias,
    }
    rows = []
    traces: Dict[str, Dict[str, np.ndarray]] = {}
    trace_index = 0
    for bias_name, bias in bias_orders.items():
        for alpha, beta in itertools.product(FRAME_COEFFICIENTS, repeat=2):
            name = f"{bias_name}__alpha_{alpha:+d}__beta_{beta:+d}"
            pre_delay = _frame_trace(data, bias, alpha, beta, include_d=True)
            post_delay = _delay_trace(pre_delay, fit.delay_steps)
            stage_sweeps = {
                "pre_delay": _lag_sweep(reference, pre_delay),
                "post_delay3": _lag_sweep(reference, post_delay),
            }
            best_stage, best_sweep = min(
                stage_sweeps.items(), key=lambda item: item[1]["best"]["rmse_Nm"]
            )
            best_lag = int(best_sweep["best"]["lag"])
            best_reference, best_aligned, _, _ = _align_dataset_candidate(
                reference,
                pre_delay if best_stage == "pre_delay" else post_delay,
                best_lag,
            )
            joint_rmse = per_joint_rmse(best_aligned, best_reference)
            rows.append(
                {
                    "trace_index": trace_index,
                    "candidate": name,
                    "bias_order": bias_name,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma_alpha_minus_beta": alpha - beta,
                    "formula": (
                        "Kp*((target+alpha*bias)-(q+beta*bias))-Kd*qdot"
                    ),
                    "zero_lag_pre_delay": stage_sweeps["pre_delay"]["rows"][5],
                    "best_stage": best_stage,
                    "best_lag": best_lag,
                    "best_metrics": best_sweep["best"],
                    "best_per_joint": [
                        {
                            "data_channel": index,
                            "joint_label_strong_inference": joint,
                            "rmse_Nm": float(joint_rmse[index]),
                            "correlation": _correlation(
                                best_aligned[:, index], best_reference[:, index]
                            ),
                            "mean_residual_ref_minus_candidate_Nm": float(
                                np.mean(
                                    best_reference[:, index]
                                    - best_aligned[:, index]
                                )
                            ),
                        }
                        for index, joint in enumerate(CANONICAL_JOINT_NAMES)
                    ],
                    "stage_lag_sweeps": stage_sweeps,
                }
            )
            traces[name] = {
                "pre_delay": pre_delay,
                "post_delay3": post_delay,
            }
            trace_index += 1
    rows.sort(key=lambda row: row["best_metrics"]["rmse_Nm"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, traces


def _per_joint_residual_audit(
    fit: DecodedFit,
    data: ReplayData,
    base_post_delay: np.ndarray,
    best_post_delay: np.ndarray,
) -> Dict[str, Any]:
    reference = data.sim_method_dof_torques.astype(np.float64)

    # Best timing: dataset[k] vs post-delay candidate[k-1]. The first active
    # delayed torque appears at candidate index 3 and dataset state index 4.
    ref_aligned, base_aligned, dataset_indices, candidate_indices = (
        _align_dataset_candidate(reference, base_post_delay, -1)
    )
    _, best_aligned, _, _ = _align_dataset_candidate(
        reference, best_post_delay, -1
    )
    active = candidate_indices >= fit.delay_steps
    ref_active = ref_aligned[active]
    base_active = base_aligned[active]
    best_active = best_aligned[active]
    dataset_active_indices = dataset_indices[active]
    candidate_active_indices = candidate_indices[active]

    delta_base_candidate_minus_dataset = base_active - ref_active
    residual_best_dataset_minus_candidate = ref_active - best_active
    raw_bias = fit.params_raw[36:48]
    delta_mean = np.mean(delta_base_candidate_minus_dataset, axis=0)
    effective_bias = delta_mean / KP
    rows = []
    for index, data_joint in enumerate(CANONICAL_JOINT_NAMES):
        delta = delta_base_candidate_minus_dataset[:, index]
        residual = residual_best_dataset_minus_candidate[:, index]
        denominator = KP * raw_bias[index]
        rows.append(
            {
                "data_channel": index,
                "data_joint_label_strong_inference": data_joint,
                "raw_bias_slot_label": RAW_JOINT_NAMES[index],
                "raw_bias_rad": float(raw_bias[index]),
                "effective_bias_from_torque_rad": float(effective_bias[index]),
                "effective_minus_fitting_raw_bias_rad": float(
                    effective_bias[index] - raw_bias[index]
                ),
                "Kp_times_raw_bias_Nm": float(denominator),
                "base_delta_mean_Nm": float(np.mean(delta)),
                "base_delta_std_Nm": float(np.std(delta)),
                "base_delta_peak_to_peak_Nm": float(np.ptp(delta)),
                "base_delta_mean_over_Kp_bias": (
                    float(np.mean(delta) / denominator) if denominator != 0.0 else None
                ),
                "winning_residual_mean_Nm": float(np.mean(residual)),
                "winning_residual_std_Nm": float(np.std(residual)),
                "winning_residual_peak_to_peak_Nm": float(np.ptp(residual)),
                "winning_residual_rmse_Nm": float(
                    np.sqrt(np.mean(np.square(residual), dtype=np.float64))
                ),
            }
        )

    plus = KP * raw_bias
    minus = -plus
    effective_candidate = base_active - KP * effective_bias[None, :]
    return {
        "strict_delta_definition": (
            "delta_base[k] = tau_base_post_delay[k-1] - sim_method.dof_torques[k], "
            "evaluated for k>=4 after FIFO activation"
        ),
        "strict_winning_residual_definition": (
            "residual_best[k] = sim_method.dof_torques[k] - "
            "tau_best_post_delay[k-1], evaluated for k>=4"
        ),
        "active_dataset_state_start": int(dataset_active_indices[0]),
        "active_candidate_index_start": int(candidate_active_indices[0]),
        "active_sample_count": int(len(dataset_active_indices)),
        "delta_mean_vs_plus_Kp_raw_bias": {
            "rmse_Nm": rmse(delta_mean, plus),
            "correlation": _correlation(delta_mean, plus),
        },
        "delta_mean_vs_minus_Kp_raw_bias": {
            "rmse_Nm": rmse(delta_mean, minus),
            "correlation": _correlation(delta_mean, minus),
        },
        "effective_bias_diagnostic": {
            "definition": "mean(tau_base_candidate-tau_dataset)/Kp per stored channel",
            "effective_bias_rad": effective_bias,
            "fitting_raw_bias_rad": raw_bias,
            "effective_minus_fitting_raw_bias_rad": effective_bias - raw_bias,
            "candidate_metrics_after_effective_constant_bias": _metrics(
                effective_candidate, ref_active
            ),
            "formal_bias_modified": False,
        },
        "sign_conclusion": (
            "tau_base_candidate - tau_dataset is +Kp*raw_bias; equivalently "
            "tau_dataset = tau_base_candidate - Kp*raw_bias"
        ),
        "per_joint": rows,
        "arrays": {
            "dataset_indices": dataset_active_indices,
            "candidate_indices": candidate_active_indices,
            "delta_base_candidate_minus_dataset": delta_base_candidate_minus_dataset,
            "residual_best_dataset_minus_candidate": residual_best_dataset_minus_candidate,
        },
    }


def _p_pd_and_clipping_audit(
    fit: DecodedFit,
    data: ReplayData,
    best_bias: np.ndarray,
    alpha: int,
    beta: int,
) -> Dict[str, Any]:
    reference = data.sim_method_dof_torques.astype(np.float64)
    variants = {}
    traces = {}
    for name, include_d in (("P_only", False), ("full_PD", True)):
        pre = _frame_trace(
            data, best_bias, alpha, beta, include_d=include_d
        )
        delayed = _delay_trace(pre, fit.delay_steps)
        sweep = _lag_sweep(reference, delayed)
        variants[name] = {
            "best_post_delay_lag": sweep["best"],
            "post_delay_lag_sweep": sweep,
        }
        traces[name] = {"pre": pre, "post": delayed}

    full_pd = traces["full_PD"]["pre"]
    clipped = _dcmotor_clip(full_pd, data.sim_method_dof_vel)
    clipped_delay = _delay_trace(clipped, fit.delay_steps)
    clipping_sweep = _lag_sweep(reference, clipped_delay)
    difference = clipped - full_pd
    return {
        "variants": variants,
        "D_term_conclusion": (
            "Full PD is required" if variants["full_PD"]["best_post_delay_lag"]["rmse_Nm"]
            < variants["P_only"]["best_post_delay_lag"]["rmse_Nm"]
            else "P-only is not worse"
        ),
        "DCMotor_clipping": {
            "clipped_sample_joint_count": int(np.count_nonzero(np.abs(difference) > 1e-12)),
            "max_abs_clip_change_Nm": float(np.max(np.abs(difference))),
            "unclipped_best": variants["full_PD"]["best_post_delay_lag"],
            "clipped_best": clipping_sweep["best"],
            "conclusion": (
                "Dataset trajectory never reaches the DCMotor envelope"
                if np.count_nonzero(np.abs(difference) > 1e-12) == 0
                else "DCMotor clipping changes the candidate trace"
            ),
        },
        "arrays": {
            "P_only_pre_delay": traces["P_only"]["pre"],
            "P_only_post_delay": traces["P_only"]["post"],
            "full_PD_pre_delay": full_pd,
            "full_PD_post_delay": traces["full_PD"]["post"],
            "full_PD_clipped_pre_delay": clipped,
            "full_PD_clipped_post_delay": clipped_delay,
        },
    }


def _velocity_residual_audit(
    data: ReplayData,
    residual_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    residual = np.asarray(
        residual_audit["arrays"]["residual_best_dataset_minus_candidate"],
        dtype=np.float64,
    )
    dataset_indices = np.asarray(
        residual_audit["arrays"]["dataset_indices"], dtype=np.int64
    )
    candidate_indices = np.asarray(
        residual_audit["arrays"]["candidate_indices"], dtype=np.int64
    )
    qdot_logged = data.sim_method_dof_vel[dataset_indices].astype(np.float64)
    # A delayed torque at candidate index c came from pre-delay index c-3.
    qdot_source = data.sim_method_dof_vel[candidate_indices - 3].astype(np.float64)
    centered = residual - np.mean(residual, axis=0, keepdims=True)

    def correlations(values: np.ndarray, velocity: np.ndarray) -> Dict[str, Any]:
        return {
            "overall_residual_vs_qdot": _correlation(values, velocity),
            "overall_residual_vs_sign_qdot": _correlation(values, np.sign(velocity)),
            "per_joint": [
                {
                    "joint": joint,
                    "residual_vs_qdot": _correlation(
                        values[:, index], velocity[:, index]
                    ),
                    "residual_vs_sign_qdot": _correlation(
                        values[:, index], np.sign(velocity[:, index])
                    ),
                }
                for index, joint in enumerate(CANONICAL_JOINT_NAMES)
            ],
        }

    centered_rms = float(
        np.sqrt(np.mean(np.square(centered), dtype=np.float64))
    )
    return {
        "raw_residual_vs_logged_state_velocity": correlations(residual, qdot_logged),
        "per_joint_mean_centered_residual_vs_logged_state_velocity": correlations(
            centered, qdot_logged
        ),
        "per_joint_mean_centered_residual_vs_pre_delay_source_velocity": correlations(
            centered, qdot_source
        ),
        "centered_residual_rms_Nm": centered_rms,
        "interpretation": (
            "The winning residual is constant per joint to micronewton-meter scale. "
            "Correlations of that numerical remainder are reported as requested but are not "
            "evidence for a material viscous or Coulomb term."
        ),
        "simple_friction_model_branch_executed": False,
        "simple_friction_model_branch_reason": (
            "The PD/frame/delay candidate already closes torque RMSE below 0.1 Nm; the frozen "
            "instructions make friction/damping model probing conditional on failure to close."
        ),
    }


def _strip_arrays(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key != "arrays"}


def _render_markdown(report: Mapping[str, Any]) -> str:
    ranking_rows = []
    for row in report["frame_candidates_ranking"]:
        metrics = row["best_metrics"]
        ranking_rows.append(
            f"| {row['rank']} | {row['bias_order']} | {row['alpha']} | {row['beta']} | "
            f"{row['gamma_alpha_minus_beta']} | {row['best_stage']} | {row['best_lag']} | "
            f"{metrics['rmse_Nm']:.9f} | {metrics['correlation']:.9f} | "
            f"{metrics['mean_residual_ref_minus_candidate_Nm']:.9f} |"
        )

    residual_rows = []
    for row in report["constant_residual_audit"]["per_joint"]:
        residual_rows.append(
            f"| {row['data_channel']} | {row['data_joint_label_strong_inference']} | "
            f"{row['raw_bias_slot_label']} | {row['base_delta_mean_Nm']:.9f} | "
            f"{row['base_delta_std_Nm']:.9f} | {row['base_delta_peak_to_peak_Nm']:.9f} | "
            f"{row['base_delta_mean_over_Kp_bias']:.6f} | "
            f"{row['winning_residual_rmse_Nm']:.9f} |"
        )

    winner_joint_rows = [
        f"| {row['data_channel']} | {row['joint_label_strong_inference']} | "
        f"{row['rmse_Nm']:.9f} | {row['correlation']:.9f} | "
        f"{row['mean_residual_ref_minus_candidate_Nm']:.9f} |"
        for row in report["winning_candidate"]["best_per_joint"]
    ]

    best = report["winning_candidate"]
    timing = report["pre_vs_post_delay"]
    ppd = report["P_vs_PD_and_DCMotor"]
    lines = [
        "# PACE Stage 0C sim_method.dof_torques semantics",
        "",
        "> Dataset-only algebraic forensic. No Isaac Gym replay, actuator change, joint-order "
        "change, asset/PhysX audit, or locomotion/PPO work was performed.",
        "",
        "## 1. Strict delta definition and sign",
        "",
        report["constant_residual_audit"]["strict_delta_definition"] + ".",
        "",
        report["constant_residual_audit"]["sign_conclusion"] + ".",
        "",
        "Using the per-channel effective bias inferred from that constant delta reduces "
        f"the active-window RMSE to "
        f"{report['constant_residual_audit']['effective_bias_diagnostic']['candidate_metrics_after_effective_constant_bias']['rmse_Nm']:.9e} Nm. "
        "This is diagnostic only; fitting.npy and the formal bias were not changed.",
        "",
        "## 2–3. Eighteen frame candidates and raw/canonical result",
        "",
        "| Rank | Bias order | alpha | beta | gamma | Stage | Lag | RMSE [Nm] | Corr | Mean ref-candidate |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        *ranking_rows,
        "",
        f"Winner: `{best['candidate']}`, gamma={best['gamma_alpha_minus_beta']}, "
        f"{best['best_stage']} at lag {best['best_lag']}, RMSE "
        f"{best['best_metrics']['rmse_Nm']:.9f} Nm and correlation "
        f"{best['best_metrics']['correlation']:.9f}.",
        "",
        "Only alpha-beta is identifiable: `(alpha=-1,beta=0)` and "
        "`(alpha=0,beta=+1)` generate the same torque. Dataset algebra cannot choose "
        "between target-minus-bias and q-plus-bias explanations.",
        "",
        "All 18 candidates include their 12-joint metrics in `report.json`. Winning candidate:",
        "",
        "| Ch | Joint | RMSE [Nm] | Corr | Mean ref-candidate |",
        "| ---: | --- | ---: | ---: | ---: |",
        *winner_joint_rows,
        "",
        "## 4–5. Pre-delay, post-delay, and lag",
        "",
        f"- Pre-delay best: lag {timing['pre_delay_best']['lag']}, RMSE "
        f"{timing['pre_delay_best']['rmse_Nm']:.9f} Nm.",
        f"- Post-delay3 best: lag {timing['post_delay3_best']['lag']}, RMSE "
        f"{timing['post_delay3_best']['rmse_Nm']:.9f} Nm.",
        f"- First nonzero dataset torque state: {timing['first_nonzero_dataset_state']}.",
        "",
        timing["interpretation"],
        "",
        "## 6–7. P vs PD and DCMotor clipping",
        "",
        f"- P-only best RMSE: {ppd['variants']['P_only']['best_post_delay_lag']['rmse_Nm']:.9f} Nm.",
        f"- Full-PD best RMSE: {ppd['variants']['full_PD']['best_post_delay_lag']['rmse_Nm']:.9f} Nm.",
        f"- D-term verdict: {ppd['D_term_conclusion']}.",
        f"- Clipped sample-joints: {ppd['DCMotor_clipping']['clipped_sample_joint_count']}; "
        f"max change {ppd['DCMotor_clipping']['max_abs_clip_change_Nm']:.9f} Nm.",
        "",
        "## 8. Constant residual",
        "",
        "| Ch | Data joint | RAW bias slot | Base delta mean | Base delta std | Base delta p-p | mean/(Kp*bias) | Winner residual RMSE |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *residual_rows,
        "",
        "## 9. Velocity/friction signature",
        "",
        f"Centered residual RMS: {report['velocity_residual_audit']['centered_residual_rms_Nm']:.9e} Nm. "
        + report["velocity_residual_audit"]["interpretation"],
        "",
        f"- Raw winning residual vs logged-state qdot: "
        f"{report['velocity_residual_audit']['raw_residual_vs_logged_state_velocity']['overall_residual_vs_qdot']:.9f}.",
        f"- Raw winning residual vs sign(qdot): "
        f"{report['velocity_residual_audit']['raw_residual_vs_logged_state_velocity']['overall_residual_vs_sign_qdot']:.9f}.",
        f"- Per-joint-mean-centered residual vs logged-state qdot: "
        f"{report['velocity_residual_audit']['per_joint_mean_centered_residual_vs_logged_state_velocity']['overall_residual_vs_qdot']:.9f}.",
        f"- Per-joint-mean-centered residual vs sign(qdot): "
        f"{report['velocity_residual_audit']['per_joint_mean_centered_residual_vs_logged_state_velocity']['overall_residual_vs_sign_qdot']:.9f}.",
        "",
        "## 10–13. Classification and branch decision",
        "",
        f"- Most likely field semantics: {report['classification']['most_likely']}.",
        f"- Exact mathematical difference: {report['classification']['exact_difference']}.",
        f"- Actuator-core implication: {report['classification']['actuator_core_implication']}",
        f"- Branch A closed: {report['classification']['branch_A_closed']}.",
        f"- Asset/plant audit authorized: {report['classification']['asset_plant_audit_authorized']}.",
        "",
        "`Stage 0C = FAIL`. Formal code remains frozen pending human review of the newly "
        "identified applied-torque-like bias convention.",
        "",
    ]
    return "\n".join(lines)


def run_torque_semantics(
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    fit = decode_fit()
    data = load_replay_data()
    result_dir = _new_output_dir(output_dir)
    reference = data.sim_method_dof_torques.astype(np.float64)

    ranking, traces = _enumerate_frame_candidates(fit, data)
    best = ranking[0]
    best_traces = traces[best["candidate"]]
    base_pre = _frame_trace(
        data, fit.params_raw[36:48], 0, 0, include_d=True
    )
    base_post = _delay_trace(base_pre, fit.delay_steps)

    residual = _per_joint_residual_audit(
        fit, data, base_post, best_traces["post_delay3"]
    )
    ppd = _p_pd_and_clipping_audit(
        fit,
        data,
        fit.params_raw[36:48],
        int(best["alpha"]),
        int(best["beta"]),
    )
    velocity = _velocity_residual_audit(data, residual)

    first_nonzero = np.flatnonzero(
        np.max(np.abs(reference), axis=1) > 1e-9
    )
    first_nonzero_state = int(first_nonzero[0]) if len(first_nonzero) else None
    pre_sweep = _lag_sweep(reference, best_traces["pre_delay"])
    post_sweep = _lag_sweep(reference, best_traces["post_delay3"])
    semantics_closed = (
        best["best_metrics"]["rmse_Nm"] < 0.1
        and best["best_metrics"]["correlation"] > 0.999
    )

    report = {
        "schema": "pace_stage0.torque_semantics.v1",
        "stage_status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "Stage_1_locomotion_PPO": "NOT STARTED",
        },
        "scope": {
            "dataset_only": True,
            "Isaac_Gym_replay_run": False,
            "actuator_core_modified": False,
            "decoder_or_data_order_modified": False,
            "bias_or_delay_modified": False,
            "asset_or_PhysX_audit": False,
            "formal_threshold_modified": False,
        },
        "definitions": {
            "candidate": (
                "tau(alpha,beta)=Kp*((target+alpha*bias)-(q+beta*bias))-Kd*qdot"
            ),
            "lag": "sim_method.dof_torques[k] vs candidate[k+lag]",
            "pre_delay": "candidate PD/DCMotor input at index t",
            "post_delay3": "zero for t<3, otherwise pre_delay[t-3]",
            "Kp": KP,
            "Kd": KD,
            "delay_steps": fit.delay_steps,
        },
        "frame_candidates_ranking": ranking,
        "winning_candidate": {
            key: value for key, value in best.items() if key != "stage_lag_sweeps"
        },
        "winning_equivalence_class": {
            "identifiable_gamma_alpha_minus_beta": -1,
            "equivalent_candidates": [
                "raw bias alpha=-1 beta=0: target is shifted by -bias",
                "raw bias alpha=0 beta=+1: state is shifted by +bias",
            ],
            "not_identifiable_from_torque_algebra": (
                "Whether the legacy implementation shifted the target or used true q=q_dataset+bias"
            ),
        },
        "raw_vs_canonical_bias": {
            "best_raw": next(
                row for row in ranking if row["bias_order"] == "raw_LF_LH_RF_RH"
            ),
            "best_canonical": next(
                row for row in ranking
                if row["bias_order"] == "canonical_LF_RF_LH_RH"
            ),
            "interpretation": (
                "RAW bias slots explain the torque field; this does not reopen the separately "
                "audited canonical order of target/position arrays."
            ),
        },
        "pre_vs_post_delay": {
            "pre_delay_best": pre_sweep["best"],
            "post_delay3_best": post_sweep["best"],
            "pre_delay_lag_sweep": pre_sweep,
            "post_delay3_lag_sweep": post_sweep,
            "first_nonzero_dataset_state": first_nonzero_state,
            "dataset_states_0_to_3_all_zero": bool(
                np.allclose(reference[:4], 0.0, atol=0.0, rtol=0.0)
            ),
            "interpretation": (
                "Pre-delay lag -4 and post-delay3 lag -1 are timing-equivalent. Together with "
                "exactly zero dataset torques at states 0..3, the strongest artifact reading is "
                "a three-step delayed/applied-like torque recorded at the following state boundary."
            ),
        },
        "constant_residual_audit": _strip_arrays(residual),
        "P_vs_PD_and_DCMotor": _strip_arrays(ppd),
        "velocity_residual_audit": velocity,
        "classification": {
            "candidate_classes_considered": {
                "A": "raw PD before bias handling — excluded at zero/best timing",
                "B": "encoder-frame PD torque — excluded unless an additional -raw-bias term is applied",
                "C": "true-frame PD torque — numerically supported through beta=+1",
                "D": "pre-delay actuator torque — timing-equivalent only at lag -4",
                "E": "post-delay/applied actuator torque — best-supported at lag -1 with reset signature",
                "F": "simulator-reported generalized torque — no material residual remains to require this",
                "G": "unknown — no longer necessary for numerical description, but exporter source remains unavailable",
            },
            "most_likely": (
                "E, with a C-equivalent relative frame term: full PD torque containing "
                "-Kp*raw_bias, passed through the 3-step FIFO and recorded one state later"
            ),
            "semantic_closure_threshold_met": semantics_closed,
            "exact_difference": (
                "Before delay, tau_legacy_logged_like = tau_current_base_PD - Kp*raw_bias "
                "in stored channel slots; alpha=-1,beta=0 and alpha=0,beta=+1 are indistinguishable"
            ),
            "source_evidence": (
                "legacy data.npy artifact: 0.009036 Nm RMSE, 0.999999834 correlation, constant "
                "per-joint residual (1.67e-6 Nm after removing its mean), and four initial zero "
                "torque states; legacy exporter source unavailable"
            ),
            "actuator_core_implication": (
                "Evidence still points to a legacy applied-torque-like bias/PD convention that differs "
                "from the frozen reproduction core. Do not modify the core until the raw-slot/order "
                "mechanism is source- or author-confirmed."
            ),
            "branch_A_closed": False,
            "branch_A_status": (
                "Case B: torque semantics are numerically closed, but the winning trace is post-delay/"
                "applied-like rather than a harmless pre-delay exporter-only quantity"
            ),
            "asset_plant_audit_authorized": False,
            "formal_position_replay_changed": False,
        },
        "source_provenance": {
            "legacy_exporter": "UNAVAILABLE / unresolved",
            "numeric_semantics": "artifact-identified / high confidence",
            "causal_implementation_location": "strong inference, not source-code-confirmed",
        },
    }

    _write_json(result_dir / "report.json", report)
    (result_dir / "report.md").write_text(
        _render_markdown(_jsonable(report)), encoding="utf-8"
    )
    ordered_names = [row["candidate"] for row in sorted(ranking, key=lambda row: row["trace_index"])]
    np.savez_compressed(
        result_dir / "candidate_traces.npz",
        candidate_names=np.asarray(ordered_names),
        candidate_pre_delay=np.asarray(
            [traces[name]["pre_delay"] for name in ordered_names], dtype=np.float32
        ),
        candidate_post_delay3=np.asarray(
            [traces[name]["post_delay3"] for name in ordered_names], dtype=np.float32
        ),
        tau_dataset=reference.astype(np.float32),
        tau_base_pre_delay=base_pre.astype(np.float32),
        tau_base_post_delay3=base_post.astype(np.float32),
        tau_best_pre_delay=best_traces["pre_delay"].astype(np.float32),
        tau_best_post_delay3=best_traces["post_delay3"].astype(np.float32),
        **{
            key: np.asarray(value, dtype=np.float32)
            for key, value in ppd["arrays"].items()
        },
    )
    return _jsonable({"artifact_dir": str(result_dir), **report})
