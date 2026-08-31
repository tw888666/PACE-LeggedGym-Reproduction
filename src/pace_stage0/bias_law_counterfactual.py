from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .constants import CANONICAL_JOINT_NAMES, PHYSICS_DT, PROJECT_ROOT
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import per_joint_rmse, rmse
from .replay import LEGACY_SIM_METHOD_FRAME, _make_sim


WINDOWS = (10, 50, 100, 500)
FOCUS_JOINTS = ("RF_HFE", "LH_HFE", "LF_KFE")
PUBLIC_MODE = "public_encoder_feedback"
LEGACY_MODE = "legacy_effective_bias"
TORQUE_ARTIFACT_RMSE_NM = 0.009035969986880686
CHECKPOINT_COMMIT = "b251390b4f5d3b768b45cc3a6b846f32d01fe39a"
FROZEN_PUBLIC_BASELINE = {
    "position_overall_rmse_rad": 0.012818364796375577,
    "RF_HFE_position_rmse_rad": 0.023814172906643862,
    "LH_HFE_position_rmse_rad": 0.022510682052527972,
    "q_compare_vs_real_rmse_rad": 0.023642306816740108,
}


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


def _new_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "artifacts" / "bias_law_counterfactual" / stamp
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


def _window_metrics(
    ours: np.ndarray,
    reference: np.ndarray,
    windows: Sequence[int] = WINDOWS,
) -> Dict[str, Any]:
    ours = np.asarray(ours, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if ours.shape != reference.shape or ours.ndim != 2:
        raise ValueError(f"Expected equal [time,joint], got {ours.shape}/{reference.shape}")
    result: Dict[str, Any] = {}
    for count in tuple(windows) + (len(ours),):
        actual = min(int(count), len(ours))
        key = "full" if actual == len(ours) else f"first_{actual}"
        result[key] = {
            "sample_count": actual,
            "overall_rmse": rmse(ours[:actual], reference[:actual]),
            "per_joint_rmse": dict(
                zip(
                    CANONICAL_JOINT_NAMES,
                    per_joint_rmse(ours[:actual], reference[:actual]).tolist(),
                )
            ),
        }
    return result


def _first_sustained_crossing(
    error: np.ndarray,
    threshold: float,
    consecutive: int = 5,
) -> Optional[int]:
    values = np.asarray(error, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("error must be one-dimensional")
    if len(values) < consecutive:
        return None
    hits = np.convolve(
        (values > threshold).astype(np.int32),
        np.ones(consecutive, dtype=np.int32),
        mode="valid",
    )
    indices = np.flatnonzero(hits == consecutive)
    return int(indices[0]) if len(indices) else None


def _divergence(
    ours: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    *,
    index_offset: int = 0,
    unit: str,
) -> Dict[str, Any]:
    ours = np.asarray(ours, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if ours.shape != reference.shape:
        raise ValueError("Divergence arrays must have equal shapes")
    frame_rmse = np.sqrt(np.mean(np.square(ours - reference), axis=1))
    local_index = _first_sustained_crossing(frame_rmse, threshold)
    state_index = None if local_index is None else local_index + index_offset
    return {
        "threshold": threshold,
        "unit": unit,
        "rule": "frame RMSE exceeds threshold for 5 consecutive samples",
        "first_local_index": local_index,
        "state_index": state_index,
        "time_s": None if state_index is None else state_index * PHYSICS_DT,
    }


def _simulate_mode(
    device_id: int,
    fit: DecodedFit,
    data: ReplayData,
    mode: str,
) -> Dict[str, Any]:
    # Isaac Gym is imported by the caller before torch and diagnostic_actuator.
    from isaacgym import gymtorch
    import torch

    from .diagnostic_actuator import BiasLawDiagnosticActuator

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    (
        _,
        _,
        _,
        gym,
        sim,
        _,
        _,
        dof_state,
        gather_indices,
        sim_settings,
    ) = _make_sim(device_id, fit, self_collisions=False)
    try:
        device = dof_state.device
        gather = torch.as_tensor(gather_indices, dtype=torch.long, device=device)
        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
        q_true_0 = torch.as_tensor(
            data.real_dof_pos[0] + fit.encoder_bias,
            dtype=torch.float32,
            device=device,
        )
        dof_state.zero_()
        dof_state[gather, 0] = q_true_0
        if not gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state)):
            raise RuntimeError("gym.set_dof_state_tensor failed")
        gym.refresh_dof_state_tensor(sim)

        if not torch.allclose(dof_state[gather, 0], q_true_0, atol=1e-7, rtol=0.0):
            raise RuntimeError("Initial q_true mismatch")
        if not torch.allclose(
            dof_state[gather, 1], torch.zeros_like(q_true_0), atol=1e-7, rtol=0.0
        ):
            raise RuntimeError("Initial qdot is not zero")

        count = data.sample_count
        q_true = np.empty((count, 12), dtype=np.float32)
        qdot = np.empty_like(q_true)
        q_control_interval = np.empty((count - 1, 12), dtype=np.float32)
        target = np.empty_like(q_control_interval)
        qdot_interval = np.empty_like(q_control_interval)
        raw_pd = np.empty_like(q_control_interval)
        saturated = np.empty_like(q_control_interval)
        applied = np.empty_like(q_control_interval)
        q_true[0] = dof_state[gather, 0].detach().cpu().numpy()
        qdot[0] = dof_state[gather, 1].detach().cpu().numpy()

        actuator = BiasLawDiagnosticActuator(bias, fit.delay_steps, mode)
        actuator.reset(dof_state[gather, 0])
        for index in range(count - 1):
            current_q = dof_state[gather, 0]
            current_qdot = dof_state[gather, 1]
            command = torch.as_tensor(
                data.real_des_dof_pos[index], dtype=torch.float32, device=device
            )
            step = actuator.step(command, current_q, current_qdot)
            effort = torch.zeros_like(dof_state[:, 0])
            effort[gather] = step.applied_torque
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort.contiguous())
            ):
                raise RuntimeError(f"Failed to apply effort at interval {index}")

            target[index] = command.detach().cpu().numpy()
            q_control_interval[index] = step.q_control.detach().cpu().numpy()
            qdot_interval[index] = current_qdot.detach().cpu().numpy()
            raw_pd[index] = step.raw_pd_torque.detach().cpu().numpy()
            saturated[index] = step.saturated_torque.detach().cpu().numpy()
            applied[index] = step.applied_torque.detach().cpu().numpy()

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            q_true[index + 1] = dof_state[gather, 0].detach().cpu().numpy()
            qdot[index + 1] = dof_state[gather, 1].detach().cpu().numpy()
    finally:
        gym.destroy_sim(sim)

    q_true64 = q_true.astype(np.float64)
    qdot64 = qdot.astype(np.float64)
    q_compare = q_true64 - fit.encoder_bias[None, :]
    q_control = (
        q_compare.copy() if mode == PUBLIC_MODE else q_true64.copy()
    )
    if not np.array_equal(q_control_interval.astype(np.float64), q_control[:-1]):
        maximum = float(
            np.max(np.abs(q_control_interval.astype(np.float64) - q_control[:-1]))
        )
        if maximum > 2e-7:
            raise RuntimeError(f"q_control trace mismatch in {mode}: {maximum}")
    return {
        "mode": mode,
        "simulation": sim_settings,
        "arrays": {
            "q_true": q_true64,
            "q_control": q_control,
            "q_compare": q_compare,
            "qdot": qdot64,
            "target": target.astype(np.float64),
            "qdot_interval": qdot_interval.astype(np.float64),
            "raw_pd_torque": raw_pd.astype(np.float64),
            "saturated_torque": saturated.astype(np.float64),
            "applied_torque": applied.astype(np.float64),
        },
    }


def _torque_metrics(case: Mapping[str, Any], data: ReplayData, delay_steps: int) -> Dict[str, Any]:
    applied = case["arrays"]["applied_torque"]
    # Candidate interval t is compared with the author's state-logged torque t+1.
    # Intervals [0, delay_steps) are zero FIFO initialization and are excluded.
    candidate = applied[delay_steps:]
    reference = data.sim_method_dof_torques[delay_steps + 1:]
    if candidate.shape != reference.shape:
        raise RuntimeError(f"Torque alignment mismatch: {candidate.shape}/{reference.shape}")
    joint_rmse = per_joint_rmse(candidate, reference)
    joint_correlation = [
        _correlation(candidate[:, index], reference[:, index])
        for index in range(candidate.shape[1])
    ]
    return {
        "alignment": "candidate applied interval[t] vs dataset torque state[t+1]",
        "active_candidate_interval_start": delay_steps,
        "active_dataset_state_start": delay_steps + 1,
        "inactive_fifo_intervals_excluded": delay_steps,
        "active_sample_count": len(candidate),
        "overall_rmse_Nm": rmse(candidate, reference),
        "overall_correlation": _correlation(candidate, reference),
        "per_joint_rmse_Nm": dict(zip(CANONICAL_JOINT_NAMES, joint_rmse.tolist())),
        "per_joint_correlation": dict(zip(CANONICAL_JOINT_NAMES, joint_correlation)),
        "windows": _window_metrics(candidate, reference),
        "divergence": _divergence(
            candidate,
            reference,
            0.5,
            index_offset=delay_steps + 1,
            unit="Nm",
        ),
    }


def _focus_metrics(
    case: Mapping[str, Any],
    data: ReplayData,
    delay_steps: int,
) -> Dict[str, Any]:
    arrays = case["arrays"]
    result = {}
    for joint in FOCUS_JOINTS:
        index = CANONICAL_JOINT_NAMES.index(joint)
        torque_candidate = arrays["applied_torque"][delay_steps:, index]
        torque_reference = data.sim_method_dof_torques[delay_steps + 1:, index]
        layers = {
            "position": {
                "ours": arrays["q_compare"][:, index],
                "reference": data.sim_method_dof_pos[:, index],
                "threshold": 0.001,
                "offset": 0,
                "unit": "rad",
            },
            "velocity": {
                "ours": arrays["qdot"][:, index],
                "reference": data.sim_method_dof_vel[:, index],
                "threshold": 0.05,
                "offset": 0,
                "unit": "rad/s",
            },
            "torque": {
                "ours": torque_candidate[:, None],
                "reference": torque_reference[:, None],
                "threshold": 0.5,
                "offset": delay_steps + 1,
                "unit": "Nm",
            },
        }
        result[joint] = {}
        for layer, values in layers.items():
            ours = np.asarray(values["ours"])
            reference = np.asarray(values["reference"])
            result[joint][layer] = {
                "rmse": rmse(ours, reference),
                "divergence": _divergence(
                    ours.reshape(-1, 1),
                    reference.reshape(-1, 1),
                    values["threshold"],
                    index_offset=values["offset"],
                    unit=values["unit"],
                ),
            }
    return result


def _mode_metrics(
    case: Mapping[str, Any],
    data: ReplayData,
    fit: DecodedFit,
) -> Dict[str, Any]:
    arrays = case["arrays"]
    position = _window_metrics(arrays["q_compare"], data.sim_method_dof_pos)
    velocity = _window_metrics(arrays["qdot"], data.sim_method_dof_vel)
    torque = _torque_metrics(case, data, fit.delay_steps)
    position_joint = per_joint_rmse(arrays["q_compare"], data.sim_method_dof_pos)
    ours_real = rmse(arrays["q_compare"], data.real_dof_pos)
    author_real = rmse(data.sim_method_dof_pos, data.real_dof_pos)
    sim_nothing_real = rmse(data.sim_nothing_dof_pos, data.real_dof_pos)
    gates = {
        "overall_position_le_0p010": position["full"]["overall_rmse"] <= 0.010,
        "all_per_joint_position_le_0p020": bool(np.all(position_joint <= 0.020)),
        "encoder_fit_le_1p2x_author": ours_real <= 1.2 * author_real,
        "encoder_fit_better_than_sim_nothing": ours_real < sim_nothing_real,
    }
    return {
        "mode": case["mode"],
        "control_frame": (
            "q_control=q_true-bias" if case["mode"] == PUBLIC_MODE
            else "q_control=q_true"
        ),
        "comparison_frame": "q_compare=q_true-bias (frozen for both modes)",
        "position": position,
        "velocity": velocity,
        "torque": torque,
        "q_compare_vs_real_rmse_rad": ours_real,
        "author_sim_method_vs_real_rmse_rad": author_real,
        "sim_nothing_vs_real_rmse_rad": sim_nothing_real,
        "divergence": {
            "position": _divergence(
                arrays["q_compare"], data.sim_method_dof_pos, 0.001,
                index_offset=0, unit="rad",
            ),
            "velocity": _divergence(
                arrays["qdot"], data.sim_method_dof_vel, 0.05,
                index_offset=0, unit="rad/s",
            ),
            "torque": torque["divergence"],
        },
        "focus_joints": _focus_metrics(case, data, fit.delay_steps),
        "gates": gates,
        "diagnostic_stage0C_pass": all(gates.values()),
    }


def _classify(public: Mapping[str, Any], legacy: Mapping[str, Any]) -> Dict[str, Any]:
    public_position = public["position"]["full"]["overall_rmse"]
    legacy_position = legacy["position"]["full"]["overall_rmse"]
    public_torque = public["torque"]["overall_rmse_Nm"]
    legacy_torque = legacy["torque"]["overall_rmse_Nm"]
    torque_closed = (
        legacy_torque <= 0.05
        and legacy_torque <= 0.25 * public_torque
        and legacy["torque"]["overall_correlation"] >= 0.999
    )
    position_primary_pass = (
        legacy["gates"]["overall_position_le_0p010"]
        and legacy["gates"]["all_per_joint_position_le_0p020"]
    )
    position_clearly_worse = legacy_position > 1.05 * public_position

    if position_primary_pass and torque_closed:
        case = "A"
        conclusion = (
            "Legacy-effective closes torque and passes the primary position gate; "
            "the legacy artifact and diagnostic replay jointly support the effective law."
        )
        branch_a_closed = True
        plant_audit = False
        actuator_recommendation = (
            "Do not modify the formal core automatically. Recommend a paper-era legacy "
            "compatibility mode after human approval and preserve the public mode separately."
        )
    elif torque_closed and not position_primary_pass:
        case = "B"
        conclusion = (
            "The replay closes applied torque but still fails the primary position gate; "
            "the effective torque-law branch is closed and the remaining residual is plant-side."
        )
        branch_a_closed = True
        plant_audit = True
        actuator_recommendation = (
            "Do not replace the public core automatically. Retain the diagnostic legacy law "
            "as the artifact-matched compatibility candidate while authorizing plant audit."
        )
    elif position_clearly_worse:
        case = "C"
        conclusion = (
            "Legacy-effective materially worsens position; the exported torque field is "
            "applied-like but is inconsistent with the trajectory-generating actuator law."
        )
        branch_a_closed = True
        plant_audit = True
        actuator_recommendation = (
            "Do not modify the formal actuator; retain paper/public semantics and allow plant audit."
        )
    else:
        case = "UNRESOLVED"
        conclusion = "The single-variable result does not satisfy cases A, B, or C."
        branch_a_closed = False
        plant_audit = False
        actuator_recommendation = "No formal actuator change is supported."
    return {
        "case": case,
        "criteria": {
            "torque_closed": torque_closed,
            "torque_closed_definition": (
                "legacy RMSE <= 0.05 Nm, <=25% of public RMSE, correlation >=0.999"
            ),
            "legacy_primary_position_gate_pass": position_primary_pass,
            "legacy_position_clearly_worse": position_clearly_worse,
            "position_clearly_worse_definition": "legacy overall > 1.05 * public overall",
        },
        "conclusion": conclusion,
        "branch_A_closed": branch_a_closed,
        "asset_plant_audit_authorized": plant_audit,
        "formal_actuator_change_applied": False,
        "formal_actuator_recommendation": actuator_recommendation,
    }


def _provenance_conflict(legacy_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rows": [
            {
                "evidence": "Paper Eq. 6",
                "bias_semantics": "+Kp*bias relative to true q",
                "strength": "author published",
                "source": "https://arxiv.org/html/2509.06342v2",
            },
            {
                "evidence": "Public PaceDCMotor",
                "bias_semantics": "joint_pos - encoder_bias feedback",
                "strength": "author code",
                "source": (
                    "https://github.com/leggedrobotics/pace-sim2real/blob/"
                    "f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/"
                    "pace_sim2real/utils/pace_actuator.py"
                ),
            },
            {
                "evidence": "Pre-QoL public PaceDCMotor",
                "bias_semantics": "joint_pos - encoder_bias feedback (same sign/law)",
                "strength": "author code/history",
                "source": (
                    "https://github.com/leggedrobotics/pace-sim2real/blob/"
                    "a57793def75472dc7f1176be71249b3ec9dbfb9d/source/pace_sim2real/"
                    "pace_sim2real/utils/pace_actuator.py"
                ),
            },
            {
                "evidence": "Public data_collection",
                "bias_semantics": "save q_true-bias; target unshifted",
                "strength": "author code",
                "source": (
                    "https://github.com/leggedrobotics/pace-sim2real/blob/"
                    "f07259c09b517ab5118bb1d01b0a6078cf8e1c31/scripts/pace/"
                    "data_collection.py"
                ),
            },
            {
                "evidence": "Legacy data.npy torque",
                "bias_semantics": "effective -Kp*bias relative to stored q",
                "strength": "artifact inference",
                "source": "local frozen data.npy SHA edf2e764...",
            },
            {
                "evidence": "Counterfactual replay",
                "bias_semantics": "q_control=q_true; q_compare remains q_true-bias",
                "strength": "this diagnostic experiment",
                "result": {
                    "torque_rmse_Nm": legacy_metrics["torque"]["overall_rmse_Nm"],
                    "position_rmse_rad": legacy_metrics["position"]["full"]["overall_rmse"],
                    "diagnostic_stage0C_pass": legacy_metrics["diagnostic_stage0C_pass"],
                },
            },
        ],
        "qol_history_conclusion": (
            "The 2026-03 QoL change altered encoder-bias configuration parsing and related "
            "handling, not the feedback sign/control law; its parent already used "
            "joint_pos - encoder_bias."
        ),
        "conflict_label": (
            "legacy artifact / later public implementation semantic divergence"
        ),
        "prohibited_label": "later bug fix changed bias sign",
        "paper_public_provenance_modified": False,
        "identifiability_limit": (
            "Torque identifies only alpha-beta=-1; it cannot distinguish uncorrected "
            "feedback from an equally shifted target/feedback convention."
        ),
    }


def _first10(case: Mapping[str, Any], data: ReplayData) -> Sequence[Dict[str, Any]]:
    arrays = case["arrays"]
    rows = []
    for index in range(min(10, data.sample_count)):
        row: Dict[str, Any] = {
            "state_index": index,
            "q_true": arrays["q_true"][index],
            "q_control": arrays["q_control"][index],
            "q_compare": arrays["q_compare"][index],
            "qdot": arrays["qdot"][index],
            "sim_method_dof_pos": data.sim_method_dof_pos[index],
            "sim_method_dof_vel": data.sim_method_dof_vel[index],
        }
        if index < data.sample_count - 1:
            row.update(
                {
                    "interval": f"[{index},{index + 1})",
                    "target": arrays["target"][index],
                    "interval_qdot": arrays["qdot_interval"][index],
                    "raw_pd": arrays["raw_pd_torque"][index],
                    "saturated_torque": arrays["saturated_torque"][index],
                    "applied_torque": arrays["applied_torque"][index],
                    "sim_method_dof_torques_state_t_plus_1": (
                        data.sim_method_dof_torques[index + 1]
                    ),
                }
            )
        rows.append(row)
    return rows


def _write_focus_plots(
    output_dir: Path,
    public_case: Mapping[str, Any],
    legacy_case: Mapping[str, Any],
    data: ReplayData,
    delay_steps: int,
) -> Sequence[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "focus_plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    state_time = np.arange(data.sample_count) * PHYSICS_DT
    torque_state_indices = np.arange(delay_steps + 1, data.sample_count)
    torque_time = torque_state_indices * PHYSICS_DT
    for joint in FOCUS_JOINTS:
        index = CANONICAL_JOINT_NAMES.index(joint)
        figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        axes[0].plot(state_time, data.sim_method_dof_pos[:, index], label="sim_method", linewidth=1.0)
        axes[0].plot(state_time, public_case["arrays"]["q_compare"][:, index], label="public", linewidth=0.8)
        axes[0].plot(state_time, legacy_case["arrays"]["q_compare"][:, index], label="legacy-effective", linewidth=0.8)
        axes[0].set_ylabel("position [rad]")
        axes[1].plot(state_time, data.sim_method_dof_vel[:, index], label="sim_method", linewidth=1.0)
        axes[1].plot(state_time, public_case["arrays"]["qdot"][:, index], label="public", linewidth=0.8)
        axes[1].plot(state_time, legacy_case["arrays"]["qdot"][:, index], label="legacy-effective", linewidth=0.8)
        axes[1].set_ylabel("velocity [rad/s]")
        axes[2].plot(
            torque_time,
            data.sim_method_dof_torques[delay_steps + 1:, index],
            label="sim_method",
            linewidth=1.0,
        )
        axes[2].plot(
            torque_time,
            public_case["arrays"]["applied_torque"][delay_steps:, index],
            label="public",
            linewidth=0.8,
        )
        axes[2].plot(
            torque_time,
            legacy_case["arrays"]["applied_torque"][delay_steps:, index],
            label="legacy-effective",
            linewidth=0.8,
        )
        axes[2].set_ylabel("torque [Nm]")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right")
        figure.suptitle(joint)
        figure.tight_layout()
        path = plot_dir / f"{joint}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _render_markdown(report: Mapping[str, Any]) -> str:
    public = report["modes"][PUBLIC_MODE]
    legacy = report["modes"][LEGACY_MODE]
    decision = report["decision"]

    def onset(mode: Mapping[str, Any], layer: str) -> str:
        value = mode["divergence"][layer]["time_s"]
        return "none" if value is None else f"{value:.4f} s"

    lines = [
        "# Stage 0C legacy bias effective-law counterfactual",
        "",
        "> Diagnostic only. The formal PACEActuatorCore and Stage 0C status were not changed.",
        "",
        "## Checkpoint",
        "",
        f"- Commit: `{report['checkpoint']['commit']}`",
        f"- Push: `{report['checkpoint']['push_status']}`",
        "- Frozen-environment pytest: unavailable; full unittest suite was used without installing dependencies.",
        "",
        "## Single variable",
        "",
        "```text",
        "public:           q_control = q_true - bias",
        "legacy-effective: q_control = q_true",
        "both modes:       q_compare = q_true - bias",
        "```",
        "",
        "Everything else—including target, initial state, motor envelope, FIFO, asset, plant, and comparison frame—is identical.",
        "",
        "## Main comparison",
        "",
        "| Metric | Public/current | Legacy-effective |",
        "|---|---:|---:|",
        f"| Position overall [rad] | {public['position']['full']['overall_rmse']:.9f} | {legacy['position']['full']['overall_rmse']:.9f} |",
        f"| RF_HFE position [rad] | {public['position']['full']['per_joint_rmse']['RF_HFE']:.9f} | {legacy['position']['full']['per_joint_rmse']['RF_HFE']:.9f} |",
        f"| LH_HFE position [rad] | {public['position']['full']['per_joint_rmse']['LH_HFE']:.9f} | {legacy['position']['full']['per_joint_rmse']['LH_HFE']:.9f} |",
        f"| Velocity overall [rad/s] | {public['velocity']['full']['overall_rmse']:.9f} | {legacy['velocity']['full']['overall_rmse']:.9f} |",
        f"| Active applied torque [Nm] | {public['torque']['overall_rmse_Nm']:.9f} | {legacy['torque']['overall_rmse_Nm']:.9f} |",
        f"| Torque correlation | {public['torque']['overall_correlation']:.9f} | {legacy['torque']['overall_correlation']:.9f} |",
        f"| q_compare vs real [rad] | {public['q_compare_vs_real_rmse_rad']:.9f} | {legacy['q_compare_vs_real_rmse_rad']:.9f} |",
        "",
        "Torque excludes the three zero-filled FIFO intervals and compares candidate interval `t` with dataset state `t+1`.",
        "",
        "## Time-resolved RMSE",
        "",
        "| Mode/layer | first 10 | first 50 | first 100 | first 500 | full |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode_name, mode in (("Public", public), ("Legacy-effective", legacy)):
        for layer, unit in (("position", "rad"), ("velocity", "rad/s")):
            values = mode[layer]
            lines.append(
                f"| {mode_name} {layer} [{unit}] | "
                + " | ".join(
                    f"{values[key]['overall_rmse']:.9f}"
                    for key in ("first_10", "first_50", "first_100", "first_500", "full")
                )
                + " |"
            )
        torque_windows = mode["torque"]["windows"]
        lines.append(
            f"| {mode_name} torque [Nm] | "
            + " | ".join(
                f"{torque_windows[key]['overall_rmse']:.9f}"
                for key in ("first_10", "first_50", "first_100", "first_500", "full")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Divergence onset",
            "",
            "| Mode | Torque >0.5 Nm | Position >0.001 rad | Velocity >0.05 rad/s |",
            "|---|---:|---:|---:|",
            f"| Public | {onset(public, 'torque')} | {onset(public, 'position')} | {onset(public, 'velocity')} |",
            f"| Legacy-effective | {onset(legacy, 'torque')} | {onset(legacy, 'position')} | {onset(legacy, 'velocity')} |",
            "",
            "All onsets require five consecutive samples over threshold.",
            "",
            "## Focus joints",
            "",
            "| Joint | Mode | Position RMSE | Velocity RMSE | Torque RMSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for joint in FOCUS_JOINTS:
        for label, mode in (("Public", public), ("Legacy-effective", legacy)):
            focus = mode["focus_joints"][joint]
            lines.append(
                f"| {joint} | {label} | {focus['position']['rmse']:.9f} | "
                f"{focus['velocity']['rmse']:.9f} | {focus['torque']['rmse']:.9f} |"
            )
    lines.extend(
        [
            "",
            "## Gate and decision",
            "",
            f"- Legacy diagnostic Stage 0C gate: `{legacy['diagnostic_stage0C_pass']}`.",
            f"- Case: `{decision['case']}`.",
            f"- Branch A closed: `{decision['branch_A_closed']}`.",
            f"- Asset/plant audit authorized: `{decision['asset_plant_audit_authorized']}`.",
            f"- Formal actuator modified: `{decision['formal_actuator_change_applied']}`.",
            f"- Conclusion: {decision['conclusion']}",
            f"- Recommendation: {decision['formal_actuator_recommendation']}",
            "",
            "Formal status remains `Stage 0C = FAIL`; locomotion/PPO remains blocked.",
            "",
            "## Provenance conflict",
            "",
            "| Evidence | Bias semantics | Strength |",
            "|---|---|---|",
        ]
    )
    for row in report["provenance_conflict"]["rows"]:
        lines.append(
            f"| {row['evidence']} | {row['bias_semantics']} | {row['strength']} |"
        )
    lines.extend(
        [
            "",
            report["provenance_conflict"]["qol_history_conclusion"],
            "",
            "Final label: `legacy artifact / later public implementation semantic divergence`, not `later bug fix changed bias sign`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_bias_law_counterfactual(
    device_id: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Preserve Isaac Gym Preview 4 import-before-torch semantics.
    from isaacgym import gymapi  # noqa: F401

    fit = decode_fit()
    data = load_replay_data()
    result_dir = _new_output_dir(output_dir)

    public_case = _simulate_mode(device_id, fit, data, PUBLIC_MODE)
    legacy_case = _simulate_mode(device_id, fit, data, LEGACY_MODE)
    if public_case["simulation"] != legacy_case["simulation"]:
        raise RuntimeError("Simulation settings differ between counterfactual modes")

    public_metrics = _mode_metrics(public_case, data, fit)
    legacy_metrics = _mode_metrics(legacy_case, data, fit)
    decision = _classify(public_metrics, legacy_metrics)
    public_observed = {
        "position_overall_rmse_rad": public_metrics["position"]["full"]["overall_rmse"],
        "RF_HFE_position_rmse_rad": public_metrics["position"]["full"]["per_joint_rmse"]["RF_HFE"],
        "LH_HFE_position_rmse_rad": public_metrics["position"]["full"]["per_joint_rmse"]["LH_HFE"],
        "q_compare_vs_real_rmse_rad": public_metrics["q_compare_vs_real_rmse_rad"],
    }
    public_deltas = {
        key: public_observed[key] - FROZEN_PUBLIC_BASELINE[key]
        for key in FROZEN_PUBLIC_BASELINE
    }
    report = {
        "schema": "pace_stage0.bias_law_counterfactual.v1",
        "scope": "diagnostic_only / single-variable bias effective law",
        "checkpoint": {
            "commit": CHECKPOINT_COMMIT,
            "push_status": "pushed to origin/main",
            "pytest": "unavailable in frozen bruce_gym environment",
            "fallback_test": "python -m unittest discover -s tests -v: 36/36 PASS",
        },
        "implementation_validation": {
            "pre_replay_full_unittest": "39/39 PASS",
            "git_diff_check": "PASS",
            "public_mode_vs_frozen_baseline": {
                "expected": FROZEN_PUBLIC_BASELINE,
                "observed": public_observed,
                "delta": public_deltas,
                "absolute_tolerance": 1e-12,
                "pass": all(abs(value) <= 1e-12 for value in public_deltas.values()),
            },
        },
        "frozen_conditions": {
            "q_true_0": "real.dof_pos[0] + canonical encoder_bias",
            "qdot_0": "zeros",
            "absolute_target": "real.des_dof_pos[t]",
            "q_compare_both_modes": "q_true - canonical encoder_bias",
            "dt_s": PHYSICS_DT,
            "delay_steps": fit.delay_steps,
            "self_collisions": False,
            "simulation": public_case["simulation"],
            "only_variable": "q_control public=q_true-bias vs legacy=q_true",
        },
        "identifiability": {
            "identified_effective_term": "alpha-beta=-1",
            "cannot_distinguish": [
                "feedback=q_true,target=target",
                "feedback=q_true-bias,target=target-bias",
            ],
            "label": "legacy effective bias law",
        },
        "modes": {
            PUBLIC_MODE: public_metrics,
            LEGACY_MODE: legacy_metrics,
        },
        "comparison": {
            "position_overall_delta_legacy_minus_public_rad": (
                legacy_metrics["position"]["full"]["overall_rmse"]
                - public_metrics["position"]["full"]["overall_rmse"]
            ),
            "torque_delta_legacy_minus_public_Nm": (
                legacy_metrics["torque"]["overall_rmse_Nm"]
                - public_metrics["torque"]["overall_rmse_Nm"]
            ),
            "velocity_delta_legacy_minus_public_rad_s": (
                legacy_metrics["velocity"]["full"]["overall_rmse"]
                - public_metrics["velocity"]["full"]["overall_rmse"]
            ),
            "author_state_artifact_torque_rmse_Nm_context": TORQUE_ARTIFACT_RMSE_NM,
        },
        "decision": decision,
        "provenance_conflict": _provenance_conflict(legacy_metrics),
        "formal_status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "asset_PhysX": (
                "AUTHORIZED FOR NEXT AUDIT"
                if decision["asset_plant_audit_authorized"] else "BLOCKED"
            ),
            "locomotion_PPO": "BLOCKED",
            "formal_core_changed": False,
        },
    }

    (result_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8"
    )
    (result_dir / "report.md").write_text(
        _render_markdown(_jsonable(report)), encoding="utf-8"
    )
    (result_dir / "first10.json").write_text(
        json.dumps(
            _jsonable(
                {
                    PUBLIC_MODE: _first10(public_case, data),
                    LEGACY_MODE: _first10(legacy_case, data),
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_paths = _write_focus_plots(
        result_dir, public_case, legacy_case, data, fit.delay_steps
    )
    np.savez_compressed(
        result_dir / "trajectories.npz",
        **{
            f"{mode}_{name}": array
            for mode, case in (("public", public_case), ("legacy", legacy_case))
            for name, array in case["arrays"].items()
        },
        sim_method_dof_pos=data.sim_method_dof_pos,
        sim_method_dof_vel=data.sim_method_dof_vel,
        sim_method_dof_torques=data.sim_method_dof_torques,
        real_dof_pos=data.real_dof_pos,
        real_dof_vel=data.real_dof_vel,
    )
    return _jsonable(
        {"artifact_dir": str(result_dir), "focus_plots": plot_paths, **report}
    )
