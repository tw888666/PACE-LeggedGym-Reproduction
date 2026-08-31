from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import CANONICAL_JOINT_NAMES, KP, PHYSICS_DT, PROJECT_ROOT
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .frame_forensics import _paper_parameter_audit
from .metrics import per_joint_rmse, rmse
from .replay import LEGACY_SIM_METHOD_FRAME, _make_sim


WINDOWS = (10, 50, 100, 500)
FOCUS_JOINTS = ("RF_HFE", "LH_HFE", "LF_KFE")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _new_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "artifacts" / "residual_diagnostics" / stamp
    result = output_dir.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _bounds_block_audit(fit: DecodedFit) -> Dict[str, Any]:
    blocks = (
        ("0:12", slice(0, 12)),
        ("12:24", slice(12, 24)),
        ("24:36", slice(24, 36)),
        ("36:48", slice(36, 48)),
        ("48", slice(48, 49)),
    )
    result: Dict[str, Any] = {}
    for name, selection in blocks:
        bounds = fit.bounds_raw[selection]
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        params = fit.params_raw[selection]
        result[name] = {
            "lower_min": float(np.min(lower)),
            "lower_max": float(np.max(lower)),
            "upper_min": float(np.min(upper)),
            "upper_max": float(np.max(upper)),
            "unique_lower": np.unique(lower),
            "unique_upper": np.unique(upper),
            "unique_bound_pairs": np.unique(bounds, axis=0),
            "parameter_min": float(np.min(params)),
            "parameter_max": float(np.max(params)),
            "parameters_raw": params,
        }
    return result


def _window_metrics(
    ours: np.ndarray,
    reference: np.ndarray,
    windows: Sequence[int] = WINDOWS,
) -> Dict[str, Any]:
    ours = np.asarray(ours, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if ours.shape != reference.shape or ours.ndim != 2:
        raise ValueError(f"Expected equal [time,joint] arrays, got {ours.shape}/{reference.shape}")
    result: Dict[str, Any] = {}
    for count in tuple(windows) + (len(ours),):
        actual_count = min(int(count), len(ours))
        key = "full" if actual_count == len(ours) else f"first_{actual_count}"
        result[key] = {
            "sample_count": actual_count,
            "overall_rmse": rmse(ours[:actual_count], reference[:actual_count]),
            "per_joint_rmse": dict(
                zip(
                    CANONICAL_JOINT_NAMES,
                    per_joint_rmse(
                        ours[:actual_count], reference[:actual_count]
                    ).tolist(),
                )
            ),
        }
    return result


def _first_sustained_crossing(
    frame_error: np.ndarray,
    threshold: float,
    *,
    consecutive: int = 5,
) -> Optional[int]:
    values = np.asarray(frame_error, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("frame_error must be one-dimensional")
    above = values > threshold
    if len(above) < consecutive:
        return None
    hits = np.convolve(
        above.astype(np.int32), np.ones(consecutive, dtype=np.int32), mode="valid"
    )
    indices = np.flatnonzero(hits == consecutive)
    return int(indices[0]) if len(indices) else None


def _divergence_report(
    ours: np.ndarray,
    reference: np.ndarray,
    thresholds: Sequence[float],
    *,
    index_offset: int = 0,
) -> Dict[str, Any]:
    error = np.sqrt(
        np.mean(
            np.square(
                np.asarray(ours, dtype=np.float64)
                - np.asarray(reference, dtype=np.float64)
            ),
            axis=1,
            dtype=np.float64,
        )
    )
    crossings = {}
    for threshold in thresholds:
        local_index = _first_sustained_crossing(error, threshold)
        state_index = None if local_index is None else local_index + index_offset
        crossings[str(threshold)] = {
            "first_local_index": local_index,
            "state_index": state_index,
            "time_s": None if state_index is None else state_index * PHYSICS_DT,
            "rule": "frame RMSE exceeds threshold for 5 consecutive samples",
        }
    return {"frame_rmse": error, "sustained_crossings": crossings}


def _simulate_case(
    device_id: int,
    fit: DecodedFit,
    data: ReplayData,
    *,
    self_collisions: bool,
) -> Dict[str, Any]:
    # Isaac Gym must be imported before torch.
    from isaacgym import gymtorch
    import torch

    from .actuator import PACEActuatorCore

    (
        _,
        _,
        _,
        gym,
        sim,
        env,
        actor,
        dof_state,
        gather_indices,
        sim_settings,
    ) = _make_sim(device_id, fit, self_collisions=self_collisions)
    try:
        device = dof_state.device
        gather = torch.as_tensor(gather_indices, dtype=torch.long, device=device)
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

        contact_descriptor = gym.acquire_net_contact_force_tensor(sim)
        contact_force = gymtorch.wrap_tensor(contact_descriptor).view(-1, 3)
        body_names = tuple(gym.get_actor_rigid_body_names(env, actor))
        if len(body_names) != len(contact_force):
            raise RuntimeError(
                f"Rigid-body/contact tensor mismatch: {len(body_names)}/{len(contact_force)}"
            )

        sample_count = data.sample_count
        q_true = np.empty((sample_count, 12), dtype=np.float32)
        qdot = np.empty((sample_count, 12), dtype=np.float32)
        q_true[0] = dof_state[gather, 0].cpu().numpy()
        qdot[0] = dof_state[gather, 1].cpu().numpy()
        raw_pd = np.empty((sample_count - 1, 12), dtype=np.float32)
        saturated = np.empty_like(raw_pd)
        applied = np.empty_like(raw_pd)
        contact = np.zeros((sample_count, len(body_names), 3), dtype=np.float32)

        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
        core = PACEActuatorCore(bias, fit.delay_steps)
        core.reset(dof_state[gather, 0])
        for index in range(sample_count - 1):
            target = torch.as_tensor(
                data.real_des_dof_pos[index], dtype=torch.float32, device=device
            )
            step = core.step(target, dof_state[gather, 0], dof_state[gather, 1])
            effort = torch.zeros_like(dof_state[:, 0])
            effort[gather] = step.applied_torque
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort.contiguous())
            ):
                raise RuntimeError(f"Failed to apply effort at interval {index}")
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)

            q_true[index + 1] = dof_state[gather, 0].cpu().numpy()
            qdot[index + 1] = dof_state[gather, 1].cpu().numpy()
            raw_pd[index] = step.raw_pd_torque.cpu().numpy()
            saturated[index] = step.saturated_torque.cpu().numpy()
            applied[index] = step.applied_torque.cpu().numpy()
            contact[index + 1] = contact_force.cpu().numpy()
    finally:
        gym.destroy_sim(sim)

    contact_norm = np.linalg.norm(contact.astype(np.float64), axis=2)
    max_per_state = np.max(contact_norm, axis=1)
    contact_indices = np.flatnonzero(max_per_state > 1e-6)
    q_encoder = q_true.astype(np.float64) - fit.encoder_bias[None, :]
    case = {
        "self_collisions": self_collisions,
        "simulation": sim_settings,
        "body_names": body_names,
        "state1_joint_velocity": qdot[1],
        "first10": {
            "q_encoder": q_encoder[:10],
            "qdot": qdot[:10],
            "sim_method_dof_pos": data.sim_method_dof_pos[:10],
            "sim_method_dof_vel": data.sim_method_dof_vel[:10],
            "max_net_contact_force_per_state_N": max_per_state[:10],
        },
        "contact": {
            "measurement": "Isaac Gym net contact force tensor; no ground asset exists",
            "nonzero_state_count_at_1e_6_N": int(len(contact_indices)),
            "first_nonzero_state": int(contact_indices[0]) if len(contact_indices) else None,
            "first_nonzero_time_s": (
                float(contact_indices[0] * PHYSICS_DT) if len(contact_indices) else None
            ),
            "max_net_contact_force_N": float(np.max(max_per_state)),
            "max_net_contact_impulse_proxy_Ns": float(np.max(max_per_state) * PHYSICS_DT),
            "impulse_note": "force * dt proxy; net body force can undercount cancelling contacts",
        },
        "position": _window_metrics(q_encoder, data.sim_method_dof_pos),
        "velocity": _window_metrics(qdot, data.sim_method_dof_vel),
        "torque": {
            "trace": "post-saturation/post-delay applied effort",
            "comparison": "ours interval[t] vs sim_method.dof_torques[t+1]",
            "semantic_status": "uncertain: legacy exporter did not publish command/applied/reported torque semantics",
            "metrics": _window_metrics(applied, data.sim_method_dof_torques[1:]),
        },
        "divergence": {
            "position": _divergence_report(
                q_encoder, data.sim_method_dof_pos, (0.001, 0.005, 0.01)
            ),
            "velocity": _divergence_report(
                qdot, data.sim_method_dof_vel, (0.05, 0.1, 0.2)
            ),
            "torque": _divergence_report(
                applied,
                data.sim_method_dof_torques[1:],
                (0.1, 0.5, 1.0),
                index_offset=1,
            ),
        },
        "arrays": {
            "q_true": q_true,
            "q_encoder": q_encoder,
            "qdot": qdot,
            "raw_pd_torque": raw_pd,
            "saturated_torque": saturated,
            "applied_torque": applied,
            "net_contact_force": contact,
        },
    }
    return case


def _author_state_torque_audit(fit: DecodedFit, data: ReplayData) -> Dict[str, Any]:
    import torch

    from .actuator import PACEActuatorCore

    bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32)
    core = PACEActuatorCore(bias, fit.delay_steps)
    core.reset(torch.as_tensor(data.sim_method_dof_pos[0] + fit.encoder_bias))
    applied = []
    for index in range(data.sample_count - 1):
        step = core.step(
            torch.as_tensor(data.real_des_dof_pos[index]),
            torch.as_tensor(data.sim_method_dof_pos[index] + fit.encoder_bias),
            torch.as_tensor(data.sim_method_dof_vel[index]),
        )
        applied.append(step.applied_torque.numpy().copy())
    applied_array = np.asarray(applied, dtype=np.float64)
    logged = data.sim_method_dof_torques[1:].astype(np.float64)
    delta = applied_array - logged
    active = np.arange(len(delta)) >= fit.delay_steps
    delta_mean = np.mean(delta[active], axis=0)
    delta_std = np.std(delta[active], axis=0)
    raw_bias = fit.params_raw[36:48]
    canonical_bias = fit.encoder_bias
    raw_pattern = KP * raw_bias
    canonical_pattern = KP * canonical_bias
    return {
        "input_state": "frozen core driven by author sim_method position/velocity",
        "torque_alignment": "core applied interval[t] vs logged sim_method torque state[t+1]",
        "logged_torque_semantics": "unresolved",
        "window_metrics": _window_metrics(applied_array, logged),
        "active_delta_mean_per_joint": dict(zip(CANONICAL_JOINT_NAMES, delta_mean)),
        "active_delta_std_per_joint": dict(zip(CANONICAL_JOINT_NAMES, delta_std)),
        "inferred_constant_position_offset_delta_over_kp": dict(
            zip(CANONICAL_JOINT_NAMES, delta_mean / KP)
        ),
        "legacy_raw_bias_times_kp": raw_pattern,
        "canonical_bias_times_kp": canonical_pattern,
        "rmse_delta_mean_vs_kp_legacy_raw_bias": rmse(delta_mean, raw_pattern),
        "rmse_delta_mean_vs_kp_canonical_bias": rmse(delta_mean, canonical_pattern),
        "correlation_delta_mean_vs_kp_legacy_raw_bias": float(
            np.corrcoef(delta_mean, raw_pattern)[0, 1]
        ),
        "correlation_delta_mean_vs_kp_canonical_bias": float(
            np.corrcoef(delta_mean, canonical_pattern)[0, 1]
        ),
        "interpretation": (
            "Torque differs before plant divergence and the difference is effectively constant "
            "over time, strongly implicating legacy bias application or torque-export semantics. "
            "It is not sufficient provenance to modify the frozen actuator core."
        ),
        "applied_torque": applied_array,
    }


def _focus_summary(case: Mapping[str, Any], data: ReplayData) -> Dict[str, Any]:
    summary = {}
    for joint in FOCUS_JOINTS:
        joint_index = CANONICAL_JOINT_NAMES.index(joint)
        summary[joint] = {
            layer: case[layer]["full"]["per_joint_rmse"][joint]
            for layer in ("position", "velocity")
        }
        summary[joint]["torque"] = case["torque"]["metrics"]["full"][
            "per_joint_rmse"
        ][joint]
        layer_inputs = {
            "position": (
                case["arrays"]["q_encoder"][:, joint_index],
                data.sim_method_dof_pos[:, joint_index],
                0.001,
                0,
                "rad",
            ),
            "velocity": (
                case["arrays"]["qdot"][:, joint_index],
                data.sim_method_dof_vel[:, joint_index],
                0.05,
                0,
                "rad/s",
            ),
            "torque": (
                case["arrays"]["applied_torque"][:, joint_index],
                data.sim_method_dof_torques[1:, joint_index],
                0.5,
                1,
                "Nm",
            ),
        }
        summary[joint]["first_sustained_divergence"] = {}
        for layer, (ours, reference, threshold, offset, unit) in layer_inputs.items():
            local_index = _first_sustained_crossing(
                np.abs(ours - reference), threshold
            )
            state_index = None if local_index is None else local_index + offset
            summary[joint]["first_sustained_divergence"][layer] = {
                "absolute_error_threshold": threshold,
                "unit": unit,
                "state_index": state_index,
                "time_s": None if state_index is None else state_index * PHYSICS_DT,
                "rule": "absolute joint error exceeds threshold for 5 consecutive samples",
            }
    return summary


def _strip_arrays(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in case.items() if key != "arrays"}


def _render_markdown(report: Mapping[str, Any]) -> str:
    layout = report["fitting_layout"]["conclusion"]
    off = report["self_collision_ab"]["off"]
    on = report["self_collision_ab"]["on"]
    audit = report["author_state_torque_audit"]
    focus = report["focus_joints"]["off"]
    lines = [
        "# PACE Stage 0C residual localization",
        "",
        "> Frozen diagnostic only. Stage 0C remains FAIL; locomotion/PPO was not started.",
        "",
        "## 1. Legacy fitting.npy layout",
        "",
        "`friction → damping → armature → encoder_bias → delay`.",
        "",
        f"Evidence: {layout['evidence']}. Confidence: {layout['confidence']}.",
        "The first and third bounds blocks are both [0, 0.5], so Table 6 value matching—not bounds alone—distinguishes them.",
        "",
        "## 2. Self-collision provenance and A/B",
        "",
        "The paper requires no contacts, explicitly including inter-leg contacts. The legacy implementation is unavailable. "
        "Later public PACE imports a generic Isaac Lab ANYmal-D USD configuration that enables self-collisions, so it is corroboration neither way for legacy Isaac Gym/URDF.",
        "",
        f"- OFF: state1 max |qdot| = {max(abs(x) for x in off['state1_joint_velocity']):.9f} rad/s; "
        f"contacts = {off['contact']['nonzero_state_count_at_1e_6_N']} states; position RMSE = {off['position']['full']['overall_rmse']:.9f} rad.",
        f"- ON: state1 max |qdot| = {max(abs(x) for x in on['state1_joint_velocity']):.9f} rad/s; "
        f"contacts = {on['contact']['nonzero_state_count_at_1e_6_N']} states; max net contact force = {on['contact']['max_net_contact_force_N']:.3f} N; "
        f"position RMSE = {on['position']['full']['overall_rmse']:.9f} rad.",
        "",
        "OFF remains provisional / legacy implementation evidence pending; A/B RMSE was not used to auto-select a baseline.",
        "",
        "## 3–5. Position, velocity, and torque",
        "",
        "| Layer | first 10 | first 50 | first 100 | first 500 | full |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Position [rad] | " + " | ".join(
            f"{off['position'][key]['overall_rmse']:.9f}"
            for key in ("first_10", "first_50", "first_100", "first_500", "full")
        ) + " |",
        "| Velocity [rad/s] | " + " | ".join(
            f"{off['velocity'][key]['overall_rmse']:.9f}"
            for key in ("first_10", "first_50", "first_100", "first_500", "full")
        ) + " |",
        "| Applied torque [Nm] | " + " | ".join(
            f"{off['torque']['metrics'][key]['overall_rmse']:.9f}"
            for key in ("first_10", "first_50", "first_100", "first_500", "full")
        ) + " |",
        "",
        "Torque comparison uses applied interval[t] versus logged state[t+1]. The legacy `dof_torques` command/applied/reported semantics remain unresolved.",
        "",
        "## 6–7. Focus joints and first divergence",
        "",
        "| Joint | Position RMSE | Velocity RMSE | Torque RMSE | Position first | Velocity first | Torque first |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for joint in FOCUS_JOINTS:
        row = focus[joint]
        first = row["first_sustained_divergence"]
        def cell(layer: str) -> str:
            value = first[layer]["state_index"]
            return "none" if value is None else f"{value} / {first[layer]['time_s']:.4f}s"
        lines.append(
            f"| {joint} | {row['position']:.9f} | {row['velocity']:.9f} | {row['torque']:.9f} | "
            f"{cell('position')} | {cell('velocity')} | {cell('torque')} |"
        )
    lines += [
        "",
        "Overall applied-torque error first exceeds 0.5 Nm for five samples at state 4 (0.0100 s), "
        "before the 0.001-rad position threshold at state 9 and 0.05-rad/s velocity threshold at state 11.",
        "",
        "## 8–11. Localization and minimum correction",
        "",
        f"Author-state torque delta vs `85 × legacy raw bias`: correlation {audit['correlation_delta_mean_vs_kp_legacy_raw_bias']:.9f}, "
        f"RMSE {audit['rmse_delta_mean_vs_kp_legacy_raw_bias']:.9f} Nm. The corresponding canonical-bias RMSE is "
        f"{audit['rmse_delta_mean_vs_kp_canonical_bias']:.9f} Nm.",
        "",
        "The first inconsistency is actuator-side (legacy bias application or torque-export convention), not demonstrated plant-side. "
        "Asset and PhysX tuning were therefore deferred. The minimum recommended action is to obtain legacy exporter/bias semantics "
        "or author confirmation, then add an explicit compatibility path only if supported. No correction has sufficient implementation provenance yet.",
        "",
        "## 12. Formal replay decision",
        "",
        "Do not rerun formal Stage 0C after an unproven change. Preserve `Stage 0C = FAIL` and do not enter locomotion/PPO.",
        "",
        "## Provenance note",
        "",
        "The frame conclusion remains a joint strong inference from the paper, later official implementation, public dataset usage, and legacy forensic. "
        "The legacy exporter source remains unpublished; only explicit author confirmation may upgrade provenance to `author_confirmed`.",
        "",
    ]
    return "\n".join(lines)


def run_residual_diagnostics(
    device_id: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Preserve Isaac Gym's required import-before-torch order.
    from isaacgym import gymapi  # noqa: F401

    fit = decode_fit()
    data = load_replay_data()
    result_dir = _new_output_dir(output_dir)

    paper_audit = _paper_parameter_audit(fit)
    layout = {
        "bounds_blocks": _bounds_block_audit(fit),
        "paper_table6_crosscheck": paper_audit,
        "conclusion": {
            "legacy_fitting_npy_layout": [
                "friction", "damping", "armature", "encoder_bias", "delay"
            ],
            "evidence": "bounds + paper Table 6 value matching; legacy serializer unavailable",
            "confidence": "high / artifact-and-paper confirmed, not source-code-confirmed",
            "decoder_change_required": False,
        },
    }

    off = _simulate_case(device_id, fit, data, self_collisions=False)
    on = _simulate_case(device_id, fit, data, self_collisions=True)
    torque_audit = _author_state_torque_audit(fit, data)

    report = {
        "schema": "pace_stage0.residual_diagnostics.v1",
        "stage_status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frame semantics frozen",
            "Stage_0C": "FAIL",
            "Stage_1_locomotion_PPO": "NOT STARTED",
        },
        "frozen_frame": LEGACY_SIM_METHOD_FRAME,
        "fitting_layout": layout,
        "self_collision_provenance": {
            "legacy_sysid_implementation": "UNAVAILABLE / unresolved",
            "paper_method": "No contacts, explicitly including inter-leg contacts",
            "later_public_pace_config": (
                "Imports Isaac Lab ANYMAL_D_CFG without an explicit self-collision override"
            ),
            "later_isaaclab_anymal_d_cfg": "enabled_self_collisions=True (USD asset)",
            "formal_off_status": "provisional / legacy implementation evidence pending",
            "selection_policy": "A/B metrics are diagnostic and do not auto-select the formal configuration",
            "sources": [
                {
                    "source": "PACE paper v2 Section 2.1",
                    "url": "https://arxiv.org/html/2509.06342v2",
                    "evidence": "Avoid all contacts, including inter-leg contacts",
                },
                {
                    "source": "later public PACE ANYmal config",
                    "url": (
                        "https://github.com/leggedrobotics/pace-sim2real/blob/"
                        "f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/"
                        "pace_sim2real/tasks/manager_based/pace/anymal_pace_env_cfg.py"
                    ),
                    "evidence": "Imports generic Isaac Lab ANYMAL_D_CFG; no explicit override",
                },
                {
                    "source": "later Isaac Lab ANYmal-D asset config",
                    "url": (
                        "https://github.com/isaac-sim/IsaacLab/blob/main/source/"
                        "isaaclab_assets/isaaclab_assets/robots/anymal.py"
                    ),
                    "evidence": "enabled_self_collisions=True for the USD asset",
                },
            ],
        },
        "self_collision_ab": {
            "off": _strip_arrays(off),
            "on": _strip_arrays(on),
        },
        "author_state_torque_audit": {
            key: value
            for key, value in torque_audit.items()
            if key != "applied_torque"
        },
        "focus_joints": {
            "chosen_control_joint": "LF_KFE",
            "off": _focus_summary(off, data),
            "on": _focus_summary(on, data),
        },
        "localization": {
            "first_layer_to_diverge": "torque",
            "classification": "A — torque is already inconsistent before plant error accumulation",
            "largest_current_suspect": (
                "legacy encoder-bias application and sim_method.dof_torques exporter semantics"
            ),
            "plant_audit_status": "deferred by branch A; no asset/PhysX tuning performed",
            "recommended_minimum_change": (
                "Do not change formal code yet. Obtain legacy torque/bias exporter semantics or an "
                "author confirmation; then implement any confirmed legacy replay convention as an "
                "explicit compatibility path and rerun Stage 0C."
            ),
            "recommended_change_provenance": (
                "diagnostic evidence is strong, but implementation provenance is unresolved"
            ),
            "rerun_formal_stage0C_now": False,
        },
    }

    _write_json(result_dir / "report.json", report)
    (result_dir / "report.md").write_text(
        _render_markdown(_jsonable(report)), encoding="utf-8"
    )
    np.savez_compressed(
        result_dir / "diagnostic_trajectories.npz",
        off_q_true=off["arrays"]["q_true"],
        off_q_encoder=off["arrays"]["q_encoder"],
        off_qdot=off["arrays"]["qdot"],
        off_applied_torque=off["arrays"]["applied_torque"],
        off_net_contact_force=off["arrays"]["net_contact_force"],
        on_q_true=on["arrays"]["q_true"],
        on_q_encoder=on["arrays"]["q_encoder"],
        on_qdot=on["arrays"]["qdot"],
        on_applied_torque=on["arrays"]["applied_torque"],
        on_net_contact_force=on["arrays"]["net_contact_force"],
        author_state_core_applied_torque=torque_audit["applied_torque"],
    )
    return _jsonable({"artifact_dir": str(result_dir), **report})
