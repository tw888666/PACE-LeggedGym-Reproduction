from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .constants import (
    ANYMAL_ASSET_COMMIT,
    ANYMAL_ASSET_FILE,
    ANYMAL_ASSET_ROOT,
    CANONICAL_JOINT_NAMES,
    CONTROL_DECIMATION,
    EFFORT_LIMIT,
    PHYSICS_DT,
    PROJECT_ROOT,
    VELOCITY_LIMIT,
    canonical_gather_indices,
)
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import lag_sweep, per_joint_rmse, rmse
from .sampling import run_indexed_replay


class FrameSanityError(RuntimeError):
    def __init__(self, message: str, artifact_dir: Path):
        super().__init__(message)
        self.artifact_dir = artifact_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _new_artifact_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        result = output_dir.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = PROJECT_ROOT / "artifacts" / "replay_fit" / stamp
    result.mkdir(parents=True, exist_ok=False)
    return result


def _frame_sanity(data: ReplayData, fit: DecodedFit) -> Dict[str, Any]:
    raw = rmse(data.sim_method_dof_pos, data.real_dof_pos)
    encoder = rmse(
        data.sim_method_dof_pos - fit.encoder_bias[None, :], data.real_dof_pos
    )
    return {
        "rmse_sim_method_true_vs_real": raw,
        "rmse_sim_method_minus_bias_vs_real": encoder,
        "state0_rmse_sim_method_vs_real": rmse(
            data.sim_method_dof_pos[0], data.real_dof_pos[0]
        ),
        "state0_rmse_sim_method_vs_real_plus_bias": rmse(
            data.sim_method_dof_pos[0], data.real_dof_pos[0] + fit.encoder_bias
        ),
        "state0_max_abs_sim_method_minus_real": float(
            np.max(np.abs(data.sim_method_dof_pos[0] - data.real_dof_pos[0]))
        ),
        "state0_max_abs_sim_method_minus_real_plus_bias": float(
            np.max(
                np.abs(
                    data.sim_method_dof_pos[0]
                    - (data.real_dof_pos[0] + fit.encoder_bias)
                )
            )
        ),
        "encoder_frame_lower_than_raw": encoder < raw,
        "ratio_encoder_over_raw": encoder / raw if raw > 0 else None,
        "policy": (
            "Diagnostic only; never auto-select a frame. The frozen preflight expects the "
            "bias-corrected metric to be lower."
        ),
    }


def _make_sim(device_id: int, fit: DecodedFit):
    # Import order is intentional for Isaac Gym Preview 4.
    from isaacgym import gymapi, gymtorch
    import torch

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = PHYSICS_DT
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 100.0
    sim_params.physx.max_gpu_contact_pairs = 2**23
    sim_params.physx.default_buffer_size_multiplier = 5.0

    sim = gym.create_sim(device_id, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("gym.create_sim returned None")
    try:
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = False
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        options.flip_visual_attachments = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.use_physx_armature = True
        options.linear_damping = 0.0
        options.angular_damping = 0.0
        options.max_linear_velocity = 1000.0
        options.max_angular_velocity = 1000.0
        options.thickness = 0.01

        asset = gym.load_asset(sim, str(ANYMAL_ASSET_ROOT), ANYMAL_ASSET_FILE, options)
        if asset is None:
            raise RuntimeError("gym.load_asset returned None")
        dof_names = tuple(gym.get_asset_dof_names(asset))
        gather_indices = canonical_gather_indices(dof_names)
        gather_np = np.asarray(gather_indices, dtype=np.int64)

        props = gym.get_asset_dof_properties(asset)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        # Fit/data/core remain canonical. Isaac Gym properties are written in asset order.
        props["friction"][gather_np] = fit.friction.astype(props["friction"].dtype)
        props["damping"][gather_np] = fit.damping.astype(props["damping"].dtype)
        props["armature"][gather_np] = fit.armature.astype(props["armature"].dtype)
        props["effort"].fill(EFFORT_LIMIT)
        props["velocity"].fill(VELOCITY_LIMIT)

        env = gym.create_env(
            sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 2.0), 1
        )
        pose = gymapi.Transform()
        pose.p.z = 1.0
        actor = gym.create_actor(env, asset, pose, "anymal_d", 0, 0)
        gym.set_actor_dof_properties(env, actor, props)
        gym.prepare_sim(sim)

        state_tensor = gym.acquire_dof_state_tensor(sim)
        dof_state = gymtorch.wrap_tensor(state_tensor).view(-1, 2)
        if tuple(dof_state.shape) != (12, 2):
            raise RuntimeError(f"Unexpected DOF state tensor shape: {tuple(dof_state.shape)}")

        applied_props = gym.get_actor_dof_properties(env, actor)
        settings = {
            "asset_commit": ANYMAL_ASSET_COMMIT,
            "asset_file": str(ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE),
            "joint_order": list(dof_names),
            "canonical_joint_order": list(CANONICAL_JOINT_NAMES),
            "canonical_gather_indices": list(gather_indices),
            "dt": PHYSICS_DT,
            "decimation": CONTROL_DECIMATION,
            "fixed_base": True,
            "gravity": [0.0, 0.0, -9.81],
            "drive_mode_effort": bool(
                (applied_props["driveMode"] == gymapi.DOF_MODE_EFFORT).all()
            ),
            "builtin_stiffness": applied_props["stiffness"].tolist(),
            "physical_friction": applied_props["friction"].tolist(),
            "physical_damping": applied_props["damping"].tolist(),
            "physical_armature": applied_props["armature"].tolist(),
            "sim_effort_limit": applied_props["effort"].tolist(),
            "sim_velocity_limit": applied_props["velocity"].tolist(),
            "device": str(torch.device(f"cuda:{device_id}")),
        }
        if not settings["drive_mode_effort"]:
            raise RuntimeError("Isaac Gym actor is not in DOF_MODE_EFFORT")
        if not np.allclose(applied_props["stiffness"], 0.0):
            raise RuntimeError("Isaac Gym built-in stiffness is non-zero")
        return (
            gymapi,
            gymtorch,
            torch,
            gym,
            sim,
            env,
            actor,
            dof_state,
            gather_indices,
            settings,
        )
    except Exception:
        gym.destroy_sim(sim)
        raise


def replay_fit(
    device_id: int = 0,
    output_dir: Optional[Path] = None,
    acknowledge_frame_sanity_failure: bool = False,
) -> Dict[str, Any]:
    # Isaac Gym must be imported before decoder unpickling imports torch.
    from isaacgym import gymapi  # noqa: F401
    import torch

    from .actuator import PACEActuatorCore

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    artifact_dir = _new_artifact_dir(output_dir)
    fit = decode_fit()
    data = load_replay_data()
    sanity = _frame_sanity(data, fit)
    preflight = {
        "schema": "pace_stage0.replay_preflight.v1",
        "fit": fit.to_dict(),
        "data": {
            "source": str(Path("/home/xy.chen/tw/dataset/pace_data/1_in_air/anymal/data.npy")),
            "sha256": data.sha256,
            "sample_count": data.sample_count,
            "joint_count": data.joint_count,
        },
        "initialization": {
            "q_encoder_0": data.real_dof_pos[0],
            "q_true_0": data.real_dof_pos[0] + fit.encoder_bias,
            "qdot_0": np.zeros_like(data.real_dof_pos[0]),
            "real_dof_vel_0_diagnostic": data.real_dof_vel[0],
            "real_dof_vel_0_minus_baseline": data.real_dof_vel[0],
            "delay_fifo": "zeros",
        },
        "frame_sanity": sanity,
    }
    _write_json(artifact_dir / "preflight.json", preflight)
    if not sanity["encoder_frame_lower_than_raw"] and not acknowledge_frame_sanity_failure:
        raise FrameSanityError(
            "Frame sanity check failed: RMSE(sim_method-bias, real) is not lower than "
            "RMSE(sim_method, real). No frame was changed; replay stopped for manual review.",
            artifact_dir,
        )

    (
        gymapi_mod,
        gymtorch,
        torch_mod,
        gym,
        sim,
        env,
        actor,
        dof_state,
        gather_indices,
        sim_settings,
    ) = _make_sim(device_id, fit)
    try:
        device = dof_state.device
        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
        q_true_0 = torch.as_tensor(
            data.real_dof_pos[0] + fit.encoder_bias, dtype=torch.float32, device=device
        )
        qdot_0 = torch.zeros_like(q_true_0)
        gather = torch.as_tensor(gather_indices, dtype=torch.long, device=device)

        initial_state = np.zeros(12, dtype=gymapi_mod.DofState.dtype)
        initial_state["pos"][np.asarray(gather_indices)] = q_true_0.cpu().numpy()
        initial_state["vel"] = 0.0
        if not gym.set_actor_dof_states(env, actor, initial_state, gymapi_mod.STATE_ALL):
            raise RuntimeError("gym.set_actor_dof_states failed")
        gym.refresh_dof_state_tensor(sim)
        initial_sim_q = dof_state[gather, 0].clone()
        initial_sim_qdot = dof_state[gather, 1].clone()
        if not torch.allclose(initial_sim_q, q_true_0, atol=1e-7, rtol=0.0):
            raise RuntimeError("Simulator did not preserve requested initial q_true")
        if not torch.allclose(initial_sim_qdot, qdot_0, atol=1e-7, rtol=0.0):
            raise RuntimeError("Simulator did not preserve qdot_0=0")

        core = PACEActuatorCore(bias, fit.delay_steps)
        core.reset(initial_sim_q)

        def transition(t: int, target_np: np.ndarray, q_np: np.ndarray, qdot_np: np.ndarray):
            current_q = dof_state[gather, 0]
            current_qdot = dof_state[gather, 1]
            expected_q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
            expected_qdot = torch.as_tensor(qdot_np, dtype=torch.float32, device=device)
            if not torch.allclose(current_q, expected_q, atol=2e-6, rtol=0.0):
                raise RuntimeError(f"State alignment mismatch before interval {t} (q)")
            if not torch.allclose(current_qdot, expected_qdot, atol=2e-6, rtol=0.0):
                raise RuntimeError(f"State alignment mismatch before interval {t} (qdot)")
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            step = core.step(target, current_q, current_qdot)
            effort_asset_order = torch.zeros_like(dof_state[:, 0])
            effort_asset_order[gather] = step.applied_torque
            effort_asset_order = effort_asset_order.contiguous()
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort_asset_order)
            ):
                raise RuntimeError(f"Failed to apply effort tensor at interval {t}")
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            record = {
                "target": target.detach().cpu().numpy().copy(),
                "q_encoder_before": step.q_encoder.detach().cpu().numpy().copy(),
                "raw_pd_torque": step.raw_pd_torque.detach().cpu().numpy().copy(),
                "saturated_torque": step.saturated_torque.detach().cpu().numpy().copy(),
                "applied_torque": step.applied_torque.detach().cpu().numpy().copy(),
            }
            return (
                dof_state[gather, 0].detach().cpu().numpy().copy(),
                dof_state[gather, 1].detach().cpu().numpy().copy(),
                record,
            )

        sampled = run_indexed_replay(
            data.real_des_dof_pos,
            initial_sim_q.detach().cpu().numpy(),
            initial_sim_qdot.detach().cpu().numpy(),
            transition,
        )
    finally:
        gym.destroy_sim(sim)

    q_true = sampled.q_true.astype(np.float64)
    qdot = sampled.qdot.astype(np.float64)
    q_encoder = q_true - fit.encoder_bias[None, :]
    sim_method_encoder = data.sim_method_dof_pos.astype(np.float64) - fit.encoder_bias[None, :]
    interval = {
        name: np.stack([record[name] for record in sampled.interval_records], axis=0)
        for name in (
            "target",
            "q_encoder_before",
            "raw_pd_torque",
            "saturated_torque",
            "applied_torque",
        )
    }

    overall_replication = rmse(q_true, data.sim_method_dof_pos)
    per_joint = per_joint_rmse(q_true, data.sim_method_dof_pos)
    ours_real_encoder = rmse(q_encoder, data.real_dof_pos)
    sim_method_real_encoder = rmse(sim_method_encoder, data.real_dof_pos)
    sim_nothing_real = rmse(data.sim_nothing_dof_pos, data.real_dof_pos)
    gates = {
        "overall_replication_le_0p01": overall_replication <= 0.01,
        "all_per_joint_replication_le_0p02": bool(np.all(per_joint <= 0.02)),
        "encoder_fit_le_1p2x_author": ours_real_encoder <= 1.2 * sim_method_real_encoder,
        "encoder_fit_better_than_sim_nothing": ours_real_encoder < sim_nothing_real,
    }
    metrics = {
        "schema": "pace_stage0.replay_metrics.v1",
        "formal": {
            "rmse_q_true_ours_vs_sim_method": overall_replication,
            "per_joint_rmse_q_true_ours_vs_sim_method": dict(
                zip(CANONICAL_JOINT_NAMES, per_joint.tolist())
            ),
            "rmse_q_encoder_ours_vs_real": ours_real_encoder,
            "rmse_sim_method_minus_bias_vs_real": sim_method_real_encoder,
            "rmse_sim_nothing_vs_real": sim_nothing_real,
        },
        "diagnostic": {
            "rmse_q_true_ours_vs_real": rmse(q_true, data.real_dof_pos),
            "rmse_q_encoder_ours_vs_sim_method": rmse(q_encoder, data.sim_method_dof_pos),
            "rmse_sim_method_vs_real_raw": rmse(data.sim_method_dof_pos, data.real_dof_pos),
            "lag_convention": "ours[k] vs reference[k+lag] over common indices",
            "lag_rmse_q_true_ours_vs_sim_method": lag_sweep(q_true, data.sim_method_dof_pos),
        },
        "gates": gates,
        "pass": all(gates.values()),
        "frame_sanity": sanity,
    }

    first_rows = []
    for k in range(min(10, data.sample_count)):
        row: Dict[str, Any] = {
            "state_index": k,
            "q_true": q_true[k],
            "q_encoder": q_encoder[k],
            "qdot": qdot[k],
            "sim_method_dof_pos": data.sim_method_dof_pos[k],
            "sim_method_dof_pos_minus_bias": sim_method_encoder[k],
            "real_dof_pos": data.real_dof_pos[k],
            "sim_nothing_dof_pos": data.sim_nothing_dof_pos[k],
        }
        if k < data.sample_count - 1:
            row.update(
                {
                    "interval": f"[{k},{k + 1})",
                    "absolute_q_target": interval["target"][k],
                    "raw_pd_torque": interval["raw_pd_torque"][k],
                    "saturated_torque": interval["saturated_torque"][k],
                    "fifo_output_torque": interval["applied_torque"][k],
                }
            )
        first_rows.append(row)

    config = {
        "schema": "pace_stage0.replay_config.v1",
        "fit": fit.to_dict(),
        "simulation": sim_settings,
        "sampling": {
            "state_0": "initialized, not simulated",
            "target_t": "drives interval [t,t+1)",
            "state_t_plus_1": "post-simulate state",
            "state_array_length": data.sample_count,
            "interval_array_length": data.sample_count - 1,
        },
        "random_seed": 0,
        "frame_sanity_failure_acknowledged": acknowledge_frame_sanity_failure,
    }
    _write_json(artifact_dir / "config.json", config)
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "first10.json", first_rows)
    np.savez_compressed(
        artifact_dir / "trajectories.npz",
        ours_q_true=q_true,
        ours_q_encoder=q_encoder,
        ours_qdot=qdot,
        interval_target=interval["target"],
        interval_raw_pd_torque=interval["raw_pd_torque"],
        interval_saturated_torque=interval["saturated_torque"],
        interval_applied_torque=interval["applied_torque"],
        real_dof_pos=data.real_dof_pos,
        real_dof_vel=data.real_dof_vel,
        sim_method_dof_pos=data.sim_method_dof_pos,
        sim_nothing_dof_pos=data.sim_nothing_dof_pos,
    )
    return {"artifact_dir": str(artifact_dir), **metrics}
