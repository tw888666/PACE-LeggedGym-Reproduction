from __future__ import annotations

import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    ANYMAL_ASSET_FILE,
    ANYMAL_ASSET_ROOT,
    CANONICAL_JOINT_NAMES,
    EFFORT_LIMIT,
    PHYSICS_DT,
    PROJECT_ROOT,
    VELOCITY_LIMIT,
    canonical_gather_indices,
)
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import per_joint_rmse, rmse


FOCUS_JOINTS = ("RF_KFE", "LH_KFE", "RF_HFE")
HORIZONS = (8, 32, 128)
MATERIAL_THRESHOLD = 0.10
FRICTION_SEMANTICS_URL = (
    "https://forums.developer.nvidia.com/t/"
    "possible-bug-in-joint-friction-value-definition/208631"
)


@dataclass(frozen=True)
class PropertyVariant:
    name: str
    fitted_friction: bool = True
    fitted_damping: bool = True


@dataclass(frozen=True)
class Scenario:
    name: str
    reset_horizon: Optional[int]
    author_initial_state: bool = False


@dataclass(frozen=True)
class IntegrationConfig:
    name: str
    substeps: int = 1
    num_velocity_iterations: int = 0


def _variants() -> Tuple[PropertyVariant, ...]:
    return (
        PropertyVariant("baseline"),
        PropertyVariant("damping0", fitted_damping=False),
        PropertyVariant("friction0", fitted_friction=False),
        PropertyVariant("both0", fitted_friction=False, fitted_damping=False),
    )


def _scenarios() -> Tuple[Scenario, ...]:
    return (
        Scenario("one_step", 1, author_initial_state=True),
        Scenario("H8", 8),
        Scenario("H32", 32),
        Scenario("H128", 128),
        Scenario("full", None),
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _new_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "artifacts" / "p4_p5_audit" / stamp
    result = output_dir.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_snapshot() -> Dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(("git",) + args, cwd=PROJECT_ROOT, text=True).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Correlation shape mismatch: {left.shape}/{right.shape}")
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _regression_structure(residual: np.ndarray, qdot: np.ndarray) -> Dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    if residual.shape != qdot.shape:
        raise ValueError("Residual/qdot shape mismatch")
    design = np.column_stack((np.ones(len(qdot)), qdot))
    coefficient, _, _, _ = np.linalg.lstsq(design, residual, rcond=None)
    prediction = design @ coefficient
    denominator = float(np.sum(np.square(residual - np.mean(residual))))
    r_squared = (
        1.0 - float(np.sum(np.square(residual - prediction))) / denominator
        if denominator
        else 1.0
    )
    slope = float(coefficient[1])
    return {
        "slope_sign": "positive" if slope > 0.0 else "negative" if slope < 0.0 else "zero",
        "r_squared": r_squared,
        "slope_written_back_to_model": False,
    }


def _joint_axis_signs() -> Dict[str, Any]:
    urdf_path = ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE
    root = ET.parse(urdf_path).getroot()
    result: Dict[str, Any] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in CANONICAL_JOINT_NAMES:
            continue
        axis_element = joint.find("axis")
        if axis_element is None:
            raise ValueError(f"Joint {name} has no URDF axis")
        axis = np.asarray([float(value) for value in axis_element.attrib["xyz"].split()])
        dominant_index = int(np.argmax(np.abs(axis)))
        dominant_value = float(axis[dominant_index])
        if dominant_value == 0.0:
            raise ValueError(f"Joint {name} has zero URDF axis")
        result[name] = {
            "axis_xyz": axis.tolist(),
            "motion_alignment_sign": 1.0 if dominant_value > 0.0 else -1.0,
            "rule": "sign of dominant URDF joint-axis component",
        }
    if set(result) != set(CANONICAL_JOINT_NAMES):
        raise ValueError("Failed to recover all canonical joint axes from the frozen URDF")
    return result


def _create_simulation(
    device_id: int,
    fit: DecodedFit,
    variants: Sequence[PropertyVariant],
    config: IntegrationConfig,
):
    from isaacgym import gymapi, gymtorch
    import torch

    scenarios = _scenarios()
    actor_specs = tuple((variant, scenario) for variant in variants for scenario in scenarios)
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = PHYSICS_DT
    sim_params.substeps = config.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = config.num_velocity_iterations
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
        if gym.get_asset_dof_count(asset) != 12:
            raise RuntimeError(f"Expected 12 asset DOFs, got {gym.get_asset_dof_count(asset)}")
        gather_indices = canonical_gather_indices(dof_names)
        gather_np = np.asarray(gather_indices, dtype=np.int64)
        base_props = gym.get_asset_dof_properties(asset)
        base_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        base_props["stiffness"].fill(0.0)
        base_props["armature"][gather_np] = fit.armature.astype(base_props["armature"].dtype)
        base_props["effort"].fill(EFFORT_LIMIT)
        base_props["velocity"].fill(VELOCITY_LIMIT)

        actor_indices = []
        runtime_properties: Dict[str, Any] = {}
        per_row = int(math.ceil(math.sqrt(len(actor_specs))))
        for index, (variant, scenario) in enumerate(actor_specs):
            env = gym.create_env(
                sim,
                gymapi.Vec3(-1.0, -1.0, 0.0),
                gymapi.Vec3(1.0, 1.0, 2.0),
                per_row,
            )
            pose = gymapi.Transform()
            pose.p.z = 1.0
            actor = gym.create_actor(
                env, asset, pose, f"{variant.name}_{scenario.name}", index, 1
            )
            props = base_props.copy()
            props["friction"][gather_np] = (
                fit.friction if variant.fitted_friction else np.zeros(12)
            ).astype(props["friction"].dtype)
            props["damping"][gather_np] = (
                fit.damping if variant.fitted_damping else np.zeros(12)
            ).astype(props["damping"].dtype)
            gym.set_actor_dof_properties(env, actor, props)
            applied = gym.get_actor_dof_properties(env, actor)
            if not np.all(applied["driveMode"] == gymapi.DOF_MODE_EFFORT):
                raise RuntimeError("Diagnostic actor does not use DOF_MODE_EFFORT")
            if not np.allclose(applied["stiffness"], 0.0):
                raise RuntimeError("Diagnostic actor has hidden position stiffness")
            if scenario.name == "one_step":
                runtime_properties[variant.name] = {
                    field: np.asarray(applied[field][gather_np]).copy()
                    for field in ("friction", "damping", "armature", "effort", "velocity")
                }
            actor_indices.append(gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
        gym.prepare_sim(sim)
        dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(
            len(actor_specs), 12, 2
        )
        return {
            "gymapi": gymapi,
            "gymtorch": gymtorch,
            "torch": torch,
            "gym": gym,
            "sim": sim,
            "dof_state": dof_state,
            "gather_indices": gather_indices,
            "actor_indices": actor_indices,
            "actor_specs": actor_specs,
            "runtime_properties": runtime_properties,
            "config": config,
        }
    except Exception:
        gym.destroy_sim(sim)
        raise


def _run_matrix(
    device_id: int,
    fit: DecodedFit,
    data: ReplayData,
    variants: Sequence[PropertyVariant],
    config: IntegrationConfig,
) -> Dict[str, Any]:
    # Isaac Gym Preview 4 must be imported before torch.
    from isaacgym import gymtorch
    import torch

    from .actuator import PACEActuatorCore

    context = _create_simulation(device_id, fit, variants, config)
    gym = context["gym"]
    sim = context["sim"]
    dof_state = context["dof_state"]
    actor_specs = context["actor_specs"]
    scenarios = _scenarios()
    try:
        device = dof_state.device
        gather = torch.as_tensor(
            context["gather_indices"], dtype=torch.long, device=device
        )
        actor_index_tensor = torch.as_tensor(
            context["actor_indices"], dtype=torch.int32, device=device
        )
        author_q_true = torch.as_tensor(
            data.sim_method_dof_pos + fit.encoder_bias[None, :],
            dtype=torch.float32,
            device=device,
        )
        author_qdot = torch.as_tensor(
            data.sim_method_dof_vel, dtype=torch.float32, device=device
        )
        formal_q0 = torch.as_tensor(
            data.real_dof_pos[0] + fit.encoder_bias,
            dtype=torch.float32,
            device=device,
        )
        dof_state.zero_()
        for row, (_, scenario) in enumerate(actor_specs):
            dof_state[row, gather, 0] = (
                author_q_true[0] if scenario.author_initial_state else formal_q0
            )
            dof_state[row, gather, 1] = (
                author_qdot[0]
                if scenario.author_initial_state
                else torch.zeros(12, dtype=torch.float32, device=device)
            )
        if not gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state)):
            raise RuntimeError("Initial OFAT state write failed")
        gym.refresh_dof_state_tensor(sim)

        count = data.sample_count
        q_true = np.empty((len(actor_specs), count, 12), dtype=np.float32)
        qdot = np.empty_like(q_true)
        q_true[:, 0] = dof_state[:, gather, 0].detach().cpu().numpy()
        qdot[:, 0] = dof_state[:, gather, 1].detach().cpu().numpy()
        one_step_rows = {
            variant.name: next(
                row
                for row, (candidate, scenario) in enumerate(actor_specs)
                if candidate.name == variant.name and scenario.name == "one_step"
            )
            for variant in variants
        }
        applied = {
            variant.name: np.empty((count - 1, 12), dtype=np.float32)
            for variant in variants
        }
        raw_pd = {
            variant.name: np.empty((count - 1, 12), dtype=np.float32)
            for variant in variants
        }
        saturated = {
            variant.name: np.empty((count - 1, 12), dtype=np.float32)
            for variant in variants
        }

        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
        core = PACEActuatorCore(bias, fit.delay_steps)
        core.reset(dof_state[:, gather, 0])
        targets = torch.as_tensor(
            data.real_des_dof_pos, dtype=torch.float32, device=device
        )
        effort = torch.zeros((len(actor_specs), 12), dtype=torch.float32, device=device)
        maximum_reset_q_error = 0.0
        maximum_reset_qdot_error = 0.0
        for step_index in range(count - 1):
            reset_rows = []
            for row, (_, scenario) in enumerate(actor_specs):
                horizon = scenario.reset_horizon
                if horizon is None or step_index == 0 or step_index % horizon != 0:
                    continue
                dof_state[row, gather, 0] = author_q_true[step_index]
                dof_state[row, gather, 1] = author_qdot[step_index]
                reset_rows.append(row)
            if reset_rows:
                rows = torch.as_tensor(reset_rows, dtype=torch.long, device=device)
                actor_indices = actor_index_tensor[rows].contiguous()
                if not gym.set_dof_state_tensor_indexed(
                    sim,
                    gymtorch.unwrap_tensor(dof_state),
                    gymtorch.unwrap_tensor(actor_indices),
                    len(reset_rows),
                ):
                    raise RuntimeError(f"Indexed OFAT state reset failed at {step_index}")
                gym.refresh_dof_state_tensor(sim)
                maximum_reset_q_error = max(
                    maximum_reset_q_error,
                    float(
                        torch.max(
                            torch.abs(
                                dof_state[rows[:, None], gather[None, :], 0]
                                - author_q_true[step_index][None, :]
                            )
                        ).item()
                    ),
                )
                maximum_reset_qdot_error = max(
                    maximum_reset_qdot_error,
                    float(
                        torch.max(
                            torch.abs(
                                dof_state[rows[:, None], gather[None, :], 1]
                                - author_qdot[step_index][None, :]
                            )
                        ).item()
                    ),
                )

            target = targets[step_index].expand(len(actor_specs), -1)
            actuator_step = core.step(
                target, dof_state[:, gather, 0], dof_state[:, gather, 1]
            )
            effort.zero_()
            effort[:, gather] = actuator_step.applied_torque
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort.contiguous().view(-1))
            ):
                raise RuntimeError(f"OFAT effort write failed at interval {step_index}")
            for variant in variants:
                row = one_step_rows[variant.name]
                applied[variant.name][step_index] = (
                    actuator_step.applied_torque[row].detach().cpu().numpy()
                )
                raw_pd[variant.name][step_index] = (
                    actuator_step.raw_pd_torque[row].detach().cpu().numpy()
                )
                saturated[variant.name][step_index] = (
                    actuator_step.saturated_torque[row].detach().cpu().numpy()
                )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            q_true[:, step_index + 1] = dof_state[:, gather, 0].detach().cpu().numpy()
            qdot[:, step_index + 1] = dof_state[:, gather, 1].detach().cpu().numpy()
    finally:
        gym.destroy_sim(sim)

    index = {
        f"{variant.name}/{scenario.name}": row
        for row, (variant, scenario) in enumerate(actor_specs)
    }
    return {
        "variants": tuple(variants),
        "scenarios": scenarios,
        "config": config,
        "index": index,
        "q_true": q_true.astype(np.float64),
        "qdot": qdot.astype(np.float64),
        "one_step_applied_torque": {
            name: value.astype(np.float64) for name, value in applied.items()
        },
        "one_step_raw_pd_torque": {
            name: value.astype(np.float64) for name, value in raw_pd.items()
        },
        "one_step_saturated_torque": {
            name: value.astype(np.float64) for name, value in saturated.items()
        },
        "runtime_properties": context["runtime_properties"],
        "state_reset_audit": {
            "maximum_q_write_error": maximum_reset_q_error,
            "maximum_qdot_write_error": maximum_reset_qdot_error,
            "actuator_fifo_reset_at_segment_boundaries": False,
            "control_target_fifo_updates_per_400Hz_interval": 1,
        },
    }


def _evaluate_variant(
    matrix: Mapping[str, Any], variant_name: str, data: ReplayData, fit: DecodedFit
) -> Dict[str, Any]:
    reference_q = np.asarray(data.sim_method_dof_pos, dtype=np.float64)
    reference_qdot = np.asarray(data.sim_method_dof_vel, dtype=np.float64)
    one_step_row = matrix["index"][f"{variant_name}/one_step"]
    q_compare = matrix["q_true"][one_step_row] - fit.encoder_bias[None, :]
    qdot = matrix["qdot"][one_step_row]
    velocity_residual = qdot[1:] - reference_qdot[1:]
    acceleration_residual = velocity_residual / PHYSICS_DT
    applied = matrix["one_step_applied_torque"][variant_name]
    author_qdot = reference_qdot[:-1]
    position_joint = per_joint_rmse(q_compare, reference_q)
    velocity_joint = per_joint_rmse(qdot, reference_qdot)
    acceleration_joint = np.sqrt(
        np.mean(np.square(acceleration_residual), axis=0, dtype=np.float64)
    )
    focus: Dict[str, Any] = {}
    per_joint: Dict[str, Any] = {}
    for joint, name in enumerate(CANONICAL_JOINT_NAMES):
        residual = acceleration_residual[:, joint]
        item = {
            "position_rmse_rad": float(position_joint[joint]),
            "velocity_rmse_rad_s": float(velocity_joint[joint]),
            "acceleration_residual_mean_rad_s2": float(np.mean(residual)),
            "acceleration_residual_rmse_rad_s2": float(acceleration_joint[joint]),
            "corr_acc_residual_qdot": _correlation(residual, author_qdot[:, joint]),
            "corr_acc_residual_applied_torque": _correlation(residual, applied[:, joint]),
            "corr_acc_residual_sign_qdot": _correlation(
                residual, np.sign(author_qdot[:, joint])
            ),
            "regression_acc_residual_on_qdot": _regression_structure(
                residual, author_qdot[:, joint]
            ),
        }
        per_joint[name] = item
        if name in FOCUS_JOINTS:
            focus[name] = item
    horizon = {}
    for scenario in ("H8", "H32", "H128", "full"):
        row = matrix["index"][f"{variant_name}/{scenario}"]
        candidate = matrix["q_true"][row] - fit.encoder_bias[None, :]
        horizon[scenario] = {
            "position_rmse_rad": rmse(candidate, reference_q),
            "per_joint_position_rmse_rad": dict(
                zip(CANONICAL_JOINT_NAMES, per_joint_rmse(candidate, reference_q).tolist())
            ),
        }
    return {
        "one_step": {
            "position_rmse_rad": rmse(q_compare, reference_q),
            "velocity_rmse_rad_s": rmse(qdot, reference_qdot),
            "acceleration_rmse_rad_s2": float(
                np.sqrt(np.mean(np.square(acceleration_residual), dtype=np.float64))
            ),
            "per_joint": per_joint,
            "focus_joints": focus,
            "state_zero_included_in_position_velocity_rmse": True,
            "acceleration_interval_count": int(len(acceleration_residual)),
        },
        "horizons": horizon,
        "arrays": {
            "q_compare": q_compare,
            "qdot": qdot,
            "velocity_residual": velocity_residual,
            "acceleration_residual": acceleration_residual,
            "applied_torque": applied,
        },
    }


def _relative_change(candidate: float, baseline: float) -> float:
    return float(candidate / baseline - 1.0) if baseline else float("nan")


def _sensitivity(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    relative = {
        "one_step_position": _relative_change(
            candidate["one_step"]["position_rmse_rad"],
            baseline["one_step"]["position_rmse_rad"],
        ),
        "one_step_velocity": _relative_change(
            candidate["one_step"]["velocity_rmse_rad_s"],
            baseline["one_step"]["velocity_rmse_rad_s"],
        ),
        "one_step_acceleration": _relative_change(
            candidate["one_step"]["acceleration_rmse_rad_s2"],
            baseline["one_step"]["acceleration_rmse_rad_s2"],
        ),
        "H32_position": _relative_change(
            candidate["horizons"]["H32"]["position_rmse_rad"],
            baseline["horizons"]["H32"]["position_rmse_rad"],
        ),
        "H128_position": _relative_change(
            candidate["horizons"]["H128"]["position_rmse_rad"],
            baseline["horizons"]["H128"]["position_rmse_rad"],
        ),
        "full_position": _relative_change(
            candidate["horizons"]["full"]["position_rmse_rad"],
            baseline["horizons"]["full"]["position_rmse_rad"],
        ),
    }
    material_fields = (
        "one_step_velocity", "one_step_acceleration", "H32_position"
    )
    local_improvement = (
        relative["one_step_velocity"] <= -MATERIAL_THRESHOLD
        or relative["one_step_acceleration"] <= -MATERIAL_THRESHOLD
    )
    h32_improvement = relative["H32_position"] < 0.0
    h32_material_improvement = relative["H32_position"] <= -MATERIAL_THRESHOLD
    local_direction_improvement = (
        relative["one_step_velocity"] < 0.0
        or relative["one_step_acceleration"] < 0.0
    )
    return {
        "relative_change_vs_baseline": relative,
        "material_change": any(abs(relative[field]) >= MATERIAL_THRESHOLD for field in material_fields),
        "material_improvement_fields": [
            field for field in material_fields if relative[field] <= -MATERIAL_THRESHOLD
        ],
        "evidence_backed_material_improvement": bool(
            (local_improvement and h32_improvement)
            or (h32_material_improvement and local_direction_improvement)
        ),
        "decision_rule": (
            "Material change: |relative delta| >=10% in one-step velocity, one-step "
            "acceleration, or H32. Evidence-backed improvement additionally requires "
            "local-transition and H32 changes to both point toward improvement."
        ),
    }


def _mirror_diagnostic(
    evaluation: Mapping[str, Any], axis: Mapping[str, Any]
) -> Dict[str, Any]:
    arrays = evaluation["arrays"]
    rf_index = CANONICAL_JOINT_NAMES.index("RF_KFE")
    lh_index = CANONICAL_JOINT_NAMES.index("LH_KFE")
    rf_sign = axis["RF_KFE"]["motion_alignment_sign"]
    lh_sign = axis["LH_KFE"]["motion_alignment_sign"]
    residual = arrays["acceleration_residual"]
    qdot = np.asarray(arrays["qdot"][:-1], dtype=np.float64)
    tau = arrays["applied_torque"]
    rf_raw = residual[:, rf_index]
    lh_raw = residual[:, lh_index]
    rf_aligned = rf_sign * rf_raw
    lh_aligned = lh_sign * lh_raw
    raw_correlation = _correlation(rf_raw, lh_raw)
    aligned_correlation = _correlation(rf_aligned, lh_aligned)
    aligned_means = (float(np.mean(rf_aligned)), float(np.mean(lh_aligned)))
    magnitude_ratio = (
        abs(aligned_means[0]) / abs(aligned_means[1])
        if aligned_means[1] != 0.0
        else float("inf")
    )
    mean_direction_consistent = aligned_means[0] * aligned_means[1] > 0.0
    trace_consistent = bool(aligned_correlation >= 0.80)
    return {
        "rule": "Multiply qdot, generalized torque, and acceleration residual by the sign of the dominant URDF joint-axis component; joint order is unchanged.",
        "axis": {"RF_KFE": axis["RF_KFE"], "LH_KFE": axis["LH_KFE"]},
        "raw": {
            "RF_mean_rad_s2": float(np.mean(rf_raw)),
            "LH_mean_rad_s2": float(np.mean(lh_raw)),
            "RF_LH_residual_trace_correlation": raw_correlation,
        },
        "motion_aligned": {
            "RF_mean_rad_s2": aligned_means[0],
            "LH_mean_rad_s2": aligned_means[1],
            "absolute_mean_ratio_RF_over_LH": magnitude_ratio,
            "RF_LH_residual_trace_correlation": aligned_correlation,
            "RF_LH_residual_trace_rmse_rad_s2": rmse(rf_aligned, lh_aligned),
            "mean_direction_consistent": mean_direction_consistent,
            "trace_consistent_at_corr_ge_0p8": trace_consistent,
            "same_dynamic_deviation_supported": bool(mean_direction_consistent and trace_consistent),
        },
        "coordinate_transform_invariance": {
            "RF_corr_residual_qdot": _correlation(
                rf_sign * rf_raw, rf_sign * qdot[:, rf_index]
            ),
            "LH_corr_residual_qdot": _correlation(
                lh_sign * lh_raw, lh_sign * qdot[:, lh_index]
            ),
            "RF_corr_residual_tau": _correlation(
                rf_sign * rf_raw, rf_sign * tau[:, rf_index]
            ),
            "LH_corr_residual_tau": _correlation(
                lh_sign * lh_raw, lh_sign * tau[:, lh_index]
            ),
            "note": (
                "A consistent coordinate sign change leaves each within-joint correlation "
                "unchanged; it must not be used to make opposite correlation signs disappear."
            ),
        },
    }


def _timestamp_jitter_diagnostic(
    evaluation: Mapping[str, Any], data: ReplayData
) -> Dict[str, Any]:
    jitter = np.diff(np.asarray(data.time, dtype=np.float64)) - PHYSICS_DT
    velocity_residual = np.asarray(evaluation["arrays"]["velocity_residual"], dtype=np.float64)
    acceleration_residual = np.asarray(
        evaluation["arrays"]["acceleration_residual"], dtype=np.float64
    )
    author_position_velocity_residual = (
        np.diff(np.asarray(data.sim_method_dof_pos, dtype=np.float64), axis=0) / PHYSICS_DT
        - np.asarray(data.sim_method_dof_vel[1:], dtype=np.float64)
    )

    def correlations(values: np.ndarray) -> Dict[str, Any]:
        per_joint = {
            name: _correlation(values[:, joint], jitter)
            for joint, name in enumerate(CANONICAL_JOINT_NAMES)
        }
        overall = _correlation(values.reshape(-1), np.repeat(jitter, 12))
        return {"overall": overall, "per_joint": per_joint}

    velocity_corr = correlations(velocity_residual)
    acceleration_corr = correlations(acceleration_residual)
    author_corr = correlations(author_position_velocity_residual)
    finite_values = [
        abs(value)
        for group in (velocity_corr, acceleration_corr, author_corr)
        for value in (group["overall"], *group["per_joint"].values())
        if np.isfinite(value)
    ]
    maximum = max(finite_values, default=0.0)
    return {
        "jitter_definition": "recorded real.time[k+1]-real.time[k]-0.0025",
        "simulation_dt_remains_fixed_s": PHYSICS_DT,
        "variable_dt_simulation_run": False,
        "one_step_velocity_residual_correlation": velocity_corr,
        "one_step_acceleration_residual_correlation": acceleration_corr,
        "author_position_derived_velocity_residual_correlation": author_corr,
        "maximum_absolute_reported_correlation": maximum,
        "near_zero_threshold": 0.10,
        "timestamp_jitter_hypothesis": "CLOSED" if maximum < 0.10 else "NOT_CLOSED",
    }


def _strip_arrays(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "arrays"}


def _matrix_traces(prefix: str, matrix: Mapping[str, Any], fit: DecodedFit) -> Dict[str, np.ndarray]:
    traces: Dict[str, np.ndarray] = {}
    for key, row in matrix["index"].items():
        safe = key.replace("/", "__")
        traces[f"{prefix}__{safe}__q_encoder"] = (
            matrix["q_true"][row] - fit.encoder_bias[None, :]
        ).astype(np.float32)
        traces[f"{prefix}__{safe}__qdot"] = matrix["qdot"][row].astype(np.float32)
    for variant in matrix["variants"]:
        name = variant.name
        traces[f"{prefix}__{name}__one_step_applied_torque"] = matrix[
            "one_step_applied_torque"
        ][name].astype(np.float32)
        traces[f"{prefix}__{name}__one_step_raw_pd_torque"] = matrix[
            "one_step_raw_pd_torque"
        ][name].astype(np.float32)
        traces[f"{prefix}__{name}__one_step_saturated_torque"] = matrix[
            "one_step_saturated_torque"
        ][name].astype(np.float32)
    return traces


def _format_percent(value: float) -> str:
    return f"{100.0 * value:+.3f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    p4 = report["P4"]
    p5 = report["P5"]
    lines = [
        "# Stage 0C P4/P5 Narrow OFAT Audit",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "```text",
        "Stage 0A: PASS",
        "Stage 0B: frozen",
        "Stage 0C: FAIL",
        "Actuator Branch A: CLOSED",
        "Locomotion/PPO: BLOCKED",
        "```",
        "",
        "## Frozen scope and provenance",
        "",
        "This run is diagnostic-only. It does not change actuator code, fitted values, the "
        "formal replay, or any Stage 0 gate.",
        "",
        "Isaac Gym/PhysX joint `friction` is treated as a dimensionless, "
        "transmission-force/load-dependent Coulomb coefficient, not a fixed Nm torque. "
        "Joint `damping` is the velocity-dependent viscous term. The source is the "
        f"[NVIDIA engineer explanation]({FRICTION_SEMANTICS_URL}).",
        "",
        "## P4 property null ablations",
        "",
        "| Variant | one-step q | one-step qdot | one-step acc | H32 | H128 | full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "damping0", "friction0", "both0"):
        item = p4["evaluations"][name]
        lines.append(
            f"| {name} | {item['one_step']['position_rmse_rad']:.9g} | "
            f"{item['one_step']['velocity_rmse_rad_s']:.9g} | "
            f"{item['one_step']['acceleration_rmse_rad_s2']:.9g} | "
            f"{item['horizons']['H32']['position_rmse_rad']:.9g} | "
            f"{item['horizons']['H128']['position_rmse_rad']:.9g} | "
            f"{item['horizons']['full']['position_rmse_rad']:.9g} |"
        )
    lines.extend(["", "Relative change versus baseline:", ""])
    lines.extend(
        [
            "| Variant | qdot | acc | H32 | H128 | full | material change | evidence-backed improvement |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for name in ("damping0", "friction0", "both0"):
        item = p4["sensitivity"][name]
        relative = item["relative_change_vs_baseline"]
        lines.append(
            f"| {name} | {_format_percent(relative['one_step_velocity'])} | "
            f"{_format_percent(relative['one_step_acceleration'])} | "
            f"{_format_percent(relative['H32_position'])} | "
            f"{_format_percent(relative['H128_position'])} | "
            f"{_format_percent(relative['full_position'])} | "
            f"{item['material_change']} | {item['evidence_backed_material_improvement']} |"
        )
    lines.extend(["", "## Focus-joint one-step acceleration residual", ""])
    lines.extend(
        [
            "| Variant | Joint | mean | RMSE | corr(qdot) | corr(tau) | corr(sign qdot) | regression slope sign / R2 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for variant in ("baseline", "damping0", "friction0", "both0"):
        for joint in FOCUS_JOINTS:
            item = p4["evaluations"][variant]["one_step"]["focus_joints"][joint]
            regression = item["regression_acc_residual_on_qdot"]
            lines.append(
                f"| {variant} | {joint} | "
                f"{item['acceleration_residual_mean_rad_s2']:.7g} | "
                f"{item['acceleration_residual_rmse_rad_s2']:.7g} | "
                f"{item['corr_acc_residual_qdot']:.7g} | "
                f"{item['corr_acc_residual_applied_torque']:.7g} | "
                f"{item['corr_acc_residual_sign_qdot']:.7g} | "
                f"{regression['slope_sign']} / {regression['r_squared']:.7g} |"
            )
    mirror = report["mirror_symmetry"]
    jitter = report["timestamp_jitter"]
    lines.extend(
        [
            "",
            "## Mirror and timestamp diagnostics",
            "",
            f"RF/LH KFE motion-aligned mean direction consistent: "
            f"**{mirror['motion_aligned']['mean_direction_consistent']}**; residual-trace "
            f"correlation after alignment: "
            f"`{mirror['motion_aligned']['RF_LH_residual_trace_correlation']:.7g}`; "
            f"same-dynamic-deviation rule met: "
            f"**{mirror['motion_aligned']['same_dynamic_deviation_supported']}**.",
            "",
            "A consistent axis-sign coordinate transform leaves each within-joint "
            "residual/qdot and residual/torque correlation invariant; no correlation sign "
            "was altered to manufacture agreement.",
            "",
            f"Maximum absolute timestamp-jitter correlation: "
            f"`{jitter['maximum_absolute_reported_correlation']:.7g}`. Hypothesis: "
            f"**{jitter['timestamp_jitter_hypothesis']}**. Simulation remained fixed-dt.",
            "",
            "## P5 integration diagnostics",
            "",
            f"P5 executed: **{p5['executed']}**. Reason: {p5['reason']}",
            "",
        ]
    )
    if p5["executed"]:
        lines.extend(
            [
                "| Variant | one-step qdot | one-step acc | H32 | H128 | full | evidence-backed improvement |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name in ("substeps2", "velocity_iterations1"):
            item = p5["evaluations"][name]
            sensitivity = p5["sensitivity"][name]
            lines.append(
                f"| {name} | {item['one_step']['velocity_rmse_rad_s']:.9g} | "
                f"{item['one_step']['acceleration_rmse_rad_s2']:.9g} | "
                f"{item['horizons']['H32']['position_rmse_rad']:.9g} | "
                f"{item['horizons']['H128']['position_rmse_rad']:.9g} | "
                f"{item['horizons']['full']['position_rmse_rad']:.9g} | "
                f"{sensitivity['evidence_backed_material_improvement']} |"
            )
        lines.extend(
            [
                "",
                "Relative change versus P4 baseline:",
                "",
                "| Variant | qdot | acc | H32 | H128 | full | material change |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name in ("substeps2", "velocity_iterations1"):
            sensitivity = p5["sensitivity"][name]
            relative = sensitivity["relative_change_vs_baseline"]
            lines.append(
                f"| {name} | {_format_percent(relative['one_step_velocity'])} | "
                f"{_format_percent(relative['one_step_acceleration'])} | "
                f"{_format_percent(relative['H32_position'])} | "
                f"{_format_percent(relative['H128_position'])} | "
                f"{_format_percent(relative['full_position'])} | "
                f"{sensitivity['material_change']} |"
            )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- P4 case: **{decision['P4_case']}**",
            f"- Material sensitivity found: **{decision['material_sensitivity_found']}**",
            f"- Evidence-backed material improvement found: "
            f"**{decision['material_improvement_found']}**",
            f"- Provenance-backed formal change: **{decision['provenance_backed_formal_change']}**",
            f"- Stage 0C: **{decision['Stage_0C']}**",
            f"- Locomotion/PPO: **{decision['locomotion_PPO']}**",
            f"- Reproducibility-boundary review recommended: "
            f"**{decision['reproducibility_boundary_review_recommended']}**",
            f"- Next step: {decision['next_step']}",
            "",
            "No minimum-RMSE variant was promoted to the formal baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def run_p4_p5_audit(
    device_id: int = 0, output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    # Isaac Gym Preview 4 import must precede torch and fitting.npy unpickling.
    from isaacgym import gymapi  # noqa: F401
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    artifact_dir = _new_output_dir(output_dir)
    data = load_replay_data()
    fit = decode_fit()
    axes = _joint_axis_signs()
    p4_matrix = _run_matrix(
        device_id,
        fit,
        data,
        _variants(),
        IntegrationConfig("P4_baseline_integration"),
    )
    p4_raw = {
        variant.name: _evaluate_variant(p4_matrix, variant.name, data, fit)
        for variant in _variants()
    }
    p4_sensitivity = {
        name: _sensitivity(p4_raw[name], p4_raw["baseline"])
        for name in ("damping0", "friction0", "both0")
    }
    p4_improvements = [
        name
        for name, item in p4_sensitivity.items()
        if item["evidence_backed_material_improvement"]
    ]
    run_p5 = not p4_improvements

    trace = _matrix_traces("P4", p4_matrix, fit)
    p5_raw: Dict[str, Any] = {}
    p5_sensitivity: Dict[str, Any] = {}
    p5_properties: Dict[str, Any] = {}
    p5_reset_audit: Dict[str, Any] = {}
    if run_p5:
        baseline_only = (PropertyVariant("baseline"),)
        for config in (
            IntegrationConfig("substeps2", substeps=2),
            IntegrationConfig("velocity_iterations1", num_velocity_iterations=1),
        ):
            matrix = _run_matrix(device_id, fit, data, baseline_only, config)
            p5_raw[config.name] = _evaluate_variant(matrix, "baseline", data, fit)
            p5_sensitivity[config.name] = _sensitivity(
                p5_raw[config.name], p4_raw["baseline"]
            )
            p5_properties[config.name] = matrix["runtime_properties"]
            p5_reset_audit[config.name] = matrix["state_reset_audit"]
            trace.update(_matrix_traces(f"P5_{config.name}", matrix, fit))

    mirror = _mirror_diagnostic(p4_raw["baseline"], axes)
    timestamp = _timestamp_jitter_diagnostic(p4_raw["baseline"], data)
    if p4_sensitivity["damping0"]["evidence_backed_material_improvement"]:
        p4_case = "P4-A_damping_ablation_material_improvement"
        next_step = (
            "Review legacy damping application semantics (URDF/import/DOF-property and "
            "possible duplicate writes) without adjusting the fitted value."
        )
    elif p4_sensitivity["friction0"]["evidence_backed_material_improvement"]:
        p4_case = "P4-B_friction_ablation_material_improvement"
        next_step = (
            "Review legacy Isaac Gym friction-property use/version provenance; do not "
            "reinterpret the fitted coefficient as Nm."
        )
    elif p4_sensitivity["both0"]["evidence_backed_material_improvement"]:
        p4_case = "P4-coupled_both-null_material_improvement"
        next_step = (
            "Review legacy coupled damping/friction property application semantics without "
            "parameter scaling or re-identification."
        )
    else:
        p4_case = "P4-C_no_evidence_backed_material_improvement"
        next_step = "P5 completed; classify integration sensitivity before any further action."

    p5_improvements = [
        name
        for name, item in p5_sensitivity.items()
        if item["evidence_backed_material_improvement"]
    ]
    if run_p5 and p5_improvements:
        if "substeps2" in p5_improvements:
            next_step = "Seek legacy simulation substeps provenance; do not promote substeps=2 automatically."
        else:
            next_step = "Seek legacy solver velocity-iteration provenance; do not promote velocity_iterations=1 automatically."
    boundary_review = bool(run_p5 and not p5_improvements)
    if boundary_review:
        next_step = (
            "Submit a Stage 0 reproducibility-boundary review for human choice; keep the "
            "strict 0.01-rad gate unchanged and keep Stage 0C failed."
        )

    p4_material_changes = [
        name for name, item in p4_sensitivity.items() if item["material_change"]
    ]
    p5_material_changes = [
        name for name, item in p5_sensitivity.items() if item["material_change"]
    ]
    report: Dict[str, Any] = {
        "schema": "pace_stage0.p4_p5_audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "Actuator_Branch_A": "CLOSED",
            "joint_order_layout": "CLOSED",
            "target_timing_plus_minus_1": "CLOSED",
            "initial_velocity": "CLOSED",
            "float_precision": "CLOSED",
            "large_asset_PhysX_search": "FORBIDDEN",
            "locomotion_PPO": "BLOCKED",
        },
        "git": _git_snapshot(),
        "random_seed": 0,
        "reproduction": {
            "command": (
                "conda run -n bruce_gym env PYTHONPATH=src python -m "
                "pace_stage0.cli p4_p5_audit --device-id 0"
            ),
            "implementation_sha256": {
                "p4_p5_audit.py": _sha256(Path(__file__).resolve()),
                "actuator.py": _sha256(Path(__file__).resolve().with_name("actuator.py")),
                "decoder.py": _sha256(Path(__file__).resolve().with_name("decoder.py")),
                "replay.py": _sha256(Path(__file__).resolve().with_name("replay.py")),
            },
        },
        "friction_damping_provenance": {
            "source": FRICTION_SEMANTICS_URL,
            "source_type": "NVIDIA Developer Forums / NVIDIA engineer explanation",
            "Isaac_Gym_Preview_4_DOF_friction": (
                "dimensionless coefficient; PhysX estimates joint normal load from "
                "transmission force and applies Coulomb-like friction"
            ),
            "Isaac_Gym_Preview_4_DOF_damping": "velocity-dependent joint damping",
            "fixed_Nm_friction_interpretation": False,
            "legacy_PACE_property_application": "UNAVAILABLE / provenance unresolved",
        },
        "protocol": {
            "dt_s": PHYSICS_DT,
            "target_and_actuator_fifo_updates_per_control_interval": 1,
            "one_step": (
                "Author q/qdot are written before each interval; target[t] drives [t,t+1); "
                "FIFO is continuous and is not reset at interval boundaries."
            ),
            "horizons": list(HORIZONS) + ["full"],
            "horizon_reset": (
                "Author q/qdot are written before intervals t>0 divisible by H; FIFO is continuous."
            ),
            "comparison_frame": "q_encoder_ours=q_true_ours-bias vs sim_method.dof_pos",
            "formal_replay_modified": False,
            "formal_actuator_modified": False,
            "parameter_scaling_or_refit": False,
        },
        "joint_axis_motion_alignment": axes,
        "P4": {
            "variants": [variant.__dict__ for variant in _variants()],
            "integration": IntegrationConfig("P4_baseline_integration").__dict__,
            "runtime_properties": p4_matrix["runtime_properties"],
            "state_reset_audit": p4_matrix["state_reset_audit"],
            "evaluations": {name: _strip_arrays(item) for name, item in p4_raw.items()},
            "sensitivity": p4_sensitivity,
        },
        "mirror_symmetry": mirror,
        "timestamp_jitter": timestamp,
        "P5": {
            "executed": run_p5,
            "reason": (
                "P4 had no evidence-backed material improvement"
                if run_p5
                else f"Stopped because P4 improvements were {p4_improvements}"
            ),
            "allowed_configs": {
                "substeps2": IntegrationConfig("substeps2", substeps=2).__dict__,
                "velocity_iterations1": IntegrationConfig(
                    "velocity_iterations1", num_velocity_iterations=1
                ).__dict__,
            },
            "runtime_properties": p5_properties,
            "state_reset_audit": p5_reset_audit,
            "evaluations": {name: _strip_arrays(item) for name, item in p5_raw.items()},
            "sensitivity": p5_sensitivity,
        },
        "decision": {
            "P4_case": p4_case,
            "P4_material_changes": p4_material_changes,
            "P4_material_improvements": p4_improvements,
            "P5_executed": run_p5,
            "P5_material_changes": p5_material_changes,
            "P5_material_improvements": p5_improvements,
            "material_sensitivity_found": bool(
                p4_material_changes or p5_material_changes
            ),
            "material_improvement_found": bool(p4_improvements or p5_improvements),
            "provenance_backed_formal_change": False,
            "automatic_minimum_RMSE_selection": False,
            "formal_baseline_changed": False,
            "formal_gate_changed": False,
            "Stage_0C": "FAIL",
            "locomotion_PPO": "BLOCKED",
            "reproducibility_boundary_review_recommended": boundary_review,
            "next_step": next_step,
        },
    }

    trace.update(
        {
            "time": np.asarray(data.time),
            "target": np.asarray(data.real_des_dof_pos),
            "sim_method_q": np.asarray(data.sim_method_dof_pos),
            "sim_method_qdot": np.asarray(data.sim_method_dof_vel),
            "recorded_dt_jitter": np.diff(np.asarray(data.time, dtype=np.float64))
            - PHYSICS_DT,
        }
    )
    np.savez_compressed(artifact_dir / "p4_p5_traces.npz", **trace)
    _write_json(artifact_dir / "report.json", report)
    (artifact_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote: {artifact_dir / 'report.md'}")
    print(f"wrote: {artifact_dir / 'report.json'}")
    print(f"wrote: {artifact_dir / 'p4_p5_traces.npz'}")
    return _jsonable(report)
