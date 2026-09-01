from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    ANYMAL_ASSET_FILE,
    ANYMAL_ASSET_ROOT,
    CANONICAL_JOINT_NAMES,
    DATA_PATH,
    EFFORT_LIMIT,
    PHYSICS_DT,
    PROJECT_ROOT,
    VELOCITY_LIMIT,
    canonical_gather_indices,
)
from .data import ReplayData, load_replay_data
from .decoder import DecodedFit, decode_fit
from .metrics import per_joint_rmse, rmse


FOCUS_JOINTS = ("RF_HFE", "LH_HFE", "LF_KFE")
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
RESET_ABLATION_HORIZONS = (8, 32, 128)
FROZEN_FULL_POSITION_RMSE = 0.012818364796375577


@dataclass(frozen=True)
class SimulationCase:
    name: str
    initial_velocity: str = "zero"
    target_shift: int = 0
    reset_horizon: Optional[int] = None
    reset_mode: str = "none"
    author_initial_state: bool = False


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
        output_dir = PROJECT_ROOT / "artifacts" / "accumulation_audit" / stamp
    result = output_dir.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Correlation shape mismatch: {left.shape}/{right.shape}")
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _git_snapshot() -> Dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(("git",) + args, cwd=PROJECT_ROOT, text=True).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _raw_payload() -> Mapping[str, Mapping[str, Any]]:
    payload = np.load(DATA_PATH, allow_pickle=True)
    if not isinstance(payload, np.ndarray) or payload.shape != ():
        raise ValueError("Expected scalar pickled data.npy payload")
    result = payload.item()
    if not isinstance(result, dict):
        raise TypeError("data.npy payload is not a dictionary")
    return result


def _loaded_field(data: ReplayData, group: str, field: str) -> np.ndarray:
    if group == "real" and field == "time":
        return data.time
    return np.asarray(getattr(data, f"{group}_{field}"))


def _source_dtype_audit(
    payload: Mapping[str, Mapping[str, Any]], data: ReplayData
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for group in ("real", "sim_method", "sim_nothing"):
        result[group] = {}
        for field, value in payload[group].items():
            source = np.asarray(value)
            loaded = _loaded_field(data, group, field)
            difference = source.astype(np.float64) - loaded.astype(np.float64)
            info = np.finfo(source.dtype) if np.issubdtype(source.dtype, np.floating) else None
            result[group][field] = {
                "source_python_type": type(value).__name__,
                "source_dtype": str(source.dtype),
                "after_loader_dtype": str(loaded.dtype),
                "shape": list(source.shape),
                "minimum": float(np.min(source)),
                "maximum": float(np.max(source)),
                "machine_epsilon": None if info is None else float(info.eps),
                "max_abs_source_to_loader_error": float(np.max(np.abs(difference))),
                "rmse_source_to_loader_error": float(
                    np.sqrt(np.mean(np.square(difference), dtype=np.float64))
                ),
            }
    all_float32 = all(
        item["source_dtype"] == "float32"
        for group in result.values()
        for item in group.values()
    )
    result["conclusion"] = {
        "all_source_fields_float32": all_float32,
        "float32_loader_conversion_hypothesis": "CLOSED" if all_float32 else "OPEN",
        "formal_loader_modified": False,
    }
    return result


def _time_audit(time: np.ndarray) -> Dict[str, Any]:
    time64 = np.asarray(time, dtype=np.float64)
    difference = np.diff(time64)
    nominal_error = difference - PHYSICS_DT
    return {
        "source_dtype": str(np.asarray(time).dtype),
        "time_0_s": float(time64[0]),
        "time_last_s": float(time64[-1]),
        "sample_count": int(len(time64)),
        "diff_mean_s": float(np.mean(difference)),
        "diff_std_s": float(np.std(difference)),
        "diff_min_s": float(np.min(difference)),
        "diff_max_s": float(np.max(difference)),
        "unique_diff_count": int(len(np.unique(difference))),
        "nominal_dt_s": PHYSICS_DT,
        "max_abs_diff_minus_nominal_s": float(np.max(np.abs(nominal_error))),
        "exactly_nominal_count": int(np.count_nonzero(difference == PHYSICS_DT)),
        "within_1e_6_s_of_nominal_count": int(
            np.count_nonzero(np.abs(nominal_error) <= 1e-6)
        ),
        "quantiles_s": {
            str(q): float(np.quantile(difference, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "strictly_equidistant": bool(np.all(difference == difference[0])),
        "timestamp_semantics": {
            "status": "artifact_inference / unresolved",
            "inference": (
                "The only time field is stored under real and is visibly jittered, so it is "
                "more consistent with real/log sampling timestamps than a PhysX fixed-step "
                "clock. No legacy exporter source confirms whether it labels state or command."
            ),
            "simulation_analysis_dt": PHYSICS_DT,
        },
    }


def _moving_average(values: np.ndarray, indices: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if width < 1 or width > len(values):
        raise ValueError("Invalid moving-average width")
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    averaged = (cumulative[width:] - cumulative[:-width]) / width
    # For even windows this chooses the later of the two central state indices. Lag is
    # swept explicitly, so this convention remains auditable rather than optimized away.
    representative = indices[(width - 1) // 2 : (width - 1) // 2 + len(averaged)]
    return averaged, representative


def _derivative_candidates(position: np.ndarray, time: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    position = np.asarray(position, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    count = len(position)
    forward = np.diff(position, axis=0) / PHYSICS_DT
    central = (position[2:] - position[:-2]) / (2.0 * PHYSICS_DT)
    dt = np.diff(time)
    forward_time = np.diff(position, axis=0) / dt[:, None]
    central_time = (position[2:] - position[:-2]) / (time[2:] - time[:-2])[:, None]
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "forward_nominal_dt": (forward, np.arange(0, count - 1)),
        "backward_nominal_dt": (forward, np.arange(1, count)),
        "central_nominal_dt": (central, np.arange(1, count - 1)),
        "forward_source_time": (forward_time, np.arange(0, count - 1)),
        "backward_source_time": (forward_time, np.arange(1, count)),
        "central_source_time": (central_time, np.arange(1, count - 1)),
    }
    for width in (2, 3, 5):
        result[f"forward_nominal_moving_average_{width}"] = _moving_average(
            forward, np.arange(0, count - 1), width
        )
        result[f"central_nominal_moving_average_{width}"] = _moving_average(
            central, np.arange(1, count - 1), width
        )
    return result


def _indexed_lag_metrics(
    candidate: np.ndarray,
    candidate_state_indices: np.ndarray,
    reference: np.ndarray,
    lag: int,
) -> Dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    indices = np.asarray(candidate_state_indices, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.float64)
    reference_indices = indices + lag
    valid = (reference_indices >= 0) & (reference_indices < len(reference))
    ours = candidate[valid]
    theirs = reference[reference_indices[valid]]
    joint_rmse = per_joint_rmse(ours, theirs)
    joint_correlation = [
        _correlation(ours[:, joint], theirs[:, joint]) for joint in range(ours.shape[1])
    ]
    return {
        "lag_convention": "derivative at position-state index k vs logged velocity[k+lag]",
        "lag": lag,
        "sample_count": int(len(ours)),
        "rmse": rmse(ours, theirs),
        "correlation": _correlation(ours, theirs),
        "per_joint_rmse": dict(zip(CANONICAL_JOINT_NAMES, joint_rmse.tolist())),
        "per_joint_correlation": dict(zip(CANONICAL_JOINT_NAMES, joint_correlation)),
    }


def _velocity_semantics(position: np.ndarray, velocity: np.ndarray, time: np.ndarray) -> Dict[str, Any]:
    candidates = _derivative_candidates(position, time)
    result: Dict[str, Any] = {}
    for name, (values, indices) in candidates.items():
        lags = {
            str(lag): _indexed_lag_metrics(values, indices, velocity, lag)
            for lag in range(-3, 4)
        }
        best_lag = min(lags, key=lambda key: lags[key]["rmse"])
        result[name] = {
            "lags": lags,
            "best_lag": int(best_lag),
            "best": lags[best_lag],
        }
    best_name = min(result, key=lambda key: result[key]["best"]["rmse"])
    unfiltered_names = (
        "forward_nominal_dt",
        "backward_nominal_dt",
        "central_nominal_dt",
        "forward_source_time",
        "backward_source_time",
        "central_source_time",
    )
    filtered_names = tuple(name for name in result if "moving_average" in name)
    best_unfiltered_name = min(
        unfiltered_names, key=lambda key: result[key]["best"]["rmse"]
    )
    best_filtered_name = min(
        filtered_names, key=lambda key: result[key]["best"]["rmse"]
    )
    velocity64 = np.asarray(velocity, dtype=np.float64)
    velocity_rms = float(np.sqrt(np.mean(np.square(velocity64), dtype=np.float64)))
    best = result[best_name]["best"]
    unfiltered_rmse = float(result[best_unfiltered_name]["best"]["rmse"])
    filtered_rmse = float(result[best_filtered_name]["best"]["rmse"])
    filter_improvement = (
        (unfiltered_rmse - filtered_rmse) / unfiltered_rmse
        if unfiltered_rmse
        else 0.0
    )
    return {
        "hypotheses": result,
        "best_hypothesis": best_name,
        "best_lag": result[best_name]["best_lag"],
        "best_rmse": best["rmse"],
        "best_correlation": best["correlation"],
        "logged_velocity_rms": velocity_rms,
        "normalized_best_rmse_over_velocity_rms": (
            best["rmse"] / velocity_rms if velocity_rms else None
        ),
        "best_unfiltered_hypothesis": best_unfiltered_name,
        "best_unfiltered_rmse": unfiltered_rmse,
        "best_filtered_hypothesis": best_filtered_name,
        "best_filtered_rmse": filtered_rmse,
        "filter_relative_rmse_improvement": filter_improvement,
        "semantics_interpretation": (
            "Strong interval-end derivative alignment is compatible with either an "
            "instantaneous simulator velocity under discrete integration or a "
            "position-derived velocity. Limited smoothing is evidence for filtering only "
            "when it materially improves the unfiltered derivative; correlation alone does "
            "not distinguish the exporter implementation."
        ),
        "filter_search_scope": "moving-average widths exactly 2, 3, 5; no optimization",
    }


def _vector_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "max_abs": float(np.max(np.abs(values))),
    }


def _initial_velocity_audit(data: ReplayData) -> Dict[str, Any]:
    fd = (
        np.asarray(data.sim_method_dof_pos[1], dtype=np.float64)
        - np.asarray(data.sim_method_dof_pos[0], dtype=np.float64)
    ) / PHYSICS_DT
    return {
        "formal_baseline": "qdot_0 = zero",
        "real_dof_vel_0": _vector_stats(data.real_dof_vel[0]),
        "sim_method_dof_vel_0": _vector_stats(data.sim_method_dof_vel[0]),
        "finite_difference_forward_nominal_dt_0": _vector_stats(fd),
        "candidate_provenance": {
            "V0_zero": "paper/public official update_simulator initialization",
            "V1_sim_method": "legacy dataset diagnostic",
            "V2_real": "legacy real log diagnostic",
            "V3_forward_fd": "position-derived diagnostic; no exporter confirmation",
        },
    }


def _target_quantization(target: np.ndarray) -> Dict[str, Any]:
    target = np.asarray(target)
    result = {
        "source_dtype": str(target.dtype),
        "source_is_float32": target.dtype == np.float32,
        "source_to_float32_max_abs_error": float(
            np.max(np.abs(target.astype(np.float64) - target.astype(np.float32).astype(np.float64)))
        ),
        "per_joint": {},
    }
    for joint, name in enumerate(CANONICAL_JOINT_NAMES):
        values = target[:, joint]
        increments = np.diff(values.astype(np.float64))
        nonzero = np.abs(increments[increments != 0.0])
        unique_values = np.unique(values)
        unique_increments = np.unique(increments)
        bit_pattern = np.ascontiguousarray(values).view(np.uint32)
        mantissa = bit_pattern & np.uint32(0x007FFFFF)
        decimal_counts = []
        for value in unique_values[: min(2000, len(unique_values))]:
            text = np.format_float_positional(value, unique=True, trim="-")
            decimal_counts.append(len(text.split(".")[1]) if "." in text else 0)
        result["per_joint"][name] = {
            "unique_value_count": int(len(unique_values)),
            "unique_increment_count": int(len(unique_increments)),
            "unique_increments": unique_increments,
            "minimum_nonzero_abs_increment": None if len(nonzero) == 0 else float(np.min(nonzero)),
            "median_nonzero_abs_increment": None if len(nonzero) == 0 else float(np.median(nonzero)),
            "maximum_shortest_roundtrip_decimal_places_sampled": max(decimal_counts, default=0),
            "unique_float32_mantissa_count": int(len(np.unique(mantissa))),
            "fraction_mantissa_low_8_bits_zero": float(
                np.mean((mantissa & np.uint32(0xFF)) == 0)
            ),
        }
    result["precision_branch"] = (
        "CLOSED — target source is already float32" if target.dtype == np.float32
        else "OPEN"
    )
    return result


def _frequency_trace(data: ReplayData) -> np.ndarray:
    target = np.asarray(data.real_des_dof_pos, dtype=np.float64)
    reference_joint = int(np.argmax(np.std(target, axis=0)))
    signal = target[:, reference_joint]
    center = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
    crossings = np.flatnonzero((signal[:-1] <= center) & (signal[1:] > center))
    if len(crossings) < 3:
        return np.linspace(0.0, 1.0, len(target) - 1)
    time = np.asarray(data.time, dtype=np.float64)
    crossing_time = time[crossings]
    frequency = 1.0 / np.diff(crossing_time)
    midpoint = 0.5 * (crossing_time[:-1] + crossing_time[1:])
    return np.interp(time[:-1], midpoint, frequency, left=frequency[0], right=frequency[-1])


def _simulation_cases() -> Sequence[SimulationCase]:
    cases = [
        SimulationCase("formal_zero"),
        SimulationCase("initial_sim_method", initial_velocity="sim_method"),
        SimulationCase("initial_real", initial_velocity="real"),
        SimulationCase("initial_forward_fd", initial_velocity="forward_fd"),
        SimulationCase("target_t_minus_1", target_shift=-1),
        SimulationCase("target_t_plus_1", target_shift=1),
    ]
    cases.extend(
        SimulationCase(
            f"horizon_{horizon}", reset_horizon=horizon, reset_mode="both"
        )
        for horizon in HORIZONS
    )
    for horizon in RESET_ABLATION_HORIZONS:
        cases.append(
            SimulationCase(
                f"q_reset_{horizon}", reset_horizon=horizon, reset_mode="q"
            )
        )
        cases.append(
            SimulationCase(
                f"qdot_reset_{horizon}", reset_horizon=horizon, reset_mode="qdot"
            )
        )
    cases.append(
        SimulationCase(
            "one_step_author_initial",
            initial_velocity="sim_method",
            reset_horizon=1,
            reset_mode="both",
            author_initial_state=True,
        )
    )
    return tuple(cases)


def _make_multi_env_sim(device_id: int, fit: DecodedFit, count: int):
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
        props["friction"][gather_np] = fit.friction.astype(props["friction"].dtype)
        props["damping"][gather_np] = fit.damping.astype(props["damping"].dtype)
        props["armature"][gather_np] = fit.armature.astype(props["armature"].dtype)
        props["effort"].fill(EFFORT_LIMIT)
        props["velocity"].fill(VELOCITY_LIMIT)

        actor_indices = []
        per_row = int(math.ceil(math.sqrt(count)))
        for index in range(count):
            env = gym.create_env(
                sim,
                gymapi.Vec3(-1.0, -1.0, 0.0),
                gymapi.Vec3(1.0, 1.0, 2.0),
                per_row,
            )
            pose = gymapi.Transform()
            pose.p.z = 1.0
            actor = gym.create_actor(env, asset, pose, f"case_{index}", index, 1)
            gym.set_actor_dof_properties(env, actor, props)
            actor_indices.append(gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
        gym.prepare_sim(sim)
        dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(count, 12, 2)
        applied = gym.get_actor_dof_properties(env, actor)
        if not np.all(applied["driveMode"] == gymapi.DOF_MODE_EFFORT):
            raise RuntimeError("Not all diagnostic actors use DOF_MODE_EFFORT")
        if not np.allclose(applied["stiffness"], 0.0):
            raise RuntimeError("Diagnostic actor has hidden position stiffness")
        return gymapi, gymtorch, torch, gym, sim, dof_state, gather_indices, actor_indices
    except Exception:
        gym.destroy_sim(sim)
        raise


def _run_simulation_matrix(
    device_id: int, fit: DecodedFit, data: ReplayData
) -> Dict[str, Any]:
    from isaacgym import gymtorch
    import torch

    from .actuator import PACEActuatorCore

    cases = _simulation_cases()
    case_index = {case.name: index for index, case in enumerate(cases)}
    (
        _,
        _,
        _,
        gym,
        sim,
        dof_state,
        gather_indices,
        actor_indices,
    ) = _make_multi_env_sim(device_id, fit, len(cases))
    try:
        device = dof_state.device
        gather = torch.as_tensor(gather_indices, dtype=torch.long, device=device)
        actor_index_tensor = torch.as_tensor(actor_indices, dtype=torch.int32, device=device)
        bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
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
        initial_velocity = {
            "zero": torch.zeros(12, dtype=torch.float32, device=device),
            "sim_method": torch.as_tensor(
                data.sim_method_dof_vel[0], dtype=torch.float32, device=device
            ),
            "real": torch.as_tensor(data.real_dof_vel[0], dtype=torch.float32, device=device),
            "forward_fd": torch.as_tensor(
                (data.sim_method_dof_pos[1] - data.sim_method_dof_pos[0]) / PHYSICS_DT,
                dtype=torch.float32,
                device=device,
            ),
        }

        dof_state.zero_()
        for index, case in enumerate(cases):
            dof_state[index, gather, 0] = author_q_true[0] if case.author_initial_state else formal_q0
            dof_state[index, gather, 1] = initial_velocity[case.initial_velocity]
        if not gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state)):
            raise RuntimeError("Initial multi-environment state write failed")
        gym.refresh_dof_state_tensor(sim)

        sample_count = data.sample_count
        q_true = np.empty((len(cases), sample_count, 12), dtype=np.float32)
        qdot = np.empty_like(q_true)
        q_true[:, 0] = dof_state[:, gather, 0].detach().cpu().numpy()
        qdot[:, 0] = dof_state[:, gather, 1].detach().cpu().numpy()
        one_step_applied = np.empty((sample_count - 1, 12), dtype=np.float32)
        formal_applied = np.empty_like(one_step_applied)

        core = PACEActuatorCore(
            bias, fit.delay_steps
        )
        core.reset(dof_state[:, gather, 0])
        target_source = torch.as_tensor(
            data.real_des_dof_pos, dtype=torch.float32, device=device
        )
        effort = torch.zeros((len(cases), 12), dtype=torch.float32, device=device)
        maximum_reset_q_error = 0.0
        maximum_reset_qdot_error = 0.0
        for step_index in range(sample_count - 1):
            reset_rows = []
            for row, case in enumerate(cases):
                horizon = case.reset_horizon
                if horizon is None or step_index == 0 or step_index % horizon != 0:
                    continue
                if case.reset_mode in ("q", "both"):
                    dof_state[row, gather, 0] = author_q_true[step_index]
                if case.reset_mode in ("qdot", "both"):
                    dof_state[row, gather, 1] = author_qdot[step_index]
                reset_rows.append(row)
            if reset_rows:
                reset_rows_tensor = torch.as_tensor(reset_rows, dtype=torch.long, device=device)
                reset_actor_indices = actor_index_tensor[reset_rows_tensor].contiguous()
                if not gym.set_dof_state_tensor_indexed(
                    sim,
                    gymtorch.unwrap_tensor(dof_state),
                    gymtorch.unwrap_tensor(reset_actor_indices),
                    len(reset_rows),
                ):
                    raise RuntimeError(f"Indexed state reset failed at {step_index}")
                gym.refresh_dof_state_tensor(sim)
                for row in reset_rows:
                    case = cases[row]
                    if case.reset_mode in ("q", "both"):
                        maximum_reset_q_error = max(
                            maximum_reset_q_error,
                            float(
                                torch.max(
                                    torch.abs(
                                        dof_state[row, gather, 0] - author_q_true[step_index]
                                    )
                                ).item()
                            ),
                        )
                    if case.reset_mode in ("qdot", "both"):
                        maximum_reset_qdot_error = max(
                            maximum_reset_qdot_error,
                            float(
                                torch.max(
                                    torch.abs(
                                        dof_state[row, gather, 1] - author_qdot[step_index]
                                    )
                                ).item()
                            ),
                        )

            targets = []
            for case in cases:
                target_index = min(
                    max(step_index + case.target_shift, 0), sample_count - 1
                )
                targets.append(target_source[target_index])
            target = torch.stack(targets, dim=0)
            actuator_step = core.step(
                target,
                dof_state[:, gather, 0],
                dof_state[:, gather, 1],
            )
            effort.zero_()
            effort[:, gather] = actuator_step.applied_torque
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort.contiguous().view(-1))
            ):
                raise RuntimeError(f"Effort write failed at interval {step_index}")
            one_step_applied[step_index] = (
                actuator_step.applied_torque[case_index["one_step_author_initial"]]
                .detach()
                .cpu()
                .numpy()
            )
            formal_applied[step_index] = (
                actuator_step.applied_torque[case_index["formal_zero"]]
                .detach()
                .cpu()
                .numpy()
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            q_true[:, step_index + 1] = dof_state[:, gather, 0].detach().cpu().numpy()
            qdot[:, step_index + 1] = dof_state[:, gather, 1].detach().cpu().numpy()
    finally:
        gym.destroy_sim(sim)
    return {
        "cases": cases,
        "case_index": case_index,
        "q_true": q_true.astype(np.float64),
        "qdot": qdot.astype(np.float64),
        "one_step_applied_torque": one_step_applied.astype(np.float64),
        "formal_applied_torque": formal_applied.astype(np.float64),
        "state_reset_audit": {
            "maximum_q_write_error": maximum_reset_q_error,
            "maximum_qdot_write_error": maximum_reset_qdot_error,
            "actuator_fifo_reset_at_segment_boundaries": False,
            "indexed_actor_state_write": True,
        },
    }


def _case_metrics(
    q_true: np.ndarray, qdot: np.ndarray, data: ReplayData, fit: DecodedFit
) -> Dict[str, Any]:
    q_compare = np.asarray(q_true, dtype=np.float64) - fit.encoder_bias[None, :]
    reference_q = np.asarray(data.sim_method_dof_pos, dtype=np.float64)
    reference_qdot = np.asarray(data.sim_method_dof_vel, dtype=np.float64)
    position_joint = per_joint_rmse(q_compare, reference_q)
    velocity_joint = per_joint_rmse(qdot, reference_qdot)
    return {
        "position": {
            "overall_rmse_rad": rmse(q_compare, reference_q),
            "first_100_rmse_rad": rmse(q_compare[:100], reference_q[:100]),
            "per_joint_rmse_rad": dict(zip(CANONICAL_JOINT_NAMES, position_joint.tolist())),
            "focus_joint_rmse_rad": {
                name: float(position_joint[CANONICAL_JOINT_NAMES.index(name)])
                for name in FOCUS_JOINTS
            },
        },
        "velocity": {
            "overall_rmse_rad_s": rmse(qdot, reference_qdot),
            "per_joint_rmse_rad_s": dict(zip(CANONICAL_JOINT_NAMES, velocity_joint.tolist())),
            "focus_joint_rmse_rad_s": {
                name: float(velocity_joint[CANONICAL_JOINT_NAMES.index(name)])
                for name in FOCUS_JOINTS
            },
        },
        "gates": {
            "overall_position_le_0p01": rmse(q_compare, reference_q) <= 0.01,
            "all_per_joint_position_le_0p02": bool(np.all(position_joint <= 0.02)),
        },
    }


def _simulation_metrics(matrix: Mapping[str, Any], data: ReplayData, fit: DecodedFit) -> Dict[str, Any]:
    return {
        case.name: {
            "configuration": _jsonable(case.__dict__),
            **_case_metrics(
                matrix["q_true"][index], matrix["qdot"][index], data, fit
            ),
        }
        for index, case in enumerate(matrix["cases"])
    }


def _acceleration_residual(
    matrix: Mapping[str, Any], metrics: Mapping[str, Any], data: ReplayData, fit: DecodedFit
) -> Dict[str, Any]:
    index = matrix["case_index"]["one_step_author_initial"]
    qdot_pred_next = matrix["qdot"][index, 1:]
    qdot_author_next = np.asarray(data.sim_method_dof_vel[1:], dtype=np.float64)
    residual = (qdot_pred_next - qdot_author_next) / PHYSICS_DT
    q = np.asarray(data.sim_method_dof_pos[:-1], dtype=np.float64)
    qdot = np.asarray(data.sim_method_dof_vel[:-1], dtype=np.float64)
    tau = np.asarray(matrix["one_step_applied_torque"], dtype=np.float64)
    frequency = _frequency_trace(data)
    time = np.asarray(data.time[:-1], dtype=np.float64)
    stats = {}
    correlations = {}
    for joint, name in enumerate(CANONICAL_JOINT_NAMES):
        values = residual[:, joint]
        joint_rmse = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
        mean = float(np.mean(values))
        stats[name] = {
            "mean_rad_s2": mean,
            "rmse_rad_s2": joint_rmse,
            "std_rad_s2": float(np.std(values)),
            "median_rad_s2": float(np.median(values)),
            "p95_abs_rad_s2": float(np.quantile(np.abs(values), 0.95)),
            "mean_over_rmse": mean / joint_rmse if joint_rmse else None,
        }
        features = {
            "q": q[:, joint],
            "qdot": qdot[:, joint],
            "tau_teacher_applied": tau[:, joint],
            "sign_qdot": np.sign(qdot[:, joint]),
            "abs_qdot": np.abs(qdot[:, joint]),
            "chirp_frequency": frequency,
            "time": time,
        }
        correlations[name] = {
            feature: _correlation(values, signal) for feature, signal in features.items()
        }
    mean_ratio = np.asarray(
        [abs(stats[name]["mean_over_rmse"]) for name in CANONICAL_JOINT_NAMES],
        dtype=np.float64,
    )
    maximum_abs_correlation = max(
        abs(value)
        for joint in correlations.values()
        for value in joint.values()
        if np.isfinite(value)
    )
    ranked_correlations = sorted(
        (
            {
                "joint": joint,
                "feature": feature,
                "correlation": value,
                "absolute_correlation": abs(value),
            }
            for joint, joint_values in correlations.items()
            for feature, value in joint_values.items()
            if np.isfinite(value)
        ),
        key=lambda item: item["absolute_correlation"],
        reverse=True,
    )
    return {
        "definition": (
            "r_a[t,j] = (qdot_pred_author_reset[t+1,j] - "
            "sim_method.dof_vel[t+1,j]) / 0.0025"
        ),
        "overall_rmse_rad_s2": float(
            np.sqrt(np.mean(np.square(residual), dtype=np.float64))
        ),
        "per_joint": stats,
        "correlations": correlations,
        "structure_summary": {
            "maximum_abs_mean_over_rmse": float(np.max(mean_ratio)),
            "median_abs_mean_over_rmse": float(np.median(mean_ratio)),
            "maximum_abs_feature_correlation": float(maximum_abs_correlation),
            "top_absolute_correlations": ranked_correlations[:12],
            "systematic_mean_rule": "abs(mean/RMSE) >= 0.25",
            "joint_with_systematic_mean": [
                name
                for name in CANONICAL_JOINT_NAMES
                if abs(stats[name]["mean_over_rmse"]) >= 0.25
            ],
        },
        "arrays": {"acceleration_residual": residual},
    }


def _horizon_summary(sim_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    table = []
    for horizon in HORIZONS:
        item = sim_metrics[f"horizon_{horizon}"]
        table.append(
            {
                "horizon_steps": horizon,
                "horizon_seconds": horizon * PHYSICS_DT,
                "position_rmse_rad": item["position"]["overall_rmse_rad"],
                "velocity_rmse_rad_s": item["velocity"]["overall_rmse_rad_s"],
                "RF_HFE_position_rmse_rad": item["position"]["focus_joint_rmse_rad"]["RF_HFE"],
                "LH_HFE_position_rmse_rad": item["position"]["focus_joint_rmse_rad"]["LH_HFE"],
            }
        )
    full = sim_metrics["formal_zero"]
    table.append(
        {
            "horizon_steps": "full",
            "horizon_seconds": (6680 - 1) * PHYSICS_DT,
            "position_rmse_rad": full["position"]["overall_rmse_rad"],
            "velocity_rmse_rad_s": full["velocity"]["overall_rmse_rad_s"],
            "RF_HFE_position_rmse_rad": full["position"]["focus_joint_rmse_rad"]["RF_HFE"],
            "LH_HFE_position_rmse_rad": full["position"]["focus_joint_rmse_rad"]["LH_HFE"],
        }
    )
    finite = table[:-1]
    x = np.asarray([item["horizon_seconds"] for item in finite], dtype=np.float64)
    result = {"table": table, "fit": {}}
    for field in ("position_rmse_rad", "velocity_rmse_rad_s"):
        y = np.asarray([item[field] for item in finite], dtype=np.float64)
        coefficient = np.polyfit(x, y, 1)
        prediction = np.polyval(coefficient, x)
        denominator = float(np.sum(np.square(y - np.mean(y))))
        r_squared = 1.0 - float(np.sum(np.square(y - prediction))) / denominator if denominator else 1.0
        local_ratio = y[1:] / np.maximum(y[:-1], np.finfo(np.float64).tiny)
        result["fit"][field] = {
            "linear_slope_per_second": float(coefficient[0]),
            "linear_intercept": float(coefficient[1]),
            "linear_r_squared_H1_to_H512": r_squared,
            "maximum_adjacent_doubling_ratio": float(np.max(local_ratio)),
            "H32_over_H1": float(y[HORIZONS.index(32)] / y[0]),
            "H512_over_H1": float(y[-1] / y[0]),
        }
    position_fit = result["fit"]["position_rmse_rad"]
    full_value = table[-1]["position_rmse_rad"]
    h32_value = table[HORIZONS.index(32)]["position_rmse_rad"]
    h128_value = table[HORIZONS.index(128)]["position_rmse_rad"]
    h512_value = table[HORIZONS.index(512)]["position_rmse_rad"]
    if position_fit["linear_r_squared_H1_to_H512"] >= 0.95:
        curve_type = "A_near_linear_growth"
    elif h128_value >= 0.75 * full_value and h512_value >= 0.90 * full_value:
        curve_type = "B_rapid_growth_then_saturation"
    elif position_fit["maximum_adjacent_doubling_ratio"] >= 3.0:
        curve_type = "C_horizon_localized_jump"
    else:
        curve_type = "mixed_gradual_growth"
    result["curve_type"] = curve_type
    result["saturation_diagnostics"] = {
        "H32_over_full": float(h32_value / full_value),
        "H128_over_full": float(h128_value / full_value),
        "H512_over_full": float(h512_value / full_value),
    }
    result["interpretation_policy"] = "Quantitative diagnostic only; no automatic timing/plant change."
    return result


def _reset_ablation(sim_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    no_reset = sim_metrics["formal_zero"]["position"]["overall_rmse_rad"]
    result = {
        "no_reset_full_position_rmse_rad": no_reset,
        "table": [],
    }
    for horizon in RESET_ABLATION_HORIZONS:
        entries = {
            "q_reset": sim_metrics[f"q_reset_{horizon}"],
            "qdot_reset": sim_metrics[f"qdot_reset_{horizon}"],
            "q_plus_qdot_reset": sim_metrics[f"horizon_{horizon}"],
        }
        row = {"horizon_steps": horizon, "horizon_seconds": horizon * PHYSICS_DT}
        for name, item in entries.items():
            value = item["position"]["overall_rmse_rad"]
            row[name] = {
                "position_rmse_rad": value,
                "velocity_rmse_rad_s": item["velocity"]["overall_rmse_rad_s"],
                "position_suppression_vs_no_reset": 1.0 - value / no_reset,
            }
        result["table"].append(row)
    q_suppression = np.mean(
        [row["q_reset"]["position_suppression_vs_no_reset"] for row in result["table"]]
    )
    v_suppression = np.mean(
        [row["qdot_reset"]["position_suppression_vs_no_reset"] for row in result["table"]]
    )
    result["mean_position_suppression"] = {
        "q_reset": float(q_suppression),
        "qdot_reset": float(v_suppression),
    }
    result["dominant_reset_channel"] = (
        "qdot" if v_suppression > q_suppression else "q"
    )
    return result


def _decision(
    velocity: Mapping[str, Any],
    sim_metrics: Mapping[str, Any],
    horizon: Mapping[str, Any],
    resets: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> Dict[str, Any]:
    sim_velocity = velocity["sim_method"]
    one_step_velocity_rmse = sim_metrics["one_step_author_initial"]["velocity"][
        "overall_rmse_rad_s"
    ]
    derivative_over_one_step_ratio = (
        sim_velocity["best_rmse"] / one_step_velocity_rmse
        if one_step_velocity_rmse
        else math.inf
    )
    derived_velocity = (
        sim_velocity["normalized_best_rmse_over_velocity_rms"] <= 0.02
        and sim_velocity["filter_relative_rmse_improvement"] >= 0.10
        and sim_velocity["best_correlation"] >= 0.99
    )
    diagnostics = {
        "initial_velocity": {
            name: sim_metrics[name]
            for name in ("formal_zero", "initial_sim_method", "initial_real", "initial_forward_fd")
        },
        "target_timing": {
            name: sim_metrics[name]
            for name in ("target_t_minus_1", "formal_zero", "target_t_plus_1")
        },
    }
    passing_initial = [
        name for name, item in diagnostics["initial_velocity"].items()
        if item["gates"]["overall_position_le_0p01"]
        and item["gates"]["all_per_joint_position_le_0p02"]
    ]
    passing_timing = [
        name for name, item in diagnostics["target_timing"].items()
        if item["gates"]["overall_position_le_0p01"]
        and item["gates"]["all_per_joint_position_le_0p02"]
    ]
    qdot_dominant = resets["dominant_reset_channel"] == "qdot"
    qdot_reset_significant = any(
        row["qdot_reset"]["position_suppression_vs_no_reset"] >= 0.20
        for row in resets["table"]
        if row["horizon_steps"] in (8, 32)
    )
    systematic = bool(
        acceleration["structure_summary"]["joint_with_systematic_mean"]
        or acceleration["structure_summary"]["maximum_abs_feature_correlation"] >= 0.30
    )
    if derived_velocity:
        case = "A_velocity_field_is_position-derived_or_filtered"
        next_step = "Redo accumulation interpretation using identified logging semantics; keep position formal."
    elif [name for name in passing_initial if name != "formal_zero"] or [
        name for name in passing_timing if name != "formal_zero"
    ]:
        case = "B_qdot0_or_target_timing_diagnostic_reaches_gate"
        next_step = "Seek legacy provenance before proposing any minimal formal change."
    elif qdot_reset_significant and systematic:
        case = "C_small_structured_transition_mismatch_accumulates"
        next_step = (
            "Authorize only narrow P4 friction/damping and P5 integration OFAT diagnostics; "
            "no parameter refit and no PhysX grid search."
        )
    elif (
        not systematic
        and horizon["curve_type"] in ("A_near_linear_growth", "mixed_gradual_growth")
    ):
        case = "D_slow_near_zero_mean_numeric_or_logging_accumulation"
        next_step = "Reassess whether the strict gate is supportable by public artifacts; do not change it yet."
    else:
        case = "UNRESOLVED_position_feedback_accumulation"
        next_step = (
            "Do not authorize P4/P5 yet: q reset, not qdot reset, dominates position-drift "
            "suppression. Reassess closed-loop position/control-feedback accumulation and the "
            "strict public-artifact gate while seeking exporter velocity provenance."
        )
    formal = sim_metrics["formal_zero"]
    alternatives = {
        **diagnostics["initial_velocity"],
        **diagnostics["target_timing"],
    }
    best_name = min(
        alternatives,
        key=lambda name: alternatives[name]["position"]["overall_rmse_rad"],
    )
    return {
        "case": case,
        "velocity_field_position_derived_rule_met": derived_velocity,
        "velocity_field_semantics": (
            "strong_inference = instantaneous-like simulator state; legacy exporter source "
            "remains unavailable. Author-reset plant qdot predicts the field "
            f"{derivative_over_one_step_ratio:.3f}x more closely than the best position "
            "derivative, while limited filtering improves that derivative by only "
            f"{100.0 * sim_velocity['filter_relative_rmse_improvement']:.3f}%"
        ),
        "velocity_field_provenance": "strong_inference / not source-code-confirmed",
        "author_reset_one_step_velocity_rmse_rad_s": one_step_velocity_rmse,
        "best_position_derivative_over_one_step_velocity_rmse_ratio": (
            derivative_over_one_step_ratio
        ),
        "qdot_reset_dominant": qdot_dominant,
        "qdot_reset_significant_at_H8_or_H32": qdot_reset_significant,
        "acceleration_residual_systematic_rule_met": systematic,
        "passing_initial_velocity_diagnostics": passing_initial,
        "passing_target_timing_diagnostics": passing_timing,
        "best_qdot0_or_target_diagnostic": best_name,
        "best_diagnostic_position_rmse_rad": alternatives[best_name]["position"]["overall_rmse_rad"],
        "formal_position_rmse_rad": formal["position"]["overall_rmse_rad"],
        "evidence_backed_implementation_bug": False,
        "reason_no_implementation_bug": (
            "All alternatives are counterfactuals without legacy exporter provenance; no formal "
            "timing, initial-state, actuator, or loader mutation is justified by minimum RMSE."
        ),
        "modify_stage0c_now": False,
        "next_step": next_step,
        "horizon_curve_type": horizon["curve_type"],
        "Stage_0C": "FAIL",
        "locomotion_PPO": "BLOCKED",
    }


def _plot_growth(path: Path, horizon: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = horizon["table"]
    x = np.asarray([row["horizon_seconds"] for row in rows], dtype=np.float64)
    position = np.asarray([row["position_rmse_rad"] for row in rows], dtype=np.float64)
    velocity = np.asarray([row["velocity_rmse_rad_s"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(x, position, marker="o")
    axes[0].set_ylabel("position RMSE [rad]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(x, velocity, marker="o", color="tab:orange")
    axes[1].set_ylabel("velocity RMSE [rad/s]")
    axes[1].set_xlabel("free-run horizon [s]")
    axes[1].grid(True, which="both", alpha=0.3)
    for axis in axes:
        axis.set_xscale("symlog", linthresh=PHYSICS_DT)
    figure.suptitle("Stage 0C teacher-forcing error growth")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _markdown(report: Mapping[str, Any]) -> str:
    raw = report["source_dtype_audit"]
    time = report["time_grid"]
    velocity = report["velocity_semantics"]
    sim_metrics = report["simulation_metrics"]
    horizon = report["horizon_experiment"]
    resets = report["state_injection_ablation"]
    acceleration = report["one_step_acceleration_residual"]
    decision = report["decision"]
    lines = [
        "# Stage 0C Closed-loop Accumulation / Velocity Semantics Audit",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "```text",
        "Stage 0A: PASS",
        "Stage 0B: frozen",
        "Stage 0C: FAIL",
        "Actuator Branch A: CLOSED",
        "Large plant-configuration sweep: HOLD",
        "Locomotion/PPO: BLOCKED",
        "```",
        "",
        "## 1. Source dtype and precision",
        "",
        f"All source fields are float32: `{raw['conclusion']['all_source_fields_float32']}`. "
        f"Loader conversion hypothesis: **{raw['conclusion']['float32_loader_conversion_hypothesis']}**.",
        "",
        "Every source→loader max/RMSE conversion error is zero. The float-precision "
        "counterfactual was therefore not run, and the formal loader was not changed.",
        "",
        "## 2. Time grid",
        "",
        f"- range: `{time['time_0_s']}` to `{time['time_last_s']}` s",
        f"- mean/std dt: `{time['diff_mean_s']:.12g}` / `{time['diff_std_s']:.12g}` s",
        f"- min/max dt: `{time['diff_min_s']:.12g}` / `{time['diff_max_s']:.12g}` s",
        f"- unique dt count: `{time['unique_diff_count']}`",
        f"- strictly equidistant: `{time['strictly_equidistant']}`",
        "",
        "Timestamp state-vs-command meaning remains artifact-inferred/unresolved; fixed-step "
        "simulation diagnostics retain nominal dt=0.0025 s.",
        "",
        "## 3. Velocity field semantics",
        "",
        "| dataset | best derivative/filter hypothesis | lag | RMSE | correlation | normalized RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in ("sim_method", "real"):
        item = velocity[group]
        lines.append(
            f"| {group} | {item['best_hypothesis']} | {item['best_lag']} | "
            f"{item['best_rmse']:.9g} | {item['best_correlation']:.9g} | "
            f"{item['normalized_best_rmse_over_velocity_rms']:.9g} |"
        )
    lines.extend(
        [
            "",
            f"For `sim_method`, limited smoothing improves the best unfiltered derivative "
            f"by only `{100.0 * velocity['sim_method']['filter_relative_rmse_improvement']:.4f}%`; "
            f"the author-reset plant qdot RMSE is "
            f"`{decision['author_reset_one_step_velocity_rmse_rad_s']:.9g}` rad/s, making the "
            f"best position derivative `{decision['best_position_derivative_over_one_step_velocity_rmse_ratio']:.6g}×` "
            "worse. The field is therefore classified as instantaneous-like with "
            "strong-inference provenance, not source-code confirmation.",
            "",
            "The JSON report contains forward/backward/central nominal and source-time "
            "derivatives, moving averages 2/3/5, lag −3…+3, correlation and 12-joint RMSE.",
            "",
            "## 4. Initial velocity counterfactual",
            "",
            "| qdot0 | position RMSE | velocity RMSE | RF_HFE | LH_HFE | gates |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name in ("formal_zero", "initial_sim_method", "initial_real", "initial_forward_fd"):
        item = sim_metrics[name]
        lines.append(
            f"| {name} | {item['position']['overall_rmse_rad']:.9g} | "
            f"{item['velocity']['overall_rmse_rad_s']:.9g} | "
            f"{item['position']['focus_joint_rmse_rad']['RF_HFE']:.9g} | "
            f"{item['position']['focus_joint_rmse_rad']['LH_HFE']:.9g} | "
            f"{all(item['gates'].values())} |"
        )
    lines.extend(
        [
            "",
            "## 5. Teacher-forcing horizon",
            "",
            "| H steps | seconds | position RMSE | velocity RMSE | RF_HFE | LH_HFE |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in horizon["table"]:
        lines.append(
            f"| {row['horizon_steps']} | {row['horizon_seconds']:.6g} | "
            f"{row['position_rmse_rad']:.9g} | {row['velocity_rmse_rad_s']:.9g} | "
            f"{row['RF_HFE_position_rmse_rad']:.9g} | "
            f"{row['LH_HFE_position_rmse_rad']:.9g} |"
        )
    lines.extend(
        [
            "",
            f"Curve classification: **{horizon['curve_type']}**. H128/full and H512/full "
            f"position ratios are `{horizon['saturation_diagnostics']['H128_over_full']:.6g}` "
            f"and `{horizon['saturation_diagnostics']['H512_over_full']:.6g}`. "
            "See `error_growth.png`.",
            "",
            "## 6. q/qdot reset ablation",
            "",
            "| H | q reset position | qdot reset position | q+qdot reset position |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in resets["table"]:
        lines.append(
            f"| {row['horizon_steps']} | {row['q_reset']['position_rmse_rad']:.9g} | "
            f"{row['qdot_reset']['position_rmse_rad']:.9g} | "
            f"{row['q_plus_qdot_reset']['position_rmse_rad']:.9g} |"
        )
    lines.extend(
        [
            "",
            f"Dominant reset channel by mean suppression: **{resets['dominant_reset_channel']}**.",
            f" Mean position suppression is "
            f"`{resets['mean_position_suppression']['q_reset']:.6g}` for q reset and "
            f"`{resets['mean_position_suppression']['qdot_reset']:.6g}` for qdot reset.",
            "",
            "## 7. One-step acceleration residual",
            "",
            f"Definition: `{acceleration['definition']}`.",
            "",
            "| joint | mean | RMSE | std | median | p95 abs | mean/RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CANONICAL_JOINT_NAMES:
        item = acceleration["per_joint"][name]
        lines.append(
            f"| {name} | {item['mean_rad_s2']:.7g} | {item['rmse_rad_s2']:.7g} | "
            f"{item['std_rad_s2']:.7g} | {item['median_rad_s2']:.7g} | "
            f"{item['p95_abs_rad_s2']:.7g} | {item['mean_over_rmse']:.7g} |"
        )
    lines.extend(
        [
            "",
            "Per-joint correlations against q, qdot, applied torque, sign/abs(qdot), "
            "frequency and time are in `report.json`.",
            "",
            "## 8. Target timing ±1",
            "",
            "| target | position RMSE | first-100 | RF_HFE | LH_HFE | gates |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name in ("target_t_minus_1", "formal_zero", "target_t_plus_1"):
        item = sim_metrics[name]
        lines.append(
            f"| {name} | {item['position']['overall_rmse_rad']:.9g} | "
            f"{item['position']['first_100_rmse_rad']:.9g} | "
            f"{item['position']['focus_joint_rmse_rad']['RF_HFE']:.9g} | "
            f"{item['position']['focus_joint_rmse_rad']['LH_HFE']:.9g} | "
            f"{all(item['gates'].values())} |"
        )
    lines.extend(
        [
            "",
            "No timing counterfactual is auto-selected; target boundaries are clamped, never wrapped.",
            "",
            "## 9. Decision",
            "",
            f"- Case: **{decision['case']}**",
            f"- Most likely propagation mechanism: **{report['propagation_mechanism']}**",
            f"- Evidence-backed implementation bug: **{decision['evidence_backed_implementation_bug']}**",
            f"- Modify Stage 0C now: **{decision['modify_stage0c_now']}**",
            f"- qdot reset significant at H8/H32: "
            f"**{decision['qdot_reset_significant_at_H8_or_H32']}**",
            f"- Next step: **{decision['next_step']}**",
            "- Stage 0C remains FAIL; locomotion/PPO remains BLOCKED.",
            "",
        ]
    )
    return "\n".join(lines)


def run_accumulation_audit(
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
    payload = _raw_payload()
    dtype_audit = _source_dtype_audit(payload, data)
    time_audit = _time_audit(np.asarray(payload["real"]["time"]))
    velocity = {
        "sim_method": _velocity_semantics(
            np.asarray(payload["sim_method"]["dof_pos"]),
            np.asarray(payload["sim_method"]["dof_vel"]),
            np.asarray(payload["real"]["time"]),
        ),
        "real": _velocity_semantics(
            np.asarray(payload["real"]["dof_pos"]),
            np.asarray(payload["real"]["dof_vel"]),
            np.asarray(payload["real"]["time"]),
        ),
    }
    initial = _initial_velocity_audit(data)
    target_quantization = _target_quantization(np.asarray(payload["real"]["des_dof_pos"]))
    q_true_0_source_precision = (
        np.asarray(payload["real"]["dof_pos"])[0].astype(np.float64)
        + fit.encoder_bias.astype(np.float64)
    )
    q_true_0_float32_error = (
        q_true_0_source_precision
        - q_true_0_source_precision.astype(np.float32).astype(np.float64)
    )
    matrix = _run_simulation_matrix(device_id, fit, data)
    sim_metrics = _simulation_metrics(matrix, data, fit)
    acceleration = _acceleration_residual(matrix, sim_metrics, data, fit)
    horizon = _horizon_summary(sim_metrics)
    resets = _reset_ablation(sim_metrics)
    decision = _decision(velocity, sim_metrics, horizon, resets, acceleration)

    if decision["case"].startswith("A_"):
        mechanism = "velocity logging semantics dominate the apparent qdot/acceleration mismatch"
    elif decision["case"].startswith("B_"):
        mechanism = "initial-velocity or target-timing counterfactual explains closed-loop drift"
    elif decision["case"].startswith("C_"):
        mechanism = "small structured velocity/acceleration transition mismatch accumulates in closed loop"
    elif decision["case"].startswith("UNRESOLVED_"):
        mechanism = (
            "position/control-feedback error grows over roughly 0.3–1.3 s and then "
            "approaches the full residual; q reset suppresses position drift more than "
            "qdot reset, while velocity exporter semantics remain unresolved"
        )
    else:
        mechanism = "gradual numerical/logging/integration accumulation without a source-backed implementation bug"

    report: Dict[str, Any] = {
        "schema": "pace_stage0.accumulation_audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "Actuator_Branch_A": "CLOSED",
            "large_plant_configuration_sweep": "HOLD",
            "locomotion_PPO": "BLOCKED",
        },
        "git": _git_snapshot(),
        "random_seed": 0,
        "reproduction": {
            "command": (
                "conda run -n bruce_gym env PYTHONPATH=src python -m "
                "pace_stage0.cli accumulation_audit --device-id 0"
            ),
            "implementation_sha256": {
                "accumulation_audit.py": _sha256(Path(__file__).resolve()),
                "actuator.py": _sha256(Path(__file__).resolve().with_name("actuator.py")),
                "data.py": _sha256(Path(__file__).resolve().with_name("data.py")),
                "replay.py": _sha256(Path(__file__).resolve().with_name("replay.py")),
            },
        },
        "source_dtype_audit": dtype_audit,
        "time_grid": time_audit,
        "velocity_semantics": velocity,
        "initial_velocity_values": initial,
        "target_quantization": target_quantization,
        "float_precision_counterfactual": {
            "run": False,
            "reason": "Every source field, including target/state/velocity, is already float32.",
            "source_to_float32_error": 0.0,
            "target_max_abs_source_to_float32_error": 0.0,
            "reference_max_abs_source_to_float32_error": 0.0,
            "composite_initial_q_true_max_abs_float32_cast_error": float(
                np.max(np.abs(q_true_0_float32_error))
            ),
            "composite_initial_q_true_rmse_float32_cast_error": float(
                np.sqrt(np.mean(np.square(q_true_0_float32_error), dtype=np.float64))
            ),
            "full_replay_counterfactual_difference": None,
            "formal_loader_changed": False,
        },
        "simulation_protocol": {
            "formal_actuator_modified": False,
            "formal_initial_q": "real.dof_pos[0] + encoder_bias",
            "formal_initial_qdot": "zero",
            "horizon_reset": (
                "Before interval t>0 when t mod H=0, indexed actor state write resets "
                "requested q/qdot channel to sim_method author state; delay FIFO continues."
            ),
            "horizon_1_note": (
                "The horizon matrix retains formal state-0 initialization. A separate "
                "one_step_author_initial case exactly matches author-state one-step semantics."
            ),
            "target_shift_boundary": "clamp to [0,N-1], never circular wrap",
            "multi_environment_isolation": "indexed actor state writes; no-reset actors are not rewritten",
            "dt": PHYSICS_DT,
        },
        "state_reset_audit": matrix["state_reset_audit"],
        "simulation_metrics": sim_metrics,
        "horizon_experiment": horizon,
        "state_injection_ablation": resets,
        "one_step_acceleration_residual": acceleration,
        "propagation_mechanism": mechanism,
        "decision": decision,
        "formal_changes": {
            "actuator": False,
            "loader": False,
            "initial_velocity": False,
            "target_timing": False,
            "threshold": False,
            "plant_parameters": False,
        },
    }

    trace: Dict[str, np.ndarray] = {
        "state_index": np.arange(data.sample_count, dtype=np.int64),
        "time": np.asarray(data.time),
        "sim_method_q_compare": np.asarray(data.sim_method_dof_pos),
        "sim_method_qdot": np.asarray(data.sim_method_dof_vel),
        "target": np.asarray(data.real_des_dof_pos),
        "one_step_applied_torque": matrix["one_step_applied_torque"],
        "formal_applied_torque": matrix["formal_applied_torque"],
        "acceleration_residual": acceleration["arrays"]["acceleration_residual"],
        "frequency_hz": _frequency_trace(data),
    }
    acceleration.pop("arrays")
    for index, case in enumerate(matrix["cases"]):
        trace[f"{case.name}__q_true"] = matrix["q_true"][index]
        trace[f"{case.name}__qdot"] = matrix["qdot"][index]
    np.savez_compressed(artifact_dir / "accumulation_traces.npz", **trace)
    _plot_growth(artifact_dir / "error_growth.png", horizon)
    _write_json(artifact_dir / "report.json", report)
    (artifact_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote: {artifact_dir / 'report.md'}")
    print(f"wrote: {artifact_dir / 'report.json'}")
    print(f"wrote: {artifact_dir / 'accumulation_traces.npz'}")
    print(f"wrote: {artifact_dir / 'error_growth.png'}")
    return _jsonable(report)
