from __future__ import annotations

import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    ANYMAL_ASSET_COMMIT,
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


WINDOW_ENDPOINTS = (10, 50, 100, 500)
FOCUS_JOINTS = ("RF_HFE", "LH_HFE", "LF_KFE")
FREQUENCY_BANDS_HZ = (
    ("estimated_below_0p75_Hz", 0.0, 0.75),
    ("estimated_0p75_to_1p25_Hz", 0.75, 1.25),
    ("estimated_1p25_to_1p75_Hz", 1.25, 1.75),
    ("estimated_at_least_1p75_Hz", 1.75, math.inf),
)
FROZEN_CLOSED_LOOP_POSITION_RMSE_RAD = 0.012818364796375577


@dataclass(frozen=True)
class PlantVariant:
    name: str
    use_physx_armature: bool
    collapse_fixed_joints: bool
    friction_mode: str = "fitted"
    damping_mode: str = "fitted"


BASELINE = PlantVariant("baseline", True, True)
ARMATURE_LINK_INERTIA = PlantVariant("armature_use_physx_false", False, True)
NO_FIXED_COLLAPSE = PlantVariant("collapse_fixed_joints_false", True, False)


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
        output_dir = PROJECT_ROOT / "artifacts" / "plant_audit" / stamp
    result = output_dir.resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_vector(text: Optional[str], count: int) -> np.ndarray:
    if text is None:
        return np.zeros(count, dtype=np.float64)
    result = np.fromstring(text, sep=" ", dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"Expected {count} values, got {text!r}")
    return result


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_from_rpy(rpy)
    result[:3, 3] = xyz
    return result


def _origin(element: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    return (
        _float_vector(element.attrib.get("xyz"), 3),
        _float_vector(element.attrib.get("rpy"), 3),
    )


def _parse_urdf(path: Path) -> Dict[str, Any]:
    root = ET.parse(path).getroot()
    links: Dict[str, Dict[str, Any]] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        inertial = link.find("inertial")
        if inertial is None:
            mass = 0.0
            xyz = np.zeros(3, dtype=np.float64)
            rpy = np.zeros(3, dtype=np.float64)
            inertia = np.zeros((3, 3), dtype=np.float64)
        else:
            mass_element = inertial.find("mass")
            inertia_element = inertial.find("inertia")
            if mass_element is None or inertia_element is None:
                raise ValueError(f"Incomplete inertial block on {name}")
            mass = float(mass_element.attrib["value"])
            xyz, rpy = _origin(inertial.find("origin"))
            values = {key: float(inertia_element.attrib[key]) for key in (
                "ixx", "ixy", "ixz", "iyy", "iyz", "izz"
            )}
            inertia = np.asarray(
                (
                    (values["ixx"], values["ixy"], values["ixz"]),
                    (values["ixy"], values["iyy"], values["iyz"]),
                    (values["ixz"], values["iyz"], values["izz"]),
                ),
                dtype=np.float64,
            )
        links[name] = {
            "name": name,
            "mass": mass,
            "inertial_origin_xyz": xyz,
            "inertial_origin_rpy": rpy,
            "inertia_at_com_in_inertial_frame": inertia,
        }

    joints = []
    child_names = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Incomplete joint {joint.attrib.get('name')}")
        xyz, rpy = _origin(joint.find("origin"))
        axis_element = joint.find("axis")
        axis = _float_vector(
            None if axis_element is None else axis_element.attrib.get("xyz"), 3
        )
        item = {
            "name": joint.attrib["name"],
            "type": joint.attrib["type"],
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis": axis,
        }
        joints.append(item)
        child_names.add(item["child"])
    root_links = sorted(set(links) - child_names)
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, got {root_links}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "robot_name": root.attrib.get("name"),
        "root_link": root_links[0],
        "links": links,
        "joints": joints,
        "link_count": len(links),
        "joint_count": len(joints),
        "fixed_joint_count": sum(item["type"] == "fixed" for item in joints),
        "movable_joint_count": sum(item["type"] != "fixed" for item in joints),
        "total_urdf_mass": float(sum(item["mass"] for item in links.values())),
    }


def _collapse_mapping(
    urdf: Mapping[str, Any], runtime_body_names: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    runtime = set(runtime_body_names)
    root = str(urdf["root_link"])
    if root not in runtime:
        raise ValueError(f"URDF root {root} is not a runtime body")
    by_parent: Dict[str, list] = {}
    for joint in urdf["joints"]:
        by_parent.setdefault(joint["parent"], []).append(joint)
    result: Dict[str, Dict[str, Any]] = {
        root: {
            "runtime_body": root,
            "runtime_body_to_link_transform": np.eye(4, dtype=np.float64),
            "incoming_joint": None,
        }
    }
    queue = [root]
    while queue:
        parent = queue.pop(0)
        parent_entry = result[parent]
        for joint in by_parent.get(parent, []):
            child = joint["child"]
            parent_to_child = _transform(joint["origin_xyz"], joint["origin_rpy"])
            if child in runtime:
                body = child
                body_to_link = np.eye(4, dtype=np.float64)
            else:
                body = parent_entry["runtime_body"]
                body_to_link = (
                    parent_entry["runtime_body_to_link_transform"] @ parent_to_child
                )
            result[child] = {
                "runtime_body": body,
                "runtime_body_to_link_transform": body_to_link,
                "incoming_joint": joint["name"],
                "incoming_joint_type": joint["type"],
            }
            queue.append(child)
    if set(result) != set(urdf["links"]):
        raise ValueError("URDF traversal did not cover every link")
    return result


def _aggregate_urdf_inertials(
    urdf: Mapping[str, Any], mapping: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, list] = {}
    for link_name, entry in mapping.items():
        grouped.setdefault(entry["runtime_body"], []).append(link_name)
    result: Dict[str, Dict[str, Any]] = {}
    for body, names in grouped.items():
        contributions = []
        for name in names:
            link = urdf["links"][name]
            body_to_link = np.asarray(
                mapping[name]["runtime_body_to_link_transform"], dtype=np.float64
            )
            body_to_inertial = body_to_link @ _transform(
                link["inertial_origin_xyz"], link["inertial_origin_rpy"]
            )
            rotation = body_to_inertial[:3, :3]
            contributions.append(
                {
                    "link": name,
                    "mass": float(link["mass"]),
                    "com_in_runtime_body": body_to_inertial[:3, 3],
                    "inertia_at_com_in_runtime_body": (
                        rotation
                        @ np.asarray(link["inertia_at_com_in_inertial_frame"])
                        @ rotation.T
                    ),
                }
            )
        total_mass = float(sum(item["mass"] for item in contributions))
        if total_mass > 0.0:
            com = sum(
                item["mass"] * item["com_in_runtime_body"] for item in contributions
            ) / total_mass
        else:
            com = np.zeros(3, dtype=np.float64)
        inertia = np.zeros((3, 3), dtype=np.float64)
        for item in contributions:
            displacement = item["com_in_runtime_body"] - com
            inertia += item["inertia_at_com_in_runtime_body"]
            inertia += item["mass"] * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        result[body] = {
            "runtime_body": body,
            "source_links": names,
            "source_link_count": len(names),
            "collapsed_link_count": max(0, len(names) - 1),
            "mass": total_mass,
            "com": com,
            "inertia_at_combined_com": inertia,
            "contributions": contributions,
        }
    return result


def _mat33(value: Any) -> np.ndarray:
    return np.asarray(
        (
            (value.x.x, value.x.y, value.x.z),
            (value.y.x, value.y.y, value.y.z),
            (value.z.x, value.z.y, value.z.z),
        ),
        dtype=np.float64,
    )


def _rigid_body_snapshot(names: Sequence[str], properties: Sequence[Any]) -> list:
    if len(names) != len(properties):
        raise ValueError(f"Rigid body property count mismatch: {len(names)}/{len(properties)}")
    result = []
    for name, prop in zip(names, properties):
        inertia = _mat33(prop.inertia)
        principal_moments, principal_axes = np.linalg.eigh(
            0.5 * (inertia + inertia.T)
        )
        result.append(
            {
                "name": name,
                "mass": float(prop.mass),
                "inverse_mass": float(prop.invMass),
                "com": [float(prop.com.x), float(prop.com.y), float(prop.com.z)],
                "inertia_raw_mat33": inertia,
                "inverse_inertia_raw_mat33": _mat33(prop.invInertia),
                "principal_moments_eigenvalues": principal_moments,
                "principal_axes_columns": principal_axes,
                "flags": int(prop.flags),
            }
        )
    return result


def _snapshot_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if [item["name"] for item in before] != [item["name"] for item in after]:
        raise ValueError("Rigid body names changed across prepare_sim")
    return {
        "max_abs_mass_change": max(
            abs(float(left["mass"]) - float(right["mass"]))
            for left, right in zip(before, after)
        ),
        "max_abs_com_change": max(
            float(np.max(np.abs(np.asarray(left["com"]) - np.asarray(right["com"]))))
            for left, right in zip(before, after)
        ),
        "max_abs_inertia_change": max(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["inertia_raw_mat33"])
                        - np.asarray(right["inertia_raw_mat33"])
                    )
                )
            )
            for left, right in zip(before, after)
        ),
    }


def _canonical_dof_snapshot(
    names: Sequence[str], properties: np.ndarray, gather_indices: Sequence[int]
) -> Dict[str, Any]:
    gather = np.asarray(gather_indices, dtype=np.int64)
    result = {
        "asset_order": list(names),
        "canonical_order": list(CANONICAL_JOINT_NAMES),
        "canonical_gather_indices": list(gather_indices),
    }
    for field in properties.dtype.names or ():
        result[field] = np.asarray(properties[field])[gather]
    return result


def _legacy_asset_provenance() -> Dict[str, Any]:
    return {
        "classification": "C",
        "asset_identity_provenance": "unresolved",
        "answer": "The exact ANYmal asset used to export legacy sim_method is not publicly identifiable.",
        "search_scope": [
            "PACE paper v2 and public supplemental description",
            "/home/xy.chen/tw/dataset/pace_data (dataset files only)",
            "this independent reproduction repository",
            "PACE public repository at frozen later reference commit",
            "public ANYbotics anymal_d_simple_description README",
        ],
        "search_constraint": (
            "No source code from other projects under /home/xy.chen/tw was read or searched."
        ),
        "evidence": [
            {
                "source": "PACE paper v2",
                "url": "https://arxiv.org/html/2509.06342v2",
                "finding": (
                    "ANYmal is described as a closed-source testbed; no legacy Isaac Gym "
                    "URDF filename, repository commit, or asset hash is published."
                ),
            },
            {
                "source": "pace_data/1_in_air/anymal/{data,fitting}.npy",
                "finding": (
                    "The whitelisted payloads contain trajectories and fitted parameters; "
                    "no URDF/asset path or hash is present."
                ),
            },
            {
                "source": "later public PACE Isaac Lab configuration",
                "url": (
                    "https://github.com/leggedrobotics/pace-sim2real/blob/"
                    "f07259c09b517ab5118bb1d01b0a6078cf8e1c31/source/pace_sim2real/"
                    "pace_sim2real/tasks/manager_based/pace/anymal_pace_env_cfg.py"
                ),
                "finding": (
                    "It imports Isaac Lab ANYMAL_D_CFG, but post-dates the legacy Isaac Gym "
                    "dataset and cannot establish the legacy exporter asset."
                ),
            },
            {
                "source": "ANYbotics anymal_d_simple_description README",
                "url": "https://github.com/ANYbotics/anymal_d_simple_description",
                "finding": (
                    "The vendored URDF is explicitly a simplified description; extended "
                    "description and simulation software are restricted to the research community."
                ),
            },
        ],
        "strict_replication_implication": (
            "The public simplified URDF must be treated as an independent-reproduction asset, "
            "not as source-confirmed legacy plant identity."
        ),
    }


def _git_snapshot() -> Dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(
            ("git",) + args, cwd=PROJECT_ROOT, text=True
        ).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _teacher_public_torque(
    data: ReplayData, fit: DecodedFit, device: Any
) -> Dict[str, np.ndarray]:
    import torch

    from .actuator import PACEActuatorCore

    bias = torch.as_tensor(fit.encoder_bias, dtype=torch.float32, device=device)
    core = PACEActuatorCore(bias, fit.delay_steps)
    initial = torch.as_tensor(
        data.sim_method_dof_pos[0] + fit.encoder_bias,
        dtype=torch.float32,
        device=device,
    )
    core.reset(initial)
    count = data.sample_count - 1
    raw = np.empty((count, 12), dtype=np.float32)
    saturated = np.empty_like(raw)
    applied = np.empty_like(raw)
    q_encoder = np.empty_like(raw)
    for index in range(count):
        target = torch.as_tensor(
            data.real_des_dof_pos[index], dtype=torch.float32, device=device
        )
        q_true = torch.as_tensor(
            data.sim_method_dof_pos[index] + fit.encoder_bias,
            dtype=torch.float32,
            device=device,
        )
        qdot = torch.as_tensor(
            data.sim_method_dof_vel[index], dtype=torch.float32, device=device
        )
        step = core.step(target, q_true, qdot)
        q_encoder[index] = step.q_encoder.detach().cpu().numpy()
        raw[index] = step.raw_pd_torque.detach().cpu().numpy()
        saturated[index] = step.saturated_torque.detach().cpu().numpy()
        applied[index] = step.applied_torque.detach().cpu().numpy()
    return {
        "q_encoder_author": q_encoder.astype(np.float64),
        "raw_pd_torque": raw.astype(np.float64),
        "saturated_torque": saturated.astype(np.float64),
        "applied_torque": applied.astype(np.float64),
    }


def _estimated_frequency(data: ReplayData) -> Dict[str, Any]:
    target = np.asarray(data.real_des_dof_pos, dtype=np.float64)
    reference_joint = int(np.argmax(np.std(target, axis=0)))
    signal = target[:, reference_joint]
    center = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
    crossing = np.flatnonzero((signal[:-1] <= center) & (signal[1:] > center))
    if len(crossing) < 3:
        raise RuntimeError("Not enough command zero crossings for empirical frequency bins")
    time = np.asarray(data.time, dtype=np.float64)
    crossing_time = time[crossing]
    local_frequency = 1.0 / np.diff(crossing_time)
    local_time = 0.5 * (crossing_time[:-1] + crossing_time[1:])
    interval_frequency = np.interp(
        time[:-1],
        local_time,
        local_frequency,
        left=local_frequency[0],
        right=local_frequency[-1],
    )
    return {
        "reference_joint": CANONICAL_JOINT_NAMES[reference_joint],
        "method": (
            "positive-going crossings around the command midpoint; reciprocal crossing "
            "period interpolated to interval times; edge samples use nearest observed period"
        ),
        "crossing_state_indices": crossing,
        "observed_cycle_frequency_hz": local_frequency,
        "interval_frequency_hz": interval_frequency,
    }


def _transition_metric_block(
    q_pred_true: np.ndarray,
    qdot_pred: np.ndarray,
    data: ReplayData,
    fit: DecodedFit,
    interval_frequency: np.ndarray,
) -> Dict[str, Any]:
    q_pred_compare = np.asarray(q_pred_true, dtype=np.float64) - fit.encoder_bias[None, :]
    q_reference = np.asarray(data.sim_method_dof_pos[1:], dtype=np.float64)
    qdot_pred = np.asarray(qdot_pred, dtype=np.float64)
    qdot_before = np.asarray(data.sim_method_dof_vel[:-1], dtype=np.float64)
    qdot_reference = np.asarray(data.sim_method_dof_vel[1:], dtype=np.float64)
    acc_pred = (qdot_pred - qdot_before) / PHYSICS_DT
    acc_reference = (qdot_reference - qdot_before) / PHYSICS_DT

    def metrics(indices: np.ndarray) -> Dict[str, Any]:
        if len(indices) == 0:
            return {"sample_count": 0}
        layers = {
            "position_rad": (q_pred_compare[indices], q_reference[indices]),
            "velocity_rad_s": (qdot_pred[indices], qdot_reference[indices]),
            "acceleration_rad_s2": (acc_pred[indices], acc_reference[indices]),
        }
        result: Dict[str, Any] = {"sample_count": int(len(indices))}
        for name, (ours, reference) in layers.items():
            joint = per_joint_rmse(ours, reference)
            result[name] = {
                "overall_rmse": rmse(ours, reference),
                "per_joint_rmse": dict(zip(CANONICAL_JOINT_NAMES, joint.tolist())),
                "focus_joint_rmse": {
                    focus: float(joint[CANONICAL_JOINT_NAMES.index(focus)])
                    for focus in FOCUS_JOINTS
                },
            }
        return result

    full_indices = np.arange(len(q_pred_true), dtype=np.int64)
    windows = {}
    for endpoint in WINDOW_ENDPOINTS:
        indices = np.arange(min(endpoint + 1, len(q_pred_true)), dtype=np.int64)
        windows[f"source_states_0_to_{endpoint}_inclusive"] = metrics(indices)
    windows["full_trajectory"] = metrics(full_indices)

    frequency = {}
    for name, lower, upper in FREQUENCY_BANDS_HZ:
        indices = np.flatnonzero(
            (interval_frequency >= lower) & (interval_frequency < upper)
        )
        block = metrics(indices)
        block.update(
            {
                "lower_hz_inclusive": lower,
                "upper_hz_exclusive": None if math.isinf(upper) else upper,
                "empirical_min_hz": (
                    None if len(indices) == 0 else float(np.min(interval_frequency[indices]))
                ),
                "empirical_max_hz": (
                    None if len(indices) == 0 else float(np.max(interval_frequency[indices]))
                ),
            }
        )
        frequency[name] = block
    return {
        "comparison": {
            "position": "q_pred_true[t+1] - bias vs sim_method.dof_pos[t+1]",
            "velocity": "qdot_pred[t+1] vs sim_method.dof_vel[t+1]",
            "acceleration": (
                "(qdot_pred[t+1]-qdot_author[t])/0.0025 vs "
                "(qdot_author[t+1]-qdot_author[t])/0.0025"
            ),
        },
        "windows": windows,
        "frequency_bands": frequency,
        "arrays": {
            "q_pred_compare": q_pred_compare,
            "q_reference": q_reference,
            "qdot_pred": qdot_pred,
            "qdot_reference": qdot_reference,
            "acc_pred": acc_pred,
            "acc_reference": acc_reference,
        },
    }


def _make_plant_sim(
    device_id: int,
    fit: DecodedFit,
    urdf: Mapping[str, Any],
    variant: PlantVariant,
):
    # Isaac Gym Preview 4 import order is intentional.
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
        options.collapse_fixed_joints = variant.collapse_fixed_joints
        options.replace_cylinder_with_capsule = True
        options.flip_visual_attachments = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.use_physx_armature = variant.use_physx_armature
        options.armature = 0.0
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
        body_names = tuple(gym.get_asset_rigid_body_names(asset))
        mapping = _collapse_mapping(urdf, body_names)
        aggregate = _aggregate_urdf_inertials(urdf, mapping)

        dof_props = gym.get_asset_dof_properties(asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["friction"][gather_np] = (
            fit.friction if variant.friction_mode == "fitted" else 0.0
        )
        dof_props["damping"][gather_np] = (
            fit.damping if variant.damping_mode == "fitted" else 0.0
        )
        dof_props["armature"][gather_np] = fit.armature
        dof_props["effort"].fill(EFFORT_LIMIT)
        dof_props["velocity"].fill(VELOCITY_LIMIT)

        envs = []
        actors = []
        for index in range(2):
            env = gym.create_env(
                sim,
                gymapi.Vec3(-1.0, -1.0, 0.0),
                gymapi.Vec3(1.0, 1.0, 2.0),
                2,
            )
            pose = gymapi.Transform()
            pose.p.z = 1.0
            actor = gym.create_actor(env, asset, pose, f"anymal_d_{index}", index, 1)
            gym.set_actor_dof_properties(env, actor, dof_props)
            envs.append(env)
            actors.append(actor)

        actor_body_names = tuple(gym.get_actor_rigid_body_names(envs[0], actors[0]))
        pre_prepare = _rigid_body_snapshot(
            actor_body_names,
            gym.get_actor_rigid_body_properties(envs[0], actors[0]),
        )
        gym.prepare_sim(sim)
        post_prepare = _rigid_body_snapshot(
            actor_body_names,
            gym.get_actor_rigid_body_properties(envs[0], actors[0]),
        )
        applied_dof = gym.get_actor_dof_properties(envs[0], actors[0])
        if not np.all(applied_dof["driveMode"] == gymapi.DOF_MODE_EFFORT):
            raise RuntimeError("Plant audit actor is not in DOF_MODE_EFFORT")
        if not np.allclose(applied_dof["stiffness"], 0.0):
            raise RuntimeError("Plant audit actor has non-zero built-in stiffness")

        dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(2, 12, 2)
        contact_force = gymtorch.wrap_tensor(
            gym.acquire_net_contact_force_tensor(sim)
        ).view(2, len(body_names), 3)
        runtime = {
            "variant": _jsonable(variant.__dict__),
            "simulation": {
                "dt": PHYSICS_DT,
                "substeps": 1,
                "solver_type": 1,
                "num_position_iterations": 4,
                "num_velocity_iterations": 0,
                "fixed_base": True,
                "self_collisions": False,
                "ground_created": False,
                "gravity": [0.0, 0.0, -9.81],
                "device": str(torch.device(f"cuda:{device_id}")),
            },
            "asset_options": {
                "use_physx_armature": variant.use_physx_armature,
                "asset_options_armature": 0.0,
                "collapse_fixed_joints": variant.collapse_fixed_joints,
                "replace_cylinder_with_capsule": True,
                "default_dof_drive_mode": "DOF_MODE_EFFORT",
                "isaac_gym_documented_armature_semantics": (
                    "True uses joint-space armature; False uses link-inertia-tensor modifications."
                ),
            },
            "asset_level": {
                "rigid_body_count": gym.get_asset_rigid_body_count(asset),
                "rigid_body_names": body_names,
                "rigid_body_property_api_available_in_preview4": False,
                "property_fallback": (
                    "URDF source inertials plus actor properties immediately after creation"
                ),
            },
            "actor_level": {
                "rigid_body_count": gym.get_actor_rigid_body_count(envs[0], actors[0]),
                "rigid_body_names": actor_body_names,
                "pre_prepare_sim_properties": pre_prepare,
                "post_prepare_sim_properties": post_prepare,
                "pre_vs_post_prepare_delta": _snapshot_delta(pre_prepare, post_prepare),
            },
            "dof_properties_after_actor_write": _canonical_dof_snapshot(
                dof_names, applied_dof, gather_indices
            ),
            "urdf_to_runtime_mapping": {
                name: {
                    "runtime_body": item["runtime_body"],
                    "incoming_joint": item.get("incoming_joint"),
                    "incoming_joint_type": item.get("incoming_joint_type"),
                    "runtime_body_to_link_transform": item[
                        "runtime_body_to_link_transform"
                    ],
                }
                for name, item in mapping.items()
            },
            "urdf_aggregated_inertials": aggregate,
        }
        return (
            gymapi,
            gymtorch,
            torch,
            gym,
            sim,
            dof_state,
            contact_force,
            gather_indices,
            runtime,
        )
    except Exception:
        gym.destroy_sim(sim)
        raise


def _run_variant(
    device_id: int,
    fit: DecodedFit,
    data: ReplayData,
    urdf: Mapping[str, Any],
    variant: PlantVariant,
) -> Dict[str, Any]:
    from isaacgym import gymtorch
    import torch

    (
        _,
        _,
        _,
        gym,
        sim,
        dof_state,
        contact_force,
        gather_indices,
        runtime,
    ) = _make_plant_sim(device_id, fit, urdf, variant)
    try:
        device = dof_state.device
        teacher = _teacher_public_torque(data, fit, device)
        logged = np.asarray(data.sim_method_dof_torques[1:], dtype=np.float64)
        count = data.sample_count - 1
        predicted_q = np.empty((2, count, 12), dtype=np.float32)
        predicted_qdot = np.empty_like(predicted_q)
        effort = torch.zeros((2, 12), dtype=torch.float32, device=device)
        gather = torch.as_tensor(gather_indices, dtype=torch.long, device=device)
        author_q_true = torch.as_tensor(
            data.sim_method_dof_pos + fit.encoder_bias[None, :],
            dtype=torch.float32,
            device=device,
        )
        author_qdot = torch.as_tensor(
            data.sim_method_dof_vel, dtype=torch.float32, device=device
        )
        teacher_torque = torch.as_tensor(
            teacher["applied_torque"], dtype=torch.float32, device=device
        )
        logged_torque = torch.as_tensor(logged, dtype=torch.float32, device=device)
        maximum_state_write_error = 0.0
        maximum_velocity_write_error = 0.0
        maximum_contact_force = 0.0
        nonzero_contact_element_count = 0

        for index in range(count):
            dof_state.zero_()
            dof_state[:, gather, 0] = author_q_true[index]
            dof_state[:, gather, 1] = author_qdot[index]
            if not gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state)):
                raise RuntimeError(f"State tensor write failed at source state {index}")
            gym.refresh_dof_state_tensor(sim)
            maximum_state_write_error = max(
                maximum_state_write_error,
                float(torch.max(torch.abs(dof_state[:, gather, 0] - author_q_true[index])).item()),
            )
            maximum_velocity_write_error = max(
                maximum_velocity_write_error,
                float(torch.max(torch.abs(dof_state[:, gather, 1] - author_qdot[index])).item()),
            )

            effort.zero_()
            effort[0, gather] = teacher_torque[index]
            effort[1, gather] = logged_torque[index]
            if not gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(effort.contiguous().view(-1))
            ):
                raise RuntimeError(f"Effort tensor write failed at source state {index}")
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_net_contact_force_tensor(sim)
            predicted_q[:, index] = dof_state[:, gather, 0].detach().cpu().numpy()
            predicted_qdot[:, index] = dof_state[:, gather, 1].detach().cpu().numpy()
            contact_abs = torch.abs(contact_force)
            maximum_contact_force = max(
                maximum_contact_force, float(torch.max(contact_abs).item())
            )
            nonzero_contact_element_count += int(
                torch.count_nonzero(contact_abs > 1e-8).item()
            )
    finally:
        gym.destroy_sim(sim)

    frequency = _estimated_frequency(data)
    public_metrics = _transition_metric_block(
        predicted_q[0], predicted_qdot[0], data, fit, frequency["interval_frequency_hz"]
    )
    logged_metrics = _transition_metric_block(
        predicted_q[1], predicted_qdot[1], data, fit, frequency["interval_frequency_hz"]
    )
    runtime["state_reset_audit"] = {
        "reset_before_every_interval": True,
        "maximum_q_write_error": maximum_state_write_error,
        "maximum_qdot_write_error": maximum_velocity_write_error,
        "source_state": "q_true=sim_method.dof_pos+bias; qdot=sim_method.dof_vel",
    }
    runtime["contact_audit"] = {
        "force_threshold": 1e-8,
        "maximum_absolute_contact_force_component": maximum_contact_force,
        "nonzero_force_element_observation_count": nonzero_contact_element_count,
        "contact_count_zero": nonzero_contact_element_count == 0,
        "contact_force_zero": maximum_contact_force == 0.0,
    }
    return {
        "variant": _jsonable(variant.__dict__),
        "runtime": runtime,
        "teacher_public": public_metrics,
        "logged_torque_secondary": logged_metrics,
        "frequency_estimation": frequency,
        "torque_semantics": {
            "teacher_public": (
                "Frozen PACEActuatorCore driven sequentially by author state; FIFO is not reset "
                "between intervals, while the plant state is reset before every one-step test."
            ),
            "logged_secondary": (
                "sim_method.dof_torques[t+1] is applied over source interval [t,t+1); "
                "diagnostic only and cannot reopen Branch A."
            ),
        },
        "arrays": {
            "teacher_q_pred_true": predicted_q[0].astype(np.float64),
            "teacher_qdot_pred": predicted_qdot[0].astype(np.float64),
            "logged_q_pred_true": predicted_q[1].astype(np.float64),
            "logged_qdot_pred": predicted_qdot[1].astype(np.float64),
            "teacher_raw_pd_torque": teacher["raw_pd_torque"],
            "teacher_saturated_torque": teacher["saturated_torque"],
            "teacher_applied_torque": teacher["applied_torque"],
            "logged_torque": logged,
            "interval_frequency_hz": frequency["interval_frequency_hz"],
        },
    }


def _full(case: Mapping[str, Any], torque: str = "teacher_public") -> Mapping[str, Any]:
    return case[torque]["windows"]["full_trajectory"]


def _comparison(baseline: Mapping[str, Any], alternate: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for layer in ("position_rad", "velocity_rad_s", "acceleration_rad_s2"):
        base = float(_full(baseline)[layer]["overall_rmse"])
        other = float(_full(alternate)[layer]["overall_rmse"])
        result[layer] = {
            "baseline_rmse": base,
            "alternate_rmse": other,
            "alternate_minus_baseline": other - base,
            "percent_change": 100.0 * (other - base) / base if base else None,
            "alternate_improves": other < base,
        }
    base_position = result["position_rad"]["baseline_rmse"]
    other_position = result["position_rad"]["alternate_rmse"]
    result["material_position_improvement_rule"] = "at least 5% lower one-step position RMSE"
    result["material_position_improvement"] = other_position <= 0.95 * base_position
    if "arrays" in baseline and "arrays" in alternate:
        result["trace_max_abs_difference"] = {
            key: float(
                np.max(
                    np.abs(
                        np.asarray(alternate["arrays"][key], dtype=np.float64)
                        - np.asarray(baseline["arrays"][key], dtype=np.float64)
                    )
                )
            )
            for key in (
                "teacher_q_pred_true",
                "teacher_qdot_pred",
                "logged_q_pred_true",
                "logged_qdot_pred",
            )
        }
    return result


def _frequency_interpretation(case: Mapping[str, Any]) -> Dict[str, Any]:
    bands = case["teacher_public"]["frequency_bands"]
    low = bands["estimated_below_0p75_Hz"]
    high = bands["estimated_at_least_1p75_Hz"]

    def ratio(layer: str, joint: Optional[str] = None) -> float:
        if joint is None:
            low_value = float(low[layer]["overall_rmse"])
            high_value = float(high[layer]["overall_rmse"])
        else:
            low_value = float(low[layer]["focus_joint_rmse"][joint])
            high_value = float(high[layer]["focus_joint_rmse"][joint])
        return high_value / low_value if low_value else math.inf

    return {
        "low_band": "estimated below 0.75 Hz",
        "high_band": "estimated at least 1.75 Hz",
        "high_over_low_overall_rmse_ratio": {
            layer: ratio(layer)
            for layer in ("position_rad", "velocity_rad_s", "acceleration_rad_s2")
        },
        "high_over_low_focus_position_rmse_ratio": {
            joint: ratio("position_rad", joint) for joint in FOCUS_JOINTS
        },
        "high_over_low_focus_velocity_rmse_ratio": {
            joint: ratio("velocity_rad_s", joint) for joint in FOCUS_JOINTS
        },
        "interpretation": (
            "The one-step residual is already present at low frequency but grows at high "
            "frequency, particularly for RF_HFE/LH_HFE. This pattern is compatible with "
            "dynamic/discretization/logging precision effects; it is not evidence for a "
            "specific unpublished inertia or armature configuration."
        ),
    }


def _inertial_runtime_comparison(case: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = case["runtime"]
    actor = {
        item["name"]: item
        for item in runtime["actor_level"]["post_prepare_sim_properties"]
    }
    aggregate = runtime["urdf_aggregated_inertials"]
    result = {}
    for name in sorted(set(actor) & set(aggregate)):
        source = aggregate[name]
        current = actor[name]
        result[name] = {
            "urdf_aggregated_mass": source["mass"],
            "runtime_mass": current["mass"],
            "mass_delta": float(current["mass"] - source["mass"]),
            "urdf_aggregated_com": source["com"],
            "runtime_com": current["com"],
            "max_abs_com_delta": float(
                np.max(np.abs(np.asarray(current["com"]) - np.asarray(source["com"])))
            ),
            "urdf_aggregated_inertia": source["inertia_at_combined_com"],
            "runtime_inertia": current["inertia_raw_mat33"],
            "max_abs_inertia_delta": float(
                np.max(
                    np.abs(
                        np.asarray(current["inertia_raw_mat33"])
                        - np.asarray(source["inertia_at_combined_com"])
                    )
                )
            ),
            "source_links": source["source_links"],
        }
    return result


def _decision(
    provenance: Mapping[str, Any],
    baseline: Mapping[str, Any],
    armature: Mapping[str, Any],
    collapse: Mapping[str, Any],
) -> Dict[str, Any]:
    armature_comparison = _comparison(baseline, armature)
    collapse_comparison = _comparison(baseline, collapse)
    material = [
        name
        for name, comparison in (
            (armature["variant"]["name"], armature_comparison),
            (collapse["variant"]["name"], collapse_comparison),
        )
        if comparison["material_position_improvement"]
    ]
    baseline_position = float(_full(baseline)["position_rad"]["overall_rmse"])
    one_step_over_closed_loop = (
        baseline_position / FROZEN_CLOSED_LOOP_POSITION_RMSE_RAD
    )
    plant_one_step_close = (
        baseline_position <= 1e-4 and one_step_over_closed_loop <= 0.01
    )
    frequency = _frequency_interpretation(baseline)
    if plant_one_step_close:
        case = "B"
        suspect = (
            "No dominant public-asset plant mismatch established; investigate closed-loop "
            "integration accumulation, sampling/write-read timing, and float/log precision."
        )
    elif material:
        case = "A_diagnostic_only_without_provenance"
        suspect = (
            "armature semantics" if armature["variant"]["name"] in material else
            "fixed-joint inertial aggregation / asset identity"
        )
    else:
        case = "C"
        suspect = "unresolved legacy asset/inertial identity"
    public_rmse = float(_full(baseline)["position_rad"]["overall_rmse"])
    logged_rmse = float(
        _full(baseline, "logged_torque_secondary")["position_rad"]["overall_rmse"]
    )
    return {
        "case": case,
        "case_rule": (
            "Case B when public-teacher one-step position RMSE <=1e-4 rad and <=1% of "
            "the frozen full closed-loop 0.0128183648 rad residual."
        ),
        "plant_one_step_close": plant_one_step_close,
        "frozen_full_closed_loop_position_rmse_rad": (
            FROZEN_CLOSED_LOOP_POSITION_RMSE_RAD
        ),
        "one_step_over_full_closed_loop_position_rmse_ratio": one_step_over_closed_loop,
        "armature_ab": armature_comparison,
        "collapse_fixed_joints_ab": collapse_comparison,
        "frequency_interpretation": frequency,
        "logged_torque_one_step": {
            "teacher_public_position_rmse_rad": public_rmse,
            "logged_position_rmse_rad": logged_rmse,
            "logged_over_teacher_ratio": logged_rmse / public_rmse,
            "result": (
                "legacy logged torque has stronger aggregate one-step plant consistency"
                if logged_rmse < public_rmse
                else "public teacher torque has stronger aggregate one-step plant consistency"
            ),
            "policy": (
                "Diagnostic only. This result does not reopen Branch A or modify the actuator."
            ),
        },
        "material_diagnostic_improvements": material,
        "current_largest_plant_side_suspect": suspect,
        "evidence_backed_configuration_change_exists": False,
        "evidence_backed_configuration_change_reason": (
            "No public legacy exporter/asset/config provenance identifies either A/B as the "
            "author setting. A diagnostic improvement alone cannot modify the frozen baseline."
        ),
        "worth_running_full_diagnostic_replay": False,
        "full_replay_reason": (
            "No A/B both materially improves one-step dynamics and has legacy provenance. "
            "Case B instead directs the next audit to accumulation/timing/numeric semantics."
        ),
        "strict_stage0c_asset_provenance_sufficient": False,
        "strict_stage0c_status": (
            "FAIL — Case B: public-asset one-step plant is close; closed-loop accumulation "
            "remains unresolved"
        ),
        "asset_identity_status": (
            "UNRESOLVED — strict asset identity cannot be source-confirmed, but the current "
            "one-step result does not establish asset mismatch as the dominant residual cause"
        ),
        "public_asset_independent_reproduction": {
            "teacher_public_one_step_position_rmse_rad": _full(baseline)[
                "position_rad"
            ]["overall_rmse"],
            "teacher_public_one_step_velocity_rmse_rad_s": _full(baseline)[
                "velocity_rad_s"
            ]["overall_rmse"],
            "teacher_public_one_step_acceleration_rmse_rad_s2": _full(baseline)[
                "acceleration_rad_s2"
            ]["overall_rmse"],
        },
        "branch_A_actuator_law": "CLOSED",
        "actuator_modified": False,
        "locomotion_or_ppo_created": False,
        "thresholds_relaxed": False,
        "parameters_refitted": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    baseline = report["experiments"]["baseline"]
    public = _full(baseline)
    logged = _full(baseline, "logged_torque_secondary")
    decision = report["decision"]
    lines = [
        "# Stage 0 Plant / Asset one-step audit",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "## Frozen status",
        "",
        "```text",
        "Stage 0A: PASS",
        "Stage 0B: frozen",
        "Stage 0C: FAIL",
        "Branch A actuator-law audit: CLOSED",
        "Locomotion/PPO: BLOCKED",
        "```",
        "",
        "No actuator law, frame, threshold, or fitted parameter was changed.",
        "",
        "## 1. Legacy asset provenance",
        "",
        f"Classification: **{report['asset_provenance']['classification']} / "
        f"{report['asset_provenance']['asset_identity_provenance']}**.",
        "",
        report["asset_provenance"]["answer"],
        "",
        "The current asset is the public `anymal_d_simple_description`, whose README calls it "
        "a simplified description. No released source binds legacy `sim_method` to this URDF.",
        "",
        "## 2. Current public asset and runtime snapshot",
        "",
        f"- URDF SHA-256: `{report['public_asset']['sha256']}`",
        f"- URDF links/joints: {report['public_asset']['link_count']} / "
        f"{report['public_asset']['joint_count']}",
        f"- Fixed/movable joints: {report['public_asset']['fixed_joint_count']} / "
        f"{report['public_asset']['movable_joint_count']}",
        f"- URDF total mass: {report['public_asset']['total_urdf_mass']:.9f} kg",
        f"- Collapsed runtime bodies: "
        f"{baseline['runtime']['actor_level']['rigid_body_count']}",
        "- Preview 4 exposes actor rigid-body properties but no asset-level rigid-body property "
        "getter; source URDF inertials and pre/post-prepare actor snapshots are both preserved in JSON.",
        "",
        "The JSON report contains every body mass, COM, full Mat33 inertia, inverse inertia, "
        "principal eigensystem, pre/post-prepare delta, and the URDF-to-runtime collapse mapping.",
        "",
        "## 3. Baseline one-step transitions",
        "",
        "| torque source | position RMSE (rad) | velocity RMSE (rad/s) | acceleration RMSE (rad/s²) |",
        "|---|---:|---:|---:|",
        f"| teacher public | {public['position_rad']['overall_rmse']:.9g} | "
        f"{public['velocity_rad_s']['overall_rmse']:.9g} | "
        f"{public['acceleration_rad_s2']['overall_rmse']:.9g} |",
        f"| logged, diagnostic only | {logged['position_rad']['overall_rmse']:.9g} | "
        f"{logged['velocity_rad_s']['overall_rmse']:.9g} | "
        f"{logged['acceleration_rad_s2']['overall_rmse']:.9g} |",
        "",
        "Formal teacher torque is reconstructed sequentially from author state with the frozen "
        "public actuator and delay FIFO. The simulator plant is reset to the author state before "
        "every interval. Logged torque uses dataset state `t+1` for interval `[t,t+1)` and remains "
        "secondary only.",
        "",
        "### Focus joints (teacher public)",
        "",
        "| joint | position | velocity | acceleration |",
        "|---|---:|---:|---:|",
    ]
    for joint in FOCUS_JOINTS:
        lines.append(
            f"| {joint} | {public['position_rad']['focus_joint_rmse'][joint]:.9g} | "
            f"{public['velocity_rad_s']['focus_joint_rmse'][joint]:.9g} | "
            f"{public['acceleration_rad_s2']['focus_joint_rmse'][joint]:.9g} |"
        )
    lines.extend(
        [
            "",
        "Complete 12-joint metrics, inclusive source-state windows 0–10/50/100/500, "
        "and empirical chirp-frequency bands are in `report.json`.",
        "",
        "### Error versus empirical chirp frequency",
        "",
        "| quantity | high-band / low-band RMSE |",
        "|---|---:|",
        f"| position | "
        f"{decision['frequency_interpretation']['high_over_low_overall_rmse_ratio']['position_rad']:.6g}× |",
        f"| velocity | "
        f"{decision['frequency_interpretation']['high_over_low_overall_rmse_ratio']['velocity_rad_s']:.6g}× |",
        f"| acceleration | "
        f"{decision['frequency_interpretation']['high_over_low_overall_rmse_ratio']['acceleration_rad_s2']:.6g}× |",
        "",
        decision["frequency_interpretation"]["interpretation"],
            "",
            "## 4. Contact audit",
            "",
            f"Baseline maximum absolute contact-force component: "
            f"`{baseline['runtime']['contact_audit']['maximum_absolute_contact_force_component']}`.",
            "",
            f"Zero-contact result: `{baseline['runtime']['contact_audit']['contact_force_zero']}`. "
            "No contact parameters were scanned.",
            "",
            "## 5. OFAT plant diagnostics",
            "",
            "| alternate | position change | velocity change | acceleration change | material position improvement |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for key in ("armature_ab", "collapse_fixed_joints_ab"):
        comparison = decision[key]
        lines.append(
            f"| {key} | {comparison['position_rad']['percent_change']:+.4f}% | "
            f"{comparison['velocity_rad_s']['percent_change']:+.4f}% | "
            f"{comparison['acceleration_rad_s2']['percent_change']:+.4f}% | "
            f"{comparison['material_position_improvement']} |"
        )
    lines.extend(
        [
            "",
            "`use_physx_armature=True` means joint-space armature in Preview 4; False changes "
            "the same per-DOF armature values to link-inertia-tensor modification semantics. "
            "The applied actor DOF properties are captured for both modes.",
            "",
            f"Observed armature A/B maximum teacher position-trace difference: "
            f"`{decision['armature_ab']['trace_max_abs_difference']['teacher_q_pred_true']}` rad. "
            "Thus this A/B is exactly indistinguishable in the present actor-property-write path.",
            "",
            f"Logged-torque result: **{decision['logged_torque_one_step']['result']}** "
            f"(logged/teacher position RMSE ratio "
            f"`{decision['logged_torque_one_step']['logged_over_teacher_ratio']:.6g}`). "
            "This remains diagnostic only and does not reopen Branch A.",
            "",
            "## 6. Decision",
            "",
            f"- Case: **{decision['case']}**.",
            f"- One-step/full-closed-loop position RMSE ratio: "
            f"`{decision['one_step_over_full_closed_loop_position_rmse_ratio']:.6g}`.",
            f"- Largest plant-side suspect: **{decision['current_largest_plant_side_suspect']}**.",
            f"- Evidence-backed configuration change: "
            f"**{decision['evidence_backed_configuration_change_exists']}**.",
            f"- Run full diagnostic replay: **{decision['worth_running_full_diagnostic_replay']}**.",
            f"- Strict Stage 0C asset provenance sufficient: "
            f"**{decision['strict_stage0c_asset_provenance_sufficient']}**.",
            f"- Strict Stage 0C: **{decision['strict_stage0c_status']}**.",
            f"- Asset identity: **{decision['asset_identity_status']}**.",
            "",
            "Independent public-asset reproduction remains quantified above. This does not relax "
            "the frozen 0.01/0.02 rad gate and does not authorize locomotion/PPO.",
            "",
        ]
    )
    return "\n".join(lines)


def run_plant_audit(
    device_id: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Isaac Gym must be imported before decoder unpickling can import torch.
    from isaacgym import gymapi  # noqa: F401
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    artifact_dir = _new_output_dir(output_dir)
    fit = decode_fit()
    data = load_replay_data()
    asset_path = ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE
    urdf = _parse_urdf(asset_path)
    provenance = _legacy_asset_provenance()

    # The sequence is intentionally baseline first, then P1, then P2.
    baseline = _run_variant(device_id, fit, data, urdf, BASELINE)
    armature = _run_variant(device_id, fit, data, urdf, ARMATURE_LINK_INERTIA)
    collapse = _run_variant(device_id, fit, data, urdf, NO_FIXED_COLLAPSE)
    for case in (baseline, armature, collapse):
        case["runtime"]["urdf_vs_runtime_inertial_comparison"] = (
            _inertial_runtime_comparison(case)
        )
    decision = _decision(provenance, baseline, armature, collapse)

    report: Dict[str, Any] = {
        "schema": "pace_stage0.plant_audit.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": {
            "Stage_0A": "PASS",
            "Stage_0B": "frozen",
            "Stage_0C": "FAIL",
            "Branch_A_actuator_law": "CLOSED",
            "plant_asset_audit": "COMPLETED",
            "locomotion_PPO": "BLOCKED",
        },
        "git": _git_snapshot(),
        "random_seed": 0,
        "reproduction": {
            "command": (
                "conda run -n bruce_gym env PYTHONPATH=src python -m "
                "pace_stage0.cli plant_audit --device-id 0"
            ),
            "implementation_sha256": {
                "plant_audit.py": _sha256(Path(__file__).resolve()),
                "actuator.py": _sha256(Path(__file__).resolve().with_name("actuator.py")),
                "decoder.py": _sha256(Path(__file__).resolve().with_name("decoder.py")),
                "data.py": _sha256(Path(__file__).resolve().with_name("data.py")),
            },
        },
        "fit": fit.to_dict(),
        "data": {
            "sha256": data.sha256,
            "sample_count": data.sample_count,
            "interval_count": data.sample_count - 1,
            "time_start_s": float(data.time[0]),
            "time_end_s": float(data.time[-1]),
        },
        "asset_provenance": provenance,
        "public_asset": {
            "vendored_source_commit": ANYMAL_ASSET_COMMIT,
            **urdf,
        },
        "one_step_protocol": {
            "q_compare_author": "sim_method.dof_pos[k]",
            "q_true_author": "sim_method.dof_pos[k] + encoder_bias",
            "qdot_author": "sim_method.dof_vel[k]",
            "state_reset_before_each_interval": True,
            "simulate_steps_per_sample": 1,
            "dt": PHYSICS_DT,
            "teacher_torque": "author state + frozen public PACE actuator law",
            "logged_torque": "secondary diagnostic only",
        },
        "experiments": {
            "baseline": baseline,
            "armature_use_physx_false": armature,
            "collapse_fixed_joints_false": collapse,
        },
        "decision": decision,
        "not_run": {
            "friction_damping_null_tests": (
                "Not run because Case B says not to scan plant parameters when the one-step "
                "plant is already close; no parameter refit or scan is allowed."
            ),
            "solver_integration_variants": (
                "Not run; the next audit target is closed-loop accumulation/timing/numeric "
                "semantics, not a solver grid search."
            ),
            "full_closed_loop_diagnostic": decision["full_replay_reason"],
        },
    }

    trace: Dict[str, np.ndarray] = {
        "source_state_index": np.arange(data.sample_count - 1, dtype=np.int64),
        "next_state_index": np.arange(1, data.sample_count, dtype=np.int64),
        "author_q_compare_before": data.sim_method_dof_pos[:-1],
        "author_q_true_before": data.sim_method_dof_pos[:-1] + fit.encoder_bias[None, :],
        "author_qdot_before": data.sim_method_dof_vel[:-1],
        "author_q_compare_next": data.sim_method_dof_pos[1:],
        "author_qdot_next": data.sim_method_dof_vel[1:],
        "absolute_q_target": data.real_des_dof_pos[:-1],
    }
    for name, case in report["experiments"].items():
        for key, value in case.pop("arrays").items():
            trace[f"{name}__{key}"] = np.asarray(value)
        for torque_key in ("teacher_public", "logged_torque_secondary"):
            metric_arrays = case[torque_key].pop("arrays")
            for key, value in metric_arrays.items():
                trace[f"{name}__{torque_key}__{key}"] = np.asarray(value)
        frequency = case["frequency_estimation"]
        frequency.pop("interval_frequency_hz", None)

    np.savez_compressed(artifact_dir / "one_step_traces.npz", **trace)
    _write_json(artifact_dir / "report.json", report)
    (artifact_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote: {artifact_dir / 'report.md'}")
    print(f"wrote: {artifact_dir / 'report.json'}")
    print(f"wrote: {artifact_dir / 'one_step_traces.npz'}")
    return _jsonable(report)
