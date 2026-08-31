from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .constants import (
    ANYMAL_ASSET_FILE,
    ANYMAL_ASSET_ROOT,
    CANONICAL_JOINT_NAMES,
    EXPECTED_RUNTIME,
    PHYSICS_DT,
    canonical_gather_indices,
)


def _ubuntu_version() -> str:
    values: Dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    return values.get("VERSION_ID", "unknown")


def _gcc_version() -> str:
    try:
        return subprocess.check_output(
            ["gcc", "-dumpfullversion"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _smoke_isaacgym(device_id: int) -> Dict[str, Any]:
    # Preview 4 requires importing Isaac Gym before torch.
    from isaacgym import gymapi

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

    sim = gym.create_sim(device_id, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("gym.create_sim returned None")
    try:
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = False
        options.collapse_fixed_joints = True
        options.replace_cylinder_with_capsule = True
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        options.use_physx_armature = True
        asset = gym.load_asset(sim, str(ANYMAL_ASSET_ROOT), ANYMAL_ASSET_FILE, options)
        if asset is None:
            raise RuntimeError("gym.load_asset returned None")
        dof_names = tuple(gym.get_asset_dof_names(asset))
        gather_indices = canonical_gather_indices(dof_names)
        props = gym.get_asset_dof_properties(asset)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)

        env = gym.create_env(
            sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 2.0), 1
        )
        pose = gymapi.Transform()
        pose.p.z = 1.0
        actor = gym.create_actor(env, asset, pose, "anymal_d", 0, 0)
        gym.set_actor_dof_properties(env, actor, props)
        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        applied_props = gym.get_actor_dof_properties(env, actor)
        return {
            "created": True,
            "asset_file": str(ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE),
            "dof_count": len(dof_names),
            "dof_names": list(dof_names),
            "canonical_joint_names": list(CANONICAL_JOINT_NAMES),
            "canonical_gather_indices": list(gather_indices),
            "drive_mode_effort": bool(
                (applied_props["driveMode"] == gymapi.DOF_MODE_EFFORT).all()
            ),
            "builtin_stiffness_zero": bool((applied_props["stiffness"] == 0.0).all()),
        }
    finally:
        gym.destroy_sim(sim)


def check_install(device_id: int = 0) -> Dict[str, Any]:
    # Import order is intentional: Isaac Gym before torch.
    import isaacgym  # noqa: F401
    import numpy
    import torch
    import torchvision

    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": platform.python_version(),
        "isaacgym": importlib.metadata.version("isaacgym"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "ubuntu": _ubuntu_version(),
        "gcc": _gcc_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    checks = {
        "python": versions["python"] == EXPECTED_RUNTIME["python"],
        "isaacgym": versions["isaacgym"] == EXPECTED_RUNTIME["isaacgym"],
        "torch": versions["torch"] == EXPECTED_RUNTIME["torch"],
        "torchvision": versions["torchvision"] == EXPECTED_RUNTIME["torchvision"],
        "numpy": versions["numpy"] == EXPECTED_RUNTIME["numpy"],
        "ubuntu": versions["ubuntu"] == "22.04",
        "gcc": versions["gcc"].split(".", 1)[0] == "11",
        "cuda": bool(versions["cuda_available"]),
        "asset_exists": (ANYMAL_ASSET_ROOT / ANYMAL_ASSET_FILE).is_file(),
    }
    smoke: Dict[str, Any]
    try:
        smoke = _smoke_isaacgym(device_id)
        checks["isaacgym_smoke"] = bool(
            smoke["created"]
            and smoke["drive_mode_effort"]
            and smoke["builtin_stiffness_zero"]
        )
    except Exception as exc:
        smoke = {"created": False, "error": f"{type(exc).__name__}: {exc}"}
        checks["isaacgym_smoke"] = False
    return {
        "schema": "pace_stage0.install_check.v1",
        "versions": versions,
        "expected": EXPECTED_RUNTIME,
        "checks": checks,
        "smoke": smoke,
        "pass": all(checks.values()),
        "selected_cuda_device": device_id,
        "environment": {"CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV")},
    }


def print_install_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))
