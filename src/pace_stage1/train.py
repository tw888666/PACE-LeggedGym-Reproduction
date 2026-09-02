"""Exact Stage 1 PACE-derived energy-off flat seed0 ablation launcher."""

from __future__ import annotations

# Isaac Gym Preview 4 must be imported before torch/rsl_rl.
from .env import Stage1LocomotionEnv

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from pace_stage0.constants import PROJECT_ROOT

from .config import STAGE1_CONFIG, ppo_train_cfg


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_flat_seed0() -> Path:
    ablation = STAGE1_CONFIG.formal_ablation
    if importlib.metadata.version("rsl-rl") != "1.0.2":
        raise RuntimeError("formal energy-off ablation requires rsl_rl 1.0.2")
    if subprocess.call(["git", "diff", "--quiet"], cwd=str(PROJECT_ROOT)) != 0:
        raise RuntimeError("tracked worktree must match the committed ablation implementation")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        PROJECT_ROOT / "artifacts" / "ppo" / ablation.experiment_name / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "schema": "pace_stage1.pace_energy_off_flat_ablation.v1",
        "classification": "PACE-DERIVED ENERGY-OFF ABLATION / NOT AN OFFICIAL PACE BASELINE",
        "commit": commit,
        "experiment_name": ablation.experiment_name,
        "num_envs": ablation.num_envs,
        "max_iterations": ablation.max_iterations,
        "seed": ablation.seed,
        "terrain_mode": ablation.terrain_mode,
        "friction_randomization": ablation.friction_randomization,
        "pushes": ablation.pushes,
        "enforce_joint_limit_on_policy_target": (
            STAGE1_CONFIG.target_adapter.enforce_joint_limit_on_policy_target
        ),
        "sim_device": ablation.sim_device,
        "rl_device": ablation.rl_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_device_name": torch.cuda.get_device_name(0),
        "formal_ablation_PPO_started": False,
        "state": "constructing_environment",
        "created_at_unix_s": time.time(),
    }
    _write_manifest(manifest_path, manifest)
    environment = None
    try:
        environment = Stage1LocomotionEnv(
            num_envs=ablation.num_envs,
            sim_device=ablation.sim_device,
            headless=True,
            terrain_mode=ablation.terrain_mode,
            seed=ablation.seed,
            enable_pushes=ablation.pushes,
        )
        train_cfg = ppo_train_cfg()
        train_cfg["seed"] = ablation.seed
        train_cfg["runner"]["max_iterations"] = ablation.max_iterations
        train_cfg["runner"]["experiment_name"] = ablation.experiment_name
        runner = OnPolicyRunner(
            environment,
            train_cfg,
            log_dir=str(run_dir),
            device=ablation.rl_device,
        )
        manifest["formal_ablation_PPO_started"] = True
        manifest["state"] = "training"
        manifest["runner_created_at_unix_s"] = time.time()
        _write_manifest(manifest_path, manifest)
        print(f"RUN_DIR={run_dir}", flush=True)
        runner.learn(ablation.max_iterations, init_at_random_ep_len=True)
        manifest["state"] = "completed"
        manifest["completed_iterations"] = ablation.max_iterations
        manifest["completed_at_unix_s"] = time.time()
        _write_manifest(manifest_path, manifest)
    except Exception as error:
        manifest["state"] = "failed"
        manifest["error"] = repr(error)
        manifest["updated_at_unix_s"] = time.time()
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        if environment is not None:
            environment.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-seed0", action="store_true")
    args = parser.parse_args()
    if not args.flat_seed0:
        parser.error("only the exact --flat-seed0 energy-off ablation is available")
    run_flat_seed0()


if __name__ == "__main__":
    main()
