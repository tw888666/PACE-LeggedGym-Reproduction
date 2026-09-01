"""Freeze-guarded Stage 1 PPO launcher. The first authorized run is Phase A only."""

from __future__ import annotations

# Importing the environment first preserves Isaac Gym Preview 4 import order.
from .env import Stage1LocomotionEnv

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from pace_stage0.constants import PROJECT_ROOT

from .config import STAGE1_CONFIG, ppo_train_cfg
from .scheduled_runner import Stage1ScheduledOnPolicyRunner, _atomic_json


AUTHORIZATION_PATH = PROJECT_ROOT / "provenance" / "stage1_ppo_authorization.json"
STAGE0_COMMIT = "0fe25899e63a7be0a5c73a0184f7390352d9f6be"
STAGE1_TAG = "stage1-ppo-ready"
PHASE_A = asdict(STAGE1_CONFIG.phase_a_validation)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def verify_freeze() -> dict:
    authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    if authorization["status"]["Stage_1"] != "IMPLEMENTED / PPO READY":
        raise RuntimeError("Stage 1 authorization is not PPO READY")
    if authorization["status"]["PPO_started"] is not False:
        raise RuntimeError("freeze authorization must record PPO_started=false")
    head = _git("rev-parse", "HEAD")
    tag_target = _git("rev-parse", f"{STAGE1_TAG}^{{}}")
    if head != tag_target:
        raise RuntimeError(f"HEAD {head} is not frozen tag target {tag_target}")
    if _git("rev-parse", "stage0-public-reproduction-ready^{}") != STAGE0_COMMIT:
        raise RuntimeError("Stage 0 freeze tag moved")
    worktree_status = _git("status", "--short")
    if worktree_status:
        raise RuntimeError(f"worktree is dirty:\n{worktree_status}")
    version = importlib.metadata.version("rsl-rl")
    if version != "1.0.2":
        raise RuntimeError(f"rsl_rl must be 1.0.2, got {version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable")
    return {"authorization": authorization, "head": head, "rsl_rl": version}


def phase_a_train() -> Path:
    verification = verify_freeze()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "artifacts" / "ppo" / PHASE_A["experiment_name"] / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    runtime_manifest = {
        "schema": "pace_stage1.ppo_run.v1",
        **PHASE_A,
        "stage1_freeze_commit": verification["head"],
        "stage1_freeze_tag": STAGE1_TAG,
        "stage0_freeze_commit": STAGE0_COMMIT,
        "rsl_rl": verification["rsl_rl"],
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_device_name": torch.cuda.get_device_name(0),
        "PPO_started": False,
        "checkpoint_classification": "pipeline diagnostics only; excluded from paper results",
        "state": "constructing_environment",
        "created_at_unix_s": time.time(),
    }
    _atomic_json(manifest_path, runtime_manifest)

    train_cfg = ppo_train_cfg()
    train_cfg["runner"]["max_iterations"] = PHASE_A["max_iterations"]
    train_cfg["runner"]["experiment_name"] = PHASE_A["experiment_name"]
    environment = None
    try:
        environment = Stage1LocomotionEnv(
            num_envs=PHASE_A["num_envs"],
            sim_device=PHASE_A["sim_device"],
            headless=True,
            seed=PHASE_A["seed"],
        )
        runner = Stage1ScheduledOnPolicyRunner(
            environment,
            train_cfg,
            log_dir=str(run_dir),
            device=PHASE_A["rl_device"],
            runtime_manifest_path=manifest_path,
            runtime_manifest=runtime_manifest,
        )
        runtime_manifest["PPO_started"] = True
        runtime_manifest["state"] = "runner_created"
        runtime_manifest["runner_created_at_unix_s"] = time.time()
        _atomic_json(manifest_path, runtime_manifest)
        print(f"STAGE1_RUN_DIR={run_dir}", flush=True)
        print("PPO_STARTED=true", flush=True)
        runner.learn(PHASE_A["max_iterations"], init_at_random_ep_len=True)
    except Exception as error:
        runtime_manifest["state"] = "failed"
        runtime_manifest["error"] = repr(error)
        runtime_manifest["updated_at_unix_s"] = time.time()
        _atomic_json(manifest_path, runtime_manifest)
        raise
    finally:
        if environment is not None:
            environment.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-a",
        action="store_true",
        help="run the only currently authorized 1024-env/300-iteration/seed1 validation",
    )
    args = parser.parse_args()
    if not args.phase_a:
        parser.error("only the explicit --phase-a validation is authorized")
    phase_a_train()


if __name__ == "__main__":
    main()
