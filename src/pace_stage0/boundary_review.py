from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .constants import (
    ANYMAL_ASSET_COMMIT,
    CANONICAL_JOINT_NAMES,
    EXPECTED_DATA_SHA256,
    EXPECTED_FIT_SHA256,
    PACE_REFERENCE_COMMIT,
    PROJECT_ROOT,
)
from .data import load_replay_data


EVIDENCE_BASELINE_COMMIT = "206b550dc087ba9f4375fd13516439103611db3f"
LEGGED_GYM_REFERENCE_COMMIT = "8fa29acc6fd1910c3d9659eef6310bdd301cde0a"
PAPER_VERSION = "arXiv:2509.06342v2"
PAPER_URL = "https://arxiv.org/html/2509.06342v2"
EXACT_OVERALL_THRESHOLD = 0.010
EXACT_PER_JOINT_THRESHOLD = 0.020

SOURCE_REPORTS = {
    "replay": PROJECT_ROOT / "artifacts/replay_fit/20260831T073349Z/metrics.json",
    "residual": PROJECT_ROOT
    / "artifacts/residual_diagnostics/20260831T080620Z/report.json",
    "joint_order": PROJECT_ROOT
    / "artifacts/joint_order_diagnostics/20260831T083839Z/report.json",
    "torque": PROJECT_ROOT
    / "artifacts/torque_semantics/20260831T091024Z/report.json",
    "fit_order": PROJECT_ROOT
    / "artifacts/fit_order_diagnostics/final_order_audit_v2/report.json",
    "bias_law": PROJECT_ROOT
    / "artifacts/bias_law_counterfactual/final_v2/report.json",
    "plant": PROJECT_ROOT / "artifacts/plant_audit/20260831T144000Z/report.json",
    "accumulation": PROJECT_ROOT
    / "artifacts/accumulation_audit/20260901T020000Z/report.json",
    "p4_p5_first": PROJECT_ROOT
    / "artifacts/p4_p5_audit/20260901T040000Z/report.json",
    "p4_p5_final": PROJECT_ROOT
    / "artifacts/p4_p5_audit/20260901T050000Z/report.json",
    "decoded_fit": PROJECT_ROOT / "artifacts/decoded_fit.json",
    "install": PROJECT_ROOT / "artifacts/install_check.json",
    "actuator_unit": PROJECT_ROOT / "artifacts/actuator_unit.json",
}

EXPECTED_SCHEMAS = {
    "replay": "pace_stage0.replay_metrics.v2",
    "residual": "pace_stage0.residual_diagnostics.v1",
    "joint_order": "pace_stage0.joint_order_diagnostics.v1",
    "torque": "pace_stage0.torque_semantics.v1",
    "fit_order": "pace_stage0.fit_order_diagnostics.v1",
    "bias_law": "pace_stage0.bias_law_counterfactual.v1",
    "plant": "pace_stage0.plant_audit.v1",
    "accumulation": "pace_stage0.accumulation_audit.v1",
    "p4_p5_first": "pace_stage0.p4_p5_audit.v1",
    "p4_p5_final": "pace_stage0.p4_p5_audit.v1",
    "decoded_fit": "pace_stage0.decoded_fit.v1",
    "install": "pace_stage0.install_check.v1",
    "actuator_unit": "pace_stage0.actuator_unit.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sources() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, path in SOURCE_REPORTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen Stage 0 evidence is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = EXPECTED_SCHEMAS[name]
        if value.get("schema") != expected:
            raise ValueError(
                f"Unexpected schema for {name}: {value.get('schema')!r}, expected {expected!r}"
            )
        result[name] = value
    return result


def _git_snapshot() -> Dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(("git",) + args, cwd=PROJECT_ROOT, text=True).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"Frozen evidence drift for {label}: {actual} != {expected}")


def _determinism_check(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, float] = {}
    exactly_equal = True
    for variant in ("baseline", "damping0", "friction0", "both0"):
        left = first["P4"]["evaluations"][variant]
        right = second["P4"]["evaluations"][variant]
        paths = {
            "one_step_position": ("one_step", "position_rmse_rad"),
            "one_step_velocity": ("one_step", "velocity_rmse_rad_s"),
            "one_step_acceleration": ("one_step", "acceleration_rmse_rad_s2"),
        }
        for label, (group, field) in paths.items():
            lhs = float(left[group][field])
            rhs = float(right[group][field])
            fields[f"{variant}/{label}"] = abs(lhs - rhs)
            exactly_equal &= lhs == rhs
        for horizon in ("H32", "H128", "full"):
            lhs = float(left["horizons"][horizon]["position_rmse_rad"])
            rhs = float(right["horizons"][horizon]["position_rmse_rad"])
            fields[f"{variant}/{horizon}"] = abs(lhs - rhs)
            exactly_equal &= lhs == rhs
    for variant in ("substeps2", "velocity_iterations1"):
        left = first["P5"]["evaluations"][variant]
        right = second["P5"]["evaluations"][variant]
        for label, group, field in (
            ("one_step_velocity", "one_step", "velocity_rmse_rad_s"),
            ("one_step_acceleration", "one_step", "acceleration_rmse_rad_s2"),
        ):
            lhs = float(left[group][field])
            rhs = float(right[group][field])
            fields[f"{variant}/{label}"] = abs(lhs - rhs)
            exactly_equal &= lhs == rhs
        for horizon in ("H32", "H128", "full"):
            lhs = float(left["horizons"][horizon]["position_rmse_rad"])
            rhs = float(right["horizons"][horizon]["position_rmse_rad"])
            fields[f"{variant}/{horizon}"] = abs(lhs - rhs)
            exactly_equal &= lhs == rhs
    return {
        "runs": [
            str(SOURCE_REPORTS["p4_p5_first"]),
            str(SOURCE_REPORTS["p4_p5_final"]),
        ],
        "selected_metric_count": len(fields),
        "maximum_absolute_metric_difference": max(fields.values()),
        "all_selected_metrics_exactly_equal": bool(exactly_equal),
        "per_metric_absolute_difference": fields,
    }


def _closed_hypotheses(sources: Mapping[str, Any]) -> list:
    replay = sources["replay"]
    plant = sources["plant"]
    accumulation = sources["accumulation"]
    p4_p5 = sources["p4_p5_final"]
    bias = sources["bias_law"]
    collapse = plant["decision"]["collapse_fixed_joints_ab"]
    armature = plant["decision"]["armature_ab"]
    p4 = p4_p5["P4"]["sensitivity"]
    p5 = p4_p5["P5"]["sensitivity"]
    rows = [
        {
            "branch": "joint order",
            "hypothesis": "data.npy or fit vectors require an RF/LH or alternative leg permutation",
            "test": "channel signatures, all leg-block bias permutations, and same-index/swap comparisons",
            "result": "canonical data order retained; RF/LH swap changes sim_method-vs-real RMSE from 0.015845945 to 0.887019788 rad",
            "formal_change": False,
            "status": "CLOSED for implementation; legacy schema source remains unavailable",
        },
        {
            "branch": "fit block layout",
            "hypothesis": "legacy fitting.npy uses the current public CMA-ES block serialization",
            "test": "payload bounds/magnitudes and paper Table 6 cross-check",
            "result": "legacy layout is friction, damping, armature, encoder bias, delay; public-order counterfactual rejected",
            "formal_change": False,
            "status": "CLOSED / high-confidence artifact-and-paper evidence",
        },
        {
            "branch": "fit block internal joint order",
            "hypothesis": "the 12-value blocks are already LF RF LH RH",
            "test": "rounding-aware comparison of both frozen hypotheses against all Table 6 blocks",
            "result": "LF LH RF RH wins all four blocks; RAW-to-CANONICAL mapping retained",
            "formal_change": False,
            "status": "CLOSED / strong inference",
        },
        {
            "branch": "encoder-bias sign/effective law",
            "hypothesis": "legacy torque identity justifies replacing public q_true-bias feedback with q_true feedback",
            "test": "single-variable legacy-effective counterfactual replay",
            "result": (
                f"position RMSE worsens from {bias['modes']['public_encoder_feedback']['position']['full']['overall_rmse']:.9g} "
                f"to {bias['modes']['legacy_effective_bias']['position']['full']['overall_rmse']:.9g} rad; Branch A closed"
            ),
            "formal_change": False,
            "status": "CLOSED as a trajectory-generating correction; legacy torque divergence documented",
        },
        {
            "branch": "delay/FIFO",
            "hypothesis": "delay initialization/order or pre/post saturation FIFO is wrong",
            "test": "unit tests, zero-padded 3-step trace identity, and torque-field timing forensic",
            "result": "delay=3; first three interval outputs zero; applied[t]=saturated[t-3] identity error 0",
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "DCMotor clipping",
            "hypothesis": "four-quadrant envelope implementation explains replay divergence",
            "test": "four-quadrant unit tests and full trace saturation audit",
            "result": f"unit tests pass; released replay has {replay['diagnostic']['trace']['dc_motor_saturation']['changed_element_count']} clipped sample-joints",
            "formal_change": False,
            "status": "CLOSED for this trajectory",
        },
        {
            "branch": "torque timing",
            "hypothesis": "logged torque is compared at the wrong pre/post-delay state boundary",
            "test": "pre/post FIFO and lag sweep",
            "result": "three-step delayed/applied-like field recorded at the following state boundary; pre-delay lag -4 and post-delay lag -1 are equivalent",
            "formal_change": False,
            "status": "CLOSED for diagnostic alignment",
        },
        {
            "branch": "legacy torque-field semantics",
            "hypothesis": "near-exact torque algebra proves the formal actuator law must change",
            "test": "18 frame candidates plus legacy-effective causal replay",
            "result": "applied-like raw-bias identity found, but its causal counterfactual worsens position and does not close torque",
            "formal_change": False,
            "status": "CLOSED as correction basis; exporter semantics remain strong inference",
        },
        {
            "branch": "target ±1 timing",
            "hypothesis": "target is naturally shifted by one sample",
            "test": "fixed target[t-1], target[t], and target[t+1] full replays",
            "result": (
                f"RMSE t-1={accumulation['simulation_metrics']['target_t_minus_1']['position']['overall_rmse_rad']:.9g}, "
                f"t={accumulation['simulation_metrics']['formal_zero']['position']['overall_rmse_rad']:.9g}, "
                f"t+1={accumulation['simulation_metrics']['target_t_plus_1']['position']['overall_rmse_rad']:.9g} rad; none passes"
            ),
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "initial qdot",
            "hypothesis": "zero initial velocity causes the full residual",
            "test": "zero, sim_method, real, and forward-difference initial qdot",
            "result": "no candidate passes; best counterfactual is 0.012816372 rad versus formal 0.012818365 rad and lacks legacy provenance",
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "float precision",
            "hypothesis": "loader float conversion causes accumulation",
            "test": "raw payload dtype and source-to-loader error inventory",
            "result": "all source fields are already float32 and every source-to-loader error is zero",
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "timestamp jitter",
            "hypothesis": "recorded wall-clock jitter should drive variable-dt simulation",
            "test": "overall/per-joint correlation with velocity and acceleration residual; no variable-dt replay",
            "result": f"maximum absolute correlation {p4_p5['timestamp_jitter']['maximum_absolute_reported_correlation']:.9g}",
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "self collision",
            "hypothesis": "legacy replay used self-collision ON",
            "test": "OFF/ON fixed A/B with contact audit",
            "result": "OFF has zero contacts and 0.012818365 rad; ON has contacts at all 6679 transitions and 0.189955519 rad",
            "formal_change": False,
            "status": "CLOSED as a candidate improvement; exact legacy setting remains unresolved",
        },
        {
            "branch": "armature toggle",
            "hypothesis": "use_physx_armature mode explains the local residual",
            "test": "single-property OFAT",
            "result": f"position/velocity/acceleration changes are {armature['position_rad']['percent_change']:.4f}%/{armature['velocity_rad_s']['percent_change']:.4f}%/{armature['acceleration_rad_s2']['percent_change']:.4f}%",
            "formal_change": False,
            "status": "CLOSED",
        },
        {
            "branch": "collapse fixed joints",
            "hypothesis": "fixed-joint collapse mismatch materially explains the transition residual",
            "test": "single-property OFAT",
            "result": f"position/velocity/acceleration changes are {collapse['position_rad']['percent_change']:.4f}%/{collapse['velocity_rad_s']['percent_change']:.4f}%/{collapse['acceleration_rad_s2']['percent_change']:.4f}%; below material rule and no legacy provenance",
            "formal_change": False,
            "status": "CLOSED as a formal correction basis",
        },
        {
            "branch": "damping null",
            "hypothesis": "fitted DOF damping is incorrectly applied",
            "test": "damping=0 only; one-step plus H8/H32/H128/full",
            "result": f"one-step acceleration {100*p4['damping0']['relative_change_vs_baseline']['one_step_acceleration']:+.1f}% and H32 {100*p4['damping0']['relative_change_vs_baseline']['H32_position']:+.1f}% versus baseline",
            "formal_change": False,
            "status": "CLOSED; removal dramatically worsens",
        },
        {
            "branch": "friction null",
            "hypothesis": "fitted DOF friction coefficient is incorrectly applied",
            "test": "friction=0 only; one-step plus H8/H32/H128/full",
            "result": f"one-step acceleration {100*p4['friction0']['relative_change_vs_baseline']['one_step_acceleration']:+.1f}% and H32 {100*p4['friction0']['relative_change_vs_baseline']['H32_position']:+.1f}% versus baseline",
            "formal_change": False,
            "status": "CLOSED; removal worsens",
        },
        {
            "branch": "damping+friction null",
            "hypothesis": "joint properties mutually cancel and jointly cause the residual",
            "test": "both properties zero; one-step plus horizons",
            "result": f"one-step acceleration {100*p4['both0']['relative_change_vs_baseline']['one_step_acceleration']:+.1f}% and H32 {100*p4['both0']['relative_change_vs_baseline']['H32_position']:+.1f}% versus baseline",
            "formal_change": False,
            "status": "CLOSED; removal dramatically worsens",
        },
        {
            "branch": "substeps=2",
            "hypothesis": "higher internal integration resolution closes the local residual",
            "test": "dt fixed at 0.0025, one control/FIFO update, PhysX substeps 1 to 2",
            "result": f"one-step acceleration {100*p5['substeps2']['relative_change_vs_baseline']['one_step_acceleration']:+.1f}% and H32 {100*p5['substeps2']['relative_change_vs_baseline']['H32_position']:+.1f}%",
            "formal_change": False,
            "status": "CLOSED; worsens",
        },
        {
            "branch": "velocity iterations=1",
            "hypothesis": "one TGS velocity iteration closes the velocity transition residual",
            "test": "velocity iterations 0 to 1, position iterations 4 and substeps 1 frozen",
            "result": f"one-step acceleration {100*p5['velocity_iterations1']['relative_change_vs_baseline']['one_step_acceleration']:+.1f}% and H32 {100*p5['velocity_iterations1']['relative_change_vs_baseline']['H32_position']:+.1f}%",
            "formal_change": False,
            "status": "CLOSED; dramatically worsens",
        },
    ]
    if any(row["formal_change"] for row in rows):
        raise AssertionError("Boundary review must not apply a formal implementation change")
    return rows


def _provenance_matrix() -> list:
    return [
        {
            "assumption": "Kp=85, Kd=0.6",
            "tier": "A_author_confirmed_or_released_source",
            "basis": "PACE paper and released public implementation",
            "exact_legacy_caveat": None,
        },
        {
            "assumption": "absolute joint-position target, scale=1, no default offset",
            "tier": "A_author_confirmed_or_released_source",
            "basis": "released PACE SysID configuration and released des_dof_pos artifact",
            "exact_legacy_caveat": "paper-era source itself is unavailable",
        },
        {
            "assumption": "three-step saturated-torque delay",
            "tier": "B_strong_inference",
            "basis": "paper 7.5 ms at 400 Hz, fitting delay=3.240597, zero torque states 0..3, later official code",
            "exact_legacy_caveat": "legacy discretization/operator source unavailable",
        },
        {
            "assumption": "public encoder feedback q_encoder=q_true-bias",
            "tier": "A_author_confirmed_or_released_source",
            "basis": "paper model and later released collector/CMA-ES/plotting implementation",
            "exact_legacy_caveat": "legacy torque artifact has a documented divergent relative-frame identity",
        },
        {
            "assumption": "fitting.npy block layout",
            "tier": "B_strong_inference",
            "basis": "artifact bounds/magnitudes plus paper Table 6",
            "exact_legacy_caveat": "legacy serializer source unavailable",
        },
        {
            "assumption": "fitting.npy internal LF LH RF RH order before canonicalization",
            "tier": "B_strong_inference",
            "basis": "rounding-aware Table 6 comparison across all four blocks",
            "exact_legacy_caveat": "no machine-readable joint metadata",
        },
        {
            "assumption": "data.npy canonical LF RF LH RH order",
            "tier": "B_strong_inference",
            "basis": "command signature and same-index target/real/sim alignment",
            "exact_legacy_caveat": "no machine-readable joint metadata",
        },
        {
            "assumption": "sim_method.dof_pos is a bias-corrected comparison frame",
            "tier": "B_strong_inference",
            "basis": "paper, later official usage, released plotting, and legacy H0/H+/H- forensic",
            "exact_legacy_caveat": "legacy exporter unavailable",
        },
        {
            "assumption": "sim_method.dof_torques is delayed/applied-like and logged one state later",
            "tier": "B_strong_inference",
            "basis": "near-exact artifact algebra, zero prefix, and lag equivalence",
            "exact_legacy_caveat": "field producer unavailable; target-shift and state-shift terms are non-identifiable",
        },
        {
            "assumption": "sim_method.dof_vel is instantaneous-like simulator state",
            "tier": "B_strong_inference",
            "basis": "author-reset plant predicts it 8.153x better than best position derivative",
            "exact_legacy_caveat": "legacy exporter/filter source unavailable",
        },
        {
            "assumption": "exact legacy ANYmal simulation asset identity",
            "tier": "C_unresolved",
            "basis": "only public anymal_d_simple_description is available",
            "exact_legacy_caveat": "author paper-era asset hash/config not released",
        },
        {
            "assumption": "exact legacy Isaac Gym asset options and importer behavior",
            "tier": "C_unresolved",
            "basis": "current public asset/options are audited but do not identify the legacy settings",
            "exact_legacy_caveat": "collapse, armature mode, import defaults, and version-era details not source-confirmed",
        },
        {
            "assumption": "legacy self-collision setting",
            "tier": "C_unresolved",
            "basis": "paper motions are contact-free and OFF is the viable public baseline",
            "exact_legacy_caveat": "paper-era configuration source unavailable",
        },
        {
            "assumption": "exact paper-era PhysX settings",
            "tier": "C_unresolved",
            "basis": "public/reconstruction baseline is fixed and narrow OFATs were audited",
            "exact_legacy_caveat": "complete solver/integration configuration not released",
        },
    ]


def _unresolved_legacy_information() -> list:
    return [
        {
            "item": "legacy ANYmal simulation asset",
            "status": "UNRESOLVED",
            "known": "public anymal_d_simple_description commit/hash",
            "missing": "author paper-era asset identity, hash, and full inertial/import configuration",
        },
        {
            "item": "legacy Isaac Gym asset options",
            "status": "UNRESOLVED",
            "known": "audited public baseline and two narrow option OFATs",
            "missing": "complete author options/importer defaults and whether all match Preview 4 public reconstruction",
        },
        {
            "item": "legacy data.npy exporter",
            "status": "UNAVAILABLE / provenance unresolved",
            "known": "strong forensic semantics for position, velocity, and torque fields",
            "missing": "producer source, exact write boundary, internal frame conversion, and filtering/reporting operations",
        },
        {
            "item": "exact paper-era SysID simulator source",
            "status": "UNAVAILABLE",
            "known": "paper, released parameters/data, and later public Isaac Lab implementation",
            "missing": "legacy Isaac Gym environment source and complete paper-era simulator configuration",
        },
    ]


def _readiness_checklist(exact: Mapping[str, Any], deterministic: Mapping[str, Any]) -> list:
    return [
        {
            "id": "R1",
            "condition": "Released parameter decoding is stable",
            "assessment": "SATISFIED",
            "evidence": "layout and internal order are high-confidence; delay is independently corroborated",
        },
        {
            "id": "R2",
            "condition": "Public actuator semantics are evidence-backed",
            "assessment": "SATISFIED",
            "evidence": "paper and later official PACE implementation agree; legacy torque conflict was audited and its causal counterfactual worsened replay",
        },
        {
            "id": "R3",
            "condition": "Replay implementation is deterministic",
            "assessment": "SATISFIED",
            "evidence": f"two independent P4/P5 runs have exact equality across {deterministic['selected_metric_count']} selected metrics; max delta={deterministic['maximum_absolute_metric_difference']}",
        },
        {
            "id": "R4",
            "condition": "Local dynamics are close",
            "assessment": "SATISFIED_AS_EVIDENCE_NOT_A_NEW_THRESHOLD",
            "evidence": f"one-step position={exact['one_step_position_rmse_rad']:.9g} rad, velocity={exact['one_step_velocity_rmse_rad_s']:.9g} rad/s, acceleration={exact['one_step_acceleration_rmse_rad_s2']:.9g} rad/s^2",
        },
        {
            "id": "R5",
            "condition": "Full replay substantially reproduces author dynamics with residual disclosed",
            "assessment": "SATISFIED_WITH_MATERIAL_FAILURE_DISCLOSURE",
            "evidence": (
                f"exact author-trajectory residual={exact['overall_q_rmse_rad']:.9g} rad; "
                f"ours-vs-real={exact['ours_vs_real_rmse_rad']:.9g}, author-vs-real={exact['author_vs_real_rmse_rad']:.9g}, "
                f"sim_nothing-vs-real={exact['sim_nothing_vs_real_rmse_rad']:.9g}; original exact and 1.2x-author gates remain FAIL"
            ),
        },
        {
            "id": "R6",
            "condition": "No material evidence-backed implementation correction remains",
            "assessment": "SATISFIED_WITH_CURRENT_PUBLIC_EVIDENCE",
            "evidence": "all authorized branches are closed as formal-change bases; no improvement has both material effect and legacy provenance",
        },
        {
            "id": "R7",
            "condition": "Remaining discrepancy has a documented mechanism",
            "assessment": "SATISFIED",
            "evidence": "small structured velocity/acceleration transition mismatch -> position/PD feedback amplification -> saturation near 0.013 rad",
        },
        {
            "id": "R8",
            "condition": "Remaining discrepancy overlaps unresolved provenance",
            "assessment": "SATISFIED",
            "evidence": "legacy asset identity, complete paper-era simulator config, and exporter semantics remain unavailable",
        },
        {
            "id": "R9",
            "condition": "No parameter was retuned against the released replay",
            "assessment": "SATISFIED",
            "evidence": "no re-identification, residual regression write-back, parameter scaling, or RMSE-directed PhysX search was performed",
        },
        {
            "id": "R10",
            "condition": "Stage 1 can preserve this uncertainty",
            "assessment": "SATISFIABLE_ONLY_IF_MANDATORY_BANNER_IS_ADOPTED",
            "evidence": "a fixed Stage 1 provenance banner is specified by this review; Stage 1 remains blocked until human approval",
        },
    ]


def _options() -> list:
    return [
        {
            "option": "A",
            "policy": "Keep Stage 1 blocked until Stage 0C-E exact replay passes",
            "advantages": ["maximally strict historical-comparison standard", "no ambiguity about exact author-trajectory equivalence"],
            "disadvantages": ["may be impossible from public information", "makes missing author asset/config a permanent blocker for an independent reproduction"],
            "assessment": "scientifically defensible for an exact-replay-only project, but mismatched to the broader independent-reproduction objective",
        },
        {
            "option": "B",
            "policy": "Abolish the exact gate and declare current Stage 0 passed",
            "advantages": ["simple status model"],
            "disadvantages": ["erases a real 0.012818-rad failure", "conflates historical replay with independent reproduction", "invites result-based threshold movement"],
            "assessment": "REJECT",
        },
        {
            "option": "C",
            "policy": "Preserve Stage 0C-E FAIL/unresolved and add a separate Stage 0C-R readiness gate",
            "advantages": ["keeps exact 0.01/0.02 gates intact", "fully discloses residual", "allows an auditable public-information reproduction to proceed without claiming exact author simulation"],
            "disadvantages": ["requires persistent dual-status documentation", "Stage 1 results cannot be described as exact PACE simulator reproduction"],
            "assessment": "RECOMMENDED",
        },
    ]


def build_boundary_review() -> Dict[str, Any]:
    sources = _load_sources()
    replay = sources["replay"]
    accumulation = sources["accumulation"]
    p4_p5 = sources["p4_p5_final"]
    formal = replay["formal"]
    one_step = p4_p5["P4"]["evaluations"]["baseline"]["one_step"]
    horizons = p4_p5["P4"]["evaluations"]["baseline"]["horizons"]
    data = load_replay_data()
    real = np.asarray(data.real_dof_pos, dtype=np.float64)
    centered_real_scale = float(
        np.sqrt(np.mean(np.square(real - np.mean(real, axis=0)), dtype=np.float64))
    )
    exact = {
        "overall_q_rmse_rad": float(formal["rmse_q_encoder_ours_vs_sim_method"]),
        "overall_threshold_rad": EXACT_OVERALL_THRESHOLD,
        "overall_gate": "FAIL",
        "per_joint_threshold_rad": EXACT_PER_JOINT_THRESHOLD,
        "per_joint_q_rmse_rad": formal["per_joint_rmse_q_encoder_ours_vs_sim_method"],
        "failing_joints": {
            name: value
            for name, value in formal["per_joint_rmse_q_encoder_ours_vs_sim_method"].items()
            if value > EXACT_PER_JOINT_THRESHOLD
        },
        "per_joint_gate": "FAIL",
        "one_step_position_rmse_rad": float(one_step["position_rmse_rad"]),
        "one_step_velocity_rmse_rad_s": float(one_step["velocity_rmse_rad_s"]),
        "one_step_acceleration_rmse_rad_s2": float(one_step["acceleration_rmse_rad_s2"]),
        "H32_position_rmse_rad": float(horizons["H32"]["position_rmse_rad"]),
        "H128_position_rmse_rad": float(horizons["H128"]["position_rmse_rad"]),
        "full_position_rmse_rad": float(horizons["full"]["position_rmse_rad"]),
        "contact_state_count": 0,
        "ours_vs_real_rmse_rad": float(formal["rmse_q_encoder_ours_vs_real"]),
        "author_vs_real_rmse_rad": float(
            formal["rmse_sim_method_vs_real_comparison_frame"]
        ),
        "sim_nothing_vs_real_rmse_rad": float(formal["rmse_sim_nothing_vs_real"]),
        "original_encoder_fit_le_1p2x_author_gate": "FAIL",
        "real_trajectory_centered_rms_scale_rad": centered_real_scale,
    }
    exact.update(
        {
            "exact_residual_over_real_centered_scale": exact["overall_q_rmse_rad"]
            / centered_real_scale,
            "ours_real_error_over_real_centered_scale": exact["ours_vs_real_rmse_rad"]
            / centered_real_scale,
            "author_real_error_over_real_centered_scale": exact["author_vs_real_rmse_rad"]
            / centered_real_scale,
            "sim_nothing_real_error_over_real_centered_scale": exact[
                "sim_nothing_vs_real_rmse_rad"
            ]
            / centered_real_scale,
            "ours_real_improvement_vs_sim_nothing": 1.0
            - exact["ours_vs_real_rmse_rad"] / exact["sim_nothing_vs_real_rmse_rad"],
            "ours_real_over_author_real": exact["ours_vs_real_rmse_rad"]
            / exact["author_vs_real_rmse_rad"],
        }
    )
    _assert_close(exact["overall_q_rmse_rad"], 0.012818364796375577, "overall")
    _assert_close(
        exact["per_joint_q_rmse_rad"]["RF_HFE"],
        0.023814172906643862,
        "RF_HFE",
    )
    _assert_close(
        exact["per_joint_q_rmse_rad"]["LH_HFE"],
        0.022510682052527972,
        "LH_HFE",
    )
    _assert_close(exact["one_step_position_rmse_rad"], 2.5462942034564316e-05, "one-step q")
    _assert_close(exact["H32_position_rmse_rad"], 0.0030144730587311273, "H32")
    deterministic = _determinism_check(
        sources["p4_p5_first"], sources["p4_p5_final"]
    )
    if not deterministic["all_selected_metrics_exactly_equal"]:
        raise ValueError("Frozen deterministic-rerun evidence no longer matches")
    closed = _closed_hypotheses(sources)
    unresolved = _unresolved_legacy_information()
    provenance = _provenance_matrix()
    readiness = _readiness_checklist(exact, deterministic)
    review_status = {
        "Stage_0A": "PASS",
        "Stage_0B": "frozen",
        "Stage_0C_E_exact_legacy_replay": "FAIL / unresolved",
        "Stage_0C_R_reproduction_readiness": "NOT SET — RECOMMENDED PASS, awaiting human approval",
        "locomotion_PPO": "BLOCKED pending human approval",
    }
    banner = (
        "Reproduction status:\n"
        "Public-information independent reproduction.\n\n"
        "Exact PACE legacy SysID replay:\n"
        "UNRESOLVED / Stage 0C-E FAIL.\n\n"
        "Known residual:\n"
        "overall 0.012818 rad on released legacy replay.\n\n"
        "Legacy exact asset and paper-era simulator configuration:\n"
        "not publicly resolved."
    )
    return {
        "schema": "pace_stage0.reproducibility_boundary_review.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "review_type": "static evidence review and protocol design; no GPU simulation",
        "evidence_baseline_commit": EVIDENCE_BASELINE_COMMIT,
        "review_generation_git": _git_snapshot(),
        "status_before_and_after_review": review_status,
        "goal_recovery": {
            "Goal_E": "exact historical replay of released legacy sim_method trajectory under frozen 0.01/0.02 gates",
            "Goal_R": "auditable independent reproduction from paper, released parameters/data, public PACE evidence, public ANYmal asset, and frozen Isaac Gym/LeggedGym stack",
            "repository_Stage0_gate_history": "Stage 0C was explicitly implemented as Goal E",
            "overall_project_objective": "Goal R: construct the PACE ANYmal D locomotion training environment independently",
            "conclusion": "Track both goals separately; neither is a substitute for the other.",
        },
        "frozen_references": {
            "paper": {"version": PAPER_VERSION, "url": PAPER_URL},
            "PACE_source_commit": PACE_REFERENCE_COMMIT,
            "LeggedGym_reference45_commit": LEGGED_GYM_REFERENCE_COMMIT,
            "public_ANYmal_asset_commit": ANYMAL_ASSET_COMMIT,
            "data_npy_sha256": EXPECTED_DATA_SHA256,
            "fitting_npy_sha256": EXPECTED_FIT_SHA256,
        },
        "source_reports": {
            name: {
                "path": str(path),
                "schema": EXPECTED_SCHEMAS[name],
                "sha256": _sha256(path),
            }
            for name, path in SOURCE_REPORTS.items()
        },
        "exact_replay": exact,
        "deterministic_reruns": deterministic,
        "closed_implementation_hypotheses": closed,
        "unresolved_legacy_information": unresolved,
        "provenance_matrix": provenance,
        "simulator_tuning_boundary": {
            "evidence_backed_change_remaining": False,
            "continue_RMSE_directed_search_classification": "artifact-driven re-identification / simulator tuning",
            "why": (
                "The public one-step plant is already close, all authorized narrow branches "
                "either fail to improve or lack legacy provenance, and the remaining exact "
                "configuration is underdetermined by public sources."
            ),
            "forbidden_shortcut": "do not change the exact overall threshold from 0.010 to a result-selected value such as 0.013",
        },
        "dual_track_protocol": {
            "Stage_0C_E": {
                "name": "Exact Legacy Replay",
                "overall_gate_rad": EXACT_OVERALL_THRESHOLD,
                "all_per_joint_gate_rad": EXACT_PER_JOINT_THRESHOLD,
                "status": "FAIL / unresolved",
                "thresholds_changed": False,
                "meaning": "publicly available information does not reproduce the released legacy trajectory within the frozen exact gate",
            },
            "Stage_0C_R": {
                "name": "Reproduction Readiness",
                "status": "NOT SET",
                "recommendation": "PASS",
                "approval": "awaiting human approval",
                "core_semantics": "evidence checklist, not a result-selected full-replay RMSE threshold",
                "meaning_if_approved": "sufficient evidence exists to proceed with an auditable independent reproduction using publicly available information",
                "does_not_mean": "the author simulator was exactly reproduced",
                "checklist": readiness,
            },
        },
        "decision_options": _options(),
        "recommendation": {
            "option": "C",
            "exact_gate_retained": True,
            "add_independent_readiness_gate": True,
            "current_evidence_supports_recommended_R_pass": True,
            "R_pass_applied_by_this_review": False,
            "authorize_Stage1_now": False,
            "recommended_human_decision": (
                "Approve Stage 0C-R PASS and authorize Stage 1 independent reproduction "
                "while permanently retaining Stage 0C-E FAIL/unresolved."
            ),
            "recommended_tag_after_approval_and_freeze_manifest": "stage0-public-reproduction-ready",
            "tag_created": False,
            "stage0_freeze_manifest_created": False,
        },
        "mandatory_Stage1_provenance_banner_if_approved": banner,
        "post_approval_actions_not_executed": [
            "set Stage 0C-R to PASS",
            "authorize or implement locomotion/PPO",
            "create provenance/stage0_freeze.json",
            "create branch or tag",
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    exact = report["exact_replay"]
    status = report["status_before_and_after_review"]
    lines = [
        "# Stage 0 Reproducibility-Boundary Review",
        "",
        f"Generated: `{report['generated_utc']}` from evidence baseline "
        f"`{report['evidence_baseline_commit']}`.",
        "",
        "```text",
        f"Stage 0C-E: {status['Stage_0C_E_exact_legacy_replay']}",
        f"Stage 0C-R: {status['Stage_0C_R_reproduction_readiness']}",
        f"locomotion/PPO: {status['locomotion_PPO']}",
        "```",
        "",
        "This is a static evidence review. It ran no GPU simulation, changed no actuator, "
        "asset, PhysX setting, fitted parameter, loader, decoder, or exact threshold, and "
        "did not authorize Stage 1.",
        "",
        "## 1. Goal boundary",
        "",
        "The repository's Stage 0C gate was built as **Goal E: exact historical replay**. "
        "The overall project objective—independently reconstructing the PACE ANYmal D "
        "locomotion environment—is **Goal R: auditable independent reproduction**. The "
        "scientifically correct status model records both; they are not equivalent.",
        "",
        "## 2. Exact replay result",
        "",
        "| Evidence | Result | Status |",
        "|---|---:|---|",
        f"| Overall q RMSE | {exact['overall_q_rmse_rad']:.9f} rad | exact gate FAIL |",
        f"| RF_HFE | {exact['per_joint_q_rmse_rad']['RF_HFE']:.9f} rad | FAIL |",
        f"| LH_HFE | {exact['per_joint_q_rmse_rad']['LH_HFE']:.9f} rad | FAIL |",
        f"| One-step q RMSE | {exact['one_step_position_rmse_rad']:.9g} rad | very close; evidence only |",
        f"| One-step qdot RMSE | {exact['one_step_velocity_rmse_rad_s']:.9g} rad/s | structured residual |",
        f"| One-step acceleration RMSE | {exact['one_step_acceleration_rmse_rad_s2']:.9g} rad/s² | structured residual |",
        f"| H32 q RMSE | {exact['H32_position_rmse_rad']:.9g} rad | diagnostic |",
        f"| H128 q RMSE | {exact['H128_position_rmse_rad']:.9g} rad | diagnostic |",
        f"| Full q RMSE | {exact['full_position_rmse_rad']:.9g} rad | saturated |",
        f"| Contact states | {exact['contact_state_count']} | pass |",
        f"| Deterministic reruns | max selected-metric delta {report['deterministic_reruns']['maximum_absolute_metric_difference']:.1f} | pass |",
        "",
        "The exact status is **FAIL**, not “near pass”: overall exceeds 0.010 rad and "
        "RF_HFE/LH_HFE both exceed 0.020 rad.",
        "",
        "### Real-data context (not substitute gates)",
        "",
        "| Comparison | RMSE | centered real-trajectory scale |",
        "|---|---:|---:|",
        f"| author sim_method vs real | {exact['author_vs_real_rmse_rad']:.9g} | {100*exact['author_real_error_over_real_centered_scale']:.3f}% |",
        f"| ours vs real | {exact['ours_vs_real_rmse_rad']:.9g} | {100*exact['ours_real_error_over_real_centered_scale']:.3f}% |",
        f"| sim_nothing vs real | {exact['sim_nothing_vs_real_rmse_rad']:.9g} | {100*exact['sim_nothing_real_error_over_real_centered_scale']:.3f}% |",
        "",
        f"Ours reduces real-data RMSE by `{100*exact['ours_real_improvement_vs_sim_nothing']:.3f}%` "
        f"relative to `sim_nothing`, but is `{exact['ours_real_over_author_real']:.3f}×` "
        "the author's error and fails the original 1.2×-author gate. Both facts are retained.",
        "",
        "## 3. Closed implementation hypotheses",
        "",
        "| Branch | Test result | Formal change? | Status |",
        "|---|---|---|---|",
    ]
    for row in report["closed_implementation_hypotheses"]:
        lines.append(
            f"| {row['branch']} | {row['result']} | {row['formal_change']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "No listed branch establishes a material, provenance-backed formal correction.",
            "",
            "## 4. Unresolved legacy information",
            "",
            "| Item | Status | Known | Missing |",
            "|---|---|---|---|",
        ]
    )
    for row in report["unresolved_legacy_information"]:
        lines.append(
            f"| {row['item']} | {row['status']} | {row['known']} | {row['missing']} |"
        )
    lines.extend(
        [
            "",
            "Later public Isaac Lab code is evidence for public actuator semantics, not "
            "source-code confirmation of the paper-era Isaac Gym exporter/environment.",
            "",
            "## 5. Provenance matrix",
            "",
            "| Assumption | Tier | Basis | Exact-legacy caveat |",
            "|---|---|---|---|",
        ]
    )
    for row in report["provenance_matrix"]:
        lines.append(
            f"| {row['assumption']} | {row['tier']} | {row['basis']} | "
            f"{row['exact_legacy_caveat'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "## 6. Simulator-tuning boundary",
            "",
            "With no source-backed new configuration, continuing to tune damping, friction, "
            "inertia, solver iterations, substeps, or other PhysX settings against full "
            "RMSE is **artifact-driven re-identification / simulator tuning**, not "
            "evidence-backed reproduction. The exact threshold must not be changed from "
            "0.010 to a result-selected value such as 0.013.",
            "",
            "## 7. Stage 0C-R readiness checklist",
            "",
            "| ID | Condition | Assessment | Evidence |",
            "|---|---|---|---|",
        ]
    )
    for row in report["dual_track_protocol"]["Stage_0C_R"]["checklist"]:
        lines.append(
            f"| {row['id']} | {row['condition']} | {row['assessment']} | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "The checklist supports a **recommended PASS**, but this review deliberately "
            "leaves Stage 0C-R `NOT SET` pending human approval. A future PASS would mean "
            "only that public evidence is sufficient for an auditable independent "
            "reproduction; it would not mean the author simulator was reproduced exactly.",
            "",
            "## 8. Options and recommendation",
            "",
            "| Option | Policy | Assessment |",
            "|---|---|---|",
        ]
    )
    for option in report["decision_options"]:
        lines.append(
            f"| {option['option']} | {option['policy']} | {option['assessment']} |"
        )
    lines.extend(
        [
            "",
            "**Recommendation: Option C.** Permanently retain Stage 0C-E FAIL/unresolved "
            "and the original 0.01/0.02 gates. After explicit human approval, set the "
            "separate Stage 0C-R readiness gate to PASS and authorize Stage 1 as a "
            "public-information independent reproduction.",
            "",
            "## 9. Mandatory Stage 1 provenance banner",
            "",
            "```text",
            report["mandatory_Stage1_provenance_banner_if_approved"],
            "```",
            "",
            "## 10. Freeze recommendation (not executed)",
            "",
            "After approval, create `provenance/stage0_freeze.json` and the annotated tag "
            "`stage0-public-reproduction-ready`. This review created neither, did not set "
            "Stage 0C-R PASS, and did not authorize or implement locomotion/PPO.",
            "",
        ]
    )
    return "\n".join(lines)


def write_boundary_review(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    destination = (
        PROJECT_ROOT / "artifacts/reproducibility_boundary_review"
        if output_dir is None
        else output_dir.resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    report = build_boundary_review()
    (destination / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"wrote: {destination / 'report.md'}")
    print(f"wrote: {destination / 'report.json'}")
    return report
