from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from .constants import PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PACE ANYmal D frozen Stage 0")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("check_install", help="Validate the frozen runtime and Isaac Gym smoke test")
    install.add_argument("--device-id", type=int, default=0)
    install.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "install_check.json"
    )

    decode = sub.add_parser("decode_fit", help="Verify and decode the whitelisted fitting.npy")
    decode.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "decoded_fit.json"
    )

    unit = sub.add_parser("actuator_unit", help="Run frozen actuator/adapter/decoder/sampling tests")
    unit.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "actuator_unit.json"
    )

    forensic = sub.add_parser(
        "frame_forensics",
        help="Run read-only legacy data frame analysis without Isaac Gym replay",
    )
    forensic.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "frame_forensics.json",
    )
    forensic.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "frame_forensics.md",
    )

    replay = sub.add_parser("replay_fit", help="Run the full absolute-target Isaac Gym replay")
    replay.add_argument("--device-id", type=int, default=0)
    replay.add_argument("--output-dir", type=Path)

    residual = sub.add_parser(
        "residual_diagnostics",
        help="Run frozen Stage 0C residual localization and self-collision A/B",
    )
    residual.add_argument("--device-id", type=int, default=0)
    residual.add_argument("--output-dir", type=Path)

    joint_order = sub.add_parser(
        "joint_order_diagnostics",
        help="Run read-only Stage 0C data.npy joint-order diagnostics",
    )
    joint_order.add_argument("--output-dir", type=Path)

    torque_semantics = sub.add_parser(
        "torque_semantics",
        help="Run dataset-only Stage 0C sim_method.dof_torques forensic",
    )
    torque_semantics.add_argument("--output-dir", type=Path)

    fit_order = sub.add_parser(
        "fit_order_diagnostics",
        help="Audit fitting.npy block-internal joint order against paper Table 6",
    )
    fit_order.add_argument("--output-dir", type=Path)

    bias_law = sub.add_parser(
        "bias_law_counterfactual",
        help="Run diagnostic-only public vs legacy-effective bias-law replay",
    )
    bias_law.add_argument("--device-id", type=int, default=0)
    bias_law.add_argument("--output-dir", type=Path)

    plant_audit = sub.add_parser(
        "plant_audit",
        help="Run frozen teacher-forced one-step plant/asset audit and OFAT A/B",
    )
    plant_audit.add_argument("--device-id", type=int, default=0)
    plant_audit.add_argument("--output-dir", type=Path)
    return parser


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _actuator_unit(output: Path) -> bool:
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema": "pace_stage0.actuator_unit.v1",
        "tests_run": result.testsRun,
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "skipped": [str(test) for test, _ in result.skipped],
        "pass": result.wasSuccessful(),
    }
    _write_json(output, report)
    print(f"wrote: {output.resolve()}")
    return result.wasSuccessful()


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check_install":
        from .install_check import check_install, print_install_report

        report = check_install(args.device_id)
        print_install_report(report)
        _write_json(args.output, report)
        print(f"wrote: {args.output.resolve()}")
        return 0 if report["pass"] else 1
    if args.command == "decode_fit":
        from .decoder import decode_fit, write_decoded_fit

        decoded = decode_fit()
        write_decoded_fit(args.output, decoded)
        print(json.dumps(decoded.to_dict(), indent=2))
        print(f"wrote: {args.output.resolve()}")
        return 0
    if args.command == "actuator_unit":
        return 0 if _actuator_unit(args.output) else 1
    if args.command == "frame_forensics":
        from .frame_forensics import write_frame_forensics

        report = write_frame_forensics(args.json_output, args.markdown_output)
        print(json.dumps(report, indent=2))
        print(f"wrote: {args.json_output.resolve()}")
        print(f"wrote: {args.markdown_output.resolve()}")
        return 0
    if args.command == "replay_fit":
        # Importing replay lazily preserves the Isaac Gym-before-torch requirement.
        from .replay import replay_fit

        report = replay_fit(device_id=args.device_id, output_dir=args.output_dir)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1

    if args.command == "residual_diagnostics":
        from .residual_diagnostics import run_residual_diagnostics

        report = run_residual_diagnostics(
            device_id=args.device_id, output_dir=args.output_dir
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "joint_order_diagnostics":
        from .joint_order_diagnostics import run_joint_order_diagnostics

        report = run_joint_order_diagnostics(output_dir=args.output_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "torque_semantics":
        from .torque_semantics import run_torque_semantics

        report = run_torque_semantics(output_dir=args.output_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "fit_order_diagnostics":
        from .fit_order_diagnostics import run_fit_order_diagnostics

        report = run_fit_order_diagnostics(output_dir=args.output_dir)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "bias_law_counterfactual":
        # Isaac Gym Preview 4 must be imported before modules that import torch.
        from isaacgym import gymapi  # noqa: F401
        from .bias_law_counterfactual import run_bias_law_counterfactual

        report = run_bias_law_counterfactual(
            device_id=args.device_id, output_dir=args.output_dir
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plant_audit":
        # Isaac Gym Preview 4 must be imported before modules that import torch.
        from isaacgym import gymapi  # noqa: F401
        from .plant_audit import run_plant_audit

        report = run_plant_audit(
            device_id=args.device_id, output_dir=args.output_dir
        )
        print(json.dumps(report, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
