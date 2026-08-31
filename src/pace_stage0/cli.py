from __future__ import annotations

import argparse
import json
import sys
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

    replay = sub.add_parser("replay_fit", help="Run the full absolute-target Isaac Gym replay")
    replay.add_argument("--device-id", type=int, default=0)
    replay.add_argument("--output-dir", type=Path)
    replay.add_argument(
        "--acknowledge-frame-sanity-failure",
        action="store_true",
        help="Continue without changing frames after manually reviewing a failed frame sanity preflight",
    )
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
    if args.command == "replay_fit":
        # Importing replay lazily preserves the Isaac Gym-before-torch requirement.
        from .replay import FrameSanityError, replay_fit

        try:
            report = replay_fit(
                device_id=args.device_id,
                output_dir=args.output_dir,
                acknowledge_frame_sanity_failure=args.acknowledge_frame_sanity_failure,
            )
        except FrameSanityError as exc:
            print(f"FRAME_SANITY_BLOCKED: {exc}", file=sys.stderr)
            print(f"preflight artifacts: {exc.artifact_dir}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
