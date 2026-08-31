from __future__ import annotations

import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIT_PATH = Path("/home/xy.chen/tw/dataset/pace_data/1_in_air/anymal/fitting.npy")
DATA_PATH = Path("/home/xy.chen/tw/dataset/pace_data/1_in_air/anymal/data.npy")
ENERGY_RATIO_PATH = Path("/home/xy.chen/tw/dataset/pace_data/x_others/22_energy_ratio.py")

EXPECTED_FIT_SHA256 = "4436941fa5e9a5e8e1ef93d55956fcffdb4c4c4526b8ef3e145e6ea619fbe1c8"
EXPECTED_DATA_SHA256 = "edf2e7648602802c87bbe52bb5c098c41553e8c683b2d9013d6f3f4df8bca621"

RAW_JOINT_NAMES = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)
CANONICAL_JOINT_NAMES = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)
RAW_TO_CANONICAL = (0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11)


def canonical_gather_indices(source_joint_names):
    """Indices that gather a source-order joint vector into canonical order."""
    names = tuple(source_joint_names)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate joint names in source order: {names}")
    if set(names) != set(CANONICAL_JOINT_NAMES):
        raise ValueError(
            f"Joint-name set mismatch: expected {CANONICAL_JOINT_NAMES}, got {names}"
        )
    return tuple(names.index(name) for name in CANONICAL_JOINT_NAMES)

ANYMAL_ASSET_ROOT = PROJECT_ROOT / "third_party"
ANYMAL_ASSET_FILE = "anymal_d_simple_description/urdf/anymal.urdf"
ANYMAL_ASSET_COMMIT = "cc1c920093b55ce75641dbfc367f9abd6505d112"
PACE_REFERENCE_COMMIT = "f07259c09b517ab5118bb1d01b0a6078cf8e1c31"

PHYSICS_DT = 0.0025
CONTROL_DECIMATION = 1
KP = 85.0
KD = 0.6
SATURATION_EFFORT = 140.0
EFFORT_LIMIT = 89.0
VELOCITY_LIMIT = 8.5
HARD_LIMIT_SOFT_BAND = math.radians(5.0)

EXPECTED_RUNTIME = {
    "python": "3.8",
    "isaacgym": "1.0rc4",
    "torch": "1.13.1+cu117",
    "torchvision": "0.14.1+cu117",
    "numpy": "1.23.5",
}
