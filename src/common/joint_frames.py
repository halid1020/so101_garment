"""The servo frame and the URDF frame, and the conversion between them.

Every SO-101 servo was zeroed and oriented independently during LeRobot
calibration, so what a bus reports is not what the model, the IK and the
recorded dataset mean by a joint angle. One sign and one offset per joint per
arm relate the two::

    urdf_deg = sign * hw_deg + offset
    hw_deg   = sign * (urdf_deg - offset)

The numbers themselves belong to the rig and live in ``common.configs`` (fitted
with ``tool/fit_joint_offsets.py``); this module is only where the tables are
assembled and the conversion is named, so the joint-state thread, the sidecar
writer and the console's idle arm reader cannot disagree about it.

Pure: numpy and the constants, nothing else.
"""

from __future__ import annotations

import numpy as np

from common.configs import (
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)

HW_TO_URDF_OFFSETS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
}
HW_TO_URDF_SIGNS = {
    "left": np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
    "right": np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
}


def hw_to_urdf(side: str, hw_deg: "np.ndarray | list") -> np.ndarray:
    """One arm's five body joints, servo degrees → URDF degrees. Pure."""
    return HW_TO_URDF_SIGNS[side] * np.asarray(hw_deg, dtype=np.float64) + (
        HW_TO_URDF_OFFSETS[side]
    )


def urdf_to_hw(side: str, urdf_deg: "np.ndarray | list") -> np.ndarray:
    """One arm's five body joints, URDF degrees → servo degrees. Pure."""
    return HW_TO_URDF_SIGNS[side] * (
        np.asarray(urdf_deg, dtype=np.float64) - HW_TO_URDF_OFFSETS[side]
    )
