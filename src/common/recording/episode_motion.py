"""Velocity and acceleration of a recorded episode, for review.

Playing an episode back answers whether the demonstration did the right thing.
It does not answer whether it did it *smoothly*, and a jerky demonstration is a
defect a policy learns from. That property is a derivative, and the dataset
stores none: it stores joint positions at a fixed rate. So the review tool
differentiates them here.

Two decisions are worth stating, because both could reasonably have gone the
other way.

**Plain central differences, no filter.** The encoder resolves 0.088 degrees,
which at the recorded rate puts the acceleration noise floor near 20 deg/s^2
against motion whose 95th percentile is around 376 deg/s^2 -- roughly an order
of magnitude of headroom. Savitzky-Golay was measured at every window from five
to fifteen samples and made the still-frame residual WORSE, not better, because
a quadratic fit over a longer window drags in neighbouring motion. A filter here
would only blur the transients that are the thing being looked for.

**The end effector is derived, not read.** Episodes collected from the leader
arms carry no Cartesian stream at all -- ``ee_pose`` is written in quest mode
only -- so Cartesian motion is reconstructed by forward kinematics through the
platform's own URDF, in each arm's OWN base frame, which is the convention
``ee_pose`` uses when it is recorded. A derived pose and a recorded one
therefore mean the same thing. This costs about 4 ms for a 500-frame episode.

The joint half needs nothing but numpy, so it keeps working on a machine where
the kinematic model will not load; ``episode_motion`` reports that case rather
than raising, because a review tool that cannot build a model must still play
videos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.recording.features import BODY_DOF, SIDES

# Channel layout of ``observation.state``: per side the five body joints then
# the gripper, left arm first (see common/recording/features.py).
STATE_DOF = (BODY_DOF + 1) * len(SIDES)
GRIPPER_CHANNELS = tuple(s * (BODY_DOF + 1) + BODY_DOF for s in range(len(SIDES)))

# The unit of each channel's FIRST derivative. The body joints are URDF degrees,
# but a gripper channel is an open FRACTION, so its rate is a fraction per
# second and printing it under a degrees heading would be a lie.
JOINT_UNITS: list[str] = [
    "frac/s" if i in GRIPPER_CHANNELS else "deg/s" for i in range(STATE_DOF)
]

# Cached per URDF path: building the reduced model costs about a millisecond,
# which is not much, but a review session asks for one episode after another.
_KINEMATICS: dict[str, Any] = {}


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    """Central difference along time, tolerating an episode too short for one.

    ``np.gradient`` needs two samples; a one-frame episode has no rate to
    report, and zero is the honest answer rather than an exception in a
    reviewer's face.
    """
    array = np.asarray(values, dtype=float)
    if array.shape[0] < 2:
        return np.zeros_like(array)
    return np.gradient(array, dt, axis=0)


def joint_derivatives(state: np.ndarray, fps: float) -> "tuple[np.ndarray, np.ndarray]":
    """``(velocity, acceleration)`` of every state channel, both ``(N, 12)``.

    Units follow ``JOINT_UNITS`` per channel and its per-second square for the
    acceleration. Pure.
    """
    array = np.asarray(state, dtype=float)
    if array.ndim != 2 or array.shape[1] != STATE_DOF:
        raise ValueError(f"state must have shape (N, {STATE_DOF}), got {array.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt = 1.0 / float(fps)
    velocity = _gradient(array, dt)
    return velocity, _gradient(velocity, dt)


def _kinematic_model(urdf_path: str):
    """``(model, data, eef frame ids, inverse base transforms)`` for the dual arm.

    The two gripper joints are reduced out, leaving a ten-degree configuration
    ordered left five then right five -- exactly the body-joint columns of
    ``observation.state``, in radians.
    """
    cached = _KINEMATICS.get(urdf_path)
    if cached is not None:
        return cached
    import pinocchio as pin  # type: ignore[import]

    from common.configs import (
        LEFT_END_EFFECTOR_FRAME_NAME,
        RIGHT_END_EFFECTOR_FRAME_NAME,
    )
    from common.recording.sidecar import compute_world_base_transforms

    full = pin.buildModelFromUrdf(urdf_path)
    gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
    model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))
    if model.nq != BODY_DOF * len(SIDES):
        raise ValueError(
            f"{urdf_path} reduces to {model.nq} joints, expected "
            f"{BODY_DOF * len(SIDES)}"
        )
    eef = {
        "left": model.getFrameId(LEFT_END_EFFECTOR_FRAME_NAME),
        "right": model.getFrameId(RIGHT_END_EFFECTOR_FRAME_NAME),
    }
    base_inv = {
        side: np.linalg.inv(tf)
        for side, tf in compute_world_base_transforms(urdf_path).items()
    }
    cached = (model, model.createData(), eef, base_inv)
    _KINEMATICS[urdf_path] = cached
    return cached


def body_configuration(state: np.ndarray) -> np.ndarray:
    """The ``(N, 10)`` joint configuration in RADIANS, gripper columns dropped.

    ``observation.state`` interleaves each side's gripper fraction after its five
    body joints, and the gripper is not part of the arm's kinematic chain to the
    tool frame. Pure.
    """
    array = np.asarray(state, dtype=float)
    if array.ndim != 2 or array.shape[1] != STATE_DOF:
        raise ValueError(f"state must have shape (N, {STATE_DOF}), got {array.shape}")
    columns = [
        array[:, s * (BODY_DOF + 1) : s * (BODY_DOF + 1) + BODY_DOF]
        for s in range(len(SIDES))
    ]
    return np.deg2rad(np.concatenate(columns, axis=1))


def ee_trajectory(
    state: np.ndarray, urdf_path: "str | None" = None
) -> "dict[str, tuple[np.ndarray, np.ndarray]]":
    """Per side, the ``(position (N,3), rotation (N,3,3))`` of the tool frame.

    Expressed in that arm's own base frame, matching the convention the recorder
    uses for ``ee_pose``, so a pose derived here and a pose recorded there are
    the same quantity.
    """
    import pinocchio as pin  # type: ignore[import]

    if urdf_path is None:
        from common.configs import DUAL_URDF_PATH

        urdf_path = str(DUAL_URDF_PATH)
    model, data, eef, base_inv = _kinematic_model(urdf_path)
    q = body_configuration(state)
    n = len(q)
    out = {
        side: (np.empty((n, 3), dtype=float), np.empty((n, 3, 3), dtype=float))
        for side in eef
    }
    for k in range(n):
        pin.framesForwardKinematics(model, data, q[k])
        for side, fid in eef.items():
            world = np.asarray(data.oMf[fid].homogeneous, dtype=float)
            local = base_inv[side] @ world
            out[side][0][k] = local[:3, 3]
            out[side][1][k] = local[:3, :3]
    return out


def ee_derivatives(
    position: np.ndarray, rotation: np.ndarray, fps: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """``(|v|, |a|, |w|, |alpha|)`` of a pose trajectory, in SI. Pure.

    Linear speed and acceleration in m/s and m/s^2; angular speed and
    acceleration in rad/s and rad/s^2.

    The angular pair is the rotational analogue of the linear central
    difference: the rotation BETWEEN two samples is a group element, not a
    difference, so it is taken through the logarithm and only then divided by the
    interval. Angular acceleration differentiates that VECTOR before taking its
    magnitude -- the magnitude of a derivative, not the derivative of a
    magnitude, which would report zero for a turn that changes axis at constant
    speed.

    All four magnitudes are invariant to the fixed frame the poses are expressed
    in, so it does not matter that the caller works in each arm's base frame.
    """
    import pinocchio as pin  # type: ignore[import]

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt = 1.0 / float(fps)
    pos = np.asarray(position, dtype=float)
    rot = np.asarray(rotation, dtype=float)
    linear = _gradient(pos, dt)
    lin_speed = np.linalg.norm(linear, axis=1)
    lin_acc = np.linalg.norm(_gradient(linear, dt), axis=1)

    n = len(rot)
    omega = np.zeros((n, 3), dtype=float)
    for k in range(n):
        lo, hi = max(k - 1, 0), min(k + 1, n - 1)
        if hi == lo:
            continue
        omega[k] = pin.log3(rot[lo].T @ rot[hi]) / ((hi - lo) * dt)
    ang_speed = np.linalg.norm(omega, axis=1)
    ang_acc = np.linalg.norm(_gradient(omega, dt), axis=1)
    return lin_speed, lin_acc, ang_speed, ang_acc


def episode_motion(
    state: np.ndarray, fps: float, urdf_path: "str | None" = None
) -> "dict[str, Any]":
    """Everything the review view draws, in the units it displays.

    Joint rates are per ``JOINT_UNITS``; end-effector rates are m/s, m/s^2,
    deg/s and deg/s^2 -- degrees for the angular pair so a reviewer reads one
    angular convention across the whole pane.

    ``ee`` is ``None``, with the reason in ``ee_error``, whenever the kinematic
    model cannot be built or evaluated. That is deliberately not an exception:
    the joint rates are still worth showing, and the episode is still worth
    playing, on a machine where the URDF or pinocchio is unavailable.
    """
    velocity, acceleration = joint_derivatives(state, fps)
    result: dict[str, Any] = {
        "joint_vel": velocity,
        "joint_acc": acceleration,
        "joint_units": list(JOINT_UNITS),
        "ee": None,
        "ee_error": None,
    }
    try:
        trajectory = ee_trajectory(state, urdf_path)
        ee: dict[str, dict[str, np.ndarray]] = {}
        for side, (pos, rot) in trajectory.items():
            v, a, w, alpha = ee_derivatives(pos, rot, fps)
            ee[side] = {
                "v": v,
                "a": a,
                "w": np.rad2deg(w),
                "alpha": np.rad2deg(alpha),
            }
        result["ee"] = ee
    except Exception as exc:  # noqa: BLE001
        result["ee_error"] = f"{type(exc).__name__}: {exc}"
    return result
