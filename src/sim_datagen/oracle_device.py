"""Scripted-operator Quest device that drives the full teleop pipeline.

The direct oracle feeds EE targets straight into an IK method; this device
instead pretends to be a human operator so the demonstrations flow through the
*entire* production teleop stack (One-Euro filter, grip clutch, operator-frame
retargeting, translation scaling, ``dual_ik_solver_thread``, rate limiter),
wired exactly as ``tool/quest_sim_teleop.py``.

It implements the MetaQuestReader API surface the pipeline calls
(``get_hand_controller_transform_ros``, ``get_grip_value``,
``get_trigger_value``, ``get_joystick_value``, ``get_button_state``, ``stop``)
and holds both grips for the whole episode. To place a hand it INVERTS the
retargeting mapping.

Inverse-mapping derivation
--------------------------
The pipeline maps an operator-frame hand position to an EE target by
(``common.threads.dual_ik_solver`` + ``common.utils``)::

    target = E0 + (h_op(t) - h_op(0)) * translation_scale

where ``E0`` is the EE pose at grip engage (neutral FK) and ``h_op`` is the
hand position in the operator control frame,
``h_op = R_op^T (h_raw - origin)`` with ``(R_op, origin) =
compute_operator_frame(h_raw_left, h_raw_right)``. Solving for the raw reader
hand position that realises a desired EE position ``p(t)`` gives::

    h_raw(t) = H0 + R_op @ (p(t) - E0) / translation_scale

(the operator-frame origin cancels because only the delta from engage is
used). We choose the two rest hand positions level and separated purely along
the reader y-axis, so ``compute_operator_frame`` returns ``R_op = I`` and the
mapping reduces to ``h_raw(t) = H0 + (p(t) - E0) / translation_scale`` — the
operator frame is robot-aligned by construction, so a reader-frame delta is a
robot-frame delta. Hands keep identity orientation (handles pointing down);
grasps are position-only (the 5-DoF arm's soft-orientation IK).

The trigger squeezes the gripper: ``trigger(t) = 1 - grip_fraction(t)`` (the
collection loop maps trigger -> gripper control exactly like the sim tool).
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from common.utils import compute_operator_frame
from sim_benchmark.constants import SIDES

# Rest hand positions in the reader "ros head frame" (x forward, y left, z up),
# level and separated along y so the derived operator frame is the identity.
HAND_REST = {
    "left": np.array([0.35, 0.15, -0.25]),
    "right": np.array([0.35, -0.15, -0.25]),
}
# Idle time before the operator "presses" both grips: lets the arms anchor and
# the One-Euro filter settle on the stationary rest pose before engagement.
ENGAGE_DELAY_S = 1.0


class _TaskScript(Protocol):
    """The task-script surface the device wraps."""

    duration: float

    def targets(self, t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ...

    def grip_fractions(self, t: float) -> dict[str, float]:
        ...


class OracleQuestDevice:
    """MetaQuestReader-compatible scripted operator for one episode at a time."""

    def __init__(
        self,
        neutral_ik_poses: dict[str, tuple[np.ndarray, np.ndarray]],
        translation_scale: float,
    ) -> None:
        self._e0 = {side: neutral_ik_poses[side][0].copy() for side in SIDES}
        self._ts = float(translation_scale)
        # Operator frame from the (fixed) rest hand poses — identity by design.
        left_tf = np.eye(4)
        left_tf[:3, 3] = HAND_REST["left"]
        right_tf = np.eye(4)
        right_tf[:3, 3] = HAND_REST["right"]
        self._r_op, _ = compute_operator_frame(left_tf, right_tf)
        self._lock = threading.Lock()
        self._script: _TaskScript | None = None
        self._t0 = time.time()
        self._stopped = False

    # ------------------------------------------------------------------
    def load_episode(self, script: _TaskScript) -> None:
        """Arm the device with a task script; grips engage after ENGAGE_DELAY_S."""
        with self._lock:
            self._script = script
            self._t0 = time.time()

    def disengage(self) -> None:
        """Release the grips (deactivates teleop so the clutch recalibrates)."""
        with self._lock:
            self._script = None

    def engaged(self) -> bool:
        with self._lock:
            return self._script is not None and self._elapsed() >= ENGAGE_DELAY_S

    def episode_finished(self) -> bool:
        with self._lock:
            if self._script is None:
                return False
            return self._elapsed() >= ENGAGE_DELAY_S + self._script.duration

    # ------------------------------------------------------------------
    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _script_time(self) -> float:
        return max(0.0, self._elapsed() - ENGAGE_DELAY_S)

    def script_time(self) -> float:
        """Public script clock (s since grip engage) for a closed-loop servo."""
        with self._lock:
            return self._script_time()

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray:
        tf = np.eye(4)  # identity orientation: handles pointing down
        with self._lock:
            pos = HAND_REST[hand].copy()
            if self._script is not None:
                desired_ee = self._script.targets(self._script_time())[hand][0]
                pos = pos + self._r_op @ (desired_ee - self._e0[hand]) / self._ts
        tf[:3, 3] = pos
        return tf

    def get_grip_value(self, hand: str) -> float:
        with self._lock:
            engaged = self._script is not None and self._elapsed() >= ENGAGE_DELAY_S
        return 1.0 if engaged else 0.0

    def get_trigger_value(self, hand: str) -> float:
        with self._lock:
            if self._script is None or self._elapsed() < ENGAGE_DELAY_S:
                return 0.0
            frac = self._script.grip_fractions(self._script_time())[hand]
        return float(np.clip(1.0 - frac, 0.0, 1.0))

    def get_joystick_value(self, hand: str) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_button_state(self, button: str) -> bool:
        return False

    def stop(self) -> None:
        self._stopped = True
