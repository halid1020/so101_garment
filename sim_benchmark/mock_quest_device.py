"""Device-level Meta Quest mock: same API surface as MetaQuestReader.

Lets the full production teleop stack (One-Euro filtering, grip clutch,
handle-axis calibration, armplane orientation mapping, IK thread) run
without a physical headset: after a short idle it "presses" both grips
and moves both hands in a scripted pattern.

Patterns (``MockQuestReader(pattern=...)`` / ``--mock-pattern``):

- ``circle``    — 4 cm horizontal circles (the original smoke test).
- ``wrist``     — hands hold position while rotating: a 1 Hz roll
                  oscillation about the (downward) handle axis plus a
                  slower flex rock, to exercise wrist agility end-to-end.
- ``excursion`` — an exaggerated forward stroke that drags the targets
                  well past the arms' reach and back, to exercise the
                  out-of-envelope policies end-to-end.

Only the methods the pipeline actually calls are implemented:
``get_hand_controller_transform_ros``, ``get_grip_value``,
``get_trigger_value``, ``get_button_state``, ``stop``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

GRIP_DELAY_S = 2.0  # idle time before the mock presses both grips
CIRCLE_RADIUS = 0.04
CIRCLE_PERIOD_S = 6.0
WRIST_ROLL_AMP_DEG = 45.0
WRIST_ROLL_FREQ_HZ = 1.0
WRIST_FLEX_AMP_DEG = 25.0
WRIST_FLEX_FREQ_HZ = 0.4
EXCURSION_REACH = 0.45  # m of forward hand travel — far past the arm's reach
EXCURSION_PERIOD_S = 10.0
# Nominal hand rest positions in the ROS head frame (x fwd, y left, z up).
HAND_REST = {
    "left": np.array([0.35, 0.15, -0.25]),
    "right": np.array([0.35, -0.15, -0.25]),
}

MOCK_PATTERNS = ("circle", "wrist", "excursion")


class MockQuestReader:
    """Scripted stand-in for meta_quest_teleop.reader.MetaQuestReader."""

    def __init__(self, pattern: str = "circle") -> None:
        if pattern not in MOCK_PATTERNS:
            raise ValueError(f"Unknown mock pattern {pattern!r}: {MOCK_PATTERNS}")
        self.pattern = pattern
        self._t0 = time.time()
        print(
            f"🎭 Mock Quest device (pattern={pattern}): grips press at "
            f"t={GRIP_DELAY_S:.0f}s (identity orientation = handles "
            "pointing down)."
        )

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray:
        tf = np.eye(4)
        pos = HAND_REST[hand].copy()
        t_active = self._elapsed() - GRIP_DELAY_S
        if t_active > 0:
            if self.pattern == "circle":
                phase = 2 * np.pi * t_active / CIRCLE_PERIOD_S
                ramp = min(t_active / (0.25 * CIRCLE_PERIOD_S), 1.0)
                dy = CIRCLE_RADIUS * np.sin(phase)
                pos[0] += ramp * CIRCLE_RADIUS * (np.cos(phase) - 1.0)
                pos[1] += ramp * (dy if hand == "right" else -dy)
            elif self.pattern == "wrist":
                # Position held; the hand transform rotates. Roll about the
                # handle axis (world z here — handles point down) plus a
                # slower flex rock about the lateral axis.
                ramp = min(t_active / 2.0, 1.0)
                roll = (
                    ramp
                    * np.deg2rad(WRIST_ROLL_AMP_DEG)
                    * np.sin(2 * np.pi * WRIST_ROLL_FREQ_HZ * t_active)
                )
                flex = (
                    ramp
                    * np.deg2rad(WRIST_FLEX_AMP_DEG)
                    * np.sin(2 * np.pi * WRIST_FLEX_FREQ_HZ * t_active)
                )
                if hand == "left":
                    roll = -roll
                tf[:3, :3] = Rotation.from_euler("zy", [roll, flex]).as_matrix()
            else:  # excursion
                # Slow forward push far past reach, hold, and pull back.
                phase = 2 * np.pi * t_active / EXCURSION_PERIOD_S
                s = 0.5 * (1 - np.cos(phase))  # 0 -> 1 -> 0 each period
                ramp = min(t_active / (0.25 * EXCURSION_PERIOD_S), 1.0)
                pos[0] += ramp * s * EXCURSION_REACH
        tf[:3, 3] = pos
        return tf

    def get_grip_value(self, hand: str) -> float:
        return 1.0 if self._elapsed() > GRIP_DELAY_S else 0.0

    def get_trigger_value(self, hand: str) -> float:
        return 0.0

    def get_button_state(self, button: str) -> bool:
        return False

    def stop(self) -> None:
        pass
