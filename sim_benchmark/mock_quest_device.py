"""Device-level Meta Quest mock: same API surface as MetaQuestReader.

Lets the full production teleop stack (One-Euro filtering, grip clutch,
handle-axis calibration, armplane orientation mapping, IK thread) run
without a physical headset: after a short idle it "presses" both grips
and moves both hands in horizontal circles.

Only the methods the pipeline actually calls are implemented:
``get_hand_controller_transform_ros``, ``get_grip_value``,
``get_trigger_value``, ``get_button_state``, ``stop``.
"""

from __future__ import annotations

import time

import numpy as np

GRIP_DELAY_S = 2.0  # idle time before the mock presses both grips
CIRCLE_RADIUS = 0.04
CIRCLE_PERIOD_S = 6.0
# Nominal hand rest positions in the ROS head frame (x fwd, y left, z up).
HAND_REST = {
    "left": np.array([0.35, 0.15, -0.25]),
    "right": np.array([0.35, -0.15, -0.25]),
}


class MockQuestReader:
    """Scripted stand-in for meta_quest_teleop.reader.MetaQuestReader."""

    def __init__(self) -> None:
        self._t0 = time.time()
        print(
            "🎭 Mock Quest device: grips press at "
            f"t={GRIP_DELAY_S:.0f}s, hands draw {CIRCLE_RADIUS*100:.0f} cm "
            "circles (identity orientation = handles pointing down)."
        )

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def get_hand_controller_transform_ros(self, hand: str) -> np.ndarray:
        tf = np.eye(4)
        pos = HAND_REST[hand].copy()
        t_active = self._elapsed() - GRIP_DELAY_S
        if t_active > 0:
            phase = 2 * np.pi * t_active / CIRCLE_PERIOD_S
            ramp = min(t_active / (0.25 * CIRCLE_PERIOD_S), 1.0)
            dy = CIRCLE_RADIUS * np.sin(phase)
            pos[0] += ramp * CIRCLE_RADIUS * (np.cos(phase) - 1.0)
            pos[1] += ramp * (dy if hand == "right" else -dy)
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
