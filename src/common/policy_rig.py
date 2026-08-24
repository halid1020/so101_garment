"""What a rollout drives: the arms on the bench, or the same arms in the twin.

A rollout loop needs remarkably little from a robot -- an observation to send,
somewhere to put an action, and a way to stop safely. Naming that little makes
the digital twin a drop-in for the rig, so a chunking strategy can be measured a
hundred times in simulation and then run on hardware through the same client,
the same wire and the same splice.

    ``observe() -> (state12, {camera: HWC uint8})``
    ``command(action12)``
    ``shutdown()``
    ``cameras``  -- the names ``observe`` will produce

The hardware implementation stays in ``tool/run_policy_real.py``, where the
buses, the torque and the confirmation prompt already live and where they
belong: this module deliberately imports nothing that can move a real motor.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Rig(Protocol):
    """The robot, as much of it as a rollout loop needs to know."""

    cameras: "list[str]"

    def observe(self) -> "tuple[np.ndarray, dict]":
        """This tick's 12-D state and one frame per camera."""

    def command(self, action12: np.ndarray) -> None:
        """Execute a 12-D action, in the units the dataset recorded."""

    def shutdown(self) -> None:
        """Stop safely. Called on every exit path, including a crash."""


class TwinRig:
    """The MuJoCo twin behind the same three methods as the bench.

    One ``command`` is exactly 1/30 s of simulated time, so a tick here and a
    recorded frame mean the same thing. Nothing reads the wall clock: the world
    advances only when ``command`` is called, which is what lets a rollout
    inject a chosen inference delay and get the same answer every time.

    Rendering happens on whichever thread calls ``observe``, and must therefore
    be the control thread only -- ``mujoco.Renderer`` is not thread-safe, and
    the remote client fetches its chunks on a background thread.
    """

    def __init__(
        self,
        env,
        cameras: "list[str]",
        camera_wh: "tuple[int, int]" = (640, 480),
        camera_map: "dict[str, str] | None" = None,
    ) -> None:
        from tool.eval_sim_policy import decode_action

        self.env = env
        self.camera_wh = (int(camera_wh[0]), int(camera_wh[1]))
        self._decode = decode_action
        #: Applied to the names this rig REPORTS, so a caller that asks what
        #: cameras exist is told what the policy will be sent. The client does
        #: the same rename on the frames themselves.
        self.camera_map = dict(camera_map or {})
        self.cameras = sorted(self.camera_map.get(name, name) for name in cameras)
        self.ticks = 0
        self.last_action: "np.ndarray | None" = None

    def observe(self) -> "tuple[np.ndarray, dict]":
        return self.env.observe(self.camera_wh)

    def command(self, action12: np.ndarray) -> None:
        action = np.asarray(action12, dtype=float)
        q_rad, grip = self._decode(action)
        self.env.tick(q_rad, grip)
        self.last_action = action
        self.ticks += 1

    def hold(self) -> None:
        """Advance time with no new goal -- the servos keep their last target.

        The twin's actuators hold their commanded position when nothing new is
        written, so a starved queue costs the same here as it does on the bench:
        the arms stop where they are, and the world keeps moving.
        """
        if self.last_action is None:
            state, _ = self.env.observe(self.camera_wh)
            self.last_action = np.asarray(state, dtype=float)
        self.command(self.last_action)

    def shutdown(self) -> None:
        """Nothing to release: no torque, no bus, no camera thread."""
