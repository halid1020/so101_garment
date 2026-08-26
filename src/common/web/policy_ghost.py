"""The chunk a policy just returned, drawn as two ghosts you can walk around.

Twelve numbers are not a pose, and one camera angle is not a view. A returned
chunk is thirty-two poses, and the question an operator has about them -- does
that gripper clear the block -- is a question about depth, which no single fixed
render can answer. So the plan is shown in the rig's own URDF, in the viewer's
browser, and the viewer turns it.

TWO GHOSTS. The arms as they are MEASURED are drawn in transparent blue; where
the action being inspected would send them, in transparent orange. The gap
between the two is the pending motion, which is the thing worth seeing and which
neither pose shows on its own.

This is a PREVIEW, not a simulation: joint angles are written straight into the
URDF and the scene is re-solved by forward kinematics. Nothing is stepped, no
contact is computed, nothing is grasped -- and the scene holds the arms and a
ground grid, nothing else, so rig geometry stays described in exactly one place.

Rendering happens in the browser, on the viewer's GPU. This process only pushes
two joint vectors over a websocket, which is why the control loop can afford to
have a view at all.
"""

from __future__ import annotations

import threading

import numpy as np

#: The arms as they are measured NOW: cool, and the fainter of the two, because
#: it is the reference rather than the news.
NOW_RGBA = (0.25, 0.50, 1.00, 0.45)

#: Where the plan sends them: warm, and more solid, because it is what the
#: viewer is being asked to judge.
PLAN_RGBA = (1.00, 0.45, 0.10, 0.60)

#: Gripper travel of the SO-101 URDF, in radians. Read from the model at build
#: time; these are only the fallback for the pure helper's default arguments.
GRIP_LO, GRIP_HI = -0.174533, 1.745330


def action_to_urdf_cfg(
    action12, grip_lo: float = GRIP_LO, grip_hi: float = GRIP_HI
) -> np.ndarray:
    """A 12-D policy action -> the URDF's 12 actuated joints, in radians. Pure.

    The two layouts are the same by construction -- the URDF's
    ``actuated_joint_names`` is ``[left_shoulder_pan .. left_wrist_roll,
    left_gripper, right_...]`` and the recorder's action is ``[left5 deg,
    left_grip, right5 deg, right_grip]`` -- so this is a unit change on ten
    channels and a range map on two. The split is
    :func:`tool.eval_sim_policy.decode_action`, the same one the simulated
    evaluation uses, so a chunk previewed here and a chunk rolled out cannot
    mean different things.
    """
    from tool.eval_sim_policy import decode_action

    q_rad, grip = decode_action(np.asarray(action12, dtype=float))
    cfg = np.zeros(12, dtype=float)
    cfg[0:5] = q_rad[0:5]
    cfg[6:11] = q_rad[5:10]
    for i, side in ((5, "left"), (11, "right")):
        cfg[i] = grip_lo + float(np.clip(grip[side], 0.0, 1.0)) * (grip_hi - grip_lo)
    return cfg


class GhostPair:
    """One viser scene, two SO-101 ghosts, behind one lock.

    Serialised because the control loop and the page both reach it, and viser's
    scene handles are not documented as thread-safe. Pushing two twelve-element
    vectors is cheap enough that the lock is never held long.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8768,
        urdf_path: "str | None" = None,
    ) -> None:
        from common.configs import DUAL_URDF_PATH
        from common.visualizer_core import RobotVisualizerCore

        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._core = RobotVisualizerCore(
            str(urdf_path or DUAL_URDF_PATH),
            host=host,
            port=self.port,
            robot_color=NOW_RGBA,
            ghost_color=PLAN_RGBA,
        )
        # The page owns the transport and the numbers; viser's own panel would
        # only offer a second, disagreeing set of controls.
        self._core.server.gui.configure_theme(
            control_layout="collapsible",
            show_logo=False,
            show_share_button=False,
            dark_mode=True,
        )
        limits = self._gripper_limits()
        self.grip_lo, self.grip_hi = limits
        # Both start at the URDF's neutral pose rather than nowhere, so the
        # scene is a rig from the first frame the browser draws.
        self.show(None, None)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def show(self, now12, plan12) -> None:
        """Blue to the measured pose, orange to the planned one.

        Either may be ``None`` -- before the first tick there is no measured
        pose, and before the first chunk there is no plan -- in which case that
        ghost is hidden rather than left showing a stale pose it no longer
        stands for.
        """
        with self._lock:
            for value, setter, shower in (
                (now12, self._core.update_robot_pose, self._show_now),
                (plan12, self._core.update_ghost_robot_pose, self._show_plan),
            ):
                ok = value is not None and len(value) == 12
                shower(bool(ok))
                if ok:
                    setter(action_to_urdf_cfg(value, self.grip_lo, self.grip_hi))

    def stop(self) -> None:
        try:
            self._core.stop()
        except Exception:  # noqa: BLE001 - stopping twice must not raise
            pass

    # -- internals -----------------------------------------------------
    def _show_now(self, flag: bool) -> None:
        self._core.urdf_vis.show_visual = flag

    def _show_plan(self, flag: bool) -> None:
        self._core.ghost_robot_urdf.show_visual = flag

    def _gripper_limits(self) -> "tuple[float, float]":
        """The gripper joint's travel, from the model rather than from memory."""
        try:
            limits = self._core.urdf_vis.get_actuated_joint_limits()
            lo, hi = limits["left_gripper"]
            return float(lo), float(hi)
        except Exception:  # noqa: BLE001 - a URDF without limits is not fatal
            return GRIP_LO, GRIP_HI
