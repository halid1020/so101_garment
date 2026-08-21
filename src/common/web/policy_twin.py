"""The chunk a policy just returned, drawn as the rig's own twin.

Twelve numbers are not a pose. A returned chunk is thirty-two of them and an
operator cannot read a grasp out of a table -- but they can see one, and the rig
already has a geometrically faithful model of itself (``sim_twin``), so the plan
can simply be shown as the arms that would execute it.

This is a PREVIEW, not a simulation: joint angles are written straight into
``qpos`` and the scene is re-solved with ``mj_forward``. Nothing is stepped, no
contact is computed, nothing is grasped. The question being answered is "where
does this plan put the arms", and physics would only add a second opinion about
the answer -- one the real arms are already giving.

Rendering is offscreen (``MUJOCO_GL=egl``) and single-threaded behind a lock:
one MuJoCo renderer is not safe to share, and the control loop that must not
miss its 30 Hz tick is never the thread that draws.
"""

from __future__ import annotations

import threading

import numpy as np

#: The framing sim_benchmark's GIFs use -- both arms and the board, from the
#: front-left. Proven to show the rig; kept identical so views are comparable.
CAMERA_LOOKAT = (0.25, 0.0, 0.06)
CAMERA_DISTANCE = 0.95
CAMERA_AZIMUTH = 160.0
CAMERA_ELEVATION = -30.0


class TwinPreview:
    """One twin, one renderer, one lock. Feed it 12-D actions, get RGB frames."""

    def __init__(self, width: int = 480, height: int = 360) -> None:
        import mujoco

        from sim_benchmark.constants import GRIPPER_JOINTS
        from sim_twin.scene import TwinSim

        self._mujoco = mujoco
        self._lock = threading.Lock()
        # Contacts cost time and answer nothing here: this is forward kinematics
        # with a camera, so the cheap collision-free twin is the right one.
        self.twin = TwinSim(all_collisions=False)
        self.model = self.twin.model
        self.data = self.twin.data
        self.twin.reset()

        self._grip_qpos = {}
        for name in GRIPPER_JOINTS:
            side = name.split("_")[0]
            joint = self.model.joint(name)
            lo, hi = (float(v) for v in joint.range)
            self._grip_qpos[side] = (int(joint.qposadr[0]), lo, hi)

        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = list(CAMERA_LOOKAT)
        self.camera.distance = CAMERA_DISTANCE
        self.camera.azimuth = CAMERA_AZIMUTH
        self.camera.elevation = CAMERA_ELEVATION

    def render(self, action12) -> np.ndarray:
        """A 12-D action (URDF degrees + gripper fractions) -> one RGB frame."""
        # The same split the simulated evaluation uses, so a chunk previewed
        # here and a chunk rolled out in the twin cannot mean different things.
        from tool.eval_sim_policy import decode_action

        q_rad, grip = decode_action(np.asarray(action12, dtype=float))
        with self._lock:
            self.data.qpos[self.twin.arm_qpos_idx] = q_rad
            for side, frac in grip.items():
                adr, lo, hi = self._grip_qpos[side]
                self.data.qpos[adr] = lo + float(np.clip(frac, 0.0, 1.0)) * (hi - lo)
            self._mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, self.camera)
            return np.asarray(self.renderer.render()).copy()

    def close(self) -> None:
        with self._lock:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001 - closing twice must not raise
                pass
