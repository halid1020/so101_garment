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

TWO POSES AT ONCE. :meth:`TwinPreview.render_pair` draws the arms as they ARE,
in transparent blue, and where an action would send them, in transparent orange,
in the same scene. The gap between the two ghosts is the pending motion, which
is the thing an operator actually wants to see and which neither pose shows on
its own. It works by rendering the scene once and then calling ``mjv_addGeoms``
for the second pose with ``mjCAT_DYNAMIC`` only -- so the arms and the payload
are drawn twice but the board, the table and the lights are not.

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

#: The arms as they are measured NOW: cool, and the fainter of the two, because
#: it is the reference rather than the news.
NOW_RGBA = (0.25, 0.50, 1.00, 0.35)

#: Where the plan sends them: warm, and more solid, because it is what the
#: viewer is being asked to judge.
PLAN_RGBA = (1.00, 0.45, 0.10, 0.55)

#: A pose drawn on its own is not competing with anything, so it need not be
#: see-through.
SOLO_ALPHA = 0.9


class TwinPreview:
    """One twin, one renderer, one lock. Feed it 12-D actions, get RGB frames.

    Rendering is serialised on ``self._lock``: one MuJoCo renderer cannot be
    shared, and the scene is mutated in place between the two poses.
    """

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
        # Needed only by mjv_addGeoms, which will not accept nulls; the defaults
        # are what update_scene already uses.
        self._vopt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = list(CAMERA_LOOKAT)
        self.camera.distance = CAMERA_DISTANCE
        self.camera.azimuth = CAMERA_AZIMUTH
        self.camera.elevation = CAMERA_ELEVATION

    def render_pair(self, now12, plan12) -> "np.ndarray | None":
        """Measured pose in blue, planned pose in orange, one RGB frame.

        Either may be ``None`` -- before the first tick there is no measured
        pose, and before the first chunk there is no plan -- in which case only
        the other is drawn, solid, since nothing is being compared to it.
        ``None`` back means there was nothing at all to draw.
        """
        poses = [p for p in (now12, plan12) if p is not None and len(p) == 12]
        if not poses:
            return None
        solo = len(poses) == 1
        with self._lock:
            self._pose(poses[0])
            self.renderer.update_scene(self.data, self.camera)
            first = self.renderer.scene.ngeom
            colour = NOW_RGBA if (now12 is not None and len(now12) == 12) else PLAN_RGBA
            self._tint(0, first, colour, solid=solo)
            if not solo:
                self._pose(poses[1])
                # Dynamic only: the board and the table have not moved, and a
                # second transparent copy of them would fog the whole picture.
                self._mujoco.mjv_addGeoms(
                    self.model,
                    self.data,
                    self._vopt,
                    self._pert,
                    int(self._mujoco.mjtCatBit.mjCAT_DYNAMIC),
                    self.renderer.scene,
                )
                self._tint(first, self.renderer.scene.ngeom, PLAN_RGBA)
            return np.asarray(self.renderer.render()).copy()

    def _tint(self, start: int, stop: int, rgba, solid: bool = False) -> None:
        """Recolour the MOVING geoms in ``[start, stop)``. Caller holds the lock.

        Static geoms are left alone: the table and the board are the scene both
        ghosts stand in, and tinting them would make the picture unreadable.
        """
        dynamic = int(self._mujoco.mjtCatBit.mjCAT_DYNAMIC)
        scene = self.renderer.scene
        for i in range(start, min(stop, scene.ngeom)):
            geom = scene.geoms[i]
            if int(geom.category) != dynamic:
                continue
            geom.rgba[:3] = rgba[:3]
            geom.rgba[3] = SOLO_ALPHA if solid else rgba[3]

    def _pose(self, action12) -> None:
        """Write a 12-D action into qpos and re-solve. Caller holds the lock."""
        # The same split the simulated evaluation uses, so a chunk previewed
        # here and a chunk rolled out in the twin cannot mean different things.
        from tool.eval_sim_policy import decode_action

        q_rad, grip = decode_action(np.asarray(action12, dtype=float))
        self.data.qpos[self.twin.arm_qpos_idx] = q_rad
        for side, frac in grip.items():
            adr, lo, hi = self._grip_qpos[side]
            self.data.qpos[adr] = lo + float(np.clip(frac, 0.0, 1.0)) * (hi - lo)
        self._mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        with self._lock:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001 - closing twice must not raise
                pass
