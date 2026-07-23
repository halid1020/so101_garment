"""Rig-true benchmark scene: the IK methods driven against the digital twin.

``DualArmSim`` (scene.py) is an abstract flat scene — two arms on a slab at
z=0, collisions off. ``RigBenchSim`` instead runs the methods against the
full printed-rig twin (``sim_twin.scene.TwinSim``, ``all_collisions=True``):
arms raised on the printed board and adapter plates, the perforated board,
the camera tower, and every arm--arm / arm--rig / arm--table contact live.
This is what the user's TODO "use the platform rig setup" and "enable all
the collisions" ask for.

Frame bridge (verified exact). The twin mounts the arms on the board+adapter
stack, so every EE sits a fixed height above where the flat IK model places
it. Measured at the neutral pose the offset is a pure +36.4 mm z-translation,
identical for both arms, with zero rotation — because the kinematic chain is
the same and only the base's world placement differs. Joint angles are
frame-agnostic, so the flat-URDF IK the methods already solve produces
exactly the right joint commands for the twin; we only have to express the
twin's measured EE back in the flat IK frame to compare against the targets.
``RigBenchSim`` therefore subtracts the per-side offset in ``eef_pose`` /
``fk_eef_pose`` (so the benchmark loop sees the same frame as the plain
scene, and tracking errors stay comparable) and adds it back in
``set_target_markers`` (so the on-screen spheres sit in twin world coords).
The rig's real geometry and contacts thus affect the dynamics while the
error metric stays in the arm's own frame.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim_benchmark.constants import ARM_JOINTS as _ARM_JOINTS
from sim_benchmark.constants import SIDES
from sim_benchmark.scene import build_spec
from sim_twin.scene import TwinSim


class RigBenchSim:
    """DualArmSim-compatible wrapper over the collision-enabled twin.

    Exposes the same interface the benchmark runners use
    (neutral_q/reset/set_arm_targets/set_target_markers/step/eef_pose/
    arm_q/fk_eef_pos/fk_eef_pose), with EE poses expressed in the flat IK
    frame via the constant per-side world offset described in the module
    docstring.
    """

    def __init__(self) -> None:
        self.twin = TwinSim(all_collisions=True)
        self.model = self.twin.model
        self.data = self.twin.data
        # Separate flat IK model: dls/mink build their kinematics from this,
        # so they solve in the same flat frame as the pinocchio methods and
        # the targets (see the module docstring's frame bridge).
        self.ik_model = build_spec().compile()
        self.arm_ctrl_idx = self.twin.arm_ctrl_idx
        self.gripper_ctrl_idx = self.twin.gripper_ctrl_idx
        self.arm_qpos_idx = self.twin.arm_qpos_idx
        # Per-side flat-frame -> twin-frame world offset, fixed at neutral:
        # (twin EE at neutral) - (flat IK model EE at neutral). The chain is
        # identical, so this is a constant translation for every pose.
        self.twin.reset()
        flat_neutral = self._fk_flat(self.twin.neutral_q())
        self._world_offset = {
            side: self.twin.eef_pose(side)[0] - flat_neutral[side][:3, 3]
            for side in SIDES
        }

    def neutral_q(self) -> np.ndarray:
        return self.twin.neutral_q()

    def reset(self, q: np.ndarray | None = None) -> None:
        self.twin.reset(q)

    def set_arm_targets(self, q: np.ndarray) -> None:
        self.twin.set_arm_targets(q)

    def set_target_markers(
        self, targets: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Move the mocap spheres, mapping flat-frame targets into twin world.

        Also writes the target attitude into ``mocap_quat`` so the target
        triads (viz.py) show the commanded orientation, not just position.
        """
        for side, (pos, rot) in targets.items():
            mid = self.twin.target_mocap_id[side]
            self.data.mocap_pos[mid] = pos + self._world_offset[side]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).reshape(9))
            self.data.mocap_quat[mid] = quat

    def step(self, n_substeps: int) -> None:
        self.twin.step(n_substeps)

    def eef_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Measured EE pose in the flat IK frame (twin measurement - offset)."""
        pos, rot = self.twin.eef_pose(side)
        return pos - self._world_offset[side], rot

    def arm_q(self) -> np.ndarray:
        return self.twin.arm_q()

    def _fk_flat(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Full 4x4 flat-IK-frame EE poses for a configuration (scratch data).

        Uses the flat IK model, so this is already in the frame the targets
        live in — no offset correction needed.
        """
        if not hasattr(self, "_fk_data"):
            self._fk_data = mujoco.MjData(self.ik_model)
            self._fk_qpos_idx = np.array(
                [self.ik_model.joint(j).qposadr[0] for j in _ARM_JOINTS]
            )
            self._fk_site_id = {
                side: self.ik_model.site(f"{side}_eef_site").id for side in SIDES
            }
        self._fk_data.qpos[:] = 0.0
        self._fk_data.qpos[self._fk_qpos_idx] = q
        mujoco.mj_kinematics(self.ik_model, self._fk_data)
        poses = {}
        for side in SIDES:
            sid = self._fk_site_id[side]
            pose = np.eye(4)
            pose[:3, :3] = self._fk_data.site_xmat[sid].reshape(3, 3)
            pose[:3, 3] = self._fk_data.site_xpos[sid]
            poses[side] = pose
        return poses

    def fk_eef_pose(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Flat-IK-frame EE poses for a configuration (pure IK, no physics)."""
        return self._fk_flat(q)

    def fk_eef_pos(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return {side: pose[:3, 3] for side, pose in self.fk_eef_pose(q).items()}


def make_sim(scene: str):
    """Build the executing scene: 'rig' (collision-enabled twin) or 'plain'.

    Falls back to the plain scene with a warning if the twin assets are
    missing, so a fresh checkout can still run the cheap smoke sweep.
    """
    if scene == "rig":
        try:
            return RigBenchSim()
        except FileNotFoundError as exc:
            print(f"⚠️  Twin assets missing ({exc}); falling back to --scene plain")
    from sim_benchmark.scene import DualArmSim

    return DualArmSim()
