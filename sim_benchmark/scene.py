"""Build a MuJoCo scene with the dual SO-101 arms, a table, and EE sites.

The dual-arm URDF from ``src/so101_dual_description`` is patched in-memory
with a ``<mujoco>`` compiler block (absolute meshdir) and loaded via MjSpec,
then decorated with a floor, a table slab, position actuators for every
revolute joint, end-effector sites, and mocap spheres that visualize the
commanded teleop targets.

Robot geoms are made non-colliding: this benchmark verifies teleop tracking,
not contact dynamics, and the raw URDF collision meshes would otherwise
produce spurious self-contacts.
"""

from __future__ import annotations

import re
import tempfile

import mujoco
import numpy as np

from sim_benchmark.constants import (
    ACTUATOR_KP,
    ACTUATOR_KV,
    ARM_JOINTS,
    DESCRIPTION_DIR,
    DUAL_URDF_PATH,
    EE_FRAMES,
    GRIPPER_JOINTS,
    NEUTRAL_ARM_ANGLES_DEG,
    SIDES,
)

TARGET_RGBA = {"left": (0.9, 0.2, 0.2, 0.5), "right": (0.2, 0.4, 0.9, 0.5)}

# Payload cube half-extent (m); resting center height == PAYLOAD_HALF.
PAYLOAD_HALF = 0.011


def patched_urdf_path() -> str:
    """Return a temp URDF path with a MuJoCo compiler block injected."""
    text = DUAL_URDF_PATH.read_text()
    mujoco_block = (
        f'<mujoco><compiler meshdir="{DESCRIPTION_DIR}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>'
    )
    patched = re.sub(r"(<robot[^>]*>)", rf"\1\n  {mujoco_block}", text, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="so101_dual_mjc_", delete=False
    )
    tmp.write(patched)
    tmp.close()
    return tmp.name


def build_spec() -> mujoco.MjSpec:
    """Assemble the benchmark scene as an MjSpec."""
    spec = mujoco.MjSpec.from_file(patched_urdf_path())
    spec.modelname = "so101_dual_teleop_benchmark"

    # Timestep: 2 ms physics, control layer runs at 50 Hz on top.
    spec.option.timestep = 0.002

    # Disable all collisions on imported robot geoms (visual benchmark only).
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0

    # URDF carries no joint dynamics; without damping/armature the position
    # servos oscillate unboundedly. Values match the official SO-ARM100 MJCF
    # (STS3215 bus servos).
    for joint in spec.joints:
        joint.damping[0] = 0.60
        joint.armature = 0.028
        joint.frictionloss = 0.05

    # Lighting and floor.
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], castshadow=False)
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        pos=[0, 0, -0.4],
        rgba=[0.35, 0.35, 0.35, 1.0],
        contype=0,
        conaffinity=0,
    )
    # Table slab whose top surface is the arms' mounting plane (z = 0).
    # Collision group 2: table <-> payload only (robot geoms stay inert).
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.45, 0.02],
        pos=[0.15, 0, -0.02],
        rgba=[0.55, 0.42, 0.30, 1.0],
        contype=2,
        conaffinity=2,
    )

    # Payload cube for the pick-handover-place experiment. Grasping is
    # mocked kinematically (see DualArmSim.attach), so it only needs
    # contact with the table to rest and to land after release.
    payload = spec.worldbody.add_body(name="payload", pos=[0.30, 0.15, PAYLOAD_HALF])
    payload.add_freejoint(name="payload_free")
    payload.add_geom(
        name="payload_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PAYLOAD_HALF] * 3,
        rgba=[1.0, 0.55, 0.1, 1.0],
        contype=2,
        conaffinity=2,
    )

    # End-effector tracking sites.
    for side in SIDES:
        body = spec.body(EE_FRAMES[side])
        body.add_site(
            name=f"{side}_eef_site",
            pos=[0, 0, 0],
            size=[0.008, 0.008, 0.008],
            rgba=[0, 1, 0, 0.6],
        )

    # Mocap spheres visualizing the commanded targets.
    for side in SIDES:
        target = spec.worldbody.add_body(name=f"{side}_target", mocap=True)
        target.add_geom(
            name=f"{side}_target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            rgba=list(TARGET_RGBA[side]),
            contype=0,
            conaffinity=0,
        )

    # Position servos on every revolute joint (arms + grippers).
    for joint_name in (*ARM_JOINTS, *GRIPPER_JOINTS):
        act = spec.add_actuator(
            name=f"act_{joint_name}",
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = ACTUATOR_KP
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0] = 0.0
        act.biasprm[1] = -ACTUATOR_KP
        act.biasprm[2] = -ACTUATOR_KV

    return spec


class DualArmSim:
    """Thin wrapper over the compiled MuJoCo model for the benchmark loop."""

    def __init__(self) -> None:
        self.model = build_spec().compile()
        self.data = mujoco.MjData(self.model)

        self.arm_qpos_idx = np.array(
            [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        )
        self.arm_ctrl_idx = np.array(
            [self.model.actuator(f"act_{j}").id for j in ARM_JOINTS]
        )
        self.gripper_ctrl_idx = np.array(
            [self.model.actuator(f"act_{j}").id for j in GRIPPER_JOINTS]
        )
        self.eef_site_id = {
            side: self.model.site(f"{side}_eef_site").id for side in SIDES
        }
        self.target_mocap_id = {
            side: self.model.body(f"{side}_target").mocapid[0] for side in SIDES
        }

        payload_joint = self.model.joint("payload_free")
        qadr, vadr = payload_joint.qposadr[0], payload_joint.dofadr[0]
        self._payload_pos_sl = slice(qadr, qadr + 3)
        self._payload_quat_sl = slice(qadr + 3, qadr + 7)
        self._payload_vel_sl = slice(vadr, vadr + 6)
        # Mock-grasp state: which arm holds the payload, and the payload
        # pose captured in that arm's EE-site frame at attach time.
        self.attached_side: str | None = None
        self._attach_offset_pos = np.zeros(3)
        self._attach_offset_rot = np.eye(3)

    def neutral_q(self) -> np.ndarray:
        """Neutral arm configuration (radians, ARM_JOINTS order)."""
        return np.deg2rad(np.array(NEUTRAL_ARM_ANGLES_DEG * 2))

    def reset(self, q: np.ndarray | None = None) -> None:
        """Reset sim state to the given arm configuration (settled, zero vel)."""
        if q is None:
            q = self.neutral_q()
        self.attached_side = None
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_idx] = q
        self.data.ctrl[self.arm_ctrl_idx] = q
        self.data.ctrl[self.gripper_ctrl_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_arm_targets(self, q: np.ndarray) -> None:
        """Command the position servos (radians, ARM_JOINTS order)."""
        self.data.ctrl[self.arm_ctrl_idx] = q

    def set_target_markers(
        self, targets: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Move the mocap spheres to the commanded EE targets."""
        for side, (pos, _rot) in targets.items():
            self.data.mocap_pos[self.target_mocap_id[side]] = pos

    def step(self, n_substeps: int) -> None:
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self.attached_side is not None:
                self._follow_gripper()

    # ------------------------------------------------------------------
    # Payload / mock grasp
    # ------------------------------------------------------------------

    def set_payload_pos(self, pos: np.ndarray) -> None:
        """Place the payload (settled, zero velocity, identity attitude)."""
        self.data.qpos[self._payload_pos_sl] = pos
        self.data.qpos[self._payload_quat_sl] = [1, 0, 0, 0]
        self.data.qvel[self._payload_vel_sl] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def payload_pos(self) -> np.ndarray:
        return self.data.qpos[self._payload_pos_sl].copy()

    def try_attach(self, side: str, radius: float = 0.045) -> bool:
        """Mock grasp: attach the payload to `side`'s gripper if it is
        within `radius` of the EE site. No-op if the other arm holds it.

        Returns True if the payload is (now) held by `side`.
        """
        if self.attached_side == side:
            return True
        if self.attached_side is not None:
            return False
        ee_pos, ee_rot = self.eef_pose(side)
        if np.linalg.norm(self.payload_pos() - ee_pos) > radius:
            return False
        self.attached_side = side
        self._attach_offset_pos = ee_rot.T @ (self.payload_pos() - ee_pos)
        quat = self.data.qpos[self._payload_quat_sl]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        self._attach_offset_rot = ee_rot.T @ rot.reshape(3, 3)
        return True

    def release(self, side: str) -> None:
        """Release the payload if `side` is holding it (it falls freely)."""
        if self.attached_side == side:
            self.attached_side = None

    def _follow_gripper(self) -> None:
        """Kinematically pin the held payload to the gripper frame."""
        assert self.attached_side is not None
        ee_pos, ee_rot = self.eef_pose(self.attached_side)
        self.data.qpos[self._payload_pos_sl] = ee_pos + ee_rot @ self._attach_offset_pos
        rot = ee_rot @ self._attach_offset_rot
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(9))
        self.data.qpos[self._payload_quat_sl] = quat
        self.data.qvel[self._payload_vel_sl] = 0.0

    def eef_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Measured EE pose: (position(3), rotation(3,3)) in world frame."""
        sid = self.eef_site_id[side]
        pos = self.data.site_xpos[sid].copy()
        rot = self.data.site_xmat[sid].reshape(3, 3).copy()
        return pos, rot

    def arm_q(self) -> np.ndarray:
        """Measured arm joint angles (radians, ARM_JOINTS order)."""
        return self.data.qpos[self.arm_qpos_idx].copy()

    def fk_eef_pos(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """EE positions for an arm configuration, without touching sim state.

        Used to isolate pure IK error (commanded q vs target) from the
        servo-tracking error measured in the stepped physics.
        """
        return {side: pose[:3, 3] for side, pose in self.fk_eef_pose(q).items()}

    def fk_eef_pose(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Full 4x4 EE poses for an arm configuration (scratch data only)."""
        if not hasattr(self, "_fk_data"):
            self._fk_data = mujoco.MjData(self.model)
        self._fk_data.qpos[:] = 0.0
        self._fk_data.qpos[self.arm_qpos_idx] = q
        mujoco.mj_kinematics(self.model, self._fk_data)
        poses = {}
        for side in SIDES:
            sid = self.eef_site_id[side]
            pose = np.eye(4)
            pose[:3, :3] = self._fk_data.site_xmat[sid].reshape(3, 3)
            pose[:3, 3] = self._fk_data.site_xpos[sid]
            poses[side] = pose
        return poses
