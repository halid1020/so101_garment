"""What a recorded frame means on THIS rig: two arms, ten joints, two grippers.

``actoris_harena``'s recorder paces the episode, pauses on a stale camera,
samples every stream at one reference time, assembles the frame and writes the
depth beside it. None of that is about a robot. This is the part that is, behind
the ``ObservationBuilder`` protocol the recorder takes.

Three methods, and each returns ``None`` or ``False`` rather than guessing, so a
frame that cannot be built honestly is skipped instead of written partial.
"""

from __future__ import annotations

import numpy as np
from actoris_harena.recording import features as feat

from common.data_manager_dual import DualDataManager, RobotActivityState
from common.robot_schema import SCHEMA


class DualArmObservations:
    """The dual SO-101's state, action and EE, sampled at a reference time."""

    def __init__(self, data_manager: DualDataManager) -> None:
        self.dm = data_manager
        #: World-to-base transforms, inverted once. Every recorded EE pose is
        #: expressed in ITS OWN ARM's base frame rather than the world's, so a
        #: policy learns a reach relative to the arm doing the reaching and the
        #: two arms' labels stay comparable. Computed lazily on first use: the
        #: pinocchio model behind it costs a second to build, and a session that
        #: is not recording EE should not pay for it.
        self._world_base_inv: "dict | None" = None

    def _base_inverse(self, side: str) -> np.ndarray:
        if self._world_base_inv is None:
            from common.recording.sidecar import compute_world_base_transforms

            self._world_base_inv = {
                s: np.linalg.inv(tf)
                for s, tf in compute_world_base_transforms().items()
            }
        return self._world_base_inv[side]

    def state_and_action(
        self, t_ref: float, drifts: "dict[str, float]"
    ) -> "tuple[np.ndarray, np.ndarray, bool] | None":
        dm = self.dm
        joints_res = dm.get_current_joint_angles_at(t_ref)
        if joints_res is None:
            return None  # no joint state yet
        measured, joint_drift = joints_res
        if len(measured) < SCHEMA.body_dof * len(SCHEMA.limbs):
            return None
        drifts["joints"] = joint_drift

        gripper_open: "dict[str, float]" = {}
        for side in SCHEMA.limbs:
            g = dm.get_current_gripper_open_value_at(side, t_ref)
            gripper_open[side] = 0.0 if g is None else g[0]
            drifts[f"grip_{side}"] = float("nan") if g is None else g[1]

        state = feat.build_observation_state(measured, gripper_open, SCHEMA)
        last_commands = {s: dm.get_last_sent_command(s) for s in SCHEMA.limbs}
        action = feat.build_action(state, last_commands, SCHEMA, t_ref)
        # A side whose command was too stale fell back to its measured state,
        # which teaches a spurious hold. The recorder tallies these.
        fell_back = len(feat.fresh_limbs(last_commands, SCHEMA, t_ref)) < len(
            SCHEMA.limbs
        )
        return state, action, fell_back

    def ee(
        self, t_ref: float, drifts: "dict[str, float]"
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """Sample measured + target EE (both in own base frame) at ``t_ref``.

        Returns ``(ee_pose_14, ee_target_14)`` or ``None`` if a measured EE is
        not yet available. The target falls back to the measured pose when it
        is stale/missing (no fresh IK target — e.g. a homing move), mirroring
        the joint action's fallback so the label is always defined.
        """
        pose_vecs: dict = {}
        target_vecs: dict = {}
        for side in SCHEMA.limbs:
            m = self.dm.get_current_end_effector_pose_at(side, t_ref)
            if m is None:
                return None
            world_pose, ee_drift = m
            base_pose = self._base_inverse(side) @ world_pose
            pose_vecs[side] = feat.pose_to_vec7(base_pose)
            drifts[f"ee_{side}"] = ee_drift

            tgt = self.dm.get_target_pose_at(side, t_ref)
            if tgt is not None and abs(tgt[1]) < feat.ACTION_FRESH_S:
                target_vecs[side] = feat.pose_to_vec7(self._base_inverse(side) @ tgt[0])
            else:
                target_vecs[side] = pose_vecs[side]  # fallback to measured
        ee_pose_vec = np.concatenate([pose_vecs["left"], pose_vecs["right"]])
        ee_target_vec = np.concatenate([target_vecs["left"], target_vecs["right"]])
        return ee_pose_vec, ee_target_vec

    def teleop_active(self) -> bool:
        return bool(self.dm.get_teleop_active())

    def armed(self) -> bool:
        return self.dm.get_robot_activity_state() != RobotActivityState.DISABLED
