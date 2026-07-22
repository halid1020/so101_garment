"""Shared twin pick-and-place environment for collection and evaluation.

``PickPlaceTwinEnv`` wraps the digital-twin payload scene so the collector and
the policy-eval harness build observations identically. It owns the IK<->world
frame bridge (measured at the neutral pose, as in ``tool/quest_sim_teleop.py``)
so oracle/policy targets stay in the IK frame while the bar, target zone and
success test live in the twin world.

One control tick is 20 physics substeps: the payload-mode timestep is 1/600 s
and control runs at 30 Hz, so a tick is exactly 1/30 s of simulation and eval
ticks match training frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin

from common.configs import GRIPPER_OPEN_MAX_FRAC
from common.recording.features import SIDES as FEATURE_SIDES
from common.recording.features import build_observation_state
from sim_benchmark.constants import (
    ARM_JOINT_SUFFIXES,
    DUAL_URDF_PATH,
    EE_FRAMES,
    NEUTRAL_ARM_ANGLES_DEG,
    SIDES,
)
from sim_datagen.oracle import CUBE_HALF
from sim_twin.scene import TwinSim

# Dataset feature name -> twin camera name. The feature names match the real
# recorder (common.recording) so policies transfer between sim and hardware.
CAMERAS = {
    "scene": "rgb_scene",
    "wrist_left": "rgb_wrist_left",
    "wrist_right": "rgb_wrist_right",
}

# Twin table top world height (build_spec puts the table top at world z = 0).
TABLE_TOP_WORLD_Z = 0.0
SUCCESS_RADIUS = 0.02  # m XY payload-to-target distance that counts as placed
SETTLE_VEL = 0.02  # m/s payload speed below which it counts as settled
SETTLE_Z_TOL = 0.01  # m payload-z tolerance around its resting height
N_SUBSTEPS = 20  # physics substeps per 30 Hz control tick (1/30 / (1/600))

# The two task language strings the policy is conditioned on.
TASKS = {
    "handover": "pick up the block, hand it over, and place it on the marked target",
    "single": "pick up the block and place it on the marked target",
}


class PickPlaceTwinEnv:
    """Twin payload scene with an IK<->world bridge and LeRobot observations."""

    def __init__(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r} (choose from {sorted(TASKS)})")
        self.task = task
        self.sim = TwinSim(all_collisions=True, payload=True)
        self.sim.reset()

        # IK-frame neutral EE poses (dual-URDF FK) and the per-side IK->world
        # offset that bridges oracle/policy targets to the twin world.
        self._pin_model, self._pin_data, neutral_poses = self._build_ik_fk()
        self.neutral_ik_poses = neutral_poses
        self.world_offset: dict[str, np.ndarray] = {}
        for side in SIDES:
            meas, _ = self.sim.eef_pose(side)
            self.world_offset[side] = meas - neutral_poses[side][0]
        self.scene_offset = np.mean([self.world_offset[s] for s in SIDES], axis=0)
        # IK-frame table top: world table top minus the (pure-z) offset.
        self.table_z = TABLE_TOP_WORLD_Z - float(self.scene_offset[2])

        self._target_world_xy = np.zeros(2)
        # Per-episode resting height of the cube centre (reset() overwrites it).
        self._payload_rest_z = CUBE_HALF

    # ------------------------------------------------------------------
    def _build_ik_fk(
        self,
    ) -> tuple[pin.Model, Any, dict[str, tuple[np.ndarray, np.ndarray]]]:
        full = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
        gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
        model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))
        data = model.createData()
        q = pin.neutral(model)
        for side in SIDES:
            idx = [
                model.joints[model.getJointId(f"{side}_{sfx}")].idx_q
                for sfx in ARM_JOINT_SUFFIXES
            ]
            q[idx] = np.deg2rad(NEUTRAL_ARM_ANGLES_DEG)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        poses = {}
        for side in SIDES:
            placement = data.oMf[model.getFrameId(EE_FRAMES[side])]
            poses[side] = (placement.translation.copy(), placement.rotation.copy())
        return model, data, poses

    # ------------------------------------------------------------------
    def _ik_xy_to_world(self, xy: np.ndarray, ik_z: float) -> np.ndarray:
        return np.array([xy[0], xy[1], ik_z]) + self.scene_offset

    def reset(self, scenario: Any) -> None:
        """Reset the arms to neutral and place the cube and target for ``scenario``.

        Both tasks use an upright 2.2 cm cube resting at ``payload_xy``. The
        single task places it at ``target_xy`` with one arm; the relay
        (handover) task has the left arm lay it at the midline and the right
        arm carry it on to ``target_xy`` on the right (see sim_datagen.oracle).
        """
        self.sim.reset()
        # TwinSim.reset opens the jaws to the full mechanical range; re-command
        # the capped open so the episode starts exactly at the real rig's cap.
        for side in SIDES:
            self.sim.set_gripper_frac(side, GRIPPER_OPEN_MAX_FRAC)
        payload_xy = np.asarray(scenario.payload_xy, dtype=float)
        target_xy = np.asarray(scenario.target_xy, dtype=float)
        rest_z_ik = self.table_z + CUBE_HALF
        quat = None  # upright cube

        payload_world = self._ik_xy_to_world(payload_xy, rest_z_ik)
        self.sim.set_payload_pose(payload_world, quat)
        self._payload_rest_z = float(payload_world[2])

        target_world = self._ik_xy_to_world(target_xy, self.table_z)
        # Sit the disc just above the table top so it renders on the surface.
        target_world[2] = TABLE_TOP_WORLD_Z + 0.001
        self.sim.set_target_zone(target_world)
        self._target_world_xy = target_world[:2].copy()
        # Let the prism settle onto the table before the episode starts.
        for _ in range(N_SUBSTEPS * 3):
            self.sim.step(1)

    def tick(self, q_rad_10: np.ndarray, grip_frac: dict[str, float]) -> None:
        """Command arm joints (IK-frame radians) and gripper fractions, then step.

        ``grip_frac`` is trigger-like (1 = open command, 0 = full squeeze); it
        maps to the jaw exactly as the real rig's capped trigger mapping does
        (common.threads.dual_joint_state): commanded opening = fraction x
        GRIPPER_OPEN_MAX_FRAC of the mechanical range.

        """
        self.sim.set_arm_targets(np.asarray(q_rad_10, dtype=float))
        for side in SIDES:
            self.sim.set_gripper_frac(side, grip_frac[side] * GRIPPER_OPEN_MAX_FRAC)
        self.sim.step(N_SUBSTEPS)

    def observe(
        self, camera_wh: tuple[int, int]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return (state12, {feature_name: HWC uint8 image})."""
        width, height = camera_wh
        joints_deg = np.rad2deg(self.sim.arm_q())
        # Raw full-mechanical-range open fractions, exactly like the real
        # recorder: with the capped trigger mapping they live in [0, 0.3].
        gripper_open = {
            side: self.sim.gripper_open_frac(side) for side in FEATURE_SIDES
        }
        state = build_observation_state(joints_deg, gripper_open)
        images = {
            name: np.ascontiguousarray(
                self.sim.render_camera(cam, width=width, height=height)
            )
            for name, cam in CAMERAS.items()
        }
        return state, images

    def payload_settled(self) -> bool:
        pos = self.sim.payload_pos()
        vel = float(np.linalg.norm(self.sim.payload_vel()))
        z_ok = abs(pos[2] - self._payload_rest_z) < SETTLE_Z_TOL
        return z_ok and vel < SETTLE_VEL

    def place_error(self) -> float:
        """Payload-to-target XY distance in the twin world (m)."""
        return float(np.linalg.norm(self.sim.payload_pos()[:2] - self._target_world_xy))

    def payload_dropped(self) -> bool:
        """True if the bar has fallen well below its resting height."""
        return bool(self.sim.payload_pos()[2] < self._payload_rest_z - 0.05)

    def success(self) -> bool:
        """Placed within radius and settled (grip state checked by the caller)."""
        return self.place_error() < SUCCESS_RADIUS and self.payload_settled()
