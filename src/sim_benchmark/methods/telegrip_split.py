"""Telegrip-style split IK: position-only DLS + direct analytic wrist.

Port of the teleoperation algorithm of Telegrip
(https://github.com/DipFlip/telegrip) to the benchmark interface. Telegrip
never runs IK on the wrist: the three proximal joints (shoulder_pan,
shoulder_lift, elbow_flex) track the hand *position* with a position-only
DLS solve, while wrist_flex and wrist_roll are set *directly* from the
controller orientation. The wrist therefore bypasses every task-space
cost/damping trade-off — that direct 1:1 joint mapping is what makes
Telegrip's wrist feel agile.

Telegrip itself maps controller rotation deltas to wrist-angle deltas. Our
method interface receives an absolute target pose instead, so the wrist
angles are recovered analytically from the target rotation:

- The SO-101's shoulder_lift/elbow_flex/wrist_flex axes are parallel, so the
  EE tip elevation is affine in (q_lift, q_elbow, q_flex):
  ``elev = e0 + s2*q2 + s3*q3 + s4*q4`` with slopes of exactly +-1. Solving
  for q_flex given the currently commanded q2, q3 yields the wrist_flex that
  realizes the target tip elevation.
- wrist_roll rotates the EE about its tip axis, so the signed roll angle of
  the target's local z about the tip (measured against a horizontal
  reference) is affine in q5: ``roll = r0 + s5*q5``.

The affine coefficients are NOT hand-derived from the URDF (the wrist_roll
joint carries a non-trivial rpy that makes sign bookkeeping treacherous);
they are calibrated numerically by finite differences at construction and
guarded by test/test_telegrip.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

from common.config_parser import load_method_params
from common.roll_ratchet import gripper_roll_about_tip
from sim_benchmark.constants import ARM_JOINT_SUFFIXES, ARM_JOINTS, SIDES
from sim_benchmark.methods.base import Targets, TeleopMethod

_DEG_TIP_Z = 0.995  # |tip_z| above this: roll reference is degenerate


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _tip_elevation(rot: np.ndarray) -> float:
    """Elevation angle of the EE tip axis (local x) above the horizon."""
    tip = rot[:, 0]
    return float(np.arctan2(tip[2], np.hypot(tip[0], tip[1])))


# Shared with the production ratchet logic (common.roll_ratchet).
_tip_roll = gripper_roll_about_tip


class TelegripSplit(TeleopMethod):
    """Position-only 3-joint DLS + direct analytic wrist (Telegrip port)."""

    name = "telegrip"

    def __init__(self, sim_model: mujoco.MjModel) -> None:
        super().__init__(sim_model)
        # Damping, gain and clamp from src/ik_conf/methods/telegrip.yaml.
        # gain is higher than dls (0.6) as no orientation task competes for the
        # proximal joints; max_joint_vel clamps both DLS and wrist joints.
        params = load_method_params(self.name)
        self.DAMPING = params["damping"]
        self.GAIN = params["gain"]
        self.MAX_JOINT_VEL = params["max_joint_vel"]
        self.model = sim_model
        self.data = mujoco.MjData(sim_model)  # private kinematic scratchpad
        self.q = np.zeros(len(ARM_JOINTS))

        self._qpos_idx = np.array([sim_model.joint(j).qposadr[0] for j in ARM_JOINTS])
        # Per side: qpos indices of the 5 arm joints in ARM_JOINTS order and
        # dof columns of the 3 proximal joints.
        self._side_slice = {
            side: slice(5 * i, 5 * i + 5) for i, side in enumerate(SIDES)
        }
        self._dof_idx3 = {
            side: np.array(
                [
                    sim_model.joint(f"{side}_{sfx}").dofadr[0]
                    for sfx in ARM_JOINT_SUFFIXES[:3]
                ]
            )
            for side in SIDES
        }
        self._site_id = {side: sim_model.site(f"{side}_eef_site").id for side in SIDES}
        self._q_low = np.array([sim_model.joint(j).range[0] for j in ARM_JOINTS])
        self._q_high = np.array([sim_model.joint(j).range[1] for j in ARM_JOINTS])

        # Numeric wrist-map calibration (per side): elevation affine model
        # elev = e0 + s2*q2 + s3*q3 + s4*q4 and roll model roll = r0 + s5*q5.
        self._wrist_cal = {side: self._calibrate_wrist(side) for side in SIDES}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _fk_rot(self, side: str, q_arm: np.ndarray) -> np.ndarray:
        """EE site rotation for one arm at q_arm (other arm at zero)."""
        self.data.qpos[self._qpos_idx] = 0.0
        self.data.qpos[self._qpos_idx[self._side_slice[side]]] = q_arm
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xmat[self._site_id[side]].reshape(3, 3).copy()

    def _calibrate_wrist(self, side: str) -> dict[str, float]:
        """Finite-difference the affine elevation/roll models at a safe pose."""
        qb = np.array([0.1, 0.3, 0.5, 0.3, 0.2])  # away from degeneracies
        delta = 0.1
        elev_b = _tip_elevation(self._fk_rot(side, qb))
        slopes = []
        for j in (1, 2, 3):  # lift, elbow, flex
            qp = qb.copy()
            qp[j] += delta
            slope = (_tip_elevation(self._fk_rot(side, qp)) - elev_b) / delta
            s = float(np.sign(slope))
            if abs(abs(slope) - 1.0) > 0.05 or s == 0.0:
                raise RuntimeError(
                    f"telegrip wrist calibration: elevation slope for joint "
                    f"{j} is {slope:.3f}, expected +-1 (parallel-axis "
                    "assumption violated — did the URDF change?)"
                )
            slopes.append(s)
        s2, s3, s4 = slopes
        e0 = elev_b - (s2 * qb[1] + s3 * qb[2] + s4 * qb[3])

        roll_b = _tip_roll(self._fk_rot(side, qb))
        qp = qb.copy()
        qp[4] += delta
        roll_p = _tip_roll(self._fk_rot(side, qp))
        assert roll_b is not None and roll_p is not None
        slope = _wrap_pi(roll_p - roll_b) / delta
        s5 = float(np.sign(slope))
        if abs(abs(slope) - 1.0) > 0.05 or s5 == 0.0:
            raise RuntimeError(
                f"telegrip wrist calibration: roll slope is {slope:.3f}, "
                "expected +-1 (roll axis is not the tip axis?)"
            )
        r0 = _wrap_pi(roll_b - s5 * qb[4])
        return {"e0": e0, "s2": s2, "s3": s3, "s4": s4, "r0": r0, "s5": s5}

    # ------------------------------------------------------------------
    # TeleopMethod interface
    # ------------------------------------------------------------------

    def reset(self, q0: np.ndarray) -> None:
        self.q = q0.copy()

    def solve(self, targets: Targets, dt: float) -> np.ndarray:
        max_step = self.MAX_JOINT_VEL * dt
        q_new = self.q.copy()

        # 1) Analytic wrist per arm from the target rotation, using the
        #    currently commanded lift/elbow angles.
        for side in SIDES:
            _, rot_t = targets[side]
            cal = self._wrist_cal[side]
            sl = self._side_slice[side]
            q_arm = q_new[sl]

            elev_t = _tip_elevation(rot_t)
            q4_des = cal["s4"] * (
                elev_t - cal["e0"] - cal["s2"] * q_arm[1] - cal["s3"] * q_arm[2]
            )
            q_arm[3] += np.clip(q4_des - q_arm[3], -max_step, max_step)

            roll_t = _tip_roll(rot_t)
            if roll_t is not None:  # near-vertical tip: hold previous roll
                q5_des = cal["s5"] * _wrap_pi(roll_t - cal["r0"])
                q_arm[4] += np.clip(_wrap_pi(q5_des - q_arm[4]), -max_step, max_step)

        # 2) Position-only DLS on the 3 proximal joints, wrist held fixed.
        self.data.qpos[self._qpos_idx] = q_new
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for side in SIDES:
            pos_t, _ = targets[side]
            sid = self._site_id[side]
            pos_c = self.data.site_xpos[sid]
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            jac = jacp[:, self._dof_idx3[side]]
            jjt = jac @ jac.T + self.DAMPING**2 * np.eye(3)
            dq3 = jac.T @ np.linalg.solve(jjt, pos_t - pos_c)
            vel = np.clip(dq3 * self.GAIN / dt, -self.MAX_JOINT_VEL, self.MAX_JOINT_VEL)
            sl = self._side_slice[side]
            q_new[sl.start : sl.start + 3] += vel * dt

        # 3) Joint-limit clamp on everything.
        self.q = np.clip(q_new, self._q_low, self._q_high)
        return self.q.copy()
