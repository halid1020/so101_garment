#!/usr/bin/env python3
"""Unit tests for the workspace envelope and out-of-envelope policies.

Run via: python -m unittest test.unit.test_workspace_envelope
(requires PYTHONPATH=.:src, as set by `source setup.sh`)
"""

import unittest

import numpy as np
import pinocchio as pin

from src.common.configs import (
    NEUTRAL_JOINT_ANGLES_DUAL,
    WORKSPACE_R_MAX,
    WORKSPACE_R_MIN,
    WORKSPACE_SOFT_MARGIN,
)
from src.common.workspace_envelope import (
    FreezePolicy,
    ProjectPolicy,
    SlowdownPolicy,
    WarnOnlyPolicy,
    build_envelopes,
    derive_workspace_radii,
    make_policies,
)

_URDF = "src/so101_dual_description/robot.urdf"


def _reduced_model() -> pin.Model:
    full = pin.buildModelFromUrdf(_URDF)
    gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
    return pin.buildReducedModel(full, gripper_ids, pin.neutral(full))


class TestEnvelopeGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _reduced_model()
        cls.envelopes = build_envelopes(cls.model)
        cls.rng = np.random.default_rng(42)

    def test_radii_match_urdf(self):
        """The hardcoded WORKSPACE_R_MIN/MAX must match a fresh FK sweep of
        the URDF within 5 mm — guards against silent URDF drift."""
        r_min, r_max = derive_workspace_radii(self.model, grid=41)
        self.assertAlmostEqual(r_min, WORKSPACE_R_MIN, delta=0.005)
        self.assertAlmostEqual(r_max, WORKSPACE_R_MAX, delta=0.005)

    def test_neutral_pose_inside(self):
        """The teleop neutral EE pose must be comfortably inside."""
        data = self.model.createData()
        q = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        for side, env in self.envelopes.items():
            p = data.oMf[self.model.getFrameId(f"{side}_eef_link")].translation
            self.assertGreater(env.margin(p), 0.05)

    def test_projection_idempotent_and_feasible(self):
        env = self.envelopes["left"]
        for _ in range(500):
            p = self.rng.uniform([-0.3, -0.3, -0.2], [0.9, 0.7, 0.6])
            q1 = env.project(p)
            q2 = env.project(q1)
            # Projected points are inside (tiny numerical slack).
            self.assertGreaterEqual(env.margin(q1), -1e-6)
            # Idempotent to numerical precision.
            self.assertLess(np.linalg.norm(q2 - q1), 1e-5)
            # Inside points are left untouched.
            if env.margin(p) > 0:
                self.assertTrue(np.allclose(env.project(p), p))

    def test_projection_never_below_floor(self):
        env = self.envelopes["right"]
        for _ in range(200):
            p = self.rng.uniform([-0.3, -0.6, -0.3], [0.9, 0.4, 0.05])
            self.assertGreaterEqual(env.project(p)[2], env.z_floor - 1e-9)


class TestOOEPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envelopes = build_envelopes(_reduced_model())
        cls.env = cls.envelopes["left"]
        # A comfortable interior point and a far-out point.
        cls.p_in = np.array([0.30, 0.15, 0.12])
        cls.p_out = np.array([0.80, 0.15, 0.12])
        assert cls.env.is_inside(cls.p_in) and not cls.env.is_inside(cls.p_out)

    def test_make_policies_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            make_policies("bogus", self.envelopes)

    def test_warn_passthrough(self):
        pol = WarnOnlyPolicy("left", self.env)
        p, status = pol.apply(self.p_out.copy(), 0.0)
        np.testing.assert_array_equal(p, self.p_out)
        self.assertFalse(status.inside)
        self.assertFalse(status.clamped)

    def test_project_clamps_outside_only(self):
        pol = ProjectPolicy("left", self.env)
        p, status = pol.apply(self.p_in.copy(), 0.0)
        np.testing.assert_array_equal(p, self.p_in)
        self.assertFalse(status.clamped)
        p, status = pol.apply(self.p_out.copy(), 1.0)
        self.assertTrue(status.clamped)
        self.assertGreaterEqual(self.env.margin(p), -1e-6)

    def test_freeze_holds_and_releases(self):
        pol = FreezePolicy("left", self.env)
        pol.apply(self.p_in.copy(), 0.0)  # seed the feasible anchor
        p, status = pol.apply(self.p_out.copy(), 1.0)
        np.testing.assert_array_equal(p, self.p_in)  # held at last feasible
        self.assertTrue(status.clamped)
        # Re-entry releases immediately.
        p_back = self.p_in + np.array([0.01, 0.0, 0.0])
        p, status = pol.apply(p_back.copy(), 2.0)
        np.testing.assert_array_equal(p, p_back)
        self.assertFalse(status.clamped)
        # Reset clears the anchor: first out-of-envelope apply projects.
        pol.reset()
        p, _ = pol.apply(self.p_out.copy(), 3.0)
        self.assertGreaterEqual(self.env.margin(p), -1e-6)

    def test_slowdown_attenuates_outward_not_tangential(self):
        pol = SlowdownPolicy("left", self.env)
        # Start just inside the soft band near the outer boundary.
        env = self.env
        piv = env.pivot(self.p_in)
        v = self.p_in - piv
        v /= np.linalg.norm(v)
        p_edge = piv + v * (env.r_max - 0.5 * WORKSPACE_SOFT_MARGIN)
        pol.apply(p_edge.copy(), 0.0)
        # Outward step is attenuated...
        step_out = 0.01 * v
        p1, _ = pol.apply(p_edge + step_out, 0.02)
        moved_out = float((p1 - p_edge) @ v)
        self.assertLess(moved_out, 0.01)
        self.assertGreaterEqual(moved_out, 0.0)
        # ...more strongly than when far inside.
        pol2 = SlowdownPolicy("left", self.env)
        pol2.apply(self.p_in.copy(), 0.0)
        p2, _ = pol2.apply(self.p_in + step_out, 0.02)
        moved_far = float((p2 - self.p_in) @ v)
        self.assertGreater(moved_far, moved_out)
        # Tangential slide near the edge stays full-rate.
        pol3 = SlowdownPolicy("left", self.env)
        pol3.apply(p_edge.copy(), 0.0)
        tang = np.cross(v, [0.0, 0.0, 1.0])
        tang /= np.linalg.norm(tang)
        p3, _ = pol3.apply(p_edge + 0.01 * tang, 0.02)
        self.assertGreater(float((p3 - p_edge) @ tang), 0.009)

    def test_slowdown_never_emits_outside(self):
        pol = SlowdownPolicy("left", self.env)
        rng = np.random.default_rng(7)
        p = self.p_in.copy()
        for k in range(300):
            p = p + rng.uniform(-0.02, 0.03, 3)  # biased outward random walk
            out, _ = pol.apply(p.copy(), k * 0.01)
            self.assertGreaterEqual(self.env.margin(out), -1e-6)


class TestOOEEpisodeIntegration(unittest.TestCase):
    """Drive the benchmark IK adapter through an out-of-envelope episode
    with the project policy: joint velocity must stay bounded and the arm
    must recover after re-entry."""

    def test_bounded_velocity_and_recovery(self):
        from sim_benchmark.method_adapter import MethodIKAdapter
        from sim_benchmark.mock_quest import envelope_radial

        max_vel = 3.0
        dt = 0.01
        adapter = MethodIKAdapter(
            "pink_relaxed",
            dt=dt,
            max_joint_vel=max_vel,
            initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
        )
        envelopes = build_envelopes(adapter.urdf_model)
        policies = make_policies("project", envelopes)
        traj = envelope_radial()
        poses = adapter.get_current_end_effector_poses()
        initial = {
            side: (pose[:3, 3].copy(), pose[:3, :3].copy())
            for side, pose in (
                ("left", poses["left_eef_link"]),
                ("right", poses["right_eef_link"]),
            )
        }
        q_prev = adapter.get_current_configuration()
        raw_final = None
        for k in range(int(traj.duration / dt)):
            t = k * dt
            raw = traj.targets(t, initial)
            targets = {}
            for side in ("left", "right"):
                p, _status = policies[side].apply(raw[side][0], t)
                targets[f"{side}_eef_link"] = (p, raw[side][1])
            adapter.set_target_poses(targets)
            adapter.solve_ik()
            q = adapter.get_current_configuration()
            self.assertLessEqual(
                np.abs(q - q_prev).max() / dt,
                max_vel + 1e-6,
                msg=f"joint velocity exceeded at t={t:.2f}s",
            )
            q_prev = q
            raw_final = raw
        # After the return stroke, the commanded FK must be back on the
        # (feasible) raw target.
        poses = adapter.get_current_end_effector_poses()
        assert raw_final is not None
        for side in ("left", "right"):
            err = np.linalg.norm(poses[f"{side}_eef_link"][:3, 3] - raw_final[side][0])
            self.assertLess(err, 0.010, msg=f"{side} did not recover: {err:.4f} m")


if __name__ == "__main__":
    unittest.main()
