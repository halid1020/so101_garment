"""Motion derivatives shown in the dataset review view.

The joint half is arithmetic and is tested against trajectories whose
derivatives are known exactly. The end-effector half goes through the platform's
URDF, and is tested by invariants that a wrong axis, a degree/radian slip or a
transposed rotation would each break -- a sweep of one wrist joint has to come
back as exactly that angular rate.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_episode_motion
"""

import unittest

import numpy as np

from common.recording.episode_motion import (
    GRIPPER_CHANNELS,
    JOINT_UNITS,
    STATE_DOF,
    body_configuration,
    ee_derivatives,
    ee_trajectory,
    episode_motion,
    joint_derivatives,
)

FPS = 30.0


def _state(n: int) -> np.ndarray:
    return np.zeros((n, STATE_DOF), dtype=float)


class TestJointDerivatives(unittest.TestCase):
    def test_a_constant_rate_ramp_gives_that_rate_and_no_acceleration(self):
        n = 40
        state = _state(n)
        state[:, 1] = np.arange(n) * (60.0 / FPS)  # 60 deg/s
        velocity, acceleration = joint_derivatives(state, FPS)
        np.testing.assert_allclose(velocity[:, 1], 60.0, atol=1e-9)
        # The edges of a second central difference see the ramp's ends, so the
        # interior is what carries the claim.
        np.testing.assert_allclose(acceleration[2:-2, 1], 0.0, atol=1e-6)

    def test_a_quadratic_gives_its_constant_acceleration(self):
        n = 40
        t = np.arange(n) / FPS
        state = _state(n)
        state[:, 2] = 0.5 * 120.0 * t**2  # 120 deg/s^2
        _velocity, acceleration = joint_derivatives(state, FPS)
        np.testing.assert_allclose(acceleration[2:-2, 2], 120.0, rtol=1e-6)

    def test_untouched_channels_stay_still(self):
        state = _state(20)
        state[:, 0] = np.arange(20)
        velocity, acceleration = joint_derivatives(state, FPS)
        np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(acceleration[:, 1:], 0.0, atol=1e-12)

    def test_a_single_frame_episode_reports_no_rate_rather_than_raising(self):
        velocity, acceleration = joint_derivatives(_state(1), FPS)
        self.assertEqual(velocity.shape, (1, STATE_DOF))
        np.testing.assert_allclose(velocity, 0.0)
        np.testing.assert_allclose(acceleration, 0.0)

    def test_a_gripper_channel_is_not_measured_in_degrees(self):
        # A gripper column is an open FRACTION, so its rate cannot share the
        # heading the body joints use.
        self.assertEqual(len(JOINT_UNITS), STATE_DOF)
        for channel in GRIPPER_CHANNELS:
            self.assertEqual(JOINT_UNITS[channel], "frac/s")
        body = set(range(STATE_DOF)) - set(GRIPPER_CHANNELS)
        self.assertTrue(all(JOINT_UNITS[i] == "deg/s" for i in body))

    def test_a_wrongly_shaped_state_is_refused(self):
        with self.assertRaises(ValueError):
            joint_derivatives(np.zeros((10, 7)), FPS)
        with self.assertRaises(ValueError):
            joint_derivatives(_state(10), 0.0)


class TestBodyConfiguration(unittest.TestCase):
    def test_the_gripper_columns_are_dropped_and_the_rest_is_radians(self):
        state = _state(3)
        state[:, :] = np.arange(STATE_DOF)  # channel index as its value
        q = body_configuration(state)
        self.assertEqual(q.shape, (3, 10))
        # Left five, then right five: channels 0-4 and 6-10, in radians.
        np.testing.assert_allclose(
            q[0], np.deg2rad([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]), atol=1e-12
        )


class TestEndEffectorMotion(unittest.TestCase):
    def test_a_still_arm_has_no_motion_of_any_kind(self):
        state = _state(20)
        trajectory = ee_trajectory(state)
        for position, rotation in trajectory.values():
            v, a, w, alpha = ee_derivatives(position, rotation, FPS)
            for series in (v, a, w, alpha):
                np.testing.assert_allclose(series, 0.0, atol=1e-9)

    def test_a_wrist_sweep_comes_back_as_exactly_that_angular_rate(self):
        # wrist_roll turns the tool frame about its own axis, so the frame's
        # angular speed IS the joint rate. This is the assertion that catches a
        # radian/degree slip or a rotation composed the wrong way round.
        n, rate = 60, 90.0  # deg/s
        state = _state(n)
        state[:, 4] = np.arange(n) * (rate / FPS)  # left wrist_roll
        position, rotation = ee_trajectory(state)["left"]
        _v, _a, w, _alpha = ee_derivatives(position, rotation, FPS)
        np.testing.assert_allclose(np.rad2deg(w[2:-2]), rate, rtol=1e-6)

    def test_both_arms_are_the_same_arm_in_their_own_base_frames(self):
        # The bases are fixed by pure translation in the dual URDF, so the same
        # joint values must put each tool frame at the same base-frame pose.
        # A base transform applied to the wrong side would break this.
        state = _state(2)
        state[:, [1, 2, 3]] = [20.0, -30.0, 15.0]
        state[:, [7, 8, 9]] = [20.0, -30.0, 15.0]
        trajectory = ee_trajectory(state)
        np.testing.assert_allclose(
            trajectory["left"][0], trajectory["right"][0], atol=1e-9
        )
        np.testing.assert_allclose(
            trajectory["left"][1], trajectory["right"][1], atol=1e-9
        )

    def test_angular_acceleration_measures_a_turning_axis(self):
        # |alpha| is the magnitude of the angular velocity's derivative, not the
        # derivative of its magnitude: a turn that holds its speed while changing
        # axis really is accelerating, and reporting zero for it would hide the
        # snap this whole view exists to show.
        n = 60
        state = _state(n)
        half = n // 2
        state[:half, 4] = np.arange(half) * (90.0 / FPS)  # roll, then
        state[half:, 4] = state[half - 1, 4]
        state[half:, 3] = np.arange(n - half) * (90.0 / FPS)  # flex, same speed
        position, rotation = ee_trajectory(state)["left"]
        _v, _a, w, alpha = ee_derivatives(position, rotation, FPS)
        speed = np.rad2deg(w)
        # Speed is near-constant across the handover, but the axis swings.
        self.assertLess(abs(speed[half - 3] - speed[half + 3]), 5.0)
        self.assertGreater(np.rad2deg(alpha[half - 1 : half + 2]).max(), 100.0)


class TestEpisodeMotion(unittest.TestCase):
    def test_every_series_has_one_value_per_frame(self):
        n = 25
        state = _state(n)
        state[:, 1] = np.arange(n)
        motion = episode_motion(state, FPS)
        self.assertEqual(motion["joint_vel"].shape, (n, STATE_DOF))
        self.assertEqual(motion["joint_acc"].shape, (n, STATE_DOF))
        self.assertIsNotNone(motion["ee"])
        for side in ("left", "right"):
            for key in ("v", "a", "w", "alpha"):
                self.assertEqual(motion["ee"][side][key].shape, (n,))

    def test_the_angular_channels_are_reported_in_degrees(self):
        n, rate = 40, 90.0
        state = _state(n)
        state[:, 4] = np.arange(n) * (rate / FPS)
        motion = episode_motion(state, FPS)
        np.testing.assert_allclose(motion["ee"]["left"]["w"][2:-2], rate, rtol=1e-6)

    def test_a_model_that_will_not_load_leaves_the_joint_rates_standing(self):
        # A review tool must still show what it can, and still play the video,
        # on a machine where the kinematic model is unavailable. Losing the
        # Cartesian pane is not a reason to lose the episode.
        motion = episode_motion(_state(10), FPS, urdf_path="/no/such/robot.urdf")
        self.assertIsNone(motion["ee"])
        self.assertTrue(motion["ee_error"])
        self.assertEqual(motion["joint_vel"].shape, (10, STATE_DOF))


if __name__ == "__main__":
    unittest.main()
