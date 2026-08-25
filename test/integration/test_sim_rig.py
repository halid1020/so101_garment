"""The twin standing where the bench stands, checked against the same protocol.

``tool/run_policy.py --sim`` rehearses the whole deployment procedure without a
robot, and that is only worth anything if the twin answers every call the loop
makes on the bench. So this exercises the protocol -- observe, command, enable,
hold, frames, shutdown -- against the real MuJoCo scene, since the point of the
rehearsal is that nothing is stubbed out.

Needs MuJoCo (MUJOCO_GL=egl); it builds the payload scene.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from common.policy_rig import RAMP_S, FrameCache, TwinRig  # noqa: E402


class SimRigTestCase(unittest.TestCase):
    env = None

    @classmethod
    def setUpClass(cls):
        from sim_datagen.env import CAMERAS, PickPlaceTwinEnv

        cls.CAMERAS = CAMERAS
        cls.env = PickPlaceTwinEnv("handover")

    def rig(self, camera_map=None, fps=30.0):
        self.env.reset(self._scenario())
        return TwinRig(
            self.env,
            cameras=list(self.CAMERAS),
            camera_wh=(160, 120),
            camera_map=camera_map,
            fps=fps,
        )

    def _scenario(self):
        from tool.eval_sim_policy import _scenario_for_seed

        return _scenario_for_seed("handover", 14)


class TestObservation(SimRigTestCase):
    def test_one_observe_gives_a_state_and_every_camera(self):
        state, images = self.rig().observe()

        self.assertEqual(np.asarray(state).shape, (12,))
        self.assertEqual(sorted(images), sorted(self.CAMERAS))
        for frame in images.values():
            self.assertEqual(frame.shape, (120, 160, 3))


class TestFrames(SimRigTestCase):
    """The live view reads pictures the same way on either rig."""

    def test_nothing_is_published_before_the_first_observe(self):
        rig = self.rig()

        self.assertIsNone(rig.frames.get_rgb_image("scene"))

    def test_after_one_observe_every_camera_is_readable_by_name(self):
        rig = self.rig()
        rig.observe()

        for name in rig.cameras:
            self.assertIsNotNone(rig.frames.get_rgb_image(name))

    def test_a_camera_nobody_has_is_none_rather_than_an_error(self):
        rig = self.rig()
        rig.observe()

        self.assertIsNone(rig.frames.get_rgb_image("no_such_camera"))


class TestCameraRename(SimRigTestCase):
    """A checkpoint trained before a stream was renamed still has to run."""

    def test_the_reported_names_are_the_ones_the_policy_will_be_sent(self):
        rig = self.rig(camera_map={"wrist_camera_left": "wrist_left"})

        self.assertIn("wrist_left", rig.cameras)
        self.assertNotIn("wrist_camera_left", rig.cameras)

    def test_and_the_frames_are_readable_under_those_names(self):
        rig = self.rig(camera_map={"wrist_camera_left": "wrist_left"})
        rig.observe()

        self.assertIsNotNone(rig.frames.get_rgb_image("wrist_left"))
        # observe() still returns the raw names: the remote client does the
        # rename on the frames it sends, exactly as it does for run_policy_sim.
        _, images = rig.observe()
        self.assertIn("wrist_camera_left", images)


class TestCommandAndHold(SimRigTestCase):
    def test_a_command_advances_the_world_by_one_control_tick(self):
        rig = self.rig()
        before, _ = rig.observe()
        rig.command(np.asarray(before, dtype=float))

        self.assertEqual(rig.ticks, 1)

    def test_a_hold_advances_time_without_a_new_goal(self):
        # A paused rollout must not freeze simulated time, or the arms would
        # never be seen to settle in preview.
        rig = self.rig()
        state, _ = rig.observe()
        rig.command(np.asarray(state, dtype=float))
        rig.hold()

        self.assertEqual(rig.ticks, 2)

    def test_holding_before_anything_was_commanded_uses_the_measured_pose(self):
        rig = self.rig()
        rig.hold()

        self.assertEqual(rig.ticks, 1)
        self.assertIsNotNone(rig.last_action)


class TestEnable(SimRigTestCase):
    """Where torque would be taken on the bench, and a ramp happens on both."""

    def test_it_ramps_toward_the_first_action_rather_than_teleporting(self):
        rig = self.rig()
        start, _ = rig.observe()
        target = np.asarray(start, dtype=float).copy()
        target[0] += 10.0  # ten degrees of left shoulder pan

        rig.enable(target)

        self.assertEqual(rig.ticks, int(RAMP_S * 30.0))
        moved, _ = rig.observe()
        self.assertGreater(abs(moved[0] - start[0]), 1.0)

    def test_the_ramp_lasts_RAMP_S_at_whatever_rate_the_run_ticks_at(self):
        # The sim's control rate is a parameter, and a ramp counted in ticks at
        # an assumed 30 Hz would last the wrong number of seconds at any other.
        rig = self.rig(fps=25.0)
        start, _ = rig.observe()

        rig.enable(np.asarray(start, dtype=float))

        self.assertEqual(rig.ticks, int(RAMP_S * 25.0))

    def test_shutdown_releases_nothing_and_raises_nothing(self):
        rig = self.rig()
        rig.shutdown()


class TestReset(SimRigTestCase):
    """Another attempt on the same scene, which is what the page's Reset asks."""

    def test_it_puts_the_scene_back_where_the_scenario_spawned_it(self):
        rig = self.rig()
        at_spawn, _ = rig.observe()
        away = np.asarray(at_spawn, dtype=float).copy()
        away[0] += 15.0  # fifteen degrees of left shoulder pan
        for _ in range(40):
            rig.command(away)
        moved, _ = rig.observe()
        self.assertGreater(abs(moved[0] - at_spawn[0]), 1.0)

        rig.reset(self._scenario())

        back, _ = rig.observe()
        np.testing.assert_allclose(back, at_spawn, atol=1e-3)

    def test_and_forgets_the_pose_the_last_attempt_ended_in(self):
        # hold() re-commands last_action, so keeping it across a reset would
        # drive the arms to a goal chosen for a scene that is no longer there.
        rig = self.rig()
        state, _ = rig.observe()
        rig.command(np.asarray(state, dtype=float))

        rig.reset(self._scenario())

        self.assertIsNone(rig.last_action)
        self.assertEqual(rig.ticks, 0)


class TestFrameCache(unittest.TestCase):
    """Pure enough to check on its own; the view's whole interface is here."""

    def test_the_latest_frame_wins_and_the_rest_survive(self):
        cache = FrameCache()
        cache.publish({"a": np.zeros((2, 2, 3), np.uint8), "b": np.ones(1)})
        cache.publish({"a": np.full((2, 2, 3), 7, np.uint8)})

        self.assertEqual(int(cache.get_rgb_image("a")[0, 0, 0]), 7)
        self.assertIsNotNone(cache.get_rgb_image("b"))
        cache.request_shutdown()


if __name__ == "__main__":
    unittest.main()
