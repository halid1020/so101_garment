"""The two ghosts, in a real viser scene.

``action_to_urdf_cfg`` is checked as arithmetic in the unit tests. What is
checked here is the claim the panel actually makes: that pointing the twin at an
action MOVES the orange ghost, that the blue one follows the measured pose
independently, and that a pose nobody has yet is hidden rather than left showing
a stale one. It needs a real server and two real URDF loads, which is why it
lives here and not beside the arithmetic.
"""

import socket
import unittest

import numpy as np

from common.web.policy_ghost import GhostPair


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def poses(urdf) -> list:
    """Every joint frame's placement, to compare one configuration to another."""
    return [
        (tuple(np.round(f.position, 5)), tuple(np.round(f.wxyz, 5)))
        for f in urdf._joint_frames
    ]


NOW = np.array([0, -10, 20, 25, 0, 0.0, 0, -10, 20, 25, 0, 0.0], float)
PLAN = np.array([60, -10, 20, 25, 0, 0.5, -60, -10, 20, 25, 0, 0.5], float)


class GhostSceneTestCase(unittest.TestCase):
    ghosts: GhostPair

    @classmethod
    def setUpClass(cls):
        cls.ghosts = GhostPair(port=free_port())

    @classmethod
    def tearDownClass(cls):
        cls.ghosts.stop()

    @property
    def blue(self):
        return self.ghosts._core.urdf_vis

    @property
    def orange(self):
        return self.ghosts._core.ghost_robot_urdf


class TestTwoPoses(GhostSceneTestCase):
    def test_the_ghosts_stand_apart_when_the_plan_is_not_the_present(self):
        self.ghosts.show(NOW, PLAN)

        self.assertNotEqual(poses(self.blue), poses(self.orange))

    def test_and_coincide_exactly_when_it_is(self):
        # The gap between them IS the pending motion, so no gap must mean no
        # motion -- not a constant offset from two different conversions.
        self.ghosts.show(PLAN, PLAN)

        self.assertEqual(poses(self.blue), poses(self.orange))

    def test_each_ghost_moves_when_its_own_pose_changes(self):
        self.ghosts.show(NOW, NOW)
        before = poses(self.blue)
        self.ghosts.show(PLAN, NOW)

        self.assertNotEqual(before, poses(self.blue))
        self.assertEqual(poses(self.orange), before)


class TestMissingPoses(GhostSceneTestCase):
    def test_a_plan_that_does_not_exist_yet_is_hidden_not_stale(self):
        self.ghosts.show(NOW, PLAN)
        self.ghosts.show(NOW, None)

        self.assertFalse(self.orange.show_visual)
        self.assertTrue(self.blue.show_visual)

    def test_the_measured_pose_before_the_first_tick_is_hidden_too(self):
        self.ghosts.show(None, PLAN)

        self.assertFalse(self.blue.show_visual)
        self.assertTrue(self.orange.show_visual)

    def test_a_pose_of_the_wrong_width_is_refused_rather_than_drawn(self):
        self.ghosts.show(NOW, PLAN)
        self.ghosts.show(NOW, np.zeros(6))

        self.assertFalse(self.orange.show_visual)

    def test_both_come_back_when_both_exist_again(self):
        self.ghosts.show(None, None)
        self.ghosts.show(NOW, PLAN)

        self.assertTrue(self.blue.show_visual)
        self.assertTrue(self.orange.show_visual)


class TestServed(GhostSceneTestCase):
    def test_the_page_has_somewhere_to_point_its_iframe(self):
        import urllib.request

        with urllib.request.urlopen(self.ghosts.url, timeout=10) as response:
            self.assertEqual(response.status, 200)
            # No X-Frame-Options and no CSP, or the panel would refuse to embed.
            headers = {k.lower() for k in response.headers}
            self.assertNotIn("x-frame-options", headers)
            self.assertNotIn("content-security-policy", headers)


if __name__ == "__main__":
    unittest.main()
