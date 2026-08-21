"""The twin preview: a policy's plan, drawn as the arms that would execute it.

MuJoCo-backed, so it lives here rather than in the unit tier. What is worth
checking is cheap and real: a frame comes out at the size asked for, two
different actions do not produce the same picture (a preview that ignores its
input would be worse than none), and a gripper fraction moves something.

Run:  PYTHONPATH=.:src MUJOCO_GL=egl python -m unittest \
          test.integration.test_policy_twin
"""

import unittest

import numpy as np

from common.web.policy_twin import TwinPreview

REST = np.array([0.0, -100.0, 90.0, 50.0, 0.0, 0.2, 0.0, -100.0, 90.0, 50.0, 0.0, 0.2])


class TestTwinPreview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.twin = TwinPreview(width=240, height=180)

    @classmethod
    def tearDownClass(cls):
        cls.twin.close()

    def test_a_pose_renders_at_the_size_asked_for(self):
        frame = self.twin.render(REST)

        self.assertEqual(frame.shape, (180, 240, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_the_frame_is_a_picture_of_something(self):
        frame = self.twin.render(REST)

        # A blank render (a broken camera, an unlit scene) has no spread at all.
        self.assertGreater(float(frame.std()), 5.0)

    def test_a_different_plan_looks_different(self):
        raised = REST.copy()
        raised[1] = -20.0

        self.assertFalse(
            np.array_equal(self.twin.render(REST), self.twin.render(raised))
        )

    def test_closing_the_gripper_changes_the_picture(self):
        closed = REST.copy()
        closed[5] = 0.0
        closed[11] = 0.0

        self.assertFalse(
            np.array_equal(self.twin.render(REST), self.twin.render(closed))
        )

    def test_it_is_a_preview_not_a_simulation(self):
        # Nothing is stepped, so rendering the same plan twice is identical:
        # no drift, no settling, no contact -- only where the plan puts the arms.
        first = self.twin.render(REST)

        self.assertTrue(np.array_equal(first, self.twin.render(REST)))


if __name__ == "__main__":
    unittest.main()
