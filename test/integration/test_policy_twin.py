"""The twin preview: a policy's plan, drawn as the arms that would execute it.

MuJoCo-backed, so it lives here rather than in the unit tier. What is worth
checking is cheap and real: a frame comes out at the size asked for, two
different actions do not produce the same picture (a preview that ignores its
input would be worse than none), and a gripper fraction moves something.

Since the preview draws the arms as they ARE beside where a plan sends them, the
second thing worth checking is that the two ghosts really are two: a second set
of MOVING geoms, coloured apart, without a second copy of the table and the
board.

Run:  PYTHONPATH=.:src MUJOCO_GL=egl python -m unittest \
          test.integration.test_policy_twin
"""

import unittest

import numpy as np

from common.web.policy_twin import NOW_RGBA, PLAN_RGBA, SOLO_ALPHA, TwinPreview

REST = np.array([0.0, -100.0, 90.0, 50.0, 0.0, 0.2, 0.0, -100.0, 90.0, 50.0, 0.0, 0.2])


def raised(lift: float = -20.0) -> np.ndarray:
    """REST with both shoulders somewhere else, so the ghosts do not overlap."""
    other = REST.copy()
    other[1] = other[7] = lift
    return other


class TwinCase(unittest.TestCase):
    twin: TwinPreview

    @classmethod
    def setUpClass(cls):
        cls.twin = TwinPreview(width=240, height=180)

    @classmethod
    def tearDownClass(cls):
        cls.twin.close()

    def draw(self, action) -> np.ndarray:
        """One pose on its own -- the old single-pose preview."""
        return self.twin.render_pair(action, None)

    def moving_geoms(self) -> int:
        import mujoco

        scene = self.twin.renderer.scene
        dynamic = int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        return sum(
            1 for i in range(scene.ngeom) if int(scene.geoms[i].category) == dynamic
        )

    def moving_colours(self) -> set:
        import mujoco

        scene = self.twin.renderer.scene
        dynamic = int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        return {
            tuple(round(float(c), 3) for c in scene.geoms[i].rgba)
            for i in range(scene.ngeom)
            if int(scene.geoms[i].category) == dynamic
        }


class TestTwinPreview(TwinCase):
    def test_a_pose_renders_at_the_size_asked_for(self):
        frame = self.draw(REST)

        self.assertEqual(frame.shape, (180, 240, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_the_frame_is_a_picture_of_something(self):
        frame = self.draw(REST)

        # A blank render (a broken camera, an unlit scene) has no spread at all.
        self.assertGreater(float(frame.std()), 5.0)

    def test_a_different_plan_looks_different(self):
        self.assertFalse(np.array_equal(self.draw(REST), self.draw(raised())))

    def test_closing_the_gripper_changes_the_picture(self):
        closed = REST.copy()
        closed[5] = 0.0
        closed[11] = 0.0

        self.assertFalse(np.array_equal(self.draw(REST), self.draw(closed)))

    def test_it_is_a_preview_not_a_simulation(self):
        # Nothing is stepped, so rendering the same plan twice is identical:
        # no drift, no settling, no contact -- only where the plan puts the arms.
        first = self.draw(REST)

        self.assertTrue(np.array_equal(first, self.draw(REST)))

    def test_a_pose_alone_is_drawn_solid(self):
        # Nothing is being compared to it, so it need not be see-through.
        self.draw(REST)

        self.assertEqual(self.moving_colours(), {NOW_RGBA[:3] + (SOLO_ALPHA,)})


class TestGhostPair(TwinCase):
    def test_the_second_pose_adds_its_own_moving_geoms(self):
        self.draw(REST)
        alone = self.moving_geoms()

        self.twin.render_pair(REST, raised())

        self.assertEqual(self.moving_geoms(), 2 * alone)

    def test_the_scenery_is_not_drawn_twice(self):
        # Only mjCAT_DYNAMIC is added for the second pose: a transparent second
        # copy of the table and the board would fog the whole picture.
        self.draw(REST)
        scenery = self.twin.renderer.scene.ngeom - self.moving_geoms()

        self.twin.render_pair(REST, raised())

        self.assertEqual(self.twin.renderer.scene.ngeom - self.moving_geoms(), scenery)

    def test_the_two_ghosts_are_coloured_apart(self):
        self.twin.render_pair(REST, raised())

        self.assertEqual(self.moving_colours(), {NOW_RGBA, PLAN_RGBA})

    def test_a_pair_does_not_look_like_either_pose_on_its_own(self):
        both = self.twin.render_pair(REST, raised())

        self.assertFalse(np.array_equal(both, self.draw(REST)))
        self.assertFalse(np.array_equal(both, self.twin.render_pair(None, raised())))

    def test_nothing_to_draw_is_no_frame_rather_than_a_blank_one(self):
        # The caller then shows the frame it already has, which is the whole
        # reason the page cannot flicker.
        self.assertIsNone(self.twin.render_pair(None, None))

    def test_a_pose_of_the_wrong_width_is_not_a_pose(self):
        self.assertIsNone(self.twin.render_pair(np.zeros(6), None))


if __name__ == "__main__":
    unittest.main()
