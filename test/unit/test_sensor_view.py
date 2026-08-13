"""Unit tests for the pure helpers of src/common/sensor_view.py."""

import unittest

import numpy as np

from common.sensor_view import (
    VIEW_HIDDEN_CAMERAS,
    CollectionStatus,
    FrameRateCounter,
    ViewPanel,
    _age_color,
    _vec5_to_dict,
    colourise_depth,
    compose_sensor_view_frame,
    footer_state_badge,
    progress_bar_text,
    side_joint_dict,
    visible_view_captures,
)


class TestFrameRateCounter(unittest.TestCase):
    def test_same_object_counts_once(self):
        c = FrameRateCounter(window_s=2.0)
        frame = np.zeros((2, 2, 3))
        c.tick(frame, now=0.0)
        c.tick(frame, now=0.1)  # same object: not a new frame
        self.assertAlmostEqual(c.hz(now=0.2), 0.5)  # 1 frame / 2 s window

    def test_distinct_objects_count(self):
        c = FrameRateCounter(window_s=2.0)
        for i in range(6):
            c.tick(np.zeros((2, 2, 3)), now=0.1 * i)
        self.assertAlmostEqual(c.hz(now=0.5), 3.0)  # 6 frames / 2 s

    def test_old_stamps_expire(self):
        c = FrameRateCounter(window_s=1.0)
        c.tick(np.zeros(1), now=0.0)
        c.tick(np.zeros(1), now=5.0)
        self.assertAlmostEqual(c.hz(now=5.0), 1.0)  # only the recent one

    def test_none_frame_ignored(self):
        c = FrameRateCounter()
        c.tick(None, now=0.0)
        self.assertEqual(c.hz(now=0.0), 0.0)

    def test_no_drops_without_expected_hz(self):
        c = FrameRateCounter()  # expected_hz unset → drop detection off
        c.tick(object(), now=0.0)
        c.tick(object(), now=5.0)  # huge gap, but no baseline to judge it
        self.assertEqual(c.drops, 0)

    def test_drops_counted_against_expected_period(self):
        c = FrameRateCounter(expected_hz=30.0)  # nominal 33.3 ms/frame
        c.tick(object(), now=0.0)
        c.tick(object(), now=1.0 / 30.0)  # on-time: no drop
        self.assertEqual(c.drops, 0)
        c.tick(object(), now=1.0 / 30.0 + 3.0 / 30.0)  # ~3 periods gap → 2 dropped
        self.assertEqual(c.drops, 2)


class TestAgeColor(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(_age_color(0.005), (0, 255, 0))  # green ≤ ½ frame
        self.assertEqual(_age_color(0.020), (0, 210, 255))  # amber ≤ 1 frame
        self.assertEqual(_age_color(0.050), (0, 0, 255))  # red beyond
        self.assertEqual(_age_color(None), (0, 0, 255))  # missing → red


class TestSideJointDict(unittest.TestCase):
    def _vec(self):
        # left joints 0-4, right joints 5-9 (URDF degrees)
        return np.arange(10, dtype=float) * 10.0

    def test_right_side_uses_upper_slice(self):
        d = side_joint_dict(self._vec(), "right", 0.5)
        self.assertEqual(d["shoulder_pan"], 50.0)  # index 5
        self.assertEqual(d["wrist_roll"], 90.0)  # index 9
        self.assertEqual(d["gripper"], 0.5)

    def test_left_side_uses_lower_slice(self):
        d = side_joint_dict(self._vec(), "left", None)
        self.assertEqual(d["shoulder_pan"], 0.0)  # index 0
        self.assertEqual(d["wrist_roll"], 40.0)  # index 4
        self.assertIsNone(d["gripper"])  # no gripper value yet

    def test_none_vector_gives_all_none_joints(self):
        d = side_joint_dict(None, "left", None)
        self.assertEqual(
            set(d),
            {
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            },
        )
        self.assertTrue(all(v is None for v in d.values()))

    def test_gripper_kept_when_vector_present(self):
        d = side_joint_dict(self._vec(), "left", 0.25)
        self.assertEqual(d["gripper"], 0.25)


class TestColouriseDepth(unittest.TestCase):
    def test_shape_and_dtype(self):
        depth = np.full((8, 12), 1000, dtype=np.uint16)  # 1 m at 0.001 scale
        out = colourise_depth(depth, scale_m=0.001)
        self.assertEqual(out.shape, (8, 12, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_zero_pixels_stay_black(self):
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        depth[0, 0] = 0  # no return
        out = colourise_depth(depth, scale_m=0.001)
        self.assertTrue(np.all(out[0, 0] == 0))
        self.assertFalse(np.all(out[1, 1] == 0))

    def test_near_and_far_differ(self):
        near = colourise_depth(np.full((2, 2), 200, np.uint16), scale_m=0.001)
        far = colourise_depth(np.full((2, 2), 2000, np.uint16), scale_m=0.001)
        self.assertFalse(np.array_equal(near, far))

    def test_zero_scale_falls_back(self):
        # A capture that never populated depth_scale (0.0) must not divide by
        # zero or blow up — it falls back to a sane default.
        out = colourise_depth(np.full((3, 3), 500, np.uint16), scale_m=0.0)
        self.assertEqual(out.shape, (3, 3, 3))


class TestComposeSensorViewFrame(unittest.TestCase):
    def _cols(self):
        d = _vec5_to_dict(np.zeros(5), 0.5)
        return {"left": d, "right": d}

    def test_composites_rgb_and_depth_panels(self):
        rgb = ViewPanel(
            label="central",
            image_bgr=np.zeros((48, 64, 3), np.uint8),
            fallback_hw=(48, 64),
            line1="central 30Hz",
            line2="drift 5ms  drop 0",
            line2_color=(0, 255, 0),
        )
        depth = ViewPanel(
            label="central_depth",
            image_bgr=colourise_depth(np.full((48, 64), 1000, np.uint16)),
            fallback_hw=(48, 64),
            line1="central_depth 30Hz",
        )
        out = compose_sensor_view_frame(
            [rgb, depth],
            self._cols(),
            self._cols(),
            "cmd",
            joint_strip=("j", (0, 255, 0)),
        )
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[2], 3)
        self.assertGreater(out.shape[0], 0)

    def test_missing_image_uses_black_fallback(self):
        panel = ViewPanel(
            label="scene",
            image_bgr=None,
            fallback_hw=(48, 64),
            line1="scene 0Hz",
        )
        out = compose_sensor_view_frame(
            [panel], self._cols(), self._cols(), "cmd", joint_strip=None
        )
        self.assertEqual(out.ndim, 3)

    def test_no_panels_still_renders_joint_cells(self):
        out = compose_sensor_view_frame(
            [], self._cols(), self._cols(), "cmd", joint_strip=None
        )
        self.assertGreater(out.shape[0], 0)

    def test_footer_adds_height_and_still_renders(self):
        base = compose_sensor_view_frame(
            [], self._cols(), self._cols(), "cmd", joint_strip=None
        )
        with_footer = compose_sensor_view_frame(
            [],
            self._cols(),
            self._cols(),
            "cmd",
            joint_strip=None,
            key_help=["Y enable", "A record", "Q quit"],
            status=CollectionStatus(
                state_label="RECORDING",
                episodes_done=3,
                episodes_goal=20,
                current_frames=42,
                recording=True,
            ),
        )
        self.assertEqual(with_footer.ndim, 3)
        # A footer block only adds rows; it never removes the joint cells.
        self.assertGreaterEqual(with_footer.shape[0], base.shape[0])

    def test_saving_status_still_renders(self):
        out = compose_sensor_view_frame(
            [],
            self._cols(),
            self._cols(),
            "cmd",
            joint_strip=None,
            status=CollectionStatus(
                state_label="SAVING",
                episodes_done=4,
                episodes_goal=20,
                current_frames=812,
                recording=False,
            ),
        )
        self.assertEqual(out.ndim, 3)
        self.assertGreater(out.shape[0], 0)


class TestFooterStateBadge(unittest.TestCase):
    def test_recording_is_red(self):
        text, col = footer_state_badge("RECORDING", True)
        self.assertIn("REC", text)
        self.assertEqual(col, (0, 0, 255))

    def test_saving_is_amber(self):
        text, col = footer_state_badge("SAVING", False)
        self.assertIn("SAVING", text)
        self.assertEqual(col, (0, 180, 255))

    def test_discarding_is_amber(self):
        text, col = footer_state_badge("DISCARDING", False)
        self.assertIn("DISCARDING", text)
        self.assertEqual(col, (0, 180, 255))

    def test_idle_is_grey(self):
        text, col = footer_state_badge("IDLE", False)
        self.assertEqual(text, "IDLE")
        self.assertEqual(col, (150, 150, 150))

    def test_recording_flag_overrides_label(self):
        # A stale label still reads as recording when the flag says so.
        text, col = footer_state_badge("", True)
        self.assertIn("REC", text)
        self.assertEqual(col, (0, 0, 255))


class TestProgressBarText(unittest.TestCase):
    def test_with_goal_shows_fraction_and_pct(self):
        self.assertEqual(progress_bar_text(3, 20), "episodes 3/20 (15%)")

    def test_no_goal_shows_bare_count(self):
        self.assertEqual(progress_bar_text(7, 0), "episodes 7")

    def test_clamps_overshoot_to_100(self):
        self.assertEqual(progress_bar_text(25, 20), "episodes 25/20 (100%)")


class _Cam:
    def __init__(self, name):
        self.name = name


class TestVisibleViewCaptures(unittest.TestCase):
    def test_scene_hidden_by_default(self):
        cams = [_Cam("scene"), _Cam("wrist_camera_left"), _Cam("central")]
        out = visible_view_captures(cams)
        self.assertEqual([c.name for c in out], ["wrist_camera_left", "central"])
        self.assertIn("scene", VIEW_HIDDEN_CAMERAS)

    def test_custom_hidden_set(self):
        cams = [_Cam("scene"), _Cam("wrist_camera_left")]
        out = visible_view_captures(cams, hidden={"wrist_camera_left"})
        self.assertEqual([c.name for c in out], ["scene"])

    def test_nothing_hidden_when_empty(self):
        cams = [_Cam("scene"), _Cam("central")]
        out = visible_view_captures(cams, hidden=set())
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
