"""Unit tests for the leader read retry and the Qt environment settling.

Both exist because a working setup was reporting problems it did not have. The
leader retry is the one with a real cost behind it: an unretried dropped status
packet does not merely print, it surrenders a control tick and leaves the
followers on a stale target for that long.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_startup_warnings
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.sensor_view import (
    choose_font_dir,
    quiet_qt_warnings,
    wayland_plugin_available,
)
from common.threads.leader_arm import read_leader_action


class _Leader:
    """A leader whose bus drops the first ``fails`` status packets."""

    def __init__(self, fails: int):
        self.fails = fails
        self.calls = 0

    def get_action(self):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError(
                "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] "
                "after 1 tries. [TxRxResult] There is no status packet!"
            )
        return {"shoulder_pan.pos": 1.0, "gripper.pos": 50.0}


class TestReadLeaderAction(unittest.TestCase):
    def test_a_clean_read_costs_one_call(self):
        leader = _Leader(0)
        self.assertEqual(read_leader_action(leader)["shoulder_pan.pos"], 1.0)
        self.assertEqual(leader.calls, 1)

    def test_a_dropped_packet_is_retried_rather_than_costing_the_tick(self):
        # The regression this guards: without the retry the caller skipped a
        # whole control period, so the followers held the previous target while
        # the leader had already moved on.
        leader = _Leader(1)
        self.assertEqual(read_leader_action(leader)["shoulder_pan.pos"], 1.0)
        self.assertEqual(leader.calls, 2)

    def test_a_bus_that_never_answers_still_raises(self):
        # A genuinely absent bus has to reach the caller's failure counter, which
        # is what ends the session; retrying must not swallow it.
        leader = _Leader(99)
        with self.assertRaises(ConnectionError):
            read_leader_action(leader)
        self.assertEqual(leader.calls, 2)

    def test_the_attempt_count_matches_what_the_followers_ask_for(self):
        # dual_joint_state.py reads with num_retry=2; the point of this change is
        # that both buses absorb an ordinary drop identically.
        leader = _Leader(99)
        with self.assertRaises(ConnectionError):
            read_leader_action(leader, attempts=3)
        self.assertEqual(leader.calls, 3)

    def test_at_least_one_attempt_is_always_made(self):
        leader = _Leader(0)
        read_leader_action(leader, attempts=0)
        self.assertEqual(leader.calls, 1)


class TestChooseFontDir(unittest.TestCase):
    def test_an_existing_configured_directory_is_left_alone(self):
        with mock.patch("common.sensor_view.Path") as path:
            path.return_value.is_dir.return_value = True
            self.assertIsNone(choose_font_dir("/somewhere/real"))

    def test_a_missing_configured_directory_falls_back_to_the_system(self):
        # This is the real case: the OpenCV wheel points QT_QPA_FONTDIR at a
        # directory inside itself that the Linux wheel does not ship, which
        # causes the font warning it was added to prevent.
        def is_dir(self):
            return str(self) != "/site-packages/cv2/qt/fonts"

        with mock.patch.object(type(__import__("pathlib").Path()), "is_dir", is_dir):
            self.assertEqual(
                choose_font_dir("/site-packages/cv2/qt/fonts", ("/usr/share/fonts",)),
                "/usr/share/fonts",
            )

    def test_no_usable_directory_leaves_the_choice_alone(self):
        # Pointing Qt at another empty directory would help nobody.
        self.assertIsNone(choose_font_dir("/nope", ("/also-nope", "/still-nope")))

    def test_an_unset_variable_is_still_repaired(self):
        self.assertEqual(
            choose_font_dir(None, ("/usr/share/fonts",)), "/usr/share/fonts"
        )


class TestWaylandPluginAvailable(unittest.TestCase):
    def test_a_build_shipping_only_xcb_has_no_wayland_plugin(self):
        with tempfile.TemporaryDirectory() as d:
            platforms = Path(d) / "platforms"
            platforms.mkdir()
            (platforms / "libqxcb.so").touch()
            self.assertFalse(wayland_plugin_available(d))

    def test_a_build_shipping_wayland_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            platforms = Path(d) / "platforms"
            platforms.mkdir()
            (platforms / "libqwayland-generic.so").touch()
            self.assertTrue(wayland_plugin_available(d))

    def test_no_plugin_path_at_all(self):
        self.assertFalse(wayland_plugin_available(None))
        self.assertFalse(wayland_plugin_available("/nonexistent"))


class TestQuietQtWarnings(unittest.TestCase):
    """Only the cases where Qt is complaining about something it forced."""

    def _run(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            changed = quiet_qt_warnings()
            return changed, dict(os.environ)

    def test_a_forced_xcb_fallback_drops_the_session_variable(self):
        # Qt reports the mismatch with XDG_SESSION_TYPE, not an ambiguous choice,
        # so setting the platform alone does not silence it (measured).
        changed, env = self._run(
            {"XDG_SESSION_TYPE": "wayland", "QT_QPA_PLATFORM_PLUGIN_PATH": "/none"}
        )
        self.assertNotIn("XDG_SESSION_TYPE", env)
        self.assertEqual(env["QT_QPA_PLATFORM"], "xcb")
        self.assertIn("XDG_SESSION_TYPE", changed)

    def test_a_build_with_a_wayland_plugin_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            platforms = Path(d) / "platforms"
            platforms.mkdir()
            (platforms / "libqwayland-generic.so").touch()
            changed, env = self._run(
                {"XDG_SESSION_TYPE": "wayland", "QT_QPA_PLATFORM_PLUGIN_PATH": d}
            )
        self.assertEqual(env.get("XDG_SESSION_TYPE"), "wayland")
        self.assertNotIn("QT_QPA_PLATFORM", changed)

    def test_an_x11_session_is_left_alone(self):
        changed, env = self._run({"XDG_SESSION_TYPE": "x11"})
        self.assertEqual(env.get("XDG_SESSION_TYPE"), "x11")
        self.assertNotIn("XDG_SESSION_TYPE", changed)

    def test_an_explicit_platform_choice_is_respected(self):
        _, env = self._run(
            {
                "XDG_SESSION_TYPE": "wayland",
                "QT_QPA_PLATFORM": "offscreen",
                "QT_QPA_PLATFORM_PLUGIN_PATH": "/none",
            }
        )
        self.assertEqual(env["QT_QPA_PLATFORM"], "offscreen")


if __name__ == "__main__":
    unittest.main()
