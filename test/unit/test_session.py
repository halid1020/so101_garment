"""Unit tests for supervising a collection session from the console.

No rig: what is exercised is the resolution of a start request into the exact
command the terminal front-end would have run, the refusals that stop a session
that cannot work, and the ladder that ends one. The supervisor itself is driven
against a stand-in script rather than the real recorder, so starting, tailing
and stopping are tested without opening a camera.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from common.recording.collection_settings import (
    SelectionError,
    resolve_new_selection,
    resolve_resume_selection,
)
from common.web.session import (
    INTERRUPT_GRACE_S,
    QUIT_GRACE_S,
    PreviewArms,
    SessionSupervisor,
    resolve_plan,
    session_refusals,
    stop_action,
    teleop_argv,
)

_CONFIG = {
    "cameras": {
        "central": {"enabled": True},
        "wrist_camera_left": {"enabled": True},
        "wrist_camera_right": {"enabled": False},
    },
    "realsense": {"rgb_name": "central_rgbd"},
}


def write_dataset(
    root: Path,
    name: str,
    episodes=2,
    cameras=("central",),
    ee=True,
    depth=False,
    fps=30,
) -> Path:
    path = root / name
    (path / "meta").mkdir(parents=True, exist_ok=True)
    features = {f"observation.images.{c}": {"dtype": "video"} for c in cameras}
    if ee:
        features["ee_pose"] = {"dtype": "float32"}
    (path / "meta" / "info.json").write_text(
        json.dumps({"fps": fps, "total_episodes": episodes, "features": features})
    )
    if depth:
        (path / "meta" / "realsense.json").write_text(
            json.dumps({"rgb_name": "central_rgbd"})
        )
    return path


class TestSharedResolvers(unittest.TestCase):
    """The console and the command line must resolve a selection identically."""

    def test_a_new_dataset_defaults_to_the_configured_cameras(self):
        sel = resolve_new_selection(
            {"central", "wrist_camera_left", "wrist_camera_right"},
            {"central", "wrist_camera_left"},
            [],
            depth=False,
            record_ee=True,
            fps=None,
        )
        self.assertEqual(sel["cameras"], {"central", "wrist_camera_left"})
        self.assertTrue(sel["ee"])

    def test_an_unknown_camera_is_refused(self):
        with self.assertRaises(SelectionError):
            resolve_new_selection({"central"}, {"central"}, ["nope"], False, True, None)

    def test_a_resume_follows_the_dataset_and_reports_what_it_ignored(self):
        settings = {
            "cameras": {"central", "wrist_camera_left"},
            "depth": False,
            "depth_rgb_name": None,
            "ee": True,
            "fps": 30,
        }
        sel, warnings = resolve_resume_selection(
            settings, "central_rgbd", ["central"], True, True, 60, leader=False
        )
        self.assertEqual(sel["cameras"], {"central", "wrist_camera_left"})
        self.assertEqual(sel["fps"], 30)
        self.assertEqual(len(warnings), 4)

    def test_an_ee_dataset_cannot_be_resumed_with_the_leader_arms(self):
        settings = {
            "cameras": {"central"},
            "depth": False,
            "depth_rgb_name": None,
            "ee": True,
            "fps": 30,
        }
        with self.assertRaises(SelectionError):
            resolve_resume_selection(
                settings, None, [], False, False, None, leader=True
            )


class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_good_request_has_no_refusals(self):
        self.assertEqual(
            session_refusals(self.root, "towel", "fold it", False, False, False), []
        )

    def test_a_second_session_is_refused(self):
        reasons = session_refusals(self.root, "towel", "fold it", False, False, True)
        self.assertTrue(any("already running" in r for r in reasons))

    def test_a_bad_name_is_refused(self):
        reasons = session_refusals(self.root, "../x", "fold it", False, False, False)
        self.assertTrue(reasons)

    def test_a_missing_instruction_is_refused(self):
        reasons = session_refusals(self.root, "towel", "  ", False, False, False)
        self.assertTrue(any("instruction" in r for r in reasons))

    def test_a_stillborn_dataset_is_refused_with_the_remedy(self):
        reasons = session_refusals(self.root, "towel", "fold it", False, True, False)
        self.assertTrue(any("Delete it" in r for r in reasons))

    def test_a_read_only_drive_is_refused(self):
        os.chmod(self.root, 0o555)
        try:
            reasons = session_refusals(
                self.root, "towel", "fold it", False, False, False
            )
            self.assertTrue(reasons)
        finally:
            os.chmod(self.root, 0o755)


class TestPlanAndCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_new_dataset_plans_the_configured_cameras(self):
        plan = resolve_plan(self.root, "towel", "fold it", {}, _CONFIG)
        self.assertFalse(plan["resuming"])
        self.assertEqual(plan["cameras"], ["central", "wrist_camera_left"])
        self.assertIn("--enable-camera", plan["flags"])
        self.assertIn("--disable-camera", plan["flags"])
        self.assertEqual(plan["refusals"], [])

    def test_leader_mode_never_plans_the_ee_features(self):
        plan = resolve_plan(self.root, "towel", "fold it", {"input": "leader"}, _CONFIG)
        self.assertFalse(plan["ee"])
        self.assertIn("--no-record-ee", plan["flags"])

    def test_resuming_follows_the_recorded_streams(self):
        write_dataset(self.root, "towel", cameras=("central",), fps=30)
        plan = resolve_plan(
            self.root, "towel", "fold it", {"streams": ["wrist_camera_left"]}, _CONFIG
        )
        self.assertTrue(plan["resuming"])
        self.assertEqual(plan["cameras"], ["central"])
        self.assertTrue(plan["warnings"])

    def test_the_command_matches_the_recorder_the_cli_would_have_run(self):
        plan = resolve_plan(self.root, "towel", "fold it", {}, _CONFIG)
        argv = teleop_argv(
            self.root, "towel", "fold it", plan, {"goal": 20}, monitor_port=8766
        )
        self.assertIn("--record", argv)
        self.assertEqual(argv[argv.index("--repo-id") + 1], "towel")
        self.assertEqual(
            argv[argv.index("--dataset-root") + 1], str(self.root / "towel")
        )
        self.assertEqual(argv[argv.index("--task") + 1], "fold it")
        self.assertEqual(argv[argv.index("--monitor-port") + 1], "8766")
        self.assertEqual(argv[argv.index("--episode-goal") + 1], "20")
        self.assertNotIn("--resume", argv)

    def test_a_resumed_session_passes_resume(self):
        write_dataset(self.root, "towel")
        plan = resolve_plan(self.root, "towel", "fold it", {}, _CONFIG)
        argv = teleop_argv(self.root, "towel", "fold it", plan, {}, 8766)
        self.assertIn("--resume", argv)


class TestLaunch(unittest.TestCase):
    """How the session is launched, which decides what it can read."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_session_is_given_no_stdin(self):
        # Inherited, stdin would be the terminal the console itself was started
        # in -- and a leader session reads its control keys from stdin, so it
        # would put that terminal into raw mode underneath the operator's shell
        # and then race the shell for every keystroke. It is driven from the
        # page instead.
        supervisor = SessionSupervisor(self.tmp, monitor_port=8766)
        plan = resolve_plan(self.tmp, "towel", "fold it", {}, _CONFIG)
        seen = {}

        class FakeProc:
            pid = 4242
            stdout = io.StringIO("")

            def poll(self):
                return None

        def fake_popen(argv, **kwargs):
            seen.update(kwargs)
            seen["argv"] = argv
            return FakeProc()

        with mock.patch("subprocess.Popen", fake_popen):
            supervisor.start("towel", "fold it", plan, {})

        self.assertEqual(seen["stdin"], subprocess.DEVNULL)
        # And it still outlives the console that started it.
        self.assertTrue(seen["start_new_session"])


class TestStopLadder(unittest.TestCase):
    def test_quitting_is_given_time_before_anything_harsher(self):
        self.assertEqual(stop_action(0.0), "wait")
        self.assertEqual(stop_action(QUIT_GRACE_S - 0.1), "wait")

    def test_an_ignored_quit_becomes_an_interrupt(self):
        self.assertEqual(stop_action(QUIT_GRACE_S), "interrupt")
        self.assertEqual(stop_action(INTERRUPT_GRACE_S - 0.1), "interrupt")

    def test_an_ignored_interrupt_becomes_a_terminate(self):
        self.assertEqual(stop_action(INTERRUPT_GRACE_S), "terminate")

    def test_the_ladder_never_reaches_a_kill(self):
        self.assertNotIn("kill", {stop_action(t) for t in (0, 10, 25, 60, 600, 10_000)})


class TestSupervisor(unittest.TestCase):
    """Drives a stand-in for the recorder: starting, tailing and ending it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / "fake_teleop.py"
        self.script.write_text(
            "import sys, time\n"
            "print('recorder up', flush=True)\n"
            "print(' '.join(sys.argv[1:]), flush=True)\n"
            "time.sleep(30)\n"
        )
        self.session = SessionSupervisor(
            self.root, teleop=self.script, python=sys.executable
        )

    def tearDown(self):
        proc = self.session._proc
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        self.tmp.cleanup()

    def _start(self):
        plan = resolve_plan(self.root, "towel", "fold it", {}, _CONFIG)
        return self.session.start("towel", "fold it", plan, {})

    def test_nothing_runs_before_a_start(self):
        state = self.session.state()
        self.assertFalse(state["running"])
        self.assertIsNone(state["pid"])

    def test_starting_records_what_is_running(self):
        state = self._start()
        self.assertTrue(state["running"])
        self.assertEqual(state["name"], "towel")
        self.assertEqual(state["task"], "fold it")
        self.assertTrue(self.session.running())

    def test_the_output_tail_is_captured(self):
        self._start()
        for _ in range(200):
            if self.session.state()["tail"]:
                break
            time.sleep(0.02)
        self.assertIn("recorder up", self.session.state()["tail"][0])

    def test_a_second_start_is_refused(self):
        self._start()
        plan = resolve_plan(self.root, "towel", "fold it", {}, _CONFIG)
        with self.assertRaises(RuntimeError):
            self.session.start("towel", "fold it", plan, {})

    def test_the_ladder_ends_a_session_that_ignores_the_quit_key(self):
        self._start()
        self.session.signal_stop()
        # The stand-in never honours a quit, so the ladder must escalate. Pull
        # its clock back rather than waiting out the real grace periods.
        self.session._stopping_since -= INTERRUPT_GRACE_S
        self.assertEqual(self.session.escalate(), "terminate")
        for _ in range(200):
            if not self.session.running():
                break
            time.sleep(0.02)
        self.assertFalse(self.session.running())

    def test_escalating_without_a_stop_request_does_nothing(self):
        self._start()
        self.assertEqual(self.session.escalate(), "done")
        self.assertTrue(self.session.running())


class TestPreviewArms(unittest.TestCase):
    """The idle reading of the followers, assembled without a bus in sight."""

    def test_nothing_is_reported_before_an_arm_answers(self):
        arms = PreviewArms()
        self.assertIsNone(arms.snapshot())
        arms._buses = {"left": object()}  # opened, but no reading yet
        self.assertIsNone(arms.snapshot())

    def test_a_side_that_is_open_reports_and_the_other_stays_empty(self):
        arms = PreviewArms()
        arms._buses = {"left": object()}
        arms._state = {"left": ([1.0, 2.0, 3.0, 4.0, 5.0], 0.25)}
        arms._read_at = {"left": time.monotonic()}
        snap = arms.snapshot()
        self.assertEqual(snap["source"], "preview")
        self.assertEqual(snap["joints"]["left"]["state"]["shoulder_pan"], 1.0)
        self.assertEqual(snap["joints"]["left"]["state"]["gripper"], 0.25)
        self.assertIsNone(snap["joints"]["right"]["state"]["shoulder_pan"])

    def test_nothing_is_commanding_the_arms_so_the_command_half_is_empty(self):
        # The console never writes to a bus it opened for reading, and the table
        # must not suggest otherwise.
        arms = PreviewArms()
        arms._buses = {"left": object(), "right": object()}
        arms._state = {
            "left": ([0.0] * 5, 0.0),
            "right": ([0.0] * 5, 0.0),
        }
        arms._read_at = {"left": time.monotonic(), "right": time.monotonic()}
        snap = arms.snapshot()
        self.assertFalse(snap["teleop_active"])
        for side in ("left", "right"):
            self.assertFalse(snap["joints"][side]["fresh"])
            self.assertIsNone(snap["joints"][side]["command"]["wrist_roll"])

    def test_an_unassigned_rig_is_refused_with_where_to_fix_it(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "sensor_map.yaml"
            empty.write_text("cameras: {}\n")
            with self.assertRaises(RuntimeError) as caught:
                PreviewArms().start(empty)
        self.assertIn("Signals", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
