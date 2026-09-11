"""Route tests for the rig console's dataset API (aiohttp TestClient).

Runs the real application over a temporary collection directory holding
synthetic datasets -- a dataset the console can see is a ``meta/info.json``, so
no recording, no hardware and no LeRobot dataset load is needed. What is checked
here is the contract the browser depends on: what the listing carries, what the
destructive and whole-drive routes do, and that a refused merge says why instead
of starting one.

The console's own state (its remembered directories, its render cache) is
redirected into the temporary directory, so a test can never rewrite what the
machine remembers.
"""

import argparse
import asyncio
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from actoris_harena.web.jobs import MAX_JOBS, prune_jobs
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from tool.rig_web import build_app

_FEATURES = {
    "action": {"dtype": "float32", "shape": [12]},
    "observation.images.central": {"dtype": "video", "shape": [480, 640, 3]},
}


def write_dataset(root: Path, name: str, episodes: int = 2, **over) -> Path:
    path = root / name
    (path / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": "so101_dual",
        "total_episodes": episodes,
        "total_frames": 100 * episodes,
        "features": _FEATURES,
    }
    info.update(over)
    (path / "meta" / "info.json").write_text(json.dumps(info))
    return path


class ConsoleTestCase(AioHTTPTestCase):
    dir_given = True

    async def get_application(self):
        # The console stores its state under plain string keys, as it has since
        # it was the standalone dataset browser; aiohttp only recommends its
        # typed keys. Filtered here rather than at import because the test
        # runner resets the warning filters around each run.
        warnings.filterwarnings("ignore", category=web.NotAppKeyWarning)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "drive"
        self.root.mkdir()
        write_dataset(self.root, "cube-pnp", episodes=3)
        write_dataset(self.root, "cube-pnp-2", episodes=2)
        write_dataset(self.root, "stillborn", episodes=0)
        (self.root / "cube-pnp.trash-20260819-110341").mkdir()
        # Everything the console writes about itself goes here, not into the
        # machine's real output directory.
        self._outputs = os.environ.get("SO101_OUTPUT_DIR")
        os.environ["SO101_OUTPUT_DIR"] = str(Path(self.tmp.name) / "outputs")
        app = build_app(
            argparse.Namespace(
                dir=str(self.root) if self.dir_given else None,
                fps=None,
                prerender=False,
                monitor_port=8799,
                mount_dir=str(Path(self.tmp.name) / "mounts"),
            )
        )
        # No test may read the machine's real assignments: a route that opens
        # devices would then open the arms and cameras of whatever rig is
        # plugged into the machine running the tests.
        app["sensor_map_path"] = Path(self.tmp.name) / "sensor_map.yaml"
        # The same file, for the resolvers that reach the map directly rather
        # than through the app. Without this the plan is judged against the
        # rig actually attached to whatever machine runs the suite, so the
        # camera and arm refusals fire or stay silent according to what happens
        # to be plugged in -- and a test that passes only while the rig is
        # connected is not testing the console.
        self._sensor_patch = mock.patch(
            "tool.test_sensor_rates.SENSOR_MAP_PATH", app["sensor_map_path"]
        )
        self._sensor_patch.start()
        return app

    async def tearDownAsync(self):
        await super().tearDownAsync()
        self._sensor_patch.stop()
        if self._outputs is None:
            os.environ.pop("SO101_OUTPUT_DIR", None)
        else:
            os.environ["SO101_OUTPUT_DIR"] = self._outputs
        self.tmp.cleanup()

    async def post(self, url, payload):
        return await self.client.post(url, json=payload)


class TestListing(ConsoleTestCase):
    async def test_index_and_static_are_served(self):
        for url in (
            "/",
            "/static/app.css",
            "/static/datasets.js",
            "/static/training.js",
            # The Training tab is three files now: the launch form, the runs
            # view, and the drawing they share. A page that loads a script the
            # server does not have fails silently in the browser.
            "/static/training_jobs.js",
            "/static/chart.js",
        ):
            resp = await self.client.get(url)
            self.assertEqual(resp.status, 200, url)

    def test_every_element_a_script_reaches_for_exists(self):
        """A typo'd id kills the whole script at load, silently.

        There is no build step and no module system here: every static script
        runs at top level against the one page, so `$('#t-refersh')` throws on
        load and everything after it in that file never runs -- with nothing in
        the server log and nothing on the page but a tab that does not react.
        MEASURED as exactly that while the Training tab was split in two.

        An id the script CREATES itself is fine, so a template literal in the
        same file counts as a definition.
        """
        import re

        static = Path(__file__).resolve().parents[2] / "src/common/web/static"
        markup = set()
        for page in ("index.html", "policy.html"):
            markup |= set(re.findall(r'id="([^"]+)"', (static / page).read_text()))
        for script in sorted(static.glob("*.js")):
            text = script.read_text()
            made = set(re.findall(r'id="([^"]+)"', text))
            used = set(re.findall(r"""\$\(['"]#([\w-]+)['"]\)""", text))
            used |= set(re.findall(r"""querySelector\(['"]#([\w-]+)['"]\)""", text))
            self.assertEqual(
                sorted(used - markup - made), [], f"{script.name} reaches for these"
            )

    async def test_the_page_and_its_scripts_are_revalidated(self):
        # A browser given a validator but no freshness rule may guess one and
        # keep a script from cache: the page then runs against a script written
        # for an older one. MEASURED as exactly that, so both must say no-cache.
        for url in ("/", "/static/app.js", "/static/app.css", "/static/training.js"):
            resp = await self.client.get(url)
            self.assertEqual(resp.headers["Cache-Control"], "no-cache", url)

    async def test_console_reports_the_drive(self):
        body = await (await self.client.get("/api/console")).json()
        self.assertEqual(body["root"], str(self.root))

    async def test_listing_carries_what_the_pane_shows(self):
        body = await (await self.client.get("/api/datasets")).json()
        by_name = {d["name"]: d for d in body}
        self.assertEqual(by_name["cube-pnp"]["episodes"], 3)
        self.assertEqual(by_name["cube-pnp"]["fps"], 30)
        self.assertEqual(by_name["cube-pnp"]["streams"], ["central"])
        self.assertGreater(by_name["cube-pnp"]["bytes"], 0)
        self.assertFalse(by_name["cube-pnp"]["stillborn"])

    async def test_an_empty_dataset_is_listed_so_it_can_be_deleted(self):
        body = await (await self.client.get("/api/datasets")).json()
        names = {d["name"]: d for d in body}
        self.assertTrue(names["stillborn"]["stillborn"])

    async def test_trash_directories_are_not_datasets(self):
        body = await (await self.client.get("/api/datasets")).json()
        self.assertNotIn("cube-pnp.trash-20260819-110341", [d["name"] for d in body])


class TestRenaming(ConsoleTestCase):
    async def test_renaming_a_dataset(self):
        resp = await self.post("/api/datasets/cube-pnp/rename", {"new": "cube_pnp"})
        self.assertEqual(resp.status, 200)
        self.assertTrue((self.root / "cube_pnp").is_dir())
        self.assertFalse((self.root / "cube-pnp").exists())

    async def test_rename_refuses_a_bad_name(self):
        resp = await self.post("/api/datasets/cube-pnp/rename", {"new": "../out"})
        self.assertEqual(resp.status, 400)

    async def test_rename_refuses_an_occupied_name(self):
        resp = await self.post("/api/datasets/cube-pnp/rename", {"new": "cube-pnp-2"})
        self.assertEqual(resp.status, 409)

    async def test_rename_of_a_working_directory_is_a_404(self):
        resp = await self.post(
            "/api/datasets/cube-pnp.trash-20260819-110341/rename", {"new": "back"}
        )
        self.assertEqual(resp.status, 404)


class TestDeleting(ConsoleTestCase):
    async def test_deleting_removes_the_dataset(self):
        resp = await self.post("/api/datasets/stillborn/remove", {})
        self.assertEqual(resp.status, 200)
        self.assertFalse((self.root / "stillborn").exists())

    async def test_deleting_an_unknown_dataset_is_a_404(self):
        resp = await self.post("/api/datasets/nope/remove", {})
        self.assertEqual(resp.status, 404)


class TestNameCheck(ConsoleTestCase):
    async def test_a_free_name_is_ok(self):
        body = await (
            await self.post("/api/datasets/check-name", {"name": "towel"})
        ).json()
        self.assertTrue(body["ok"])

    async def test_a_taken_name_is_reported(self):
        body = await (
            await self.post("/api/datasets/check-name", {"name": "cube-pnp"})
        ).json()
        self.assertFalse(body["ok"])
        self.assertTrue(body["exists"])

    async def test_a_bad_name_says_why(self):
        body = await (
            await self.post("/api/datasets/check-name", {"name": "my dataset"})
        ).json()
        self.assertFalse(body["ok"])
        self.assertIn("spaces", body["problem"])


class TestMergeChecks(ConsoleTestCase):
    async def test_two_matching_datasets_may_merge(self):
        body = await (
            await self.post(
                "/api/datasets/merge-check",
                {"names": ["cube-pnp", "cube-pnp-2"], "name": "cube-all"},
            )
        ).json()
        self.assertTrue(body["ok"], body)

    async def test_a_different_camera_set_is_refused_with_a_reason(self):
        features = dict(_FEATURES)
        features["observation.images.scene"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
        }
        write_dataset(self.root, "scene-set", episodes=1, features=features)
        body = await (
            await self.post(
                "/api/datasets/merge-check",
                {"names": ["cube-pnp", "scene-set"], "name": "mixed"},
            )
        ).json()
        self.assertFalse(body["ok"])
        self.assertTrue(any("scene" in r for r in body["reasons"]))

    async def test_starting_a_refused_merge_is_a_400(self):
        resp = await self.post(
            "/api/datasets/merge", {"names": ["cube-pnp"], "name": "solo"}
        )
        self.assertEqual(resp.status, 400)

    async def test_an_unknown_job_is_a_404(self):
        resp = await self.client.get("/api/datasets/jobs/deadbeef")
        self.assertEqual(resp.status, 404)


class TestCollectTab(ConsoleTestCase):
    """With no session running, the Collect routes must say so, not fail."""

    async def test_the_session_route_reports_an_idle_console(self):
        body = await (await self.client.get("/api/session")).json()
        self.assertFalse(body["running"])
        self.assertIsNone(body["monitor"])
        self.assertEqual(body["preview"], [])

    async def test_the_live_tiles_are_empty_without_a_session_or_preview(self):
        body = await (await self.client.get("/api/live/streams")).json()
        self.assertEqual(body["source"], "preview")
        self.assertEqual(body["streams"], [])

    async def test_a_live_stream_without_a_source_is_refused(self):
        resp = await self.client.get("/api/live/central.mjpg")
        self.assertEqual(resp.status, 409)

    async def test_the_batched_frames_route_answers_an_idle_console(self):
        # It must ANSWER rather than refuse: the live pane polls it continuously
        # and an idle console is the normal state, not an error.
        body = await (await self.client.get("/api/live/frames")).json()
        self.assertEqual(body["source"], "preview")
        self.assertEqual(body["streams"], [])
        self.assertEqual(body["frames"], {})
        self.assertEqual(body["missing"], [])

    async def test_the_batched_frames_route_carries_every_preview_camera(self):
        # One response for N cameras. Each tile used to hold its own endless
        # multipart stream, which spends one of the browser's ~6 connections per
        # origin, so past four tiles the rest never loaded.
        import numpy as np

        names = ["central", "left_arm_left_gripper", "wrist_camera_left"]
        frames = {n: np.full((8, 8, 3), 9, dtype=np.uint8) for n in names}
        frames["left_arm_left_gripper"] = None  # published nothing yet
        preview = self.app["preview"]
        preview.stream_names = lambda: names
        preview.frame = lambda n: frames[n]
        preview.missing = lambda: [{"name": "wrist_camera_right", "reason": "x"}]

        body = await (await self.client.get("/api/live/frames")).json()
        self.assertEqual(body["streams"], names)
        # A camera between frames keeps its tile rather than blanking it.
        self.assertEqual(sorted(body["frames"]), ["central", "wrist_camera_left"])
        self.assertTrue(body["frames"]["central"])
        self.assertEqual(body["missing"][0]["name"], "wrist_camera_right")

    async def test_pressing_the_episode_key_without_a_session_is_refused(self):
        resp = await self.post("/api/session/episode", {})
        self.assertEqual(resp.status, 409)

    async def test_an_episode_without_a_row_says_which_of_the_two_it_is(self):
        # Waiting for a session to commit and cleaning up after one that never
        # did need opposite reactions, and the console knows which it is: the
        # only session that could still be writing is the one it started.
        resp = await self.client.get("/api/datasets/cube-pnp/episodes/0/playback")
        self.assertEqual(resp.status, 409)
        idle = await resp.text()
        self.assertIn("no session is running", idle)

        with mock.patch.object(self.app["session"], "running", return_value=True):
            resp = await self.client.get("/api/datasets/cube-pnp/episodes/0/playback")
            busy = await resp.text()
        self.assertEqual(resp.status, 409)
        self.assertIn("still being written", busy)

    async def test_pressing_any_key_without_a_session_is_refused(self):
        resp = await self.post("/api/session/key", {"key": "y"})
        self.assertEqual(resp.status, 409)

    async def test_stopping_without_a_session_is_refused(self):
        resp = await self.post("/api/session/stop", {})
        self.assertEqual(resp.status, 409)

    async def test_a_plan_is_resolved_without_starting_anything(self):
        body = await (
            await self.post(
                "/api/session/plan", {"name": "towel-fold", "task": "fold the towel"}
            )
        ).json()
        self.assertFalse(body["resuming"])
        self.assertEqual(body["refusals"], [])
        self.assertIn("--enable-camera", body["flags"])

    async def test_a_plan_for_an_existing_dataset_resumes_it(self):
        body = await (
            await self.post(
                "/api/session/plan", {"name": "cube-pnp", "task": "pick the cube"}
            )
        ).json()
        self.assertTrue(body["resuming"])
        self.assertEqual(body["cameras"], ["central"])

    async def test_a_plan_without_an_instruction_is_refused(self):
        body = await (
            await self.post("/api/session/plan", {"name": "towel-fold", "task": ""})
        ).json()
        self.assertTrue(any("instruction" in r for r in body["refusals"]))

    async def test_starting_a_refused_session_is_a_400(self):
        resp = await self.post(
            "/api/session/start", {"name": "../escape", "task": "fold"}
        )
        self.assertEqual(resp.status, 400)

    async def test_nothing_reports_joints_until_something_is_reading_them(self):
        body = await (await self.client.get("/api/session")).json()
        self.assertIsNone(body["preview_joints"])

    async def test_reading_the_arms_without_assignments_says_where_to_make_them(self):
        resp = await self.post("/api/preview/arms/start", {})
        self.assertEqual(resp.status, 409)
        self.assertIn("Signals", await resp.text())

    async def test_stopping_a_reader_that_never_started_is_harmless(self):
        resp = await self.post("/api/preview/arms/stop", {})
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["arms"], [])

    async def test_the_form_says_how_each_input_mode_is_driven(self):
        body = await (await self.client.get("/api/collect/config")).json()
        quest = body["controls"]["quest"]
        self.assertEqual(quest[0]["key"], "Y")
        self.assertEqual(quest[0]["where"], "headset")
        # Leader mode has no headset, and a session the console started has no
        # keyboard either, so enabling is offered to the page there and only
        # there.
        leader = body["controls"]["leader"]
        self.assertEqual(leader[0]["where"], "session keyboard or this page")
        self.assertEqual(
            {s["key"] for s in quest if "this page" in s["where"]}, {"A", "Q"}
        )
        self.assertEqual(
            {s["key"] for s in leader if "this page" in s["where"]}, {"Y", "A", "Q"}
        )

    async def test_the_collection_form_offers_this_machines_cameras(self):
        body = await (await self.client.get("/api/collect/config")).json()
        self.assertTrue(body["cameras"])
        self.assertIn("name", body["cameras"][0])

    async def test_the_form_says_which_cameras_are_actually_plugged_in(self):
        # The page must not pre-tick a camera that is not there: the recorder
        # opens every selected one before it creates the dataset, so a session
        # started on it exits at once.
        from unittest import mock

        real = "/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.2:1.0-video-index0"
        path = Path(self.root) / "sensor_map.yaml"
        path.write_text(f"cameras:\n  central: {real}\n  scene: {path}/gone\n")
        with mock.patch("tool.test_sensor_rates.SENSOR_MAP_PATH", path):
            body = await (await self.client.get("/api/collect/config")).json()
        by_name = {c["name"]: c for c in body["cameras"]}
        self.assertFalse(by_name["scene"]["present"])
        # A camera with no assignment is not judged: a bare device index says
        # nothing about whether it is plugged in.
        self.assertTrue(by_name["wrist_camera_left"]["present"])


class TestSensorsTab(ConsoleTestCase):
    """Assignment writes a real per-machine file, so these point it at a copy."""

    def setUp(self):
        # An assignment is stored as the device's STABLE alias, which is looked
        # up in this machine's /dev -- so on the rig itself the made-up devices
        # below would resolve to whatever is really plugged in, and these tests
        # would pass or fail depending on the hardware attached. The lookup is
        # exercised by the desktop tool's own tests; here it stands aside.
        patcher = mock.patch(
            "common.web.sensors_api.stable_device_path", side_effect=lambda d: d
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()

    async def get_application(self):
        app = await super().get_application()
        app["sensor_map_path"] = self.root / "sensor_map.yaml"
        # Device discovery opens every capture node on the machine; the tab's
        # own scan button is the only thing that should ever do that, so the
        # tests hand it a fixed set instead.
        app["sensor_candidates"] = {
            "cameras": ["/dev/video0", "/dev/video2"],
            "serial": ["/dev/ttyACM0"],
            "realsense": [{"serial": "0123", "name": "D435"}],
        }
        return app

    async def test_an_unassigned_rig_lists_every_name_and_the_devices(self):
        body = await (await self.client.get("/api/sensors")).json()
        self.assertIn("wrist_camera_left", body["names"])
        self.assertEqual(body["candidates"]["cameras"], ["/dev/video0", "/dev/video2"])
        self.assertTrue(all(c["device"] is None for c in body["overview"]["cameras"]))

    async def test_assigning_a_camera_persists_and_shows_up(self):
        resp = await self.post(
            "/api/sensors/camera/assign",
            {"device": "/dev/video2", "name": "wrist_camera_left"},
        )
        self.assertEqual(resp.status, 200)
        body = await (await self.client.get("/api/sensors")).json()
        row = next(
            c for c in body["overview"]["cameras"] if c["name"] == "wrist_camera_left"
        )
        self.assertEqual(row["device"], "/dev/video2")
        self.assertTrue(row["present"])
        self.assertTrue((self.root / "sensor_map.yaml").is_file())

    async def test_a_connected_device_reports_the_name_it_carries(self):
        await self.post(
            "/api/sensors/camera/assign",
            {"device": "/dev/video2", "name": "wrist_camera_left"},
        )
        await self.post(
            "/api/sensors/arm/assign",
            {"port": "/dev/ttyACM0", "role": "follower", "side": "left"},
        )
        body = await (await self.client.get("/api/sensors")).json()
        self.assertEqual(
            body["overview"]["bound"],
            {"/dev/video2": "wrist_camera_left", "/dev/ttyACM0": "follower left"},
        )

    async def test_assigning_an_unknown_name_is_refused(self):
        resp = await self.post(
            "/api/sensors/camera/assign", {"device": "/dev/video2", "name": "nope"}
        )
        self.assertEqual(resp.status, 400)

    async def test_assigning_an_arm_side_persists(self):
        resp = await self.post(
            "/api/sensors/arm/assign",
            {"port": "/dev/ttyACM0", "role": "follower", "side": "right"},
        )
        self.assertEqual(resp.status, 200)
        body = await (await self.client.get("/api/sensors")).json()
        row = next(
            a
            for a in body["overview"]["arms"]
            if a["role"] == "follower" and a["side"] == "right"
        )
        self.assertEqual(row["device"], "/dev/ttyACM0")

    async def test_an_unknown_role_is_refused(self):
        resp = await self.post(
            "/api/sensors/arm/assign",
            {"port": "/dev/ttyACM0", "role": "gripper", "side": "right"},
        )
        self.assertEqual(resp.status, 400)

    async def test_clearing_an_assignment(self):
        await self.post(
            "/api/sensors/camera/assign", {"device": "/dev/video2", "name": "central"}
        )
        resp = await self.post(
            "/api/sensors/clear", {"kind": "camera", "key": "central"}
        )
        self.assertEqual(resp.status, 200)
        body = await (await self.client.get("/api/sensors")).json()
        row = next(c for c in body["overview"]["cameras"] if c["name"] == "central")
        self.assertIsNone(row["device"])

    async def test_the_depth_camera_is_assigned_by_serial(self):
        resp = await self.post("/api/sensors/realsense/assign", {"serial": "0123"})
        self.assertEqual(resp.status, 200)
        body = await (await self.client.get("/api/sensors")).json()
        self.assertTrue(body["overview"]["realsense"]["present"])

    async def test_ticks_without_an_open_port_say_so(self):
        body = await (await self.client.get("/api/sensors/arm/ticks")).json()
        self.assertEqual(body["joints"], [])
        self.assertIn("no port", body["error"])

    async def test_a_request_without_a_device_is_refused(self):
        resp = await self.post("/api/sensors/camera/assign", {"name": "central"})
        self.assertEqual(resp.status, 400)


class TestJobs(ConsoleTestCase):
    """Slow whole-dataset work is a job, and the dock lists it."""

    async def _settle(self, job_id):
        for _ in range(100):
            body = await (await self.client.get(f"/api/datasets/jobs/{job_id}")).json()
            if body["state"] != "running":
                return body
            await asyncio.sleep(0.05)
        raise AssertionError(f"job {job_id} never finished")

    async def test_deleting_a_dataset_is_a_job(self):
        body = await (await self.post("/api/datasets/stillborn/remove", {})).json()
        job = await self._settle(body["job"])
        self.assertEqual(job["kind"], "delete")
        self.assertEqual(job["state"], "done")
        listed = await (await self.client.get("/api/jobs")).json()
        self.assertIn(body["job"], [j["id"] for j in listed])

    async def test_removing_marked_episodes_for_good_is_a_job(self):
        # The rewrite itself is LeRobot's, and far too slow for a test; what is
        # checked here is that the operator gets a job to watch instead of a
        # request that may die half an hour later.
        await self.post("/api/datasets/cube-pnp/delete", {"episodes": [1]})
        with mock.patch(
            "common.web.datasets_api.compact_dataset", return_value=2
        ) as compact:
            body = await (await self.post("/api/datasets/cube-pnp/compact", {})).json()
            job = await self._settle(body["id"])
        self.assertEqual(job["kind"], "compact")
        self.assertEqual(job["state"], "done")
        self.assertIn("2 recording", job["message"])
        self.assertEqual(compact.call_args.args[1], "cube-pnp")

    async def test_a_rewrite_that_fails_says_why_in_the_dock(self):
        await self.post("/api/datasets/cube-pnp/delete", {"episodes": [1]})
        boom = FileNotFoundError("meta/episodes/chunk-000/file-001.parquet")
        with mock.patch("common.web.datasets_api.compact_dataset", side_effect=boom):
            body = await (await self.post("/api/datasets/cube-pnp/compact", {})).json()
            job = await self._settle(body["id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("file-001.parquet", job["message"])

    async def test_removing_nothing_is_refused_before_a_job_is_made(self):
        resp = await self.post("/api/datasets/cube-pnp/compact", {})
        self.assertEqual(resp.status, 400)
        listed = await (await self.client.get("/api/jobs")).json()
        self.assertEqual(listed, [])

    async def test_repairing_a_dataset_that_counts_a_phantom_is_a_job(self):
        # The fixture's datasets are an info.json and nothing else, so every
        # episode they count is one nobody wrote -- the fault at full strength.
        result = {"episodes": 1, "renumbered": 1}
        with mock.patch(
            "common.web.datasets_api.repair_phantom_episodes", return_value=result
        ) as repair:
            body = await (await self.post("/api/datasets/cube-pnp/repair", {})).json()
            job = await self._settle(body["id"])

        self.assertEqual(job["kind"], "repair")
        self.assertEqual(job["state"], "done")
        self.assertIn("1 recording", job["message"])
        self.assertEqual(repair.call_args.args[0].name, "cube-pnp")

    async def test_a_dataset_with_nothing_behind_it_is_never_emptied(self):
        # Every episode missing is a drive that is not mounted far more often
        # than a dataset that truly holds none, so the repair refuses rather
        # than rewriting the metadata that says what used to be here.
        body = await (await self.post("/api/datasets/cube-pnp/repair", {})).json()
        job = await self._settle(body["id"])

        self.assertEqual(job["state"], "failed")
        self.assertIn("no recordings at all", job["message"])
        self.assertTrue((self.root / "cube-pnp" / "meta" / "info.json").exists())

    async def test_repairing_a_whole_dataset_is_refused_with_a_reason(self):
        with mock.patch(
            "common.web.datasets_api.dataset_integrity",
            return_value={
                "ok": True,
                "repairable": False,
                "summary": "",
                "phantom": [],
            },
        ):
            resp = await self.post("/api/datasets/cube-pnp/repair", {})

        self.assertEqual(resp.status, 400)
        self.assertIn("nothing to repair", await resp.text())

    async def test_damage_that_needs_a_decision_is_refused_not_repaired(self):
        report = {
            "ok": False,
            "repairable": False,
            "summary": "episode(s) [2] have a recording but no metadata",
            "phantom": [],
        }
        with mock.patch(
            "common.web.datasets_api.dataset_integrity", return_value=report
        ):
            resp = await self.post("/api/datasets/cube-pnp/repair", {})

        self.assertEqual(resp.status, 400)
        self.assertIn("no metadata", await resp.text())
        listed = await (await self.client.get("/api/jobs")).json()
        self.assertEqual(listed, [])

    async def test_a_merge_may_delete_its_sources_once_it_has_worked(self):
        # The merge itself is LeRobot's; what is checked here is the console's
        # promise about the sources -- they go only after the output exists.
        def fake_merge(root, names, out_name, progress=None):
            write_dataset(Path(root), out_name, episodes=5)
            return Path(root) / out_name

        with mock.patch("common.web.lifecycle_api.merge_datasets", fake_merge):
            started = await (
                await self.post(
                    "/api/datasets/merge",
                    {
                        "names": ["cube-pnp", "cube-pnp-2"],
                        "name": "cube-all",
                        "delete_sources": True,
                    },
                )
            ).json()
            job = await self._settle(started["id"])
        self.assertEqual(job["state"], "done", job["message"])
        self.assertTrue((self.root / "cube-all").is_dir())
        self.assertFalse((self.root / "cube-pnp").exists())
        self.assertFalse((self.root / "cube-pnp-2").exists())

    async def test_a_failed_merge_keeps_every_source(self):
        def angry_merge(root, names, out_name, progress=None):
            raise RuntimeError("no room on the drive")

        with mock.patch("common.web.lifecycle_api.merge_datasets", angry_merge):
            started = await (
                await self.post(
                    "/api/datasets/merge",
                    {
                        "names": ["cube-pnp", "cube-pnp-2"],
                        "name": "cube-all",
                        "delete_sources": True,
                    },
                )
            ).json()
            job = await self._settle(started["id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("no room", job["message"])
        self.assertTrue((self.root / "cube-pnp").is_dir())
        self.assertTrue((self.root / "cube-pnp-2").is_dir())


class TestJobPruning(unittest.TestCase):
    def test_running_jobs_stay_and_old_finished_ones_go(self):
        now = 10_000.0
        jobs = {
            "run": {"id": "run", "state": "running", "started": 0.0},
            "old": {"id": "old", "state": "done", "started": 0.0, "finished": 1.0},
            "new": {"id": "new", "state": "done", "started": now, "finished": now},
        }
        prune_jobs(jobs, now)
        self.assertEqual(sorted(jobs), ["new", "run"])

    def test_only_the_most_recent_finished_jobs_are_kept(self):
        now = 100.0
        jobs = {
            str(i): {"id": str(i), "state": "done", "started": now, "finished": now + i}
            for i in range(MAX_JOBS + 5)
        }
        prune_jobs(jobs, now)
        self.assertEqual(len(jobs), MAX_JOBS)
        self.assertNotIn("0", jobs)


class TestChoosingTheDirectory(ConsoleTestCase):
    async def test_the_current_directory_is_described(self):
        body = await (await self.client.get("/api/roots")).json()
        self.assertEqual(body["root"], str(self.root))
        self.assertEqual(body["kind"], "local")
        self.assertTrue(body["writable"])
        self.assertIsNone(body["busy"])

    async def test_browsing_flags_a_directory_that_holds_datasets(self):
        body = await (
            await self.client.get("/api/roots/browse?path=" + str(Path(self.tmp.name)))
        ).json()
        drive = next(e for e in body["entries"] if e["name"] == "drive")
        self.assertTrue(drive["collection"])

    async def test_browsing_somewhere_that_does_not_exist_is_a_400(self):
        resp = await self.client.get("/api/roots/browse?path=/nowhere-at-all")
        self.assertEqual(resp.status, 400)

    async def test_switching_directory_changes_what_is_listed(self):
        other = Path(self.tmp.name) / "second"
        other.mkdir()
        write_dataset(other, "towel-fold", episodes=1)
        body = await (await self.post("/api/roots/use", {"path": str(other)})).json()
        self.assertEqual(body["root"], str(other))
        self.assertEqual(body["recent"][0]["path"], str(other))
        listed = await (await self.client.get("/api/datasets")).json()
        self.assertEqual([d["name"] for d in listed], ["towel-fold"])
        # The session would record into the new directory too, not the old one.
        self.assertEqual(str(self.app["session"].root), str(other))

    async def test_switching_to_something_that_is_not_a_directory_is_a_400(self):
        resp = await self.post("/api/roots/use", {"path": "/definitely/not/here"})
        self.assertEqual(resp.status, 400)

    async def test_switching_is_refused_while_a_job_runs(self):
        self.app["jobs"]["busy"] = {
            "id": "busy",
            "kind": "merge",
            "name": "cube-all",
            "state": "running",
            "started": 0.0,
        }
        resp = await self.post("/api/roots/use", {"path": str(self.root)})
        self.assertEqual(resp.status, 409)
        self.assertIn("cube-all", await resp.text())


class TestWithoutADirectory(ConsoleTestCase):
    """The console may be started with no drive at all; the page then asks."""

    dir_given = False

    async def test_the_console_reports_no_directory(self):
        body = await (await self.client.get("/api/console")).json()
        self.assertIsNone(body["root"])

    async def test_the_dataset_routes_refuse_and_say_why(self):
        resp = await self.client.get("/api/datasets")
        self.assertEqual(resp.status, 409)
        self.assertIn("collection directory", await resp.text())

    async def test_the_page_the_dock_and_the_directory_routes_still_answer(self):
        for url in ("/", "/api/jobs", "/api/roots", "/api/session"):
            resp = await self.client.get(url)
            self.assertEqual(resp.status, 200, url)

    async def test_the_reading_half_of_the_training_tab_needs_no_drive(self):
        # This is how the tab is actually used: from a laptop watching a GPU
        # box, with no collection drive mounted at all. Only the three routes
        # that read a dataset off the drive require one.
        #
        # Pointed at a machine that is THIS one: the checked-in destinations
        # file names CREATE, and asking it anything from a test means a real
        # SSH and a real timeout, which put this one test at a hundred seconds.
        path = Path(self.tmp.name) / "destinations.yaml"
        path.write_text(
            "here:\n  kind: local\n"
            f"  repo: {Path.cwd()}\n"
            f"  scratch: {Path(self.tmp.name) / 'scratch'}\n"
            "  stage: '{scratch}/local'\n  limits: {}\n"
        )
        self.app["destinations_file"] = path
        for url in (
            "/api/training/machines",
            "/api/training/discovered",
            "/api/training/runs",
            "/api/training/projects",
        ):
            resp = await self.client.get(url)
            self.assertEqual(resp.status, 200, url)
        resp = await self.client.get("/api/training/config")
        self.assertEqual(resp.status, 409)

    async def test_choosing_one_makes_the_datasets_appear(self):
        resp = await self.post("/api/roots/use", {"path": str(self.root)})
        self.assertEqual(resp.status, 200)
        listed = await (await self.client.get("/api/datasets")).json()
        self.assertIn("cube-pnp", [d["name"] for d in listed])


if __name__ == "__main__":
    unittest.main()


class TestTrainingTab(ConsoleTestCase):
    """Sending a dataset to a GPU machine, without a GPU machine.

    Every refusal here is the SAME rule tool/train_launch.py applies, so a run
    started in the browser cannot be one the terminal would have refused. What
    these check is that the routes carry it faithfully -- and that nothing
    reaches for a machine while the operator is still choosing.
    """

    def _dest_file(self):
        path = Path(self.tmp.name) / "destinations.yaml"
        path.write_text(
            "box:\n"
            "  ssh: box\n"
            "  kind: ssh\n"
            "  repo: ~/repo\n"
            "  scratch: ~/scratch\n"
            "  stage: '{scratch}/local'\n"
            "  limits: {pi05: {batch: 2}}\n"
        )
        self.app["destinations_file"] = path
        return path

    async def _plan(self, **over):
        self._dest_file()
        body = {"dataset": "cube-pnp", "dest": "box", "policies": ["act"]}
        body.update(over)
        response = await self.post("/api/training/plan", body)
        # A refusal comes back as JSON at 200; a bad request is plain text, the
        # way the rest of the console reports one.
        if response.status != 200:
            return response.status, await response.text()
        return response.status, await response.json()

    async def test_the_form_offers_the_datasets_on_this_drive(self):
        self._dest_file()
        body = await (await self.client.get("/api/training/config")).json()
        self.assertIn("cube-pnp", [d["name"] for d in body["datasets"]])

    async def test_the_form_says_which_policies_this_lerobot_can_train(self):
        # A gated policy is listed rather than hidden: hiding it would make the
        # pin look like a missing feature instead of a one-line change.
        self._dest_file()
        body = await (await self.client.get("/api/training/config")).json()
        names = {p["name"] for p in body["policies"]}
        self.assertIn("act", names)
        self.assertIn("fastwam", names)
        for policy in body["policies"]:
            self.assertEqual(policy["available"], policy["problem"] is None)

    async def test_the_form_offers_the_machines_the_config_names(self):
        self._dest_file()
        body = await (await self.client.get("/api/training/config")).json()
        self.assertEqual([d["name"] for d in body["destinations"]], ["box"])

    async def test_a_workable_plan_has_no_refusals(self):
        status, plan = await self._plan()
        self.assertEqual(status, 200)
        self.assertEqual(plan["refusals"], [])

    async def test_a_plan_says_what_a_default_will_become(self):
        # The page shows the number the run will actually use, which for a
        # machine with a measured ceiling is not the policy's default.
        status, plan = await self._plan(policies=["pi05"], cameras="central")
        self.assertEqual(plan["rows"][0]["resolved_batch"], 2)

    async def test_a_refusal_comes_back_at_200_so_the_page_can_show_it(self):
        # Not a 400: the operator is still choosing, and a failed request would
        # empty the form rather than annotate it.
        status, plan = await self._plan(cameras="nosuchcamera")
        self.assertEqual(status, 200)
        self.assertTrue(any("nosuchcamera" in r for r in plan["refusals"]))

    async def test_a_plan_touches_no_machine(self):
        # 'box' does not exist. Planning must still answer, because refusing a
        # row is a local decision and asking an unreachable machine about it
        # would make the form unusable off the VPN.
        status, plan = await self._plan()
        self.assertEqual(status, 200)

    async def test_starting_a_refused_run_is_refused(self):
        self._dest_file()
        response = await self.post(
            "/api/training/start",
            {
                "dataset": "cube-pnp",
                "dest": "box",
                "policies": ["act"],
                "cameras": "nosuchcamera",
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("nosuchcamera", await response.text())

    async def test_a_dataset_that_is_not_here_is_a_404(self):
        status, _ = await self._plan(dataset="never-collected")
        self.assertEqual(status, 404)

    async def test_a_machine_that_is_not_configured_lists_the_ones_that_are(self):
        self._dest_file()
        response = await self.post(
            "/api/training/plan",
            {"dataset": "cube-pnp", "dest": "hal9000", "policies": ["act"]},
        )
        self.assertEqual(response.status, 400)
        self.assertIn("box", await response.text())

    async def test_choosing_no_policy_is_refused(self):
        self._dest_file()
        response = await self.post(
            "/api/training/plan", {"dataset": "cube-pnp", "dest": "box", "policies": []}
        )
        self.assertEqual(response.status, 400)

    async def test_nothing_launched_yet_is_an_empty_list(self):
        body = await (await self.client.get("/api/training/runs")).json()
        self.assertEqual(body, [])

    async def test_a_launch_is_handed_the_collection_directory_and_the_name(self):
        # THE bug this class exists for. Handed the dataset's own directory as
        # the collection directory, rsync copied every dataset on the drive
        # into a folder named after the drive.
        self._dest_file()
        seen = {}

        def fake_launch(dest, collection_dir, dataset, rows, info, **kw):
            seen.update(collection_dir=collection_dir, dataset=dataset, rows=rows)
            return {"id": "run-1", "dest": dest["name"], "dataset": dataset}

        with mock.patch("tool.train_launch.launch", fake_launch), mock.patch(
            "common.web.training_api.reachable", return_value=None
        ):
            response = await self.post(
                "/api/training/start",
                {"dataset": "cube-pnp", "dest": "box", "policies": ["act"]},
            )
            self.assertEqual(response.status, 200, await response.text())
            job_id = (await response.json())["id"]
            for _ in range(100):
                await asyncio.sleep(0.05)
                job = await (
                    await self.client.get(f"/api/datasets/jobs/{job_id}")
                ).json()
                if job["state"] != "running":
                    break

        self.assertEqual(job["state"], "done", job["message"])
        self.assertEqual(seen["collection_dir"], self.root)
        self.assertEqual(seen["dataset"], "cube-pnp")
        self.assertEqual(seen["rows"][0]["policy"], "act")

    async def test_a_launch_that_fails_is_reported_in_the_dock(self):
        # It happens on a worker thread, so the request has long since answered
        # 200 -- the record is the only place a failure can be seen.
        self._dest_file()

        def boom(*a, **k):
            raise OSError("the machine said no")

        with mock.patch("tool.train_launch.launch", boom), mock.patch(
            "common.web.training_api.reachable", return_value=None
        ):
            response = await self.post(
                "/api/training/start",
                {"dataset": "cube-pnp", "dest": "box", "policies": ["act"]},
            )
            job_id = (await response.json())["id"]
            for _ in range(100):
                await asyncio.sleep(0.05)
                job = await (
                    await self.client.get(f"/api/datasets/jobs/{job_id}")
                ).json()
                if job["state"] != "running":
                    break
        self.assertEqual(job["state"], "failed")
        self.assertIn("the machine said no", job["message"])

    async def test_an_unreachable_machine_stops_a_launch_before_it_stages(self):
        self._dest_file()
        with mock.patch(
            "common.web.training_api.reachable", return_value="are you on the VPN?"
        ):
            response = await self.post(
                "/api/training/start",
                {"dataset": "cube-pnp", "dest": "box", "policies": ["act"]},
            )
        self.assertEqual(response.status, 502)
        self.assertIn("VPN", await response.text())


class TestWatchingARun(ConsoleTestCase):
    """The progress panel, against a run directory on this machine.

    ``kind: local`` is what makes this testable without a GPU box: the routes
    take exactly the same path they take for thanos or CREATE -- discover, read
    the log, parse it -- with a shell where the ssh would be. So this exercises
    the real command, the real filter and the real parser, and only the machine
    is different.
    """

    def _local_dest(self, outputs: Path) -> Path:
        path = Path(self.tmp.name) / "destinations.yaml"
        path.write_text(
            "here:\n"
            "  kind: local\n"
            f"  repo: {Path.cwd()}\n"
            f"  scratch: {outputs}\n"
            "  stage: '{scratch}/local'\n"
            "  limits: {}\n"
        )
        self.app["destinations_file"] = path
        return path

    def _write_run(self, run="towel__all", policy="act", lines=8, finished=True):
        """A run directory shaped exactly like the drivers leave one."""
        outputs = Path(self.tmp.name) / "scratch"
        logs = outputs / "so101_outputs" / "vla_real_long" / run / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        head = (
            "INFO 2026-08-24 11:50:53 ot_train.py:222 {'batch_size': 8,\n"
            " 'log_freq': 100,\n 'save_freq': 10000,\n 'steps': 800,\n"
            " 'wandb': {'enable': False}}\n"
        )
        body = "".join(
            f"INFO 2026-08-24 11:5{i % 10}:09 ot_train.py:596 step:{(i + 1) * 100} "
            f"smpl:800 ep:2 epch:0.0{i} loss:{10.0 - i:.3f} grdn:210.415 lr:1.0e-05 "
            f"updt_s:0.135 data_s:0.013 smp/s:54 mem_gb:5.7{i}\n"
            for i in range(lines)
        )
        end = (
            "INFO 2026-08-24 14:23:34 ot_train.py:641 Checkpoint policy after step 800\n"
            "INFO 2026-08-24 14:23:35 ot_train.py:721 End of training\n"
            if finished
            else ""
        )
        (logs / f"train_{policy}.log").write_text(head + body + end)
        self._local_dest(outputs)
        return outputs

    async def test_a_machine_that_is_here_answers_without_a_network(self):
        self._write_run()
        machines = await (await self.client.get("/api/training/machines")).json()
        self.assertEqual([m["name"] for m in machines], ["here"])
        self.assertTrue(machines[0]["ok"])
        # And it says what it measured about itself, which no remote does.
        self.assertIsNotNone(machines[0]["measured"])

    async def test_runs_are_discovered_not_listed_from_what_we_launched(self):
        # Nine of the run directories on CREATE were started from a terminal.
        # A viewer that listed only the launcher's own records would show an
        # empty page beside a full drive.
        self._write_run(run="towel__all", policy="act")
        found = await (await self.client.get("/api/training/discovered")).json()
        self.assertEqual(found["problems"], {})
        self.assertEqual(len(found["runs"]), 1)
        entry = found["runs"][0]
        self.assertEqual(entry["run"], "towel__all")
        self.assertEqual(entry["policy"], "act")
        # With nothing recorded, the directory still names its dataset.
        self.assertEqual(entry["dataset"], "towel")

    async def test_the_curve_comes_back_with_an_exact_step_axis(self):
        self._write_run(lines=8)
        response = await self.client.get(
            "/api/training/progress?dest=here&run=towel__all&policy=act"
        )
        body = await response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(
            [p["step"] for p in body["points"]], list(range(100, 900, 100))
        )
        self.assertEqual(body["summary"]["state"], "done")
        self.assertEqual(body["summary"]["checkpoint"], 800)
        self.assertEqual(body["problems"], [])

    async def test_a_run_still_going_is_not_called_finished(self):
        self._write_run(finished=False)
        body = await (
            await self.client.get(
                "/api/training/progress?dest=here&run=towel__all&policy=act"
            )
        ).json()
        self.assertEqual(body["summary"]["state"], "running")
        self.assertIsNone(body["summary"]["checkpoint"])

    async def test_a_log_that_is_not_there_says_so_rather_than_raising(self):
        self._write_run()
        body = await (
            await self.client.get(
                "/api/training/progress?dest=here&run=towel__all&policy=diffusion"
            )
        ).json()
        self.assertFalse(body["ok"])
        self.assertIn("no log at", body["problem"])

    async def test_a_name_that_could_not_be_sent_is_refused(self):
        self._write_run()
        response = await self.client.get(
            "/api/training/progress?dest=here&run=towel;rm -rf /&policy=act"
        )
        self.assertEqual(response.status, 400)
        self.assertIn("plain name", await response.text())

    async def test_every_metric_in_the_log_comes_back_as_a_series(self):
        # The page builds its panel grid from `metrics`, so a metric the parser
        # produced but the payload dropped is a panel that silently never
        # appears.
        self._write_run(lines=4)
        body = await (
            await self.client.get(
                "/api/training/progress?dest=here&run=towel__all&policy=act"
            )
        ).json()
        self.assertIn("train/loss", body["metrics"])
        self.assertIn("train/gpu_mem_gb", body["metrics"])
        self.assertEqual(
            [p["step"] for p in body["series"]["train/loss"]],
            list(range(100, 500, 100)),
        )

    async def test_a_repo_metric_line_survives_the_whole_route(self):
        # Emitted by a policy, kept by the far-side grep, parsed, and served --
        # with no name of it written anywhere in between.
        from common.training import metrics

        outputs = self._write_run(lines=2)
        log = outputs / "so101_outputs/vla_real_long/towel__all/logs/train_act.log"
        log.write_text(
            log.read_text()
            + metrics.format_line(200, {"eval/psnr_central": 31.2})
            + "\n"
        )
        body = await (
            await self.client.get(
                "/api/training/progress?dest=here&run=towel__all&policy=act"
            )
        ).json()
        self.assertIn("eval/psnr_central", body["metrics"])
        self.assertEqual(body["series"]["eval/psnr_central"][0]["value"], 31.2)

    async def test_a_real_run_has_no_rollout_success_curve(self):
        # Per-checkpoint success only exists where a simulator produced it. A
        # real run showing one would be reporting a measurement nobody made.
        self._write_run()
        body = await (
            await self.client.get(
                "/api/training/progress?dest=here&run=towel__all&policy=act"
            )
        ).json()
        self.assertNotIn("eval/success_rate", body["metrics"])
        self.assertIsNone(body["selected"])

    async def test_a_cell_that_is_not_three_segments_is_refused(self):
        self._write_run()
        response = await self.client.get(
            "/api/training/progress?dest=here&run=towel__all&policy=act&cell=a/b"
        )
        self.assertEqual(response.status, 400)
        self.assertIn("<mode>/<task>/<policy>", await response.text())

    async def test_the_page_is_told_which_policies_are_ours(self):
        # Without this the operator sees nine unstructured checkboxes and
        # nothing says `act` and `so101_act` are the same model.
        body = await (await self.client.get("/api/training/config")).json()
        by_name = {p["name"]: p for p in body["policies"]}
        self.assertFalse(by_name["act"]["local"])
        self.assertTrue(by_name["so101_act"]["local"])
        self.assertEqual(by_name["so101_act"]["ported_from"], "act")
        self.assertIsNone(by_name["act"]["ported_from"])

    # ── Projects ────────────────────────────────────────────────────────────

    async def _projects(self):
        return await (await self.client.get("/api/training/projects")).json()

    async def test_the_built_in_groupings_exist_before_any_project_does(self):
        # Twenty-one run directories already exist on these machines and none
        # of them was put in a project by anyone, so the view has to be useful
        # with the store empty.
        body = await self._projects()
        self.assertEqual(body["projects"], [])
        self.assertEqual(
            [b["name"] for b in body["builtin"]], ["__all__", "__unassigned__"]
        )

    async def test_a_project_is_created_and_runs_move_in_and_out(self):
        self._write_run()
        created = await self.client.post(
            "/api/training/projects", json={"name": "port parity"}
        )
        self.assertEqual(created.status, 200)
        key = "here|towel__all|act"
        await self.client.patch(
            "/api/training/projects/port%20parity", json={"add": [key]}
        )
        body = await self._projects()
        self.assertEqual(body["projects"][0]["runs"], [key])
        await self.client.patch(
            "/api/training/projects/port%20parity", json={"remove": [key]}
        )
        self.assertEqual((await self._projects())["projects"][0]["runs"], [])

    async def test_a_duplicate_name_comes_back_as_a_sentence(self):
        await self.client.post("/api/training/projects", json={"name": "ports"})
        again = await self.client.post("/api/training/projects", json={"name": "ports"})
        self.assertEqual(again.status, 400)
        self.assertIn("already a project", await again.text())

    async def test_deleting_a_project_says_the_runs_are_kept(self):
        await self.client.post("/api/training/projects", json={"name": "ports"})
        response = await self.client.delete("/api/training/projects/ports")
        body = await response.json()
        self.assertTrue(body["runs_kept"])
        self.assertEqual((await self._projects())["projects"], [])

    async def test_a_rename_keeps_the_membership(self):
        await self.client.post("/api/training/projects", json={"name": "ports"})
        key = "here|towel__all|act"
        await self.client.patch("/api/training/projects/ports", json={"add": [key]})
        await self.client.patch(
            "/api/training/projects/ports", json={"name": "port parity"}
        )
        body = await self._projects()
        self.assertEqual(body["projects"][0]["name"], "port parity")
        self.assertEqual(body["projects"][0]["runs"], [key])

    async def test_something_that_is_not_a_run_key_is_refused(self):
        await self.client.post("/api/training/projects", json={"name": "ports"})
        response = await self.client.patch(
            "/api/training/projects/ports", json={"add": ["towel__all"]}
        )
        self.assertEqual(response.status, 400)
        self.assertIn("not a run key", await response.text())

    async def test_the_answer_is_cached_so_polling_costs_one_read(self):
        outputs = self._write_run()
        url = "/api/training/progress?dest=here&run=towel__all&policy=act"
        first = await (await self.client.get(url)).json()
        # Change the log underneath: a second call inside the TTL must still
        # show the first answer, which is what bounds how often several run
        # cards on a page disturb a machine.
        log = outputs / "so101_outputs/vla_real_long/towel__all/logs/train_act.log"
        log.unlink()
        second = await (await self.client.get(url)).json()
        self.assertEqual(first["summary"], second["summary"])
        # ...and asking for it fresh goes back to the file.
        third = await (await self.client.get(url + "&refresh=1")).json()
        self.assertFalse(third["ok"])
