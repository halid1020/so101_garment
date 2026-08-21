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

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from common.web.jobs import MAX_JOBS, prune_jobs
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
        return app

    async def tearDownAsync(self):
        await super().tearDownAsync()
        if self._outputs is None:
            os.environ.pop("SO101_OUTPUT_DIR", None)
        else:
            os.environ["SO101_OUTPUT_DIR"] = self._outputs
        self.tmp.cleanup()

    async def post(self, url, payload):
        return await self.client.post(url, json=payload)


class TestListing(ConsoleTestCase):
    async def test_index_and_static_are_served(self):
        for url in ("/", "/static/app.css", "/static/datasets.js"):
            resp = await self.client.get(url)
            self.assertEqual(resp.status, 200, url)

    async def test_the_page_and_its_scripts_are_revalidated(self):
        # A browser given a validator but no freshness rule may guess one and
        # keep a script from cache: the page then runs against a script written
        # for an older one. MEASURED as exactly that, so both must say no-cache.
        for url in ("/", "/static/app.js", "/static/app.css"):
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

    async def test_choosing_one_makes_the_datasets_appear(self):
        resp = await self.post("/api/roots/use", {"path": str(self.root)})
        self.assertEqual(resp.status, 200)
        listed = await (await self.client.get("/api/datasets")).json()
        self.assertIn("cube-pnp", [d["name"] for d in listed])


if __name__ == "__main__":
    unittest.main()
