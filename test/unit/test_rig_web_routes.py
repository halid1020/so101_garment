"""Route tests for the rig console's dataset API (aiohttp TestClient).

Runs the real application over a temporary collection directory holding
synthetic datasets -- a dataset the console can see is a ``meta/info.json``, so
no recording, no hardware and no LeRobot dataset load is needed. What is checked
here is the contract the browser depends on: what the listing carries, that the
destructive routes are gated by ``--allow-delete``, and that a refused merge
says why instead of starting one.
"""

import argparse
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

# The console stores its state under plain string keys, as this application has
# since it was the standalone dataset browser. aiohttp only recommends its typed
# keys; the advice would be one refactor of every handler, and it is not what
# these tests are about.
warnings.filterwarnings("ignore", category=web.NotAppKeyWarning)

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
    allow_delete = False

    async def get_application(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_dataset(self.root, "cube-pnp", episodes=3)
        write_dataset(self.root, "cube-pnp-2", episodes=2)
        write_dataset(self.root, "stillborn", episodes=0)
        (self.root / "cube-pnp.trash-20260819-110341").mkdir()
        return build_app(
            argparse.Namespace(
                dir=str(self.root),
                allow_delete=self.allow_delete,
                fps=None,
                prerender=False,
            )
        )

    async def tearDownAsync(self):
        await super().tearDownAsync()
        self.tmp.cleanup()

    async def post(self, url, payload):
        return await self.client.post(url, json=payload)


class TestListing(ConsoleTestCase):
    async def test_index_and_static_are_served(self):
        for url in ("/", "/static/app.css", "/static/datasets.js"):
            resp = await self.client.get(url)
            self.assertEqual(resp.status, 200, url)

    async def test_console_reports_the_drive_and_the_mode(self):
        body = await (await self.client.get("/api/console")).json()
        self.assertEqual(body["root"], str(self.root))
        self.assertFalse(body["allow_delete"])

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


class TestGating(ConsoleTestCase):
    async def test_dataset_deletion_is_refused_without_allow_delete(self):
        resp = await self.post("/api/datasets/cube-pnp/remove", {})
        self.assertEqual(resp.status, 403)
        self.assertTrue((self.root / "cube-pnp").is_dir())

    async def test_renaming_does_not_need_allow_delete(self):
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
    allow_delete = True

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


if __name__ == "__main__":
    unittest.main()
