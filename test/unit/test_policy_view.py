"""Route tests for the rollout view (aiohttp TestClient).

The view is the one part of a rollout a person can touch while the arms are
under torque, so what is checked here is the contract between the page and the
run: the status carries what the page draws, each button reaches the throttle,
and a request the run cannot honour is refused with a reason rather than
silently ignored.

Nothing here opens a camera, a bus or a renderer -- the view is handed stubs,
which is also the point of its design: it reads a snapshot and a frame store,
and owns neither.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_view
"""

import unittest
import warnings

import numpy as np
from aiohttp.test_utils import AioHTTPTestCase

from common.policy_run import RunControl
from common.web.policy_view import PolicyView, chunk_payload, pending_payload

CAMERAS = ["central", "wrist_camera_left"]


def frame(value: int = 120) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


class StubSource:
    """A remote source that has answered once."""

    chunked = True
    actions = 32
    threshold = 27
    round_trip_s = 0.55
    server_infer_s = 0.52
    last_chunk_seq = 3
    last_chunk_at = 1000.0

    def __init__(self, sent=True, chunk=True):
        self.last_chunk = np.tile(np.arange(12.0), (32, 1)) if chunk else None
        self._sent = (np.arange(12.0), {c: frame() for c in CAMERAS}) if sent else None

    def last_sent(self):
        return self._sent


class StubDataManager:
    def get_rgb_image(self, name):
        return frame(200)


class ViewTestCase(AioHTTPTestCase):
    source_kwargs: dict = {}

    async def get_application(self):
        warnings.filterwarnings("ignore", message=".*app\\[.*")
        self.control = RunControl()
        self.control.publish(hz=30.0, task="pick up the cube", ticks_total=900)
        self.source = StubSource(**self.source_kwargs)
        self.view = PolicyView(
            self.control, self.source, StubDataManager(), CAMERAS, port=0
        )
        return self.view.build_app()

    async def status(self):
        return await (await self.client.get("/api/status")).json()

    async def mode(self, mode: str):
        return await self.client.post("/api/mode", json={"mode": mode})


class TestStatus(ViewTestCase):
    async def test_it_carries_what_the_page_draws(self):
        body = await self.status()

        self.assertEqual(body["mode"], "run")
        self.assertEqual(body["cameras"], CAMERAS)
        self.assertEqual(body["task"], "pick up the cube")
        self.assertEqual(body["chunk"]["seq"], 3)
        self.assertEqual(body["chunk"]["n"], 32)
        self.assertEqual(len(body["chunk"]["actions"][0]), 12)
        self.assertEqual(len(body["sent_state"]), 12)
        self.assertEqual(body["threshold"], 27)

    async def test_it_follows_the_loop_between_polls(self):
        self.control.publish(tick=42, queue=9, state=[0.0] * 12)

        body = await self.status()

        self.assertEqual(body["tick"], 42)
        self.assertEqual(body["queue"], 9)


class TestThrottle(ViewTestCase):
    async def test_every_button_reaches_the_run(self):
        for mode in ("hold", "step", "run"):
            response = await self.mode(mode)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["mode"], mode)
            self.assertEqual(self.control.mode, mode)

    async def test_stop_ends_the_run_without_changing_its_mode(self):
        response = await self.mode("stop")

        self.assertEqual(response.status, 200)
        self.assertTrue(self.control.stopping)
        self.assertEqual(self.control.mode, "run")

    async def test_an_unknown_mode_is_refused_with_the_ones_that_exist(self):
        response = await self.mode("faster")

        self.assertEqual(response.status, 400)
        self.assertIn("run", await response.text())
        self.assertEqual(self.control.mode, "run")


class TestPictures(ViewTestCase):
    async def test_the_frames_the_policy_was_sent_are_served_as_jpeg(self):
        response = await self.client.get("/shown/central.jpg")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "image/jpeg")
        self.assertTrue((await response.read()).startswith(b"\xff\xd8"))

    async def test_a_camera_the_run_does_not_have_is_not_found(self):
        self.assertEqual((await self.client.get("/stream/nose.mjpg")).status, 404)


class TestNothingSentYet(ViewTestCase):
    source_kwargs = {"sent": False, "chunk": False}

    async def test_the_page_gets_nulls_rather_than_an_error(self):
        body = await self.status()

        self.assertIsNone(body["chunk"])
        self.assertIsNone(body["sent_state"])

    async def test_a_frame_that_was_never_sent_is_not_found(self):
        self.assertEqual((await self.client.get("/shown/central.jpg")).status, 404)


class TestLocalRun(ViewTestCase):
    """Inference in the same process: there is no chunk to step through."""

    async def get_application(self):
        app = await super().get_application()
        self.source.chunked = False
        return app

    async def test_stepping_is_refused_with_the_reason(self):
        response = await self.mode("step")

        self.assertEqual(response.status, 400)
        self.assertIn("chunk", await response.text())
        self.assertEqual(self.control.mode, "run")


class TestChunkPayload(unittest.TestCase):
    def test_a_source_that_has_not_answered_yet_has_no_plan(self):
        self.assertIsNone(chunk_payload(StubSource(chunk=False)))

    def test_the_plan_is_rounded_but_whole(self):
        payload = chunk_payload(StubSource())

        self.assertEqual(payload["n"], 32)
        self.assertEqual(payload["actions"][0][11], 11.0)


class TestArming(ViewTestCase):
    """Consent to move the arms, when a run delegates it to the page."""

    async def get_application(self):
        warnings.filterwarnings("ignore", message=".*app\\[.*")
        self.control = RunControl()
        self.control.publish(dry_run=False)
        self.source = StubSource()
        self.view = PolicyView(
            self.control,
            self.source,
            StubDataManager(),
            CAMERAS,
            port=0,
            arm_from_view=True,
        )
        return self.view.build_app()

    async def test_the_page_reports_that_it_must_arm_this_run(self):
        body = await self.status()
        self.assertTrue(body["arm_from_view"])
        self.assertFalse(body["armed"])

    async def test_arming_is_recorded_and_does_not_change_the_mode(self):
        before = self.control.mode
        await self.mode("arm")
        self.assertTrue(self.control.armed)
        self.assertEqual(self.control.mode, before)

    async def test_arming_twice_is_harmless(self):
        for _ in range(2):
            await self.mode("arm")
        self.assertTrue(self.control.armed)


class TestArmingRefused(ViewTestCase):
    """A run that took its consent at the terminal keeps it there."""

    async def test_the_page_cannot_open_a_second_door_to_the_torque(self):
        response = await self.mode("arm")
        self.assertEqual(response.status, 400)
        self.assertFalse(self.control.armed)

    async def test_and_does_not_offer_the_button(self):
        body = await self.status()
        self.assertFalse(body["arm_from_view"])


class TestPortAlreadyTaken(unittest.TestCase):
    """A view that cannot bind must say so, not serve somebody else's run."""

    def test_start_reports_failure_rather_than_pretending(self):
        import socket

        taken = socket.socket()
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        self.addCleanup(taken.close)
        port = taken.getsockname()[1]

        view = PolicyView(
            RunControl(), StubSource(), StubDataManager(), CAMERAS, port=port
        )
        self.addCleanup(view.stop)
        self.assertFalse(view.start())
        self.assertIsNotNone(view.error)

    def test_a_free_port_starts_cleanly(self):
        view = PolicyView(
            RunControl(), StubSource(), StubDataManager(), CAMERAS, port=0
        )
        self.addCleanup(view.stop)
        self.assertTrue(view.start())
        self.assertIsNone(view.error)


class TestTwinAnchoring(unittest.TestCase):
    """The twin must animate ONE sequence, not alternate between two."""

    def test_the_two_payloads_are_distinguishable(self):
        # They share a seq -- they describe the same plan -- so 'kind' is the
        # only thing that stops the animation swapping lists mid-walk.
        source = StubSource()
        source.pending = lambda: np.zeros((4, 12))
        plan = chunk_payload(source)
        queued = pending_payload(source)
        self.assertEqual(plan["seq"], queued["seq"])
        self.assertNotEqual(plan["kind"], queued["kind"])

    def test_a_source_with_nothing_queued_has_no_pending_payload(self):
        self.assertIsNone(pending_payload(StubSource()))


if __name__ == "__main__":
    unittest.main()
