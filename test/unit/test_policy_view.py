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

        self.task = "pick up the cube"

    def last_sent(self):
        return self._sent

    def set_task(self, task):
        self.task = task
        return task


class _SpliceSource(StubSource):
    """A chunked source that records the splice it was asked for."""

    def __init__(self):
        super().__init__()
        self.strategy = "append"
        self.params: dict = {}

    def set_strategy(self, strategy=None, **params):
        from common.chunking import STRATEGIES, ChunkingError

        if strategy is not None and strategy not in STRATEGIES:
            raise ChunkingError(f"unknown strategy: {strategy}")
        if strategy is not None:
            self.strategy = strategy
        self.params.update(params)
        return self.settings()

    def settings(self):
        return {
            "strategy": self.strategy,
            "execute_ratio": self.params.get("execute_ratio", 0.5),
            "blend_window": self.params.get("blend_window", 5),
            "new_weight": self.params.get("new_weight", 0.7),
            "ramp_kind": "linear",
            "blocking": self.strategy in ("sync", "receding"),
        }


class StubTwin:
    """Stands in for the viser scene: records the pair it was pointed at."""

    url = "http://127.0.0.1:8768/"

    def __init__(self):
        self.calls: list = []

    def show(self, now12, plan12):
        self.calls.append((now12, plan12))

    def stop(self):
        pass


class StubDataManager:
    def get_rgb_image(self, name):
        return frame(200)


class ViewTestCase(AioHTTPTestCase):
    source_kwargs: dict = {}
    #: What the run can put back between attempts: None, "sim" or "manual".
    resettable: "str | None" = None
    #: Whether this run delegated its 'the arms will move' consent to the page.
    arm_from_view: bool = False

    async def get_application(self):
        warnings.filterwarnings("ignore", message=".*app\\[.*")
        self.control = RunControl()
        self.control.publish(hz=30.0, task="pick up the cube", ticks_total=900)
        self.source = StubSource(**self.source_kwargs)
        # twin_port=0: no viser server in a unit test. The twin is stubbed in
        # the cases that care about it.
        self.view = PolicyView(
            self.control,
            self.source,
            StubDataManager(),
            CAMERAS,
            port=0,
            twin_port=0,
            resettable=self.resettable,
            arm_from_view=self.arm_from_view,
        )
        return self.view.build_app()

    async def status(self):
        return await (await self.client.get("/api/status")).json()

    async def mode(self, mode: str):
        return await self.client.post("/api/mode", json={"mode": mode})

    async def reset(self):
        return await self.client.post("/api/reset")


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

    async def test_stop_ends_the_trial_and_holds_rather_than_ending_the_run(self):
        response = await self.mode("stop")

        self.assertEqual(response.status, 200)
        self.assertTrue(self.control.stop_requested())
        self.assertEqual(self.control.mode, "hold")

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


class TestResetNotOffered(ViewTestCase):
    """A run with no scene it can begin again does not pretend it has one."""

    async def test_the_page_is_told_there_is_nothing_to_reset(self):
        self.assertIsNone((await self.status())["resettable"])

    async def test_and_asking_anyway_is_refused_rather_than_ignored(self):
        response = await self.reset()

        self.assertEqual(response.status, 400)
        self.assertFalse(self.control.reset_requested())


class TestResetInTheTwin(ViewTestCase):
    """A rehearsal can put its own scene back, so the loop is simply told."""

    resettable = "sim"

    async def test_the_page_is_told_the_scene_can_be_put_back(self):
        self.assertEqual((await self.status())["resettable"], "sim")

    async def test_resetting_under_a_running_policy_is_allowed_now(self):
        # It releases the arms as part of what it does, so the motion it would
        # have interrupted is over either way; a Hold click first was friction.
        response = await self.reset()

        self.assertEqual(response.status, 200)
        self.assertTrue(self.control.reset_requested())

    async def test_a_run_may_begin_again_and_the_loop_hears_it_once(self):
        await self.mode("hold")

        response = await self.reset()

        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["instruction"], "")
        self.assertTrue(self.control.reset_requested())
        self.assertFalse(self.control.reset_requested())

    async def test_and_the_throttle_is_dropped_to_hold(self):
        # Whatever it was doing, it is not doing it to arms about to go free.
        await self.mode("preview")
        await self.reset()

        self.assertEqual(self.control.mode, "hold")


class TestResetOnTheBench(ViewTestCase):
    """No button tidies a real table, so the answer says who has to."""

    resettable = "manual"

    async def test_the_answer_carries_the_instruction_rather_than_moving_anything(self):
        await self.mode("hold")

        body = await (await self.reset()).json()

        self.assertEqual(body["resettable"], "manual")
        self.assertIn("by hand", body["instruction"])
        self.assertTrue(self.control.reset_requested())


class TestTheArmsHaveBeenReleased(ViewTestCase):
    """After a trial ends, the page must ask for consent again before motion."""

    arm_from_view = True

    async def test_the_page_reports_whether_the_arms_are_live(self):
        self.assertFalse((await self.status())["torque"])
        self.control.publish(torque=True)
        self.assertTrue((await self.status())["torque"])

    async def test_running_a_disarmed_run_is_refused_with_the_reason(self):
        response = await self.mode("run")

        self.assertEqual(response.status, 400)
        self.assertIn("released", await response.text())
        self.assertEqual(self.control.mode, "run")

    async def test_and_so_is_stepping_one_chunk(self):
        self.assertEqual((await self.mode("step")).status, 400)

    async def test_but_holding_and_previewing_move_nothing_and_are_allowed(self):
        for mode in ("hold", "preview"):
            self.assertEqual((await self.mode(mode)).status, 200)

    async def test_arming_again_lets_the_next_trial_run(self):
        await self.mode("arm")

        self.assertEqual((await self.mode("run")).status, 200)


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
            RunControl(),
            StubSource(),
            StubDataManager(),
            CAMERAS,
            port=port,
            twin_port=0,
        )
        self.addCleanup(view.stop)
        self.assertFalse(view.start())
        self.assertIsNotNone(view.error)

    def test_a_free_port_starts_cleanly(self):
        view = PolicyView(
            RunControl(),
            StubSource(),
            StubDataManager(),
            CAMERAS,
            port=0,
            twin_port=0,
        )
        self.addCleanup(view.stop)
        self.assertTrue(view.start())
        self.assertIsNone(view.error)


class TestPendingPayload(unittest.TestCase):
    """What is still going to be executed, which is not what the policy said."""

    def test_the_queue_is_reported_separately_from_the_plan(self):
        source = StubSource()
        source.pending = lambda: np.zeros((4, 12))
        plan = chunk_payload(source)
        queued = pending_payload(source)
        self.assertEqual(plan["n"], 32)
        self.assertEqual(queued["n"], 4)
        self.assertNotEqual(plan["kind"], queued["kind"])

    def test_a_source_with_nothing_queued_has_no_pending_payload(self):
        self.assertIsNone(pending_payload(StubSource()))


class TestTwinFrame(ViewTestCase):
    """One action, named by the caller. See policy_view.handle_twin."""

    async def get_application(self):
        app = await super().get_application()
        self.twin = StubTwin()
        self.view._twin = self.twin
        self.control.publish(state=[7.0] * 12)
        return app

    async def test_the_named_action_is_what_the_twin_is_pointed_at(self):
        response = await self.client.get("/twin/at?seq=3&i=5")

        # Nothing comes back: the picture is drawn by viser, in the browser.
        self.assertEqual(response.status, 204)
        now, plan = self.twin.calls[-1]
        self.assertEqual(list(now), [7.0] * 12)
        np.testing.assert_allclose(plan, self.source.last_chunk[5])

    async def test_an_index_past_the_end_wraps_rather_than_failing(self):
        self.assertEqual((await self.client.get("/twin/at?seq=3&i=99")).status, 204)
        np.testing.assert_allclose(
            self.twin.calls[-1][1], self.source.last_chunk[99 % 32]
        )

    async def test_a_plan_that_has_been_replaced_is_a_conflict_not_a_pose(self):
        # The page is one poll behind. It must be told so it can re-read the
        # status -- and meanwhile keep the pose it already has, rather than
        # having the scene moved to an action of a different plan.
        response = await self.client.get("/twin/at?seq=2&i=0")

        self.assertEqual(response.status, 409)
        self.assertIn("#3", await response.text())
        self.assertEqual(self.twin.calls, [])

    async def test_a_seq_that_is_not_a_number_is_refused(self):
        self.assertEqual((await self.client.get("/twin/at?seq=soon")).status, 400)

    async def test_the_rendered_twin_is_gone(self):
        # A single fixed camera angle could not answer the question; see
        # policy_ghost. Both the stream and the still image are retired.
        self.assertEqual((await self.client.get("/twin.jpg?seq=3&i=0")).status, 404)
        self.assertEqual((await self.client.get("/twin.mjpg")).status, 404)

    async def test_the_page_is_told_where_to_point_the_iframe(self):
        self.assertEqual((await self.status())["twin_url"], StubTwin.url)


class TestTwinBeforeAnyPlan(ViewTestCase):
    source_kwargs = {"sent": False, "chunk": False}

    async def test_there_is_nothing_to_point_at_yet(self):
        self.view._twin = StubTwin()

        self.assertEqual((await self.client.get("/twin/at?seq=0")).status, 404)


class TestTwinThatWouldNotStart(ViewTestCase):
    """A missing 3D view is survivable; a missing PAGE is not."""

    async def test_the_page_is_told_why_rather_than_pointed_at_nothing(self):
        self.view.twin_error = "OSError: address already in use"

        status = await self.status()

        self.assertIsNone(status["twin_url"])
        self.assertIn("address already in use", status["twin_error"])

    async def test_and_the_route_says_so_instead_of_failing_obscurely(self):
        self.assertEqual((await self.client.get("/twin/at?seq=3&i=0")).status, 404)


class TestTaskRoute(ViewTestCase):
    """A run may be started with no task and given one here."""

    async def test_setting_it_reaches_the_source_and_the_snapshot(self):
        response = await self.client.post("/api/task", json={"task": "fold it"})

        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["task"], "fold it")
        self.assertEqual(self.source.task, "fold it")
        self.assertEqual(self.control.snapshot()["task"], "fold it")

    async def test_surrounding_space_is_not_a_task(self):
        response = await self.client.post("/api/task", json={"task": "   "})

        self.assertEqual(response.status, 400)
        self.assertEqual(self.source.task, "pick up the cube")


class TestJudgingAnAttempt(ViewTestCase):
    """The verdict route: the one control here that touches nothing."""

    async def get_application(self):
        app = await super().get_application()
        self.said: "list[tuple[str, str]]" = []
        self.trial = 0

        def on_outcome(outcome, notes):
            self.said.append((outcome, notes))
            return {"trial": self.trial, "outcome": outcome, "notes": notes, "t": 1.0}

        self.view.on_outcome = on_outcome
        self.view.logging = True
        return app

    async def judge(self, **body):
        return await self.client.post("/api/outcome", json=body)

    async def test_a_verdict_reaches_the_run_and_comes_back(self):
        response = await self.judge(outcome="success", notes="clean fold")

        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["outcome"], "success")
        self.assertEqual(self.said, [("success", "clean fold")])

    async def test_the_page_is_told_what_it_already_recorded(self):
        # So a reload does not lose which button was lit.
        await self.judge(outcome="failure", notes="missed the hem")

        verdicts = (await self.status())["verdicts"]
        self.assertEqual(verdicts["0"]["outcome"], "failure")

    async def test_notes_are_optional(self):
        self.assertEqual((await self.judge(outcome="discard")).status, 200)
        self.assertEqual(self.said, [("discard", "")])

    async def test_an_outcome_nothing_can_score_is_refused(self):
        for bad in ("", "maybe", "SUCCESS"):
            response = await self.judge(outcome=bad)
            self.assertEqual(response.status, 400)
        self.assertEqual(self.said, [])

    async def test_judging_moves_nothing(self):
        # Deliberate: it is allowed mid-run precisely because it is inert.
        before = self.control.mode
        await self.judge(outcome="success")
        self.assertEqual(self.control.mode, before)

    async def test_looking_again_replaces_what_the_page_shows(self):
        await self.judge(outcome="failure")
        await self.judge(outcome="success", notes="it had folded after all")

        verdicts = (await self.status())["verdicts"]
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts["0"]["outcome"], "success")


class TestJudgingARunThatKeepsNoLog(ViewTestCase):
    """--no-log: three buttons whose clicks would go nowhere."""

    async def test_the_status_says_nothing_is_being_kept(self):
        self.assertFalse((await self.status())["logging"])

    async def test_a_verdict_is_refused_rather_than_dropped(self):
        response = await self.client.post("/api/outcome", json={"outcome": "success"})

        self.assertEqual(response.status, 400)
        self.assertIn("no log", await response.text())


class TestStrategyRoute(ViewTestCase):
    """Changing the splice from the page."""

    async def get_application(self):
        warnings.filterwarnings("ignore", message=".*app\\[.*")
        self.control = RunControl()
        self.source = _SpliceSource()
        # twin_port=0: no viser server in a unit test. The twin is stubbed in
        # the cases that care about it.
        self.view = PolicyView(
            self.control,
            self.source,
            StubDataManager(),
            CAMERAS,
            port=0,
            twin_port=0,
        )
        return self.view.build_app()

    async def test_a_known_strategy_is_applied_and_echoed_back(self):
        r = await self.client.post("/api/strategy", json={"strategy": "receding"})
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["strategy"], "receding")
        self.assertEqual(self.source.strategy, "receding")

    async def test_its_numbers_travel_with_it(self):
        await self.client.post(
            "/api/strategy", json={"strategy": "receding", "execute_ratio": 0.25}
        )
        self.assertEqual(self.source.params["execute_ratio"], 0.25)

    async def test_an_unknown_strategy_is_refused(self):
        r = await self.client.post("/api/strategy", json={"strategy": "clever"})
        self.assertEqual(r.status, 400)

    async def test_the_run_is_told_so_it_can_close_a_measurement(self):
        seen = []
        self.view.on_splice_change = seen.append
        await self.client.post("/api/strategy", json={"strategy": "replace"})
        self.assertEqual([s["strategy"] for s in seen], ["replace"])

    async def test_the_page_is_offered_the_whole_list(self):
        body = await self.status()
        self.assertIn("receding", body["strategies"])
        self.assertEqual(body["splice"]["strategy"], "append")


class TestStrategyRouteLocalRun(ViewTestCase):
    """A local run has no chunk to splice, so the control does not apply."""

    async def test_it_is_refused_rather_than_silently_ignored(self):
        r = await self.client.post("/api/strategy", json={"strategy": "replace"})
        self.assertEqual(r.status, 400)


if __name__ == "__main__":
    unittest.main()
