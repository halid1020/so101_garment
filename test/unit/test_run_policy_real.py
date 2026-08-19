"""Unit tests for tool/run_policy_real.py that need no hardware and no policy.

Two things are checked here. First, that a 12-D policy action reaches per-side
hardware goals through the same conversion a recorded action uses. Second, the
remote-inference client: what it does when a chunk is late (the arms are under
torque, so "hold then stop" has to be a decision, not an accident) and that it
consumes a chunk in order, refills before running dry, and gives up on a request
the server has already rejected once.

The remote tests run against a stub HTTP server in this process, so they exercise
the real client -- its threading, its handshake and its queue -- without a GPU.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from common.policy_wire import decode_request, encode_chunk
from tool.run_policy_real import (
    RemoteActionSource,
    policy_action_to_goals,
    stall_decision,
)

_BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class TestPolicyActionToGoals(unittest.TestCase):
    def test_splits_into_both_sides_with_gripper(self):
        # left body 0..4, left gripper (idx5), right body 6..10, right gripper (idx11)
        action = np.arange(12, dtype=float)
        goals = policy_action_to_goals(action)
        self.assertEqual(set(goals), {"left", "right"})
        for side in ("left", "right"):
            self.assertEqual(set(goals[side]), {*_BODY, "gripper"})

    def test_gripper_fraction_scaled_to_0_100(self):
        action = np.zeros(12)
        action[5] = 0.5  # left gripper fraction
        action[11] = 1.0  # right gripper fraction
        goals = policy_action_to_goals(action)
        self.assertAlmostEqual(goals["left"]["gripper"], 50.0)
        self.assertAlmostEqual(goals["right"]["gripper"], 100.0)

    def test_matches_manual_offset_sign_conversion(self):
        # With the shipped signs (+1) and offsets, hw = sign*(urdf - offset).
        from common.configs import LEFT_ARM_HW_TO_URDF_OFFSETS_DEG as OFF
        from common.configs import LEFT_ARM_HW_TO_URDF_SIGNS as SGN

        urdf = np.array([10.0, 20.0, -5.0, 0.0, 90.0])
        action = np.zeros(12)
        action[0:5] = urdf
        goals = policy_action_to_goals(action)
        expected = np.array(SGN) * (urdf - np.array(OFF))
        got = np.array([goals["left"][j] for j in _BODY])
        np.testing.assert_allclose(got, expected)


class TestStallDecision(unittest.TestCase):
    """What to do on a tick with no action to command."""

    def test_an_available_action_is_served(self):
        self.assertEqual(stall_decision(True, 99.0, 0.5, 2.0), "serve")

    def test_a_brief_gap_holds_quietly(self):
        # A chunk boundary or one slow round trip is normal; saying so every
        # tick would bury the message that matters.
        self.assertEqual(stall_decision(False, 0.1, 0.5, 2.0), "hold")

    def test_a_longer_gap_warns_but_keeps_holding(self):
        self.assertEqual(stall_decision(False, 0.6, 0.5, 2.0), "warn")

    def test_past_the_abort_threshold_it_stops(self):
        # Beyond this a hold is no longer a pause, and the arms should not stay
        # under torque waiting for a host that may never answer.
        self.assertEqual(stall_decision(False, 2.0, 0.5, 2.0), "abort")
        self.assertEqual(stall_decision(False, 5.0, 0.5, 2.0), "abort")


class _StubHandler(BaseHTTPRequestHandler):
    """A policy host that answers with a known chunk, or with a chosen failure."""

    def log_message(self, *_args):  # keep the test output clean
        pass

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        state = self.server.state
        if self.path == "/reset":
            payload = json.dumps(
                {
                    "session": "session-1",
                    "policy_type": "act",
                    "cameras": ["central"],
                    "n_obs_steps": 1,
                    "n_action_steps": 4,
                    "action_dim": 12,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        state["requests"].append(decode_request(body))
        if state["status"] != 200:
            self.send_error(state["status"], "refused")
            return
        served = len(state["requests"])
        chunk = np.tile(np.arange(12, dtype=np.float32), (4, 1)) + served * 100
        chunk += np.arange(4, dtype=np.float32)[:, None]
        payload = encode_chunk(chunk)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestRemoteActionSource(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.state = {"requests": [], "status": 200}
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    @staticmethod
    def _observation(seed):
        return np.full(12, float(seed), dtype=np.float32), {
            "central": np.full((8, 8, 3), seed % 255, dtype=np.uint8)
        }

    def _drain(self, source, timeout=5.0):
        deadline = time.monotonic() + timeout
        while source.depth == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        return source.take()

    def test_handshake_sizes_the_window_and_the_chunk(self):
        source = RemoteActionSource(self.url, "t")
        self.assertEqual(source.n_obs_steps, 1)
        self.assertEqual(source.actions, 4)
        self.assertEqual(source.cameras, ["central"])
        self.assertEqual(source.session, "session-1")

    def test_asking_for_more_actions_than_the_policy_plans_is_capped(self):
        # Executing past the end of a chunk would mean repeating stale actions.
        source = RemoteActionSource(self.url, "t", actions_per_chunk=999)
        self.assertEqual(source.actions, 4)

    def test_a_chunk_is_consumed_in_order(self):
        source = RemoteActionSource(self.url, "t")
        source.offer(*self._observation(1))
        first = self._drain(source)
        self.assertIsNotNone(first)
        self.assertAlmostEqual(float(first[0]), 100.0)
        self.assertAlmostEqual(float(source.take()[0]), 101.0)
        self.assertAlmostEqual(float(source.take()[0]), 102.0)

    def test_the_queue_is_refilled_before_it_runs_dry(self):
        # The whole point of prefetching: the next chunk is asked for while the
        # current one is still being executed, so the network stays out of the
        # control loop.
        source = RemoteActionSource(self.url, "t", prefetch=2)
        source.offer(*self._observation(1))
        self.assertIsNotNone(self._drain(source))
        source.take()  # depth is now 2, at the prefetch mark
        source.offer(*self._observation(2))
        deadline = time.monotonic() + 5.0
        while len(self.server.state["requests"]) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.server.state["requests"]), 2)

    def test_the_window_travels_with_the_request(self):
        source = RemoteActionSource(self.url, "the task")
        source.offer(*self._observation(7))
        self._drain(source)
        sent = self.server.state["requests"][0]
        self.assertEqual(sent["task"], "the task")
        self.assertEqual(sent["session"], "session-1")
        self.assertEqual(len(sent["steps"]), 1)
        self.assertAlmostEqual(float(sent["steps"][0][0][0]), 7.0)

    def test_a_rejected_request_is_fatal(self):
        # 4xx means the observation or the session is wrong; repeating it cannot
        # help, and the run should stop rather than keep the arms live.
        self.server.state["status"] = 400
        source = RemoteActionSource(self.url, "t")
        source.offer(*self._observation(1))
        deadline = time.monotonic() + 5.0
        while source.fatal is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(source.fatal)
        self.assertIn("400", source.fatal)

    def test_a_server_error_is_retried_not_fatal(self):
        # A host that failed once may answer the next request; only the stall
        # policy decides how long that is allowed to go on.
        self.server.state["status"] = 500
        source = RemoteActionSource(self.url, "t")
        source.offer(*self._observation(1))
        deadline = time.monotonic() + 5.0
        while source.last_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(source.fatal)
        self.assertIn("500", source.last_error or "")
        self.assertEqual(source.depth, 0)


if __name__ == "__main__":
    unittest.main()
