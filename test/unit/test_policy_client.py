"""Unit tests for the splice strategies as the remote client applies them.

``test_run_policy_real`` already covers the client's transport behaviour -- the
handshake, the window, what is fatal. These are about the one thing that changed
when the strategies arrived: what the queue holds after a chunk lands, and that
an unflagged run still behaves exactly as it did.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from common.chunking import ChunkingError
from common.policy_client import RemoteActionSource
from common.policy_wire import decode_request, encode_chunk

CHUNK = 8


class _StubHandler(BaseHTTPRequestHandler):
    """A host whose chunk rows are numbered, so a dropped row is visible."""

    def log_message(self, *_args):  # keep the test output clean
        pass

    def _send(self, payload: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        state = self.server.state
        if self.path == "/reset":
            self._send(
                json.dumps(
                    {
                        "session": "session-1",
                        "policy_type": "act",
                        "cameras": ["central"],
                        "n_obs_steps": 1,
                        "n_action_steps": CHUNK,
                        "action_dim": 12,
                    }
                ).encode(),
                "application/json",
            )
            return

        state["requests"].append(decode_request(body))
        served = len(state["requests"])
        # Row k of request n is (n * 100 + k), in every channel.
        rows = np.arange(CHUNK, dtype=np.float32)[:, None] + served * 100
        self._send(encode_chunk(np.tile(rows, (1, 12))), "application/octet-stream")


class RemoteSourceCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.state = {"requests": []}
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    @staticmethod
    def observation(seed=1):
        return np.full(12, float(seed), dtype=np.float32), {
            "central": np.full((8, 8, 3), seed % 255, dtype=np.uint8)
        }

    def source(self, **kwargs):
        return RemoteActionSource(self.url, "task", hz=30.0, **kwargs)

    def fill(self, source):
        """Offer until the first chunk has landed."""
        for _ in range(400):
            source.offer(*self.observation())
            if source.depth:
                return
            time.sleep(0.01)
        self.fail("no chunk arrived")


class TestDefaultIsUnchanged(RemoteSourceCase):
    def test_an_unflagged_client_still_appends(self):
        source = self.source()
        self.assertEqual(source.strategy, "append")
        self.fill(source)
        # Nothing is discarded, so the whole chunk is queued.
        self.assertEqual(source.depth, CHUNK)
        self.assertEqual(source.dropped_stale, 0)

    def test_a_second_chunk_lands_behind_the_first(self):
        source = self.source()
        self.fill(source)
        first = source.take()
        source.drain()
        source._splice_in(np.full((CHUNK, 12), 7.0), delay=3)
        self.assertEqual(source.depth, CHUNK)
        self.assertIsNotNone(first)


class TestAligningStrategies(RemoteSourceCase):
    def test_replace_discards_the_rows_whose_moment_has_passed(self):
        source = self.source(strategy="replace")
        self.fill(source)
        # Whatever the measured round trip was, the queue is short by exactly
        # the ticks it consumed, and the leading row is that far into the chunk.
        self.assertEqual(source.depth, CHUNK - source.last_delay)
        self.assertEqual(source.dropped_stale, source.last_delay)
        self.assertAlmostEqual(float(source.take()[0]) % 100, source.last_delay)

    def test_blend_joins_continuously_to_what_was_queued(self):
        source = self.source(strategy="blend", blend_window=4)
        source.drain()
        source._queue.extend(np.full(12, 5.0) for _ in range(6))
        source._splice_in(np.full((CHUNK, 12), 9.0), delay=0)
        # The first action is the one that was already going to be executed.
        np.testing.assert_allclose(source.take(), np.full(12, 5.0))

    def test_ensemble_averages_the_overlap(self):
        source = self.source(strategy="ensemble", new_weight=0.75)
        source.drain()
        source._queue.extend(np.zeros(12) for _ in range(CHUNK))
        source._splice_in(np.ones((CHUNK, 12)), delay=0)
        np.testing.assert_allclose(source.take(), np.full(12, 0.75))

    def test_a_stale_round_trip_can_empty_the_queue(self):
        # Longer in flight than the chunk is long: everything planned was for a
        # tick that has gone by, and the caller must see a stall, not an error.
        source = self.source(strategy="replace")
        source.drain()
        source._splice_in(np.ones((CHUNK, 12)), delay=CHUNK + 5)
        self.assertEqual(source.depth, 0)
        self.assertIsNone(source.take())


class TestBoundaryOffset(RemoteSourceCase):
    """Where the new plan actually STARTS, which is not where it landed."""

    def test_appending_puts_the_join_behind_every_leftover(self):
        source = self.source(strategy="append")
        source.drain()
        source._queue.extend(np.zeros(12) for _ in range(7))
        source._splice_in(np.ones((CHUNK, 12)), delay=2)
        self.assertEqual(source.boundary_offset, 7)

    def test_replacing_puts_the_join_at_the_next_tick(self):
        for strategy in ("replace", "blend", "ensemble", "rtc"):
            source = self.source(strategy=strategy)
            source.drain()
            source._queue.extend(np.zeros(12) for _ in range(7))
            source._splice_in(np.ones((CHUNK, 12)), delay=2)
            self.assertEqual(source.boundary_offset, 0, msg=strategy)


class TestSync(RemoteSourceCase):
    def test_it_asks_only_once_the_queue_is_empty(self):
        self.assertEqual(self.source(strategy="sync").threshold, 0)

    def test_offer_blocks_so_the_chunk_is_there_when_it_returns(self):
        source = self.source(strategy="sync")
        source.offer(*self.observation())
        # No polling: a synchronous client has its plan by the time offer ends.
        self.assertEqual(source.depth, CHUNK)


class TestCameraMap(RemoteSourceCase):
    def test_a_camera_is_renamed_on_the_way_out(self):
        # The twin renders 'scene'; this checkpoint was trained on 'central'.
        source = self.source(camera_map={"scene": "central"})
        state = np.zeros(12, dtype=np.float32)
        source.offer(state, {"scene": np.zeros((8, 8, 3), dtype=np.uint8)})
        for _ in range(400):
            if self.server.state["requests"]:
                break
            source.offer(state, {"scene": np.zeros((8, 8, 3), dtype=np.uint8)})
            time.sleep(0.01)
        self.assertEqual(self.server.state["requests"][0]["cameras"], ["central"])

    def test_what_the_policy_was_shown_is_recorded_under_the_sent_name(self):
        source = self.source(camera_map={"scene": "central"})
        source.offer(
            np.zeros(12, dtype=np.float32),
            {"scene": np.zeros((8, 8, 3), dtype=np.uint8)},
        )
        self.assertEqual(sorted(source.last_sent()[1]), ["central"])


class TestRefusals(RemoteSourceCase):
    def test_an_unknown_strategy_is_refused_before_the_handshake(self):
        with self.assertRaises(ChunkingError):
            self.source(strategy="clever")


if __name__ == "__main__":
    unittest.main()
