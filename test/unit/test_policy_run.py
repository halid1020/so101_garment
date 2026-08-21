"""The throttle on an autonomous rollout, and when the next chunk is asked for.

Both are pure arithmetic that decides whether a robot moves, so both are tested
away from the robot. The step machine is the interesting one: one press must
execute one chunk and stop, whatever the chunk's length, and a mode change must
invalidate a plan made before it -- executing a plan drawn from a two-minute-old
photograph is the failure this design exists to prevent.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_run
"""

import unittest

from common.policy_run import RunControl, gate, prefetch_threshold


class TestGate(unittest.TestCase):
    def test_running_serves_whenever_there_is_something_to_serve(self):
        self.assertEqual(gate("run", 5, 0), ("serve", 0))
        self.assertEqual(gate("run", 0, 0), ("wait", 0))

    def test_holding_never_serves_however_full_the_queue(self):
        self.assertEqual(gate("hold", 32, 0), ("hold", 0))

    def test_a_step_takes_its_budget_from_the_chunk_that_landed(self):
        # Nothing queued yet: the press has been made, the chunk is coming.
        self.assertEqual(gate("step", 0, 0), ("wait", 0))
        # A 32-action chunk arrives; the budget is the rest of it.
        self.assertEqual(gate("step", 32, 0), ("serve", 31))
        self.assertEqual(gate("step", 31, 31), ("serve", 30))

    def test_a_step_waits_rather_than_ending_early_on_an_empty_queue(self):
        # Mid-chunk with nothing queued is a slow link, not the end of the plan.
        self.assertEqual(gate("step", 0, 7), ("wait", 7))


class TestPrefetchThreshold(unittest.TestCase):
    def test_a_slow_round_trip_raises_the_threshold(self):
        # 0.6 s at 30 Hz spends 18 actions; a third of a 32-action chunk is 10,
        # which is why a diffusion rollout stutters on the default.
        self.assertEqual(prefetch_threshold(10, None, 0.6, 30.0, 32), 27)

    def test_a_fast_policy_keeps_its_static_default(self):
        self.assertEqual(prefetch_threshold(33, None, 0.018, 30.0, 100), 33)

    def test_it_never_reaches_the_chunk_length(self):
        # A request would otherwise be in flight permanently.
        self.assertEqual(prefetch_threshold(10, None, 9.0, 30.0, 32), 31)

    def test_an_explicit_prefetch_is_obeyed_in_both_directions(self):
        self.assertEqual(prefetch_threshold(10, 4, 9.0, 30.0, 32), 4)
        self.assertEqual(prefetch_threshold(10, 40, 0.0, 30.0, 32), 40)

    def test_no_measurement_yet_leaves_the_default_alone(self):
        self.assertEqual(prefetch_threshold(10, None, 0.0, 30.0, 32), 10)


class TestRunControl(unittest.TestCase):
    def test_a_run_serves_every_tick(self):
        control = RunControl()
        self.assertEqual([control.decide(9) for _ in range(3)], ["serve"] * 3)
        self.assertEqual(control.mode, "run")

    def test_one_step_executes_one_chunk_and_then_holds(self):
        control = RunControl()
        control.request("step")
        depths = [0, 3, 2, 1, 9]  # the chunk lands on the second tick
        served = [control.decide(d) for d in depths]
        self.assertEqual(served, ["wait", "serve", "serve", "serve", "hold"])
        self.assertEqual(control.mode, "hold")

    def test_changing_mode_marks_the_queue_stale_exactly_once(self):
        control = RunControl()
        control.request("hold")
        self.assertTrue(control.queue_stale())
        self.assertFalse(control.queue_stale())

    def test_asking_for_the_mode_already_in_force_changes_nothing(self):
        control = RunControl()
        control.request("run")
        self.assertFalse(control.queue_stale())

    def test_stop_is_a_request_not_a_mode(self):
        control = RunControl()
        self.assertEqual(control.request("stop"), "run")
        self.assertTrue(control.stopping)
        self.assertEqual(control.mode, "run")

    def test_an_unknown_mode_is_refused(self):
        control = RunControl()
        with self.assertRaises(ValueError):
            control.request("faster")

    def test_the_snapshot_carries_the_mode_with_whatever_was_published(self):
        control = RunControl()
        control.publish(tick=7, queue=12)
        control.request("hold")
        snapshot = control.snapshot()
        self.assertEqual(snapshot["tick"], 7)
        control.publish(tick=8)
        self.assertEqual(control.snapshot()["mode"], "hold")


if __name__ == "__main__":
    unittest.main()
