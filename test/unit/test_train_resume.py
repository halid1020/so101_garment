"""Resuming a run the machine went down under, rather than starting it again.

thanos rebooted mid-run twice in two days. Each cost the WHOLE run, because the
driver reused a FINISHED checkpoint but deleted a partial one -- while
lerobot-train has supported resuming from the last checkpoint all along.

Two things are checked here. The driver's three-way decision, exercised as the
shell function it is: reuse what is finished, resume what is not, delete only a
directory with no checkpoint in it at all. And the parser's side, because a
resumed run appends to the log it already had, so one file then holds two
attempts end to end and every line of the second one needs its true step.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_train_resume
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from actoris_harena.training.progress import parse_log, parse_resumes, resolve_steps

DRIVER = Path(__file__).resolve().parents[2] / "test/system/long_vla_real.sh"


def checkpoint_step(run: Path) -> str:
    """Call the driver's own shell function, so the test cannot drift from it."""
    script = (
        f"set -e; source_only() {{ :; }};\n"
        # Pull just the function out: sourcing the whole driver would run it.
        f'eval "$(sed -n "/^checkpoint_step()/,/^}}/p" {DRIVER})"\n'
        f'checkpoint_step "{run}"'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


class TestHowFarARunGot(unittest.TestCase):
    """The step is the name of the directory `checkpoints/last` points at."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def make(self, name: "str | None"):
        run = self.tmp / f"act-{name}"
        (run / "checkpoints").mkdir(parents=True)
        if name is not None:
            (run / "checkpoints" / name / "pretrained_model").mkdir(parents=True)
            (run / "checkpoints" / "last").symlink_to(name)
        return run

    def test_nothing_written_yet_is_zero(self):
        self.assertEqual(checkpoint_step(self.make(None)), "0")

    def test_a_directory_that_does_not_exist_is_zero(self):
        self.assertEqual(checkpoint_step(self.tmp / "never"), "0")

    def test_a_leading_zero_is_not_read_as_octal(self):
        # 070000 is a valid octal-looking string and 08 is not one at all;
        # without 10# the second is a hard shell error, mid-run.
        self.assertEqual(checkpoint_step(self.make("070000")), "70000")
        self.assertEqual(checkpoint_step(self.make("008000")), "8000")

    def test_a_checkpoint_not_named_for_a_step_is_treated_as_none(self):
        # Better to start again than to compare a step against a word.
        self.assertEqual(checkpoint_step(self.make("pretrained")), "0")

    def test_a_link_with_no_pretrained_model_under_it_is_zero(self):
        # A checkpoint interrupted DURING its own save: the directory is there
        # and the weights are not.
        run = self.tmp / "act"
        (run / "checkpoints" / "010000").mkdir(parents=True)
        (run / "checkpoints" / "last").symlink_to("010000")
        self.assertEqual(checkpoint_step(run), "0")


class TestReadingALogThatWasResumed(unittest.TestCase):
    """One file, two attempts, and every line needing its true step."""

    # Trimmed from a real pair of runs: six steps, a reboot, then six more.
    # tqdm's total is what is LEFT after a resume, which is the whole problem.
    FIRST = (
        "INFO 2026-08-29 14:50:01 ot_train.py:500 {'batch_size': 2,\n"
        " 'log_freq': 3,\n"
        " 'steps': 12,\n"
        "Training:  50%|## | 3/12 [00:01<00:03, 1.63step/s]"
        "INFO 2026-08-29 14:50:04 ot_train.py:596 step:3 loss:82.512 grdn:9.0\n"
        "Training: 100%|###| 6/12 [00:03<00:00, 1.70step/s]"
        "INFO 2026-08-29 14:50:07 ot_train.py:596 step:6 loss:42.301 grdn:8.0\n"
    )
    RESUME = (
        "  ↻ resuming /x/train/act at step 6 of 12\n"
        "Training:  50%|## | 3/6 [00:01<00:01, 1.63step/s]"
        "INFO 2026-08-29 14:52:26 ot_train.py:596 step:9 loss:17.892 grdn:7.0\n"
        "Training: 100%|###| 6/6 [00:03<00:00, 1.70step/s]"
        "INFO 2026-08-29 14:52:28 ot_train.py:596 step:12 loss:13.560 grdn:6.0\n"
        "INFO 2026-08-29 14:52:29 ot_train.py:700 End of training\n"
    )

    def test_the_marker_says_where_the_second_attempt_starts(self):
        marks = parse_resumes(self.FIRST + self.RESUME)
        self.assertEqual([(m["step"], m["total"]) for m in marks], [(6, 12)])

    def test_both_attempts_get_their_true_steps(self):
        parsed = parse_log(self.FIRST + self.RESUME)
        self.assertEqual([p["step"] for p in parsed["points"]], [3, 6, 9, 12])
        # And the log's own step: field agrees with every one of them.
        self.assertEqual(parsed["problems"], [])

    def test_the_first_attempt_is_not_shifted_by_the_second(self):
        # The bug this replaced: inferring the offset from the totals moved the
        # FIRST attempt too, because both carry the same total when the target
        # has not changed. [3, 6] must stay [3, 6].
        parsed = parse_log(self.FIRST + self.RESUME)
        self.assertEqual([p["step"] for p in parsed["points"]][:2], [3, 6])

    def test_an_unresumed_run_is_unaffected(self):
        parsed = parse_log(self.FIRST)
        self.assertEqual([p["step"] for p in parsed["points"]], [3, 6])
        self.assertEqual(parsed["resumed_at"], [])

    def test_a_raised_target_comes_from_the_resume_not_the_stale_head(self):
        # The head still states what the FIRST attempt was given.
        head = self.FIRST.replace("'steps': 12,", "'steps': 6,")
        self.assertEqual(parse_log(head + self.RESUME)["total_steps"], 12)

    def test_a_slurm_log_with_no_bars_counts_per_attempt(self):
        # CREATE disables tqdm, so the line's ordinal is the ONLY source -- and
        # it restarts at a resume like everything else does. Without segmenting,
        # the second attempt's two lines would read as steps 9 and 12 by
        # coincidence here and as nonsense on any real run.
        slurm = re.sub(r"Training:[^\n]*?step/s\]", "", self.FIRST + self.RESUME)
        self.assertNotIn("step/s]", slurm)
        parsed = parse_log(slurm)
        self.assertEqual([p["step"] for p in parsed["points"]], [3, 6, 9, 12])
        self.assertEqual({p["source"] for p in parsed["points"]}, {"count"})

    def test_two_interruptions_are_counted_as_two(self):
        again = self.RESUME.replace("at step 6 of 12", "at step 9 of 12")
        parsed = parse_log(self.FIRST + self.RESUME + again)
        self.assertEqual(len(parsed["resumed_at"]), 2)

    def test_a_resume_done_by_hand_still_falls_back_to_the_totals(self):
        # No marker -- somebody ran lerobot-train --resume themselves -- so the
        # only clue is that the bar's total is smaller than the head's. That
        # heuristic is kept for exactly this case.
        head = "INFO 2026-08-29 15:00:00 x.py:1 {'log_freq': 3,\n 'steps': 12,\n"
        no_marker = self.RESUME.replace(
            "  ↻ resuming /x/train/act at step 6 of 12\n", ""
        )
        points = parse_log(head + no_marker)["points"]
        self.assertEqual([p["step"] for p in points], [9, 12])


class TestAResumedRepoPolicyStillKnowsItsOwnPolicies(unittest.TestCase):
    """The one thing a resume may NOT take from the checkpoint.

    lerobot/configs/parser.py reads --policy.discover_packages_path off
    sys.argv and imports the package BEFORE draccus opens the file named by
    --config_path. So a checkpoint whose train_config.json says `so101_act`
    is parsed against a registry nobody populated, and the resume dies on a
    type it wrote itself. Leave the flag out and every repo-local run is
    unresumable -- on the box whose reboots are the reason resuming exists.
    """

    def resume_block(self) -> str:
        text = DRIVER.read_text(encoding="utf-8")
        start = text.index('if [ "$resume" = "1" ]; then')
        # The block ends at its own `return 0`; the fresh-start args follow.
        return text[start : text.index("return 0", start)]

    def test_the_resume_passes_the_discovery_flag(self):
        self.assertIn("--policy.discover_packages_path", self.resume_block())

    def test_it_is_guarded_by_local_policy(self):
        # An unconditional flag would put our package on lerobot's own runs.
        # Harmless today, but it would make `act` and `so101_act` differ by
        # something other than which implementation ran, which is the whole
        # point of keeping both.
        block = self.resume_block()
        guard = block.index("local_policy")
        self.assertLess(guard, block.index("--policy.discover_packages_path"))

    def test_local_policy_names_exactly_the_repo_ones(self):
        script = (
            f'eval "$(sed -n "/^local_policy()/,/^}}/p" {DRIVER})"\n'
            "for p in act diffusion pi05 fastwam so101_act so101_dreamzero; do\n"
            '  if local_policy "$p"; then echo "$p local"; else echo "$p lerobot"; fi\n'
            "done"
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        self.assertEqual(
            out.stdout.split(),
            # fmt: off
            [
                "act", "lerobot", "diffusion", "lerobot", "pi05", "lerobot",
                "fastwam", "lerobot", "so101_act", "local",
                "so101_dreamzero", "local",
            ],
            # fmt: on
        )


class TestResolveStepsDirectly(unittest.TestCase):
    def test_a_point_before_any_marker_keeps_offset_zero(self):
        points = [{"at": 0}, {"at": 100}]
        resolve_steps(points, [], log_freq=10, resumes=[{"at": 50, "step": 500}])
        self.assertEqual([p["step"] for p in points], [10, 510])


if __name__ == "__main__":
    unittest.main()
