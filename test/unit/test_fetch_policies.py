"""Bringing checkpoints back from a cluster, without a cluster.

``hpc/fetch_policies.sh`` is the return leg of ``stage_datasets.sh``: it reads a
cluster's run directory, works out which (dataset, policy) pairs actually
finished, and copies only their final checkpoints into the one-directory-per-run
layout the policy server expects. All of that is directory arithmetic over
``rsync``, and ``rsync`` is happy with local paths, so the real script can be run
end to end here against a temporary tree that stands in for the cluster.

What is worth testing is what a mistake would cost. Fetching a run that was
cancelled at its wall time would install a directory that looks like a
checkpoint and is not, and the failure would surface at the first rollout; a
filter that quietly selects nothing would look like a completed transfer.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_fetch_policies
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FETCH = REPO / "hpc" / "fetch_policies.sh"

CONFIG = {
    "type": "diffusion",
    "n_obs_steps": 2,
    "n_action_steps": 8,
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [12]},
        "observation.images.central": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.wrist_camera_left": {
            "type": "VISUAL",
            "shape": [3, 480, 640],
        },
    },
    "output_features": {"action": {"type": "ACTION", "shape": [12]}},
}


class _FetchCase(unittest.TestCase):
    """A temporary scratch holding finished and unfinished training runs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.scratch = self.tmp / "scratch"
        self.runs = self.scratch / "so101_outputs" / "vla_real_long"
        self.dest = self.tmp / "policies"

    def finished(self, dataset: str, policy: str, report: bool = True) -> Path:
        ckpt = (
            self.runs
            / dataset
            / "train"
            / policy
            / "checkpoints"
            / "last"
            / "pretrained_model"
        )
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(json.dumps({**CONFIG, "type": policy}))
        (ckpt / "model.safetensors").write_text("weights")
        if report:
            (self.runs / dataset / f"results_{policy}.md").write_text("# report\n")
        return ckpt

    def unfinished(self, dataset: str, policy: str) -> None:
        """Checkpoints exist, but the run never reached `last` -- a wall-time kill."""
        step = (
            self.runs
            / dataset
            / "train"
            / policy
            / "checkpoints"
            / "030000"
            / "pretrained_model"
        )
        step.mkdir(parents=True)
        (step / "model.safetensors").write_text("weights")

    def fetch(self, *args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["bash", str(FETCH), "--scratch", str(self.scratch), *args],
            capture_output=True,
            text=True,
        )

    def fetched(self) -> "set[str]":
        if not self.dest.exists():
            return set()
        return {p.name for p in self.dest.iterdir()}


class TestListing(_FetchCase):
    def test_it_names_every_run_and_marks_the_finished_ones(self):
        self.finished("fold-short", "diffusion")
        self.unfinished("fold-short", "act")

        result = self.fetch("--list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready", result.stdout)
        self.assertIn("unfinished", result.stdout)
        self.assertEqual(self.fetched(), set(), "--list must transfer nothing")

    def test_an_empty_run_directory_is_an_error_not_an_empty_success(self):
        self.runs.mkdir(parents=True)

        result = self.fetch("--list")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no training runs", result.stderr)

    def test_a_missing_scratch_says_so(self):
        result = self.fetch("--list")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no run directory", result.stderr)


class TestFetching(_FetchCase):
    def test_a_checkpoint_lands_under_dataset_and_policy(self):
        self.finished("fold-short", "diffusion")

        result = self.fetch("--dest", str(self.dest))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fetched(), {"fold-short-diffusion"})
        landed = self.dest / "fold-short-diffusion"
        self.assertTrue((landed / "config.json").is_file())
        self.assertTrue((landed / "model.safetensors").is_file())
        self.assertTrue((landed / "results_diffusion.md").is_file())

    def test_it_reports_what_the_checkpoint_expects(self):
        # The camera names are the reason to read the config back: a set that is
        # not the rig's cannot be run there, and this is the cheapest place to
        # see it.
        self.finished("fold-short", "diffusion")

        result = self.fetch("--dest", str(self.dest))

        self.assertIn("cameras: central, wrist_camera_left", result.stdout)
        self.assertIn("12-D", result.stdout)

    def test_an_unfinished_run_is_reported_and_left_alone(self):
        self.finished("cube-pnp", "diffusion")
        self.unfinished("cube-pnp", "act")

        result = self.fetch("--dest", str(self.dest))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fetched(), {"cube-pnp-diffusion"})
        self.assertIn("unfinished", result.stdout)

    def test_a_run_without_a_report_still_transfers(self):
        self.finished("fold-short", "act", report=False)

        result = self.fetch("--dest", str(self.dest))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fetched(), {"fold-short-act"})

    def test_the_filters_select_one_pair_out_of_four(self):
        for dataset in ("fold-short", "cube-pnp"):
            for policy in ("act", "diffusion"):
                self.finished(dataset, policy)

        result = self.fetch(
            "--dest", str(self.dest), "--datasets", "fold-short", "--only", "diffusion"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fetched(), {"fold-short-diffusion"})

    def test_a_filter_that_matches_nothing_fails_loudly(self):
        self.finished("fold-short", "diffusion")

        result = self.fetch("--dest", str(self.dest), "--datasets", "no-such-dataset")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nothing to fetch", result.stderr)
        self.assertEqual(self.fetched(), set())

    def test_as_renames_the_destination_of_a_job_id_run(self):
        # A run resubmitted without a --run-name is called after its job id,
        # which says nothing about the dataset it trained.
        self.finished("create_36618158", "act")

        result = self.fetch("--dest", str(self.dest), "--as", "cube-pnp")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.fetched(), {"cube-pnp-act"})

    def test_as_refuses_to_rename_several_runs_into_one(self):
        self.finished("fold-short", "act")
        self.finished("cube-pnp", "act")

        result = self.fetch("--dest", str(self.dest), "--as", "everything")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("2 runs are selected", result.stderr)
        self.assertEqual(self.fetched(), set())

    def test_dry_run_transfers_nothing_at_all(self):
        self.finished("fold-short", "diffusion")

        result = self.fetch("--dest", str(self.dest), "--dry-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.dest.exists(), "a dry run must not even make the dir")

    def test_a_destination_is_required_to_fetch(self):
        self.finished("fold-short", "diffusion")

        result = self.fetch()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dest is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
