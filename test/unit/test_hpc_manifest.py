"""The cluster run matrix and the submission wrapper, without a cluster.

A submission mistake is expensive in a way a local mistake is not: a wrong array
range silently trains the wrong cell, and a bad manifest row is only discovered
when the job reaches the head of a queue hours later. None of that needs Slurm to
catch, because ``hpc/submit_real.sh --dry-run`` prepares everything and prints the
``sbatch`` commands instead of running them, and ``hpc/create_real_vla.sbatch``
reads its row from a plain file.

So these tests run the real scripts: the wrapper against a temporary scratch, and
the batch script against a temporary repo whose ``setup.sh`` and training driver
are stubs that record what they were handed.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_hpc_manifest
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "hpc" / "runs.tsv"
SUBMIT = REPO / "hpc" / "submit_real.sh"
SBATCH = REPO / "hpc" / "create_real_vla.sbatch"

POLICIES = {"act", "diffusion"}


def manifest_rows(path: Path) -> "list[list[str]]":
    """Data rows of a manifest, as the shell reads them: 5 fields plus the rest."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(line.split(maxsplit=5))
    return rows


class TestRunManifest(unittest.TestCase):
    def test_every_row_has_the_six_columns(self):
        for row in manifest_rows(MANIFEST):
            self.assertEqual(len(row), 6, f"row is not 6 fields: {row}")

    def test_policies_are_ones_the_driver_trains(self):
        for row in manifest_rows(MANIFEST):
            self.assertIn(row[1], POLICIES, f"unknown policy in {row}")

    def test_numeric_columns_are_numbers_or_the_default_marker(self):
        for row in manifest_rows(MANIFEST):
            for field in row[2:5]:
                self.assertTrue(
                    field == "-" or field.isdigit(),
                    f"'{field}' is neither a number nor '-' in {row}",
                )

    def test_hours_is_always_given(self):
        # It becomes --time on the array, so it cannot fall back to a default.
        for row in manifest_rows(MANIFEST):
            self.assertTrue(row[4].isdigit(), f"hours must be explicit in {row}")

    def test_a_dataset_and_policy_pair_appears_once(self):
        pairs = [(r[0], r[1]) for r in manifest_rows(MANIFEST)]
        self.assertEqual(
            len(pairs), len(set(pairs)), f"duplicate (dataset, policy) in {pairs}"
        )


class _WrapperCase(unittest.TestCase):
    """A temporary scratch with the manifest's datasets staged into it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.scratch = self.tmp / "scratch"
        (self.scratch / "hf_lerobot" / "local").mkdir(parents=True)

    def stage(self, *datasets: str) -> None:
        for name in datasets:
            meta = self.scratch / "hf_lerobot" / "local" / name / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "info.json").write_text("{}")

    def write_manifest(self, text: str) -> Path:
        path = self.tmp / "runs.tsv"
        path.write_text(text)
        return path

    def submit(self, *args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["bash", str(SUBMIT), "--dry-run", "--scratch", str(self.scratch), *args],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )


class TestSubmissionDryRun(_WrapperCase):
    MANIFEST = (
        "# dataset policy steps batch hours extra\n"
        "alpha  act        80000   8   24  -\n"
        "alpha  diffusion  100000  32  24  -\n"
        "beta   act        -       -   36  -\n"
    )

    def setUp(self):
        super().setUp()
        self.stage("alpha", "beta")
        self.manifest = self.write_manifest(self.MANIFEST)

    def sbatch_lines(self, result) -> "list[str]":
        return [ln for ln in result.stdout.splitlines() if "DRY-RUN would submit" in ln]

    def test_rows_are_grouped_by_wall_time(self):
        # One array per distinct `hours`, so a short cell does not queue behind a
        # long reservation.
        lines = self.sbatch_lines(self.submit("--manifest", str(self.manifest)))
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(
            any("--array=0-1" in ln and "--time=24:00:00" in ln for ln in lines)
        )
        self.assertTrue(
            any("--array=0-0" in ln and "--time=36:00:00" in ln for ln in lines)
        )

    def test_the_array_reads_a_snapshot_not_the_manifest(self):
        # An array task reads its row when it starts, possibly hours later. It
        # must not read a file the user is free to edit in the meantime.
        lines = self.sbatch_lines(self.submit("--manifest", str(self.manifest)))
        for line in lines:
            self.assertIn("SO101_MANIFEST=", line)
            exported = line.split("SO101_MANIFEST=")[1].split()[0]
            self.assertNotEqual(exported, str(self.manifest))
            self.assertTrue(Path(exported).is_file(), exported)

    def test_the_snapshot_rows_match_the_array_indices(self):
        result = self.submit("--manifest", str(self.manifest))
        line = [ln for ln in self.sbatch_lines(result) if "--time=24:00:00" in ln][0]
        snapshot = Path(line.split("SO101_MANIFEST=")[1].split()[0])
        rows = manifest_rows(snapshot)
        self.assertEqual([r[0] for r in rows], ["alpha", "alpha"])
        self.assertEqual([r[1] for r in rows], ["act", "diffusion"])

    def test_filters_narrow_what_is_submitted(self):
        lines = self.sbatch_lines(
            self.submit("--manifest", str(self.manifest), "--only", "diffusion")
        )
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("--array=0-0", lines[0])

        lines = self.sbatch_lines(
            self.submit("--manifest", str(self.manifest), "--datasets", "beta")
        )
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("--time=36:00:00", lines[0])

    def test_a_filter_that_matches_nothing_is_refused(self):
        result = self.submit("--manifest", str(self.manifest), "--datasets", "gamma")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no rows selected", result.stderr)

    def test_the_repo_and_scratch_travel_with_the_job(self):
        line = self.sbatch_lines(self.submit("--manifest", str(self.manifest)))[0]
        self.assertIn(f"SO101_REPO_ROOT={REPO}", line)
        self.assertIn(f"SO101_SCRATCH={self.scratch}", line)

    def test_concurrency_caps_the_array(self):
        line = self.sbatch_lines(
            self.submit("--manifest", str(self.manifest), "--concurrency", "2")
        )[0]
        self.assertIn("%2", line)


class TestSubmissionRefusals(_WrapperCase):
    def test_an_unstaged_dataset_is_reported_instead_of_submitted(self):
        manifest = self.write_manifest("alpha act 10 2 24 -\n")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not staged", result.stderr)
        self.assertIn("alpha", result.stderr)
        self.assertIn("stage_datasets.sh", result.stderr)
        self.assertNotIn("DRY-RUN would submit", result.stdout)

    def test_an_unknown_policy_is_refused(self):
        manifest = self.write_manifest("alpha pi05 10 2 24 -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown policy", result.stderr)

    def test_a_short_row_is_refused(self):
        manifest = self.write_manifest("alpha act 10\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("needs 5 fields", result.stderr)


class TestBatchScriptPicksItsRow(unittest.TestCase):
    """The batch script, run against a stub repo, on a stub staged dataset.

    Everything it does before training is real -- row selection, the staged
    dataset check, the argument assembly -- so this exercises the mapping from
    an array index to a training command without a GPU or a cluster.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        (self.repo / "test" / "system").mkdir(parents=True)
        (self.repo / "setup.sh").write_text("true\n")
        self.captured = self.tmp / "driver_args.txt"
        (self.repo / "test" / "system" / "long_vla_real.sh").write_text(
            f'printf "%s\\n" "$@" > {self.captured}\n'
        )
        self.scratch = self.tmp / "scratch"
        for name in ("alpha", "beta"):
            (self.scratch / "hf_lerobot" / "local" / name / "meta").mkdir(parents=True)
            (
                self.scratch / "hf_lerobot" / "local" / name / "meta" / "info.json"
            ).write_text("{}")
        self.manifest = self.tmp / "runs.tsv"
        self.manifest.write_text(
            "# dataset policy steps batch hours extra\n"
            "alpha  act        -  -  24  -\n"
            "beta   diffusion  5  2  36  --policy.optimizer_lr=5e-5 --num_workers=1\n"
        )

    def run_task(self, task_id: int) -> "subprocess.CompletedProcess[str]":
        env = dict(os.environ)
        env.update(
            SLURM_ARRAY_TASK_ID=str(task_id),
            SO101_REPO_ROOT=str(self.repo),
            SO101_SCRATCH=str(self.scratch),
            SO101_MANIFEST=str(self.manifest),
        )
        return subprocess.run(
            ["bash", str(SBATCH)], capture_output=True, text=True, env=env
        )

    def driver_args(self) -> "list[str]":
        return self.captured.read_text().splitlines()

    def test_the_index_selects_the_row(self):
        result = self.run_task(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.driver_args()
        self.assertIn("--only", args)
        self.assertEqual(args[args.index("--only") + 1], "act")
        self.assertTrue(args[args.index("--dataset-root") + 1].endswith("/alpha"))

    def test_the_run_name_is_the_dataset_so_a_resubmission_resumes(self):
        # Not the job id: a requeue after a wall-time hit must reuse the
        # checkpoints the previous attempt wrote.
        self.run_task(0)
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "alpha")

    def test_default_columns_pass_no_override(self):
        self.run_task(0)
        args = self.driver_args()
        for flag in ("--steps", "--batch", "--extra"):
            self.assertNotIn(flag, args)

    def test_given_columns_reach_the_driver_intact(self):
        self.run_task(1)
        args = self.driver_args()
        self.assertEqual(args[args.index("--steps") + 1], "5")
        self.assertEqual(args[args.index("--batch") + 1], "2")
        self.assertEqual(
            args[args.index("--extra") + 1],
            "--policy.optimizer_lr=5e-5 --num_workers=1",
            "the extra column is one argument, spaces and all",
        )

    def test_an_index_past_the_end_fails_before_training(self):
        result = self.run_task(9)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no row 9", result.stdout + result.stderr)
        self.assertFalse(self.captured.exists())

    def test_an_unstaged_dataset_stops_the_job(self):
        shutil.rmtree(self.scratch / "hf_lerobot" / "local" / "alpha")
        result = self.run_task(0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("staged real dataset missing", result.stdout + result.stderr)
        self.assertFalse(self.captured.exists())


if __name__ == "__main__":
    unittest.main()
