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

POLICIES = {"act", "diffusion", "pi05"}


def manifest_rows(path: Path) -> "list[list[str]]":
    """Data rows of a manifest, as the shell reads them: 6 fields plus the rest."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(line.split(maxsplit=6))
    return rows


class TestRunManifest(unittest.TestCase):
    def test_every_row_has_the_seven_columns(self):
        for row in manifest_rows(MANIFEST):
            self.assertEqual(len(row), 7, f"row is not 7 fields: {row}")

    def test_policies_are_ones_the_driver_trains(self):
        for row in manifest_rows(MANIFEST):
            self.assertIn(row[1], POLICIES, f"unknown policy in {row}")

    def test_the_cameras_column_never_contains_a_space(self):
        # A space would shift every later column into `extra`, and the row would
        # submit and train something other than what it says.
        for row in manifest_rows(MANIFEST):
            self.assertNotIn(" ", row[2], f"cameras must be comma-joined in {row}")
            self.assertTrue(row[2], f"cameras must be given in {row}")

    def test_numeric_columns_are_numbers_or_the_default_marker(self):
        for row in manifest_rows(MANIFEST):
            for field in row[3:6]:
                self.assertTrue(
                    field == "-" or field.isdigit(),
                    f"'{field}' is neither a number nor '-' in {row}",
                )

    def test_hours_is_always_given(self):
        # It becomes --time on the array, so it cannot fall back to a default.
        for row in manifest_rows(MANIFEST):
            self.assertTrue(row[5].isdigit(), f"hours must be explicit in {row}")

    def test_a_dataset_policy_and_camera_set_appears_once(self):
        triples = [(r[0], r[1], r[2]) for r in manifest_rows(MANIFEST)]
        self.assertEqual(
            len(triples),
            len(set(triples)),
            f"duplicate (dataset, policy, cameras) in {triples}",
        )

    def test_the_ablation_covers_every_policy_on_every_camera_set(self):
        """The point of the matrix: one run per (policy, camera set) on cube-pnp-new."""
        got = {(r[1], r[2]) for r in manifest_rows(MANIFEST) if r[0] == "cube-pnp-new"}
        want = {
            (policy, cameras)
            for policy in ("act", "diffusion", "pi05")
            for cameras in ("all", "central,wrist_camera_left", "wrist_camera_left")
        }
        self.assertEqual(got, want)


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
        "# dataset policy cameras steps batch hours extra\n"
        "alpha  act        all     80000   8   24  -\n"
        "alpha  diffusion  all     100000  32  24  -\n"
        "alpha  act        wrist   80000   8   24  -\n"
        "beta   act        all     -       -   36  -\n"
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
            any("--array=0-2" in ln and "--time=24:00:00" in ln for ln in lines)
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
        self.assertEqual([r[0] for r in rows], ["alpha", "alpha", "alpha"])
        self.assertEqual([r[1] for r in rows], ["act", "diffusion", "act"])
        self.assertEqual([r[2] for r in rows], ["all", "all", "wrist"])

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

    def test_one_arm_of_the_ablation_can_be_submitted_alone(self):
        result = self.submit("--manifest", str(self.manifest), "--cameras", "wrist")
        lines = self.sbatch_lines(result)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("--array=0-0", lines[0])

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
        manifest = self.write_manifest("alpha act all 10 2 24 -\n")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not staged", result.stderr)
        self.assertIn("alpha", result.stderr)
        self.assertIn("stage_datasets.sh", result.stderr)
        self.assertNotIn("DRY-RUN would submit", result.stdout)

    def test_an_unknown_policy_is_refused(self):
        manifest = self.write_manifest("alpha smolvla all 10 2 24 -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown policy", result.stderr)

    def test_pi05_is_a_policy_the_driver_trains(self):
        manifest = self.write_manifest("alpha pi05 all 10 2 24 -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_short_row_is_refused(self):
        manifest = self.write_manifest("alpha act 10\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("needs 6 fields", result.stderr)

    def test_a_space_in_the_cameras_column_is_refused(self):
        # The shell has already split on that space by the time the row is read,
        # so the cameras check cannot see it: 'wrist_camera_left' lands in steps
        # and the real steps/batch/hours slide one column left -- leaving an
        # hours that still looks valid. The numeric columns are what catch it.
        manifest = self.write_manifest(
            "alpha act central, wrist_camera_left 10 2 24 -\n"
        )
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("steps", result.stderr)
        self.assertIn("cameras column", result.stderr)

    def test_an_empty_cameras_column_is_refused(self):
        manifest = self.write_manifest("alpha act\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)


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
        (self.repo / "tool").mkdir(parents=True)
        (self.repo / "setup.sh").write_text("true\n")
        self.captured = self.tmp / "driver_args.txt"
        (self.repo / "test" / "system" / "long_vla_real.sh").write_text(
            f'printf "%s\\n" "$@" > {self.captured}\n'
        )
        # The job builds a camera view before training. Stand in for the builder:
        # record what it was asked for, and answer with a directory, so the test
        # sees which view path the job then trains on.
        self.view_args = self.tmp / "view_args.txt"
        (self.repo / "venv" / "bin").mkdir(parents=True)
        (self.repo / "venv" / "bin" / "python").write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$@" > {self.view_args}\n'
            'ds=""; cams=""; out=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  case "$1" in\n'
            '    --dataset) ds="$2"; shift 2;;\n'
            '    --cameras) cams="$2"; shift 2;;\n'
            '    --out-dir) out="$2"; shift 2;;\n'
            "    *) shift;;\n"
            "  esac\n"
            "done\n"
            'slug="$cams"; [ "$cams" = "all" ] && slug="all"\n'
            'view="$out/$(basename "$ds")__$slug"\n'
            'mkdir -p "$view/meta" && echo "{}" > "$view/meta/info.json"\n'
            'echo "$view"\n'
        )
        (self.repo / "venv" / "bin" / "python").chmod(0o755)
        self.scratch = self.tmp / "scratch"
        for name in ("alpha", "beta"):
            (self.scratch / "hf_lerobot" / "local" / name / "meta").mkdir(parents=True)
            (
                self.scratch / "hf_lerobot" / "local" / name / "meta" / "info.json"
            ).write_text("{}")
        self.manifest = self.tmp / "runs.tsv"
        self.manifest.write_text(
            "# dataset policy cameras steps batch hours extra\n"
            "alpha  act        all                        -  -  24  -\n"
            "beta   diffusion  central,wrist_camera_left  5  2  36  "
            "--policy.optimizer_lr=5e-5 --num_workers=1\n"
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

    def view_builder_args(self) -> "list[str]":
        return self.view_args.read_text().splitlines()

    def test_the_index_selects_the_row(self):
        result = self.run_task(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.driver_args()
        self.assertIn("--only", args)
        self.assertEqual(args[args.index("--only") + 1], "act")
        self.assertTrue(args[args.index("--dataset-root") + 1].endswith("/alpha__all"))

    def test_the_row_s_cameras_reach_the_view_builder(self):
        self.run_task(1)
        args = self.view_builder_args()
        self.assertEqual(args[args.index("--cameras") + 1], "central,wrist_camera_left")
        self.assertTrue(args[args.index("--dataset") + 1].endswith("/beta"))

    def test_training_happens_on_the_view_not_the_source(self):
        # Otherwise every arm of the ablation would train on all the cameras.
        self.run_task(1)
        root = self.driver_args()[self.driver_args().index("--dataset-root") + 1]
        self.assertTrue(root.endswith("__central,wrist_camera_left"), root)

    def test_the_run_name_is_the_view_so_a_resubmission_resumes(self):
        # Not the job id: a requeue after a wall-time hit must reuse the
        # checkpoints the previous attempt wrote. It carries the camera set too,
        # so the three arms of an ablation cannot overwrite each other.
        self.run_task(0)
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "alpha__all")

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
