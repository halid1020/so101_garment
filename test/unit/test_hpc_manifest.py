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
import time
import unittest
from pathlib import Path

from actoris_harena.training.matrix import POLICY_NAMES

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "hpc" / "runs.tsv"
SUBMIT = REPO / "hpc" / "submit_real.sh"
SBATCH = REPO / "hpc" / "create_real_vla.sbatch"

# Read from the registry rather than repeated here: a second list drifts the day
# a policy is added, and its failure ("unknown policy") points at the manifest
# rather than at itself.
POLICIES = set(POLICY_NAMES)


def manifest_rows(path: Path) -> "list[list[str]]":
    """Data rows of a manifest, as the shell reads them: 7 fields plus the rest."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(line.split(maxsplit=7))
    return rows


class TestRunManifest(unittest.TestCase):
    def test_every_row_has_the_eight_columns(self):
        for row in manifest_rows(MANIFEST):
            self.assertEqual(len(row), 8, f"row is not 8 fields: {row}")

    def test_policies_are_ones_the_driver_trains(self):
        for row in manifest_rows(MANIFEST):
            self.assertIn(row[1], POLICIES, f"unknown policy in {row}")

    def test_the_wrapper_s_own_policy_list_matches_the_registry(self):
        # submit_real.sh validates a row on a login node, before anything
        # imports our Python, so it repeats POLICY_NAMES as a shell `case`.
        # That copy is only allowed to exist because this test holds it to the
        # registry -- otherwise it drifts the day a policy is added and refuses
        # a row the console happily offered.
        text = SUBMIT.read_text(encoding="utf-8")
        start = text.index('case "$policy" in')
        accepted: "set[str]" = set()
        # EVERY accepting arm, not just the first line: the list outgrew one
        # readable line once the cropped variants arrived, and a reader that
        # takes only line 1 silently stops checking the rest.
        for line in text[start:].split("\n")[1:]:
            line = line.strip()
            if line.startswith("*)") or line.startswith("esac"):
                break
            if line.endswith(") ;;"):
                accepted |= set(line[: -len(") ;;")].split("|"))
        self.assertEqual(accepted, POLICIES)

    def test_the_cameras_column_never_contains_a_space(self):
        # A space would shift every later column into `extra`, and the row would
        # submit and train something other than what it says.
        for row in manifest_rows(MANIFEST):
            self.assertNotIn(" ", row[2], f"cameras must be comma-joined in {row}")
            self.assertTrue(row[2], f"cameras must be given in {row}")

    def test_the_slots_column_never_contains_a_space(self):
        # Same reason as the cameras column: a space shifts `extra` along.
        for row in manifest_rows(MANIFEST):
            self.assertNotIn(" ", row[6], f"slots must be comma-joined in {row}")

    def test_only_pi05_rows_pin_slots(self):
        # The slots are pi0.5's; naming them for another policy would be a row
        # that reads as if it did something it cannot.
        for row in manifest_rows(MANIFEST):
            if row[6] != "-":
                self.assertEqual(row[1], "pi05", f"slots on a non-pi05 row: {row}")

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
        "# dataset policy cameras steps batch hours slots extra\n"
        "alpha  act        all     80000   8   24  -  -\n"
        "alpha  diffusion  all     100000  32  24  -  -\n"
        "alpha  act        wrist   80000   8   24  -  -\n"
        "beta   act        all     -       -   36  -  -\n"
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
        manifest = self.write_manifest("alpha act all 10 2 24 - -\n")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not staged", result.stderr)
        self.assertIn("alpha", result.stderr)
        self.assertIn("stage_datasets.sh", result.stderr)
        self.assertNotIn("DRY-RUN would submit", result.stdout)

    def test_an_unknown_policy_is_refused(self):
        manifest = self.write_manifest("alpha smolvla all 10 2 24 - -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown policy", result.stderr)

    def test_pi05_is_a_policy_the_driver_trains(self):
        manifest = self.write_manifest("alpha pi05 all 10 2 24 - -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_short_row_is_refused(self):
        manifest = self.write_manifest("alpha act 10\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("needs 7 fields", result.stderr)

    def test_a_row_written_before_the_slots_column_says_which_one_is_missing(self):
        # Reading the old 7-field shape leniently would put `extra` -- a string
        # of lerobot-train flags -- into the slots column, and the row would
        # submit and train something nobody asked for.
        manifest = self.write_manifest("alpha act all 10 2 24 -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertIn("slots", result.stderr)

    def test_a_space_in_the_slots_column_is_refused(self):
        manifest = self.write_manifest("alpha pi05 all 10 2 24 central=base, x=y -\n")
        self.stage("alpha")
        result = self.submit("--manifest", str(manifest))
        self.assertEqual(result.returncode, 2)

    def test_a_space_in_the_cameras_column_is_refused(self):
        # The shell has already split on that space by the time the row is read,
        # so the cameras check cannot see it: 'wrist_camera_left' lands in steps
        # and the real steps/batch/hours slide one column left -- leaving an
        # hours that still looks valid. The numeric columns are what catch it.
        manifest = self.write_manifest(
            "alpha act central, wrist_camera_left 10 2 24 - -\n"
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
        for name in ("alpha", "beta", "gamma"):
            (self.scratch / "hf_lerobot" / "local" / name / "meta").mkdir(parents=True)
            (
                self.scratch / "hf_lerobot" / "local" / name / "meta" / "info.json"
            ).write_text("{}")
        self.manifest = self.tmp / "runs.tsv"
        self.manifest.write_text(
            "# dataset policy cameras steps batch hours slots extra\n"
            "alpha  act        all                        -  -  24  -  -\n"
            "beta   diffusion  central,wrist_camera_left  5  2  36  -  "
            "--policy.optimizer_lr=5e-5 --num_workers=1\n"
            "gamma  pi05       central,wrist_camera_left  5  2  36  "
            "central=base,wrist_camera_left=right_wrist  -\n"
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
        for flag in ("--steps", "--batch", "--slots", "--extra"):
            self.assertNotIn(flag, args)

    def test_a_pinned_slot_map_reaches_the_driver(self):
        # Which camera pi0.5 sees through which pretrained slot is the ablation
        # itself, so it has to survive the trip from the row to the driver.
        self.run_task(2)
        args = self.driver_args()
        self.assertEqual(
            args[args.index("--slots") + 1],
            "central=base,wrist_camera_left=right_wrist",
        )

    def test_a_row_written_before_the_slots_column_is_refused_by_the_task(self):
        self.manifest.write_text("alpha act all - - 24 -\n")
        result = self.run_task(0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("slots", result.stdout + result.stderr)

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


class TestWhereTheJobLogsGo(unittest.TestCase):
    """Slurm logs belong under outputs/, which is gitignored.

    They used to land as `<job-name>-<arrayjob>_<task>.out` in the submit
    directory, which is the repo root -- so a cluster run left untracked files
    in a working tree that a later `git checkout` then refused to move past.
    """

    def test_both_sbatch_files_write_under_outputs(self):
        for name in ("create_real_vla.sbatch", "create_sim_vla.sbatch"):
            text = (REPO / "hpc" / name).read_text(encoding="utf-8")
            line = next(
                ln for ln in text.splitlines() if ln.startswith("#SBATCH --output=")
            )
            self.assertIn("outputs/runs/", line, name)

    def test_the_submitter_creates_that_directory(self):
        # Slurm does not create it, and a job whose --output path is missing
        # fails before it runs a line -- so the mkdir is load-bearing, not tidy.
        text = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('mkdir -p "$REPO_ROOT/outputs/runs"', text)
        self.assertLess(
            text.index('mkdir -p "$REPO_ROOT/outputs/runs"'),
            text.index("cmd=(sbatch"),
            "the directory must exist before sbatch is called",
        )

    def test_outputs_is_ignored_by_git(self):
        ignored = (REPO / ".gitignore").read_text(encoding="utf-8").split()
        self.assertIn("outputs", ignored)


if __name__ == "__main__":
    unittest.main()


class TestGpuBoxRunNames(unittest.TestCase):
    """The run directory a plain GPU box writes into.

    ``long_vla_real.sh`` SKIPS a policy whose ``checkpoints/last`` already
    exists, which is what makes a crashed run resumable -- and what makes a
    short probe dangerous: left under the real name, a 200-step probe would
    make the 80000-step run print "reusing checkpoint" and write a results.md
    claiming success at 200 steps. ``--run-tag`` is the way out, and it is the
    same override ``create_real_vla.sbatch`` already had.
    """

    GPU_BOX = REPO / "hpc" / "gpu_box_run.sh"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        (self.repo / "hpc").mkdir(parents=True)
        (self.repo / "test" / "system").mkdir(parents=True)
        (self.repo / "venv" / "bin").mkdir(parents=True)
        shutil.copy2(self.GPU_BOX, self.repo / "hpc" / "gpu_box_run.sh")
        (self.repo / "setup.sh").write_text("true\n")

        # Stand-ins that record what they were handed.
        self.captured = self.tmp / "driver_args"
        (self.repo / "test" / "system" / "long_vla_real.sh").write_text(
            "#!/usr/bin/env bash\n" f'printf "%s\\n" "$@" > {self.captured}\n'
        )
        view_out = self.tmp / "view_args"
        (self.repo / "venv" / "bin" / "python").write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$@" > {view_out}\n'
            'ds=""; out=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  case "$1" in --dataset) ds="$2"; shift 2;; --out-dir) out="$2"; shift 2;;'
            " *) shift;; esac\n"
            "done\n"
            'view="$out/$(basename "$ds")__all"\n'
            'mkdir -p "$view/meta" && echo "{}" > "$view/meta/info.json"\n'
            'echo "$view"\n'
        )
        (self.repo / "venv" / "bin" / "python").chmod(0o755)

        self.scratch = self.tmp / "scratch"
        (self.scratch / "local" / "alpha" / "meta").mkdir(parents=True)
        (self.scratch / "local" / "alpha" / "meta" / "info.json").write_text("{}")
        self.manifest = self.tmp / "runs.tsv"
        self.manifest.write_text("alpha act all 200 - 1 - -\n")

    def run_box(self, *args, env_extra=None):
        env = dict(os.environ)
        env["SO101_OUTPUT_DIR"] = str(self.tmp / "out")
        env.pop("SO101_RUN_TAG", None)
        env.update(env_extra or {})
        return subprocess.run(
            [
                "bash",
                str(self.repo / "hpc" / "gpu_box_run.sh"),
                "--manifest",
                str(self.manifest),
                "--scratch",
                str(self.scratch),
                *args,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    def driver_args(self):
        return self.captured.read_text().splitlines()

    def test_the_run_is_named_after_the_view_by_default(self):
        result = self.run_box()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "alpha__all")

    def test_a_run_tag_takes_the_place_of_the_view(self):
        self.run_box("--run-tag", "probe5cam")
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "probe5cam")

    def test_an_explicit_scratch_beats_an_inherited_output_dir(self):
        """The failure this prevents is a run that trains and cannot be found.

        setup.sh sets SO101_OUTPUT_DIR to the repo's own outputs/ on every
        machine, so a run launched from a shell that had sourced it used to go
        there rather than under --scratch -- while the console discovers a
        machine's runs at <scratch>/so101_outputs and would list nothing.
        MEASURED on thanos while setting up the port-parity pair.
        """
        result = self.run_box("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = [x for x in result.stdout.splitlines() if x.startswith("   outputs :")]
        self.assertEqual(line, [f"   outputs : {self.scratch}/so101_outputs"])

    def test_with_no_scratch_the_environment_still_decides(self):
        # Nobody said otherwise, so an operator who exported it deliberately
        # keeps getting what they asked for.
        env = dict(os.environ)
        env["SO101_OUTPUT_DIR"] = str(self.tmp / "chosen")
        env.pop("SO101_RUN_TAG", None)
        result = subprocess.run(
            [
                "bash",
                str(self.repo / "hpc" / "gpu_box_run.sh"),
                "--manifest",
                str(self.manifest),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn(f"   outputs : {self.tmp / 'chosen'}", result.stdout)

    def test_the_environment_spelling_matches_the_cluster_script(self):
        # create_real_vla.sbatch reads SO101_RUN_TAG; two executors that took
        # different names for one idea is how a probe ends up in the real run's
        # directory on whichever box the operator forgot about.
        self.run_box(env_extra={"SO101_RUN_TAG": "from-env"})
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "from-env")
        self.assertIn(
            "SO101_RUN_TAG", (REPO / "hpc" / "create_real_vla.sbatch").read_text()
        )

    def test_the_flag_wins_over_the_environment(self):
        self.run_box("--run-tag", "explicit", env_extra={"SO101_RUN_TAG": "from-env"})
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "explicit")

    def test_detaching_carries_every_option_to_the_copy_it_starts(self):
        # MEASURED failure: --detach rebuilt its own argument list and left
        # --run-tag out of it, so a probe ran under the REAL run's name on a
        # real box. The tests before this one all took the foreground path,
        # which is the path nobody uses -- the launcher always detaches.
        result = self.run_box("--run-tag", "probe", "--detach")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("pid=", result.stdout)

        log = Path(result.stdout.split("log=")[1].strip())
        deadline = time.time() + 30
        while time.time() < deadline and not self.captured.exists():
            time.sleep(0.1)
        self.assertTrue(
            self.captured.exists(), f"driver never ran; log:\n{log.read_text()}"
        )
        args = self.driver_args()
        self.assertEqual(args[args.index("--run-name") + 1], "probe")
