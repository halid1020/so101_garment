"""A training run as a row, and every reason it cannot work.

These are the refusals that stand between a mistyped row and a wasted GPU
reservation, so each test names the failure it prevents. None of them needs a
dataset on disk, a machine, or a card: a row is judged against an ``info.json``
mapping and a destination dict, which is what lets the same rules run here, in
``tool/train_launch.py`` and in the console's Training tab.

``hpc/runs.tsv`` itself is parsed here too. It is a tracked file that every
submission reads, so a row that this module cannot read is a broken cluster
run, not a broken test.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_train_matrix
"""

import unittest
from pathlib import Path

from actoris_harena.recording.dataset_view import PI05_SLOT_ORDER

from common.rig_profile import COMPOSITES
from common.training.matrix import (
    COLUMNS,
    POLICIES,
    POLICY_NAMES,
    MatrixError,
    format_rows,
    make_row,
    parse_rows,
    parse_slots,
    policy_available,
    resolved,
    row_cameras,
    row_refusals,
    unavailable_message,
)

REPO = Path(__file__).resolve().parents[2]
TACTILE = list(COMPOSITES["tactile_quad"])
INFO = {
    "features": {
        "observation.images.central": {"dtype": "video"},
        **{f"observation.images.{c}": {"dtype": "video"} for c in TACTILE},
        "observation.state": {"dtype": "float32"},
    }
}
CREATE = {"name": "create", "kind": "slurm", "limits": {"pi05": {"batch": 4}}}
BOX = {"name": "box", "kind": "ssh", "limits": {}}


def row(policy, cameras="all", **over):
    return make_row("ds", policy, cameras, **over)


class TestTheShippedMatrix(unittest.TestCase):
    def setUp(self):
        self.text = (REPO / "hpc" / "runs.tsv").read_text()

    def test_every_row_reads(self):
        self.assertTrue(parse_rows(self.text))

    def test_every_policy_is_one_this_repo_trains(self):
        for r in parse_rows(self.text):
            self.assertIn(r["policy"], POLICY_NAMES, r)

    def test_the_retry_matrix_reads_too(self):
        parse_rows((REPO / "hpc" / "runs_retry.tsv").read_text())

    def test_the_columns_are_the_ones_the_shell_scripts_read(self):
        # hpc/submit_real.sh and hpc/create_real_vla.sbatch read these
        # positionally with `read -r`, so the order is a contract.
        self.assertEqual(
            COLUMNS,
            (
                "dataset",
                "policy",
                "cameras",
                "steps",
                "batch",
                "hours",
                "slots",
                "extra",
            ),
        )


class TestReadingRows(unittest.TestCase):
    def test_comments_and_blank_lines_are_skipped(self):
        rows = parse_rows("# a\n\nds act all - - 24 - -\n")
        self.assertEqual(len(rows), 1)

    def test_the_last_column_keeps_its_spaces(self):
        (r,) = parse_rows("ds act all - - 24 - --lr=1 --workers=2\n")
        self.assertEqual(r["extra"], "--lr=1 --workers=2")

    def test_a_row_from_before_the_slots_column_says_which_one_is_missing(self):
        # Shifting `extra` into `slots` would read a string of lerobot-train
        # flags as a camera mapping, and the row would still submit.
        with self.assertRaises(MatrixError) as caught:
            parse_rows("ds act all - - 24 -\n")
        self.assertIn("slots", str(caught.exception))

    def test_a_shifted_row_blames_the_cameras_column(self):
        # A space in the cameras column moves every later one left; the
        # numeric columns are where that first becomes visible, and saying so
        # saves the reader looking at the wrong field.
        with self.assertRaises(MatrixError) as caught:
            parse_rows("ds act central, wrist 8 24 - -\n")
        self.assertIn("cameras", str(caught.exception))

    def test_rows_round_trip(self):
        # Every column survives; `line` is where the row was read from, which a
        # rewritten file is entitled to change.
        text = "ds act all 10 2 24 - --lr=1 --workers=2\n"
        columns = lambda rows: [{c: r[c] for c in COLUMNS} for r in rows]  # noqa: E731
        self.assertEqual(
            columns(parse_rows(format_rows(parse_rows(text)))),
            columns(parse_rows(text)),
        )

    def test_a_written_matrix_is_readable_by_the_shell_split(self):
        # The remote scripts split on whitespace, so alignment padding must not
        # become a field of its own.
        text = format_rows([row("act"), row("pi05", "central")])
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            self.assertEqual(len(line.split(None, 7)), 8, line)


class TestSlotsColumn(unittest.TestCase):
    def test_the_default_marker_means_no_pinning(self):
        self.assertIsNone(parse_slots({"slots": "-"}))

    def test_pairs_are_read(self):
        self.assertEqual(
            parse_slots({"slots": "central=base,x=left_wrist"}),
            {"central": "base", "x": "left_wrist"},
        )

    def test_something_that_is_not_a_pair_is_refused(self):
        with self.assertRaises(MatrixError):
            parse_slots({"slots": "central"})


class TestWhatARowAsksFor(unittest.TestCase):
    def test_all_stays_all_until_a_dataset_resolves_it(self):
        self.assertEqual(row_cameras(row("act")), ["all"])

    def test_a_dash_batch_takes_the_machine_s_measured_ceiling(self):
        # `-` means "whatever fits here", not "the policy default", so a
        # machine that has measured a lower one supplies it.
        self.assertEqual(resolved(row("pi05"), "batch", CREATE), 4)

    def test_a_dash_batch_falls_back_to_the_policy_default(self):
        self.assertEqual(resolved(row("pi05"), "batch", BOX), POLICIES["pi05"]["batch"])

    def test_an_explicit_batch_is_left_alone(self):
        self.assertEqual(resolved(row("pi05", batch=2), "batch", CREATE), 2)


class TestRefusals(unittest.TestCase):
    def test_a_workable_row_is_silent(self):
        self.assertEqual(row_refusals(row("act"), INFO, CREATE), [])

    def test_an_unknown_policy_lists_the_ones_that_exist(self):
        (refusal,) = row_refusals(row("smolvla"), INFO, CREATE)
        self.assertIn("act", refusal)

    def test_a_camera_the_dataset_lacks_lists_the_ones_it_has(self):
        # The failure this prevents: a typo trains on fewer cameras than the
        # experiment intended, and the result looks like a finding.
        (refusal,) = row_refusals(row("act", "centrall"), INFO, CREATE)
        self.assertIn("central", refusal)

    def test_pi05_with_more_cameras_than_slots_is_refused(self):
        refusals = row_refusals(row("pi05", "all"), INFO, CREATE)
        self.assertTrue(any("slots" in r for r in refusals))

    def test_pi05_within_its_slots_is_accepted(self):
        cameras = ",".join(["central", *TACTILE[:2]])
        self.assertEqual(row_refusals(row("pi05", cameras), INFO, CREATE), [])

    def test_an_explicit_batch_over_a_measured_ceiling_is_refused(self):
        # MEASURED: pi05 at batch 8 raised OutOfMemoryError at 39.22 GiB on a
        # 40 GB A100, hours into the reservation.
        (refusal,) = row_refusals(row("pi05", "central", batch=8), INFO, CREATE)
        self.assertIn("measured", refusal)

    def test_a_ceiling_only_applies_to_the_machine_that_measured_it(self):
        self.assertEqual(row_refusals(row("pi05", "central", batch=8), INFO, BOX), [])

    def test_a_port_is_refused_exactly_where_its_twin_is(self):
        # A ceiling is a fact about the model's activations, and a port IS the
        # model. Looking limits up by the exact policy string let so101_pi05
        # resolve to batch 8 on the machine where pi05 at batch 8 OOMed.
        (refusal,) = row_refusals(row("so101_pi05", "central", batch=8), INFO, CREATE)
        self.assertIn("measured", refusal)
        self.assertIn("pi05", refusal)

    def test_a_port_takes_its_twin_s_ceiling_for_a_dash(self):
        self.assertEqual(resolved(row("so101_pi05"), "batch", CREATE), 4)

    def test_a_destination_may_still_name_the_port_outright(self):
        # If the port ever measures differently, saying so must win over the
        # fallback rather than being averaged with it.
        dest = {
            "name": "create",
            "kind": "slurm",
            "limits": {"so101_pi05": {"batch": 2}},
        }
        self.assertEqual(resolved(row("so101_pi05"), "batch", dest), 2)

    def test_an_unstaged_dataset_is_refused_before_submission(self):
        refusals = row_refusals(row("act"), INFO, CREATE, staged=["other"])
        self.assertTrue(any("staged" in r for r in refusals))

    def test_a_row_can_be_judged_before_the_dataset_is_known(self):
        # The page refuses while the operator is still choosing, which is
        # before it has read anything off the drive.
        self.assertEqual(row_refusals(row("act")), [])

    def test_slots_naming_a_camera_the_row_does_not_record_is_refused(self):
        refusals = row_refusals(
            row("pi05", "central", slots="wrist_camera_left=base"), INFO, CREATE
        )
        self.assertTrue(any("wrist_camera_left" in r for r in refusals))


class TestTheFastwamGate(unittest.TestCase):
    """A policy the installed LeRobot has never heard of."""

    def test_the_three_that_train_today_are_available(self):
        for policy in ("act", "diffusion", "pi05"):
            self.assertTrue(policy_available(policy), policy)

    def test_availability_is_probed_not_hard_coded(self):
        # A version comparison would go stale the moment the pin moves; this
        # keeps telling the truth either way, so the test asserts the state of
        # THIS venv rather than a constant.
        import importlib.util

        self.assertEqual(
            policy_available("fastwam"),
            importlib.util.find_spec("lerobot.policies.fastwam") is not None,
        )

    def test_the_refusal_says_what_to_change(self):
        message = unavailable_message("fastwam")
        self.assertIn("LEROBOT_COMMIT", message)
        self.assertIn("install.sh", message)

    @unittest.skipIf(policy_available("fastwam"), "this venv's LeRobot ships fastwam")
    def test_a_gated_policy_is_refused_wherever_it_is_sent(self):
        for dest in (CREATE, BOX):
            refusals = row_refusals(row("fastwam", "central"), INFO, dest)
            self.assertTrue(any("fastwam" in r for r in refusals), dest["name"])


class TestFastwamImageContract(unittest.TestCase):
    """Its cameras are concatenated into ONE frame, so the count is fixed."""

    def _image_refusals(self, cameras):
        return [
            r
            for r in row_refusals(row("fastwam", cameras), INFO, BOX)
            if "concatenates" in r
        ]

    def test_five_cameras_do_not_fit(self):
        self.assertTrue(self._image_refusals(",".join(["central", *TACTILE])))

    def test_the_overhead_view_and_the_tiled_fingertips_do(self):
        # Which is the point of the composite: one view spent on four cameras
        # rather than three of them thrown away.
        self.assertEqual(self._image_refusals("central,tactile_quad"), [])

    def test_the_refusal_names_the_composite_as_the_way_out(self):
        (refusal,) = self._image_refusals(",".join(["central", *TACTILE]))
        self.assertIn("tactile_quad", refusal)

    def test_the_slot_count_follows_the_declared_image_size(self):
        height, width = POLICIES["fastwam"]["image_size"]
        self.assertEqual(width // height, 2)


class TestPolicyDefaults(unittest.TestCase):
    def test_pi05_is_capped_at_the_number_of_slots_pi05_has(self):
        self.assertEqual(POLICIES["pi05"]["max_cameras"], len(PI05_SLOT_ORDER))

    def test_every_policy_declares_what_a_dash_becomes(self):
        for name, spec in POLICIES.items():
            for key in ("steps", "batch", "hours"):
                self.assertIsInstance(spec.get(key), int, f"{name}.{key}")


if __name__ == "__main__":
    unittest.main()


class TestADeviceThatWouldBeTheCpu(unittest.TestCase):
    """The refusal that only the local machine can trigger.

    A GPU too small does not make ``long_vla_real.sh`` fail -- it falls back to
    the CPU and trains, which for an 80 000-step run means a week of work that
    looks like it is going fine. The driver itself calls that "the worst outcome
    available", and on a 4 GB laptop it is the DEFAULT outcome.
    """

    LOCAL = {
        "name": "local",
        "kind": "local",
        "ssh": "-",
        "repo": ".",
        "scratch": "~/.cache",
        "stage": "{scratch}/local",
        "limits": {},
        "measured": {
            "device": "cpu",
            "why": "this GPU has 4.1 GB and the driver wants at least 8 GB",
        },
    }

    def row(self, policy="act"):
        return make_row("fold-short", policy)

    def test_a_cpu_machine_refuses_by_default(self):
        problems = row_refusals(self.row(), None, self.LOCAL)
        self.assertEqual(len(problems), 1)
        self.assertIn("would train on the CPU", problems[0])
        # The measured reason, not a generic one: the operator has to be able
        # to tell "too small" from "no GPU at all".
        self.assertIn("4.1 GB", problems[0])

    def test_asking_for_it_outright_is_allowed(self):
        self.assertEqual(
            row_refusals(self.row(), None, {**self.LOCAL, "allow_cpu": True}), []
        )

    def test_a_machine_with_a_real_card_is_not_asked_about_it(self):
        dest = {**self.LOCAL, "measured": {"device": "cuda", "why": "24.5 GB"}}
        self.assertEqual(row_refusals(self.row(), None, dest), [])

    def test_a_machine_that_was_never_measured_is_not_refused(self):
        # Remote destinations are not probed: their ceilings are measured and
        # written into the destinations file, which is the right place for a
        # number somebody had to observe.
        dest = {k: v for k, v in self.LOCAL.items() if k != "measured"}
        self.assertEqual(row_refusals(self.row(), None, dest), [])
