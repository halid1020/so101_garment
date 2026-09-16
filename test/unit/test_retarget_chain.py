"""Which checkpoints may be read as which policy.

`lerobot/pi05_base` says `pi05`, so `--policy.path` to it would quietly load
LeRobot's class however `--only` asked. `tool/retarget_checkpoint.py` symlinks
the ~14.5 GB and rewrites one field so a repo-local pi0.5 gets a base to
finetune.

The gate deciding whether that is legitimate used to test membership of a flat
pair list, which held only the four ports. A cropped variant carries
`ported_from: None` -- it is a subclass, not a port -- so it was refused, and
`harena_pi05_crop` died on the cluster at setup, before it trained a step.

It walks the variant chain now. The direction restriction is real and kept: a
variant only ever ADDS fields to its twin, and `loading.config_as` refuses a
field the target does not declare, so two names on different chains share no
weights.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_retarget_chain
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from tool.retarget_checkpoint import _ancestry, compatible


class AncestryTest(unittest.TestCase):
    def test_a_crop_reaches_its_upstream_in_two_hops(self):
        self.assertEqual(
            _ancestry("harena_pi05_crop"), ["harena_pi05_crop", "harena_pi05", "pi05"]
        )

    def test_a_port_reaches_it_in_one(self):
        self.assertEqual(_ancestry("harena_act"), ["harena_act", "act"])

    def test_a_root_is_its_own_chain(self):
        self.assertEqual(_ancestry("act"), ["act"])

    def test_a_name_nobody_declared_is_its_own_chain(self):
        # Not an error: the caller's own refusal reports it more usefully.
        self.assertEqual(_ancestry("not_a_policy"), ["not_a_policy"])


class CompatibilityTest(unittest.TestCase):
    def test_the_case_that_failed_on_the_cluster(self):
        self.assertTrue(compatible("pi05", "harena_pi05_crop"))

    def test_every_variant_reaches_its_upstream(self):
        for upstream, variant in (
            ("act", "harena_act"),
            ("act", "harena_act_crop"),
            ("diffusion", "harena_diffusion_crop"),
            ("pi05", "harena_pi05_crop"),
            ("fastwam", "harena_fastwam"),
        ):
            with self.subTest(f"{upstream}->{variant}"):
                self.assertTrue(compatible(upstream, variant))
                self.assertTrue(compatible(variant, upstream), "symmetric")

    def test_a_variant_reaches_its_sibling_through_their_shared_twin(self):
        # harena_act_crop and harena_act share a chain, so the weights do fit.
        self.assertTrue(compatible("harena_act", "harena_act_crop"))

    def test_two_chains_are_still_refused(self):
        # The loosening must not become "anything goes" -- these share no weights.
        for a, b in (
            ("act", "harena_pi05"),
            ("pi05", "harena_act_crop"),
            ("diffusion", "harena_act"),
            ("harena_act_crop", "harena_diffusion_crop"),
            ("act", "harena_dreamzero"),
        ):
            with self.subTest(f"{a}->{b}"):
                self.assertFalse(compatible(a, b))
                self.assertFalse(compatible(b, a))

    def test_the_chain_claim_holds_against_the_real_configs(self):
        """A pair this gate admits must actually rebuild through config_as.

        The gate is a name check; this is the thing the name check stands in
        for. If a crop config ever gained a field its twin lacks in the OTHER
        direction, the gate would admit a retarget that then failed at load.
        """
        import actoris_harena.policies  # noqa: F401  -- the import IS the registration
        from lerobot.configs import PreTrainedConfig

        source = PreTrainedConfig.get_choice_class("pi05")
        target = PreTrainedConfig.get_choice_class("harena_pi05_crop")
        ours = {f.name for f in dataclasses.fields(source) if f.init}
        theirs = {f.name for f in dataclasses.fields(target) if f.init}
        self.assertEqual(
            sorted(ours - theirs),
            [],
            "pi05 carries a field the crop does not declare, so config_as would "
            "raise even though compatible() said yes",
        )
        self.assertEqual(sorted(theirs - ours), ["tactile_cameras", "tactile_crop"])


class RefusalMessageTest(unittest.TestCase):
    def test_it_names_both_chains_rather_than_the_ported_pairs(self):
        from tool.retarget_checkpoint import retarget

        source = Path(tempfile.mkdtemp())
        (source / "config.json").write_text(json.dumps({"type": "pi05"}))
        with self.assertRaises(SystemExit) as caught:
            retarget(source, "harena_act_crop", Path(tempfile.mkdtemp()) / "out")
        message = str(caught.exception)
        # The reader needs to see WHY, which is that these are two chains.
        self.assertIn("harena_act", message)
        self.assertIn("never across two", message)


class RetargetCarriesTheCropStepTest(unittest.TestCase):
    """The silent bug: a cropped pi0.5 that trained on uncropped images.

    `make_pre_post_processors` begins `if pretrained_path:` and loads the SAVED
    pipeline off disk -- it never calls the policy's factory. Every pi0.5
    finetune uses `--policy.path`, because the base carries the pretrained image
    slots, so the crop variant's factory was never invoked and its step never
    existed. MEASURED: a three-hour run scored bit-identically to plain pi0.5,
    RMSE 16.044489 both, to every digit. It was a second baseline wearing the
    crop's name, and nothing in the run said so.
    """

    def base(self) -> Path:
        src = Path(tempfile.mkdtemp()) / "base"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"type": "pi05"}))
        (src / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "rename_observations_processor",
                            "config": {},
                        },
                        {"registry_name": "normalizer_processor", "config": {}},
                    ]
                }
            )
        )
        (src / "model.safetensors").write_text("weights")
        return src

    def steps_of(self, directory: Path) -> "list[str]":
        blob = json.loads((directory / "policy_preprocessor.json").read_text())
        return [s["registry_name"] for s in blob["steps"]]

    def test_a_crop_target_gains_the_crop_step(self):
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "harena_pi05_crop", out)
        self.assertIn("so101_tactile_crop", self.steps_of(out))

    def test_it_runs_before_the_rename(self):
        """Order is not cosmetic here.

        pi0.5's rename turns the rig's cameras into openpi's slot names, so a
        crop placed after it looks for cameras that no longer exist and silently
        does nothing -- the same nothing this test exists to catch.
        """
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "harena_pi05_crop", out)
        self.assertEqual(self.steps_of(out)[0], "so101_tactile_crop")

    def test_a_plain_target_gains_nothing(self):
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "harena_pi05", out)
        self.assertEqual(
            self.steps_of(out),
            ["rename_observations_processor", "normalizer_processor"],
        )

    def test_the_shared_base_is_not_written_through(self):
        """The base is 14 GB and symlinked; editing it would poison every run."""
        from tool.retarget_checkpoint import retarget

        src = self.base()
        retarget(src, "harena_pi05_crop", Path(tempfile.mkdtemp()) / "out")
        self.assertEqual(
            self.steps_of(src),
            ["rename_observations_processor", "normalizer_processor"],
        )

    def test_the_weights_are_still_symlinked(self):
        # The whole point of the retarget is not copying 14 GB.
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "harena_pi05_crop", out)
        self.assertTrue((out / "model.safetensors").is_symlink())

    def test_the_legacy_name_works_too(self):
        """A checkpoint on disk right now says so101_pi05_crop.

        The policies were renamed so101_* -> harena_*, and the old names survive
        as registry aliases because they are written into Slurm scripts already
        submitted and into train_config.json files a resuming job reads. A
        retarget that refused the old spelling would break exactly the runs the
        shim exists to keep alive.
        """
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "so101_pi05_crop", out)
        self.assertEqual(self.steps_of(out)[0], "so101_tactile_crop")
        # And the name the caller asked for is what lands in config.json, so a
        # resuming job still finds the type it was launched with.
        self.assertEqual(
            json.loads((out / "config.json").read_text())["type"], "so101_pi05_crop"
        )

    def test_print_path_emits_the_path_and_nothing_else(self):
        """The driver captures this in a command substitution.

        `base_path="$(retarget_checkpoint.py ... --print-path)"` takes ALL of
        stdout, so one informational line here becomes part of the path. It did:
        the run died with `Repo id must be in the form 'repo_name': '  preprocessor:
        added so101_tactile_crop...'` after the GPU was already reserved.
        """
        import subprocess
        import sys

        out = Path(tempfile.mkdtemp()) / "out"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tool" / "retarget_checkpoint.py"),
                "--checkpoint",
                str(self.base()),
                "--to",
                "harena_pi05_crop",
                "--out",
                str(out),
                "--print-path",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env={
                "PYTHONPATH": f"{REPO}:{REPO / 'src'}",
                "PATH": "/usr/bin:/bin",
                "HOME": str(Path.home()),
            },
        )
        self.assertEqual(result.stdout.strip(), str(out))
        self.assertEqual(len(result.stdout.strip().splitlines()), 1)
        # The note is not lost -- it just belongs on the other stream.
        self.assertIn("preprocessor: added", result.stderr)

    def test_the_step_carries_the_measured_fraction(self):
        from tool.retarget_checkpoint import retarget

        out = Path(tempfile.mkdtemp()) / "out"
        retarget(self.base(), "harena_pi05_crop", out)
        blob = json.loads((out / "policy_preprocessor.json").read_text())
        crop = next(
            s for s in blob["steps"] if s["registry_name"] == "so101_tactile_crop"
        )
        # Rows only, width whole -- the measurement's finding, pinned.
        self.assertEqual(list(crop["config"]["fraction"]), [0.80, 1.00])


if __name__ == "__main__":
    unittest.main()
