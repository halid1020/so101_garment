"""Which checkpoints may be read as which policy.

`lerobot/pi05_base` says `pi05`, so `--policy.path` to it would quietly load
LeRobot's class however `--only` asked. `tool/retarget_checkpoint.py` symlinks
the ~14.5 GB and rewrites one field so a repo-local pi0.5 gets a base to
finetune.

The gate deciding whether that is legitimate used to test membership of a flat
pair list, which held only the four ports. A cropped variant carries
`ported_from: None` -- it is a subclass, not a port -- so it was refused, and
`so101_pi05_crop` died on the cluster at setup, before it trained a step.

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

from tool.retarget_checkpoint import _ancestry, compatible


class AncestryTest(unittest.TestCase):
    def test_a_crop_reaches_its_upstream_in_two_hops(self):
        self.assertEqual(
            _ancestry("so101_pi05_crop"), ["so101_pi05_crop", "so101_pi05", "pi05"]
        )

    def test_a_port_reaches_it_in_one(self):
        self.assertEqual(_ancestry("so101_act"), ["so101_act", "act"])

    def test_a_root_is_its_own_chain(self):
        self.assertEqual(_ancestry("act"), ["act"])

    def test_a_name_nobody_declared_is_its_own_chain(self):
        # Not an error: the caller's own refusal reports it more usefully.
        self.assertEqual(_ancestry("not_a_policy"), ["not_a_policy"])


class CompatibilityTest(unittest.TestCase):
    def test_the_case_that_failed_on_the_cluster(self):
        self.assertTrue(compatible("pi05", "so101_pi05_crop"))

    def test_every_variant_reaches_its_upstream(self):
        for upstream, variant in (
            ("act", "so101_act"),
            ("act", "so101_act_crop"),
            ("diffusion", "so101_diffusion_crop"),
            ("pi05", "so101_pi05_crop"),
            ("fastwam", "so101_fastwam"),
        ):
            with self.subTest(f"{upstream}->{variant}"):
                self.assertTrue(compatible(upstream, variant))
                self.assertTrue(compatible(variant, upstream), "symmetric")

    def test_a_variant_reaches_its_sibling_through_their_shared_twin(self):
        # so101_act_crop and so101_act share a chain, so the weights do fit.
        self.assertTrue(compatible("so101_act", "so101_act_crop"))

    def test_two_chains_are_still_refused(self):
        # The loosening must not become "anything goes" -- these share no weights.
        for a, b in (
            ("act", "so101_pi05"),
            ("pi05", "so101_act_crop"),
            ("diffusion", "so101_act"),
            ("so101_act_crop", "so101_diffusion_crop"),
            ("act", "so101_dreamzero"),
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
        from lerobot.configs import PreTrainedConfig

        import so101_policies  # noqa: F401  -- the import IS the registration

        source = PreTrainedConfig.get_choice_class("pi05")
        target = PreTrainedConfig.get_choice_class("so101_pi05_crop")
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
            retarget(source, "so101_act_crop", Path(tempfile.mkdtemp()) / "out")
        message = str(caught.exception)
        # The reader needs to see WHY, which is that these are two chains.
        self.assertIn("so101_act", message)
        self.assertIn("never across two", message)


if __name__ == "__main__":
    unittest.main()
