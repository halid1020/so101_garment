"""The shell drivers that launch a policy, and the names they route on.

The policies themselves moved to actoris_harena and are tested there. What stays
here is this rig's launch path: the driver scripts under test/system/ and the run
matrix they read, which are this project's and not the package's.
"""

import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "test" / "system" / "long_vla_real.sh"


class TestTheRealDriver(unittest.TestCase):
    def _base_policy(self, name: str) -> str:
        """Run the driver's own base_policy() on one name."""
        script = (
            f'eval "$(sed -n "/^base_policy()/,/^}}/p" {DRIVER})"\n'
            f"base_policy {name}"
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    def test_a_port_routes_to_its_upstream_twin(self):
        # A port takes its twin's defaults -- steps, batch, image size -- so the
        # driver has to know which twin each one is. Getting this wrong gives a
        # run that trains with the wrong budget and looks fine.
        for port, twin in (
            ("harena_act", "act"),
            ("harena_diffusion", "diffusion"),
            ("harena_pi05", "pi05"),
            ("harena_fastwam", "fastwam"),
        ):
            with self.subTest(policy=port):
                self.assertEqual(self._base_policy(port), twin)

    def test_a_cropped_variant_routes_through_its_port(self):
        for crop, twin in (
            ("harena_act_crop", "act"),
            ("harena_diffusion_crop", "diffusion"),
            ("harena_pi05_crop", "pi05"),
            ("harena_dreamzero_crop", "dreamzero"),
            ("harena_fastwam_crop", "fastwam"),
        ):
            with self.subTest(policy=crop):
                self.assertEqual(self._base_policy(crop), twin)

    def test_a_prediction_variant_routes_through_its_port(self):
        # harena_fastwam_predict exposes a future FastWAM already computes and
        # changes nothing about the model, so it must train on its twin's
        # budget. A variant that quietly got the default step count instead
        # would produce a run that looks fine and is not comparable to anything.
        self.assertEqual(self._base_policy("harena_fastwam_predict"), "fastwam")

    def test_a_legacy_name_still_routes(self):
        # A run matrix row or a resubmitted job may still say so101_*. The driver
        # must route it, or a resume dies where it was meant to recover.
        self.assertEqual(self._base_policy("so101_act"), "act")

    def test_an_upstream_policy_routes_to_itself(self):
        self.assertEqual(self._base_policy("act"), "act")


if __name__ == "__main__":
    unittest.main()
