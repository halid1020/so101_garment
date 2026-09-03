"""The three ported policies are still the upstream ones (source and registry).

Every ported file must be exactly what ``so101_policies._port`` derives from the
LeRobot checkout. That is fast and exact, and it catches a hand-edit the moment
it happens -- including in pi0.5, which is far too large to instantiate here.

The other half of the claim, that a port reproduces a real trained checkpoint's
actions bit for bit, needs weights and a minute of CPU, so it lives in
``test/integration/test_policy_ports_checkpoints.py``.
"""

from __future__ import annotations

import dataclasses
import importlib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _readable(path: Path) -> bool:
    """Is this file actually there? An autofs mount whose drive is detached
    raises OSError from is_file() rather than returning False."""
    try:
        return path.is_file()
    except OSError:
        return False


class PortedSourceTest(unittest.TestCase):
    """Nothing was edited by hand."""

    def setUp(self) -> None:
        from so101_policies._port import UPSTREAM, port_text, ported_files

        if not UPSTREAM.is_dir():
            self.skipTest(f"no LeRobot checkout at {UPSTREAM}")
        self.ported_files = ported_files
        self.port_text = port_text

    def test_every_ported_file_matches_upstream(self) -> None:
        for upstream, ours, name, kind in self.ported_files():
            with self.subTest(file=ours.name):
                self.assertTrue(_readable(upstream), f"upstream gone: {upstream}")
                self.assertTrue(_readable(ours), f"port missing: {ours}")
                self.assertEqual(
                    ours.read_text(),
                    self.port_text(upstream.read_text(), name, kind),
                    f"{ours} has drifted from upstream; re-run tool/port_policies.py "
                    "or, if the edit was deliberate, take this file out of _port.PORTS",
                )

    def test_port_is_not_a_no_op(self) -> None:
        """A rule that silently stopped matching would make the test above vacuous."""
        for upstream, _ours, name, kind in self.ported_files():
            with self.subTest(file=upstream.name):
                text = upstream.read_text()
                self.assertNotEqual(text, self.port_text(text, name, kind))


class RegistrationTest(unittest.TestCase):
    """LeRobot can find all three by the names this repo gives them."""

    def setUp(self) -> None:
        try:
            from lerobot.configs import PreTrainedConfig  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("LeRobot not installed")

    def test_policy_and_processor_resolve(self) -> None:
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        from so101_policies.loading import ensure_registered

        ensure_registered()
        for name, policy_cls in (
            ("so101_act", "So101ActPolicy"),
            ("so101_diffusion", "So101DiffusionPolicy"),
            ("so101_pi05", "So101Pi05Policy"),
        ):
            with self.subTest(policy=name):
                self.assertEqual(get_policy_class(name).__name__, policy_cls)
                config_cls = PreTrainedConfig.get_choice_class(name)
                module = importlib.import_module(
                    config_cls.__module__.replace("configuration_", "processor_")
                )
                # The factory looks this up by name, so its absence is only ever
                # discovered at training time without this.
                self.assertTrue(hasattr(module, f"make_{name}_pre_post_processors"))

    def test_both_implementations_coexist(self) -> None:
        """The port must not displace the original: the equivalence test loads both."""
        from lerobot.policies.factory import get_policy_class

        from so101_policies.loading import ensure_registered

        ensure_registered()
        self.assertTrue(get_policy_class("act").__module__.startswith("lerobot."))
        self.assertTrue(
            get_policy_class("so101_act").__module__.startswith("so101_policies.")
        )


class ConfigAsTest(unittest.TestCase):
    def test_refuses_a_pair_that_is_not_one(self) -> None:
        from lerobot.configs import PreTrainedConfig

        from so101_policies.loading import config_as, ensure_registered

        ensure_registered()
        act = PreTrainedConfig.get_choice_class("so101_act")()
        with self.assertRaises(ValueError) as caught:
            config_as(act, "so101_diffusion")
        self.assertIn("not a ported pair", str(caught.exception))

    def test_round_trips_every_init_field(self) -> None:
        from lerobot.configs import PreTrainedConfig

        from so101_policies.loading import config_as, ensure_registered

        ensure_registered()
        original = PreTrainedConfig.get_choice_class("act")(
            chunk_size=37, n_action_steps=5
        )
        ported = config_as(original, "so101_act")
        for field in dataclasses.fields(original):
            if field.init:
                with self.subTest(field=field.name):
                    self.assertEqual(
                        getattr(original, field.name), getattr(ported, field.name)
                    )


if __name__ == "__main__":
    unittest.main()
