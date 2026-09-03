"""A ported policy reproduces a real trained checkpoint, action for action.

This is the claim that decides whether moving the code cost anything: the
finished 80 000-step ACT run on ``fold-short-from-flattend-tactile`` must plan
exactly what it planned before. It needs the weights on this machine and about a
minute of CPU, which is why it is here and not in the unit tier -- the source
half of the same argument is in ``test/unit/test_policy_ports.py``.

Both tests skip rather than fail when the checkpoint is not on this machine.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ACT_CHECKPOINT = REPO / "outputs/policies/fold-short-from-flattend-tactile__all-act"
DIFFUSION_CHECKPOINT = (
    REPO
    / "outputs/vla_real_long/tmp_remote_diffusion/train/diffusion/checkpoints/000005/pretrained_model"
)


def _readable(path: Path) -> bool:
    """Is this file actually there? An autofs mount whose drive is detached
    raises OSError from is_file() rather than returning False."""
    try:
        return path.is_file()
    except OSError:
        return False


class CheckpointEquivalenceTest(unittest.TestCase):
    """A port that cannot reproduce a trained checkpoint is a rewrite, not a port."""

    def _compare(self, checkpoint: Path, upstream_type: str, ported_type: str) -> None:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        from so101_policies.loading import config_as, ensure_registered

        ensure_registered()
        path = str(checkpoint)
        upstream_config = PreTrainedConfig.from_pretrained(path)
        self.assertEqual(upstream_config.type, upstream_type)
        upstream_config.pretrained_path, upstream_config.device = path, "cpu"
        upstream = get_policy_class(upstream_type).from_pretrained(
            path, config=upstream_config
        )

        ported_config = config_as(upstream_config, ported_type)
        ported_config.pretrained_path, ported_config.device = path, "cpu"
        ported = get_policy_class(ported_type).from_pretrained(
            path, config=ported_config
        )

        left, right = upstream.state_dict(), ported.state_dict()
        self.assertEqual(set(left), set(right), "state_dict keys differ")
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), f"{key} differs")

        upstream.eval()
        ported.eval()
        torch.manual_seed(0)
        # A policy that reads a WINDOW of observations (diffusion's n_obs_steps)
        # wants them stacked on a step axis; one that reads a single frame (ACT)
        # must not be given that axis at all.
        steps = int(getattr(upstream_config, "n_obs_steps", 1) or 1)
        shape = (1, steps) if steps > 1 else (1,)
        batch = {
            key: torch.rand(*shape, *feature.shape)
            for key, feature in upstream_config.input_features.items()
        }

        # Seeded IMMEDIATELY BEFORE each call, not once before both: a stochastic
        # sampler (diffusion denoises over 100 timesteps) draws from the global
        # RNG, so the first call leaves the second somewhere else entirely and the
        # two would differ for a reason that has nothing to do with the port.
        def plan(policy):
            torch.manual_seed(0)
            with torch.no_grad():
                return policy.predict_action_chunk(dict(batch))

        self.assertTrue(
            torch.equal(plan(upstream), plan(ported)),
            "the two implementations planned different actions from one observation",
        )

    def test_act_reproduces_the_trained_checkpoint(self) -> None:
        if not _readable(ACT_CHECKPOINT / "model.safetensors"):
            self.skipTest(f"no ACT checkpoint at {ACT_CHECKPOINT}")
        self._compare(ACT_CHECKPOINT, "act", "so101_act")

    def test_diffusion_reproduces_a_trained_checkpoint(self) -> None:
        if not _readable(DIFFUSION_CHECKPOINT / "model.safetensors"):
            self.skipTest(f"no diffusion checkpoint at {DIFFUSION_CHECKPOINT}")
        self._compare(DIFFUSION_CHECKPOINT, "diffusion", "so101_diffusion")


if __name__ == "__main__":
    unittest.main()
