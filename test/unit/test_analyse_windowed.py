"""The attribution tool's windowed path, for a model that reads a video window.

DreamZero refuses a single frame, so its occlusion and patch maps are made on
the dataset window itself. These tests use a stand-in whose plan depends on one
camera, the state and the past actions, so each measured effect has a known
cause.
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch
from actoris_harena.action_layout import set_gripper_columns

set_gripper_columns((5, 11))


class WindowReader:
    """Plans from the SQUARES of its inputs, so a mean baseline changes them."""

    torch = torch
    device = "cpu"
    policy = types.SimpleNamespace(config=types.SimpleNamespace())

    def pre(self, batch):
        return batch

    def stat_mean(self, key):
        """A training mean of all sevens, so a replaced stream is recognisable."""
        return np.full(12, 7.0, dtype=np.float32)

    def to_device(self, batch):
        return batch

    def chunk_from(self, batch):
        tip = batch["observation.images.tip"][0]
        signal = (
            float((tip[..., :2, :2] ** 2).sum())  # only the top-left cell matters
            + float((batch["observation.state"] ** 2).sum())
            + float((batch["action"][0, :2] ** 2).sum())
        )
        return np.full((4, 12), signal, dtype=np.float32)


def window():
    tip = torch.zeros(2, 3, 4, 4)
    tip[..., 0, 0] = 1.0  # structure only in the top-left cell
    return {
        "observation.images.tip": tip,
        "observation.images.central": torch.rand(2, 3, 4, 4),
        "observation.state": torch.arange(24, dtype=torch.float32).view(2, 12),
        "action": torch.arange(48, dtype=torch.float32).view(4, 12),
        # A real window carries padding flags beside each camera.
        "observation.images.tip_is_pad": torch.zeros(2, dtype=torch.bool),
    }


def args(**overrides):
    base = dict(method=["occlusion", "patches"], patch_grid=(2, 2), patch_cameras="all")
    base.update(overrides)
    return types.SimpleNamespace(**base)


class Recording(WindowReader):
    """Keeps the last batch it planned from."""

    seen: "list[dict]" = []

    def chunk_from(self, batch):
        Recording.seen.append({k: v.clone() for k, v in batch.items()})
        return super().chunk_from(batch)


class StateBaselineTest(unittest.TestCase):
    """The joints are replaced by the training mean, not by their own average."""

    def test_dataset_baseline_uses_the_training_mean(self):
        from tool.analyse_policy_inputs import windowed_record

        Recording.seen = []
        windowed_record(Recording(), window(), args(method=["occlusion"]), 2)
        # The second plan is the one with the state replaced (state is first).
        replaced = Recording.seen[1]["observation.state"]
        self.assertTrue(torch.all(replaced == 7.0))

    def test_zeros_baseline_zeroes_state_and_images(self):
        from tool.analyse_policy_inputs import windowed_record

        Recording.seen = []
        windowed_record(
            Recording(),
            window(),
            args(method=["occlusion"], state_baseline="zeros", baseline="zeros"),
            2,
        )
        self.assertTrue(torch.all(Recording.seen[1]["observation.state"] == 0))
        last_camera = Recording.seen[-2]  # the final single-stream plan: a camera
        flat = [
            k
            for k in last_camera
            if k.startswith("observation.images.") and not k.endswith("_is_pad")
        ]
        self.assertTrue(any(torch.all(last_camera[k] == 0) for k in flat))


class WindowedRecordTest(unittest.TestCase):
    def record(self, **overrides):
        from tool.analyse_policy_inputs import windowed_record

        return windowed_record(WindowReader(), window(), args(**overrides), 2)

    def test_the_streams_include_the_past_actions(self):
        streams = self.record()["occlusion"]["streams"]
        self.assertEqual(sorted(streams), ["central", "past actions", "state", "tip"])
        self.assertGreater(streams["past actions"]["l2"], 0.0)
        self.assertEqual(streams["central"]["l2"], 0.0)  # the plan ignores it
        self.assertAlmostEqual(sum(s["share"] for s in streams.values()), 1.0)

    def test_the_patch_map_finds_the_cell_the_plan_reads(self):
        tip = np.array(self.record()["patches"]["tip"])
        self.assertEqual(tip.shape, (2, 2))
        self.assertEqual(int(tip.argmax()), 0)
        self.assertEqual(float(tip[1, 1]), 0.0)

    def test_only_the_fingertips_are_mapped_by_default(self):
        maps = self.record(patch_cameras="tactile")["patches"]
        self.assertEqual(maps, {})  # neither stand-in camera is named a fingertip


if __name__ == "__main__":
    unittest.main()
