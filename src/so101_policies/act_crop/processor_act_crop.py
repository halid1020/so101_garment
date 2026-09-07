#!/usr/bin/env python
"""The twin's pipeline, with the tactile crop put in front of it."""

from __future__ import annotations

from typing import Any

import torch

from ..act.processor_act import make_so101_act_pre_post_processors
from ..common.tactile import So101TactileCropProcessorStep
from .configuration_act_crop import So101ActCropConfig


def make_so101_act_crop_pre_post_processors(
    config: So101ActCropConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """As the twin's, with the crop inserted at index 0.

    Index 0 is load-bearing: `RenameObservationsProcessorStep` is the first step
    of every one of these pipelines, and on pi0.5 it renames the rig's cameras
    onto openpi's slot names. A crop placed after it would look for camera names
    that no longer exist and silently do nothing at all.
    """
    preprocessor, postprocessor = make_so101_act_pre_post_processors(
        config, dataset_stats
    )
    crop = So101TactileCropProcessorStep(
        fraction=config.tactile_crop,
        cameras=tuple(config.tactile_cameras),
    )
    preprocessor.steps = [crop, *preprocessor.steps]
    return preprocessor, postprocessor
