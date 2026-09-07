#!/usr/bin/env python
"""ACT with a cropped tactile front end.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override here beyond the two attributes LeRobot resolves a policy by.
Keeping the class empty is the point -- if a cropped run differs from its
baseline, the crop is the only thing it can be.
"""

from __future__ import annotations

from ..act.modeling_act import So101ActPolicy
from .configuration_act_crop import So101ActCropConfig


class So101ActCropPolicy(So101ActPolicy):
    """So101ActPolicy, trained and served on cropped tactile images."""

    config_class = So101ActCropConfig
    name = "so101_act_crop"
