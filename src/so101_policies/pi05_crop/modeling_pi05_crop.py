#!/usr/bin/env python
"""pi0.5 with a cropped tactile front end.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override here beyond the two attributes LeRobot resolves a policy by.
Keeping the class empty is the point -- if a cropped run differs from its
baseline, the crop is the only thing it can be.
"""

from __future__ import annotations

from ..pi05.modeling_pi05 import So101Pi05Policy
from .configuration_pi05_crop import So101Pi05CropConfig


class So101Pi05CropPolicy(So101Pi05Policy):
    """So101Pi05Policy, trained and served on cropped tactile images."""

    config_class = So101Pi05CropConfig
    name = "so101_pi05_crop"
