#!/usr/bin/env python
"""ACT, trained on tactile images cropped to the sensor centre.

Everything about the model is its twin's: this subclasses `So101ActConfig` and
adds two fields. It is a separate registered policy rather than a flag on the
twin because the crop must travel with the CHECKPOINT -- a run trained cropped
has to be served cropped, and LeRobot rebuilds the processor pipeline from the
policy type. A flag would let the two drift apart silently.

The port itself is untouched. `tool/port_policies.py --check` re-derives the
nine ported files from upstream byte for byte, so an edit there would stop them
being ports; a subclass in a sibling package costs that check nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import So101ActConfig
from ..common.tactile import DEFAULT_CROP, TACTILE_CAMERAS


@PreTrainedConfig.register_subclass("so101_act_crop")
@dataclass
class So101ActCropConfig(So101ActConfig):
    """So101ActConfig plus a centred crop on the fingertip cameras."""

    #: Fraction of each side kept, centred, then resized back to the source
    #: size so no downstream shape changes. 1.0 is the uncropped baseline,
    #: which makes an ablation a single flag rather than a second policy.
    tactile_crop: float = DEFAULT_CROP

    #: Which cameras are tactile. No camera carries a `tactile` flag anywhere
    #: in this repo, so the set is named rather than inferred.
    tactile_cameras: tuple[str, ...] = field(default_factory=lambda: TACTILE_CAMERAS)
