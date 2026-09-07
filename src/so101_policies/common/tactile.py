"""Crop a vision-based tactile sensor down to the part of it that touches things.

WHY. Grad-CAM on the finished five-camera ACT checkpoint shows the policy
attending to the **edges** of the fingertip images -- `right_arm_right_gripper`
saturates along its left edge and top-right corner while the gel centre stays
cold -- and it does so in frames where nothing is in contact. That is the
signature of light leaking in at the gel boundary, a known failure of
vision-based tactile sensors: the border is bright, it moves with the ambient
light rather than with the object, and a network will happily learn it.
Cropping to the centre takes the leak out of the input rather than hoping the
policy learns to ignore it.

WHAT IT DOES, and the one decision worth arguing about: it crops the central
fraction of each tactile frame and then **resizes back to the source size**.
Resizing back costs a little interpolation blur and buys the thing that makes
this a one-file change instead of a survey:

* ACT's per-camera token count stays 300, so `analysis/streams.py` keeps
  tiling. It assumes every camera contributes the same token width, and
  cropping only the fingertips would break that assumption for the very
  analysis this change is measured by.
* The diffusion encoder sizes its feature dimension from a dummy input at
  construction time; an unchanged input shape cannot disagree with it.
* pi0.5 letterboxes with `resize_with_pad`, so a changed aspect ratio would
  change how much padding each slot gets -- a second, uncontrolled difference
  between the cropped run and its baseline.

WHERE IT RUNS. As a processor step at index 0 of the preprocessor, ahead of
`RenameObservationsProcessorStep`. That ordering is load-bearing on pi0.5,
whose rename map turns `observation.images.left_arm_left_gripper` into
`observation.images.left_wrist_0_rgb` -- run the crop afterwards and it would
look for camera names that no longer exist and silently do nothing. The
preprocessor runs during training as well as inference
(`lerobot_train.py` calls it on every batch), so this changes what the policy is
trained on, not merely what it is shown at deployment.

WHAT IS NOT HERE. Contact gating -- feeding tactile only once contact is
established -- was considered and deliberately left out. "Stable contact" is a
temporal predicate and training shuffles frames, so a processor step sees a
batch and never an episode; the repo's own contact segmentation
(`analysis/phases.py`) needs a whole episode because it thresholds on quantiles
of that episode's own gripper channels. A training-time gate would have to be
re-derived from the tactile image against a no-contact reference, with its own
threshold to justify. That is a separate piece of work, not a flag on this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_IMAGES

#: The four fingertip cameras on this rig. There is no `tactile` flag on a
#: camera anywhere -- `dataset_view.COMPOSITES` hard-codes the same four names
#: for the same reason -- so the set is named here and overridable per run.
TACTILE_CAMERAS: "tuple[str, ...]" = (
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)

#: Fraction of each side kept, centred. Provisional until measured against the
#: dataset -- see `tool/measure_tactile_border.py`, which picks the largest
#: centred box that excludes the bright, temporally-flat border.
DEFAULT_CROP = 0.7


def crop_box(height: int, width: int, fraction: float) -> "tuple[int, int, int, int]":
    """The centred box keeping ``fraction`` of each side: ``(top, left, h, w)``.

    Rounded to at least one pixel, so a silly fraction produces a small image
    rather than an empty tensor and a shape error three modules away.
    """
    if not 0 < fraction <= 1:
        raise ValueError(
            f"tactile crop fraction must be in (0, 1], got {fraction!r}. "
            "1.0 keeps the whole frame, which is the uncropped baseline."
        )
    keep_h = max(1, int(round(height * fraction)))
    keep_w = max(1, int(round(width * fraction)))
    return ((height - keep_h) // 2, (width - keep_w) // 2, keep_h, keep_w)


def crop_and_restore(image: torch.Tensor, fraction: float) -> torch.Tensor:
    """Centre-crop and resize straight back, so the shape never changes.

    Accepts any leading batch dimensions; the last three are ``(C, H, W)``.
    """
    if fraction >= 1.0:
        return image
    height, width = image.shape[-2], image.shape[-1]
    top, left, keep_h, keep_w = crop_box(height, width, fraction)
    cropped = image[..., top : top + keep_h, left : left + keep_w]
    lead = cropped.shape[:-3]
    flat = cropped.reshape(-1, *cropped.shape[-3:])
    resized = F.interpolate(
        flat.float(), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.reshape(*lead, *resized.shape[-3:]).to(image.dtype)


@ProcessorStepRegistry.register(name="so101_tactile_crop")
@dataclass
class So101TactileCropProcessorStep(ObservationProcessorStep):
    """Centre-crop the tactile cameras, leave every other camera alone.

    Registered so it round-trips through `policy_preprocessor.json`: a
    checkpoint that was trained cropped must be *served* cropped, and the
    pipeline is rebuilt from that file by name. `so101_policies` has to be
    imported for the name to resolve, which
    `--policy.discover_packages_path=so101_policies` already guarantees
    everywhere these policies are trained or served.
    """

    fraction: float = DEFAULT_CROP
    cameras: "tuple[str, ...]" = field(default_factory=lambda: TACTILE_CAMERAS)

    def __post_init__(self):
        crop_box(64, 64, self.fraction)  # refuse a bad fraction at construction
        self.cameras = tuple(self.cameras)

    def _is_tactile(self, key: str) -> bool:
        # Matched on the SHORT name so the same step works whether the key is
        # `observation.images.<name>` or a bare `<name>`.
        return key.rsplit(".", 1)[-1] in self.cameras

    def observation(self, observation: "dict[str, Any]") -> "dict[str, Any]":
        if self.fraction >= 1.0:
            return observation
        out = dict(observation)
        for key, value in observation.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith(f"{OBS_IMAGES}.") and self._is_tactile(key):
                out[key] = crop_and_restore(value, self.fraction)
        return out

    def get_config(self) -> "dict[str, Any]":
        return {"fraction": float(self.fraction), "cameras": list(self.cameras)}

    def transform_features(self, features):
        # The shape is deliberately unchanged -- see the module docstring -- so
        # there is nothing to declare here.
        return features
