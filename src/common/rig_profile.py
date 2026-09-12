"""This rig's cameras, declared once for the shared view builder.

``actoris_harena.recording.dataset_view`` needs three tables that name a rig's
own cameras -- which one feeds which pi0.5 slot, which tile into a composite,
and how a name is shortened in a run-directory slug. They used to be constants
in that module, written for these arms. They are this repo's answer, and a
single-arm rig has a different one, so they live here and are handed over.

Importing this module installs them. ``common/__init__.py`` imports it, so any
code that reaches ``common`` at all has already declared them; the console's
per-rig profile (rig.yaml) will call the same ``set_profile`` for a rig it
drives from outside this repo.
"""

from pathlib import Path

from actoris_harena.action_layout import set_gripper_columns
from actoris_harena.recording.camera_profile import CameraProfile, set_profile
from actoris_harena.recording.config import set_recording_config_path
from actoris_harena.training.destinations import set_destinations_path

# pi0.5 was pretrained with three fixed camera slots under openpi's names, and
# a finetune reaches them by renaming rather than by re-deriving features: each
# slot carries what it learned about that viewpoint, so a rig camera should land
# on the slot that means the same thing. A slot left unmapped is not an error --
# pi0.5 fills it with a padded image and a zero mask, which is exactly how an
# ablated camera should read to the model.
PI05_SLOTS = {
    "central": "observation.images.base_0_rgb",
    "wrist_camera_left": "observation.images.left_wrist_0_rgb",
    "wrist_camera_right": "observation.images.right_wrist_0_rgb",
}

# Named COMPOSITES tile several cameras into ONE image feature.
#
# A policy whose architecture fixes the number of views cannot simply be given
# more of them: FastWAM concatenates its image features into a single frame of
# `policy.image_size`, so five 640x480 cameras have nowhere to go. Tiling the
# four fingertip cameras into one square feature spends one view on all four
# instead of losing three of them, and keeps every tactile signal in front of
# the model.
COMPOSITES: "dict[str, tuple[str, ...]]" = {
    "tactile_quad": (
        "left_arm_left_gripper",
        "left_arm_right_gripper",
        "right_arm_left_gripper",
        "right_arm_right_gripper",
    ),
}

# Cosmetic, but it is in run-directory names that already exist.
SLUG_ELISIONS = (("wrist_camera_", "wrist_"),)

PROFILE = CameraProfile(
    pi05_slots=dict(PI05_SLOTS),
    composites=dict(COMPOSITES),
    slug_elisions=SLUG_ELISIONS,
)

set_profile(PROFILE)


# Where this rig's training destinations are declared. The shared module used to
# derive this from its own location, which was right only while it lived in this
# repo; now the rig says.
REPO_ROOT = Path(__file__).resolve().parents[2]
set_destinations_path(
    REPO_ROOT / "src" / "conf" / "train_destinations.yaml", repo_root=REPO_ROOT
)

# Columns 5 and 11 of the 12-D action are the two grippers -- the recorder's own
# layout, five body joints then a gripper, per arm. The analyses need this and
# cannot derive it: phases segments a grasp by it, and perturb EXCLUDES those
# columns when it occludes a stream, because perturbing a gripper command
# measures something other than what the study asks.
GRIPPER_COLUMNS = (5, 11)
set_gripper_columns(GRIPPER_COLUMNS)

set_recording_config_path(REPO_ROOT / "src" / "conf" / "recording.yaml")
