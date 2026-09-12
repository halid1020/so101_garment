"""This rig's state and action layout: two five-joint arms, a gripper each.

``actoris_harena.recording.features`` holds the layout RULE -- per limb, its
body joints in URDF order then its gripper open fraction -- and takes the shape
as a :class:`RobotSchema`. This is that shape for these arms, plus the derived
names the rest of this repo already imports by the old spellings.

Nothing here is new. ``SIDES``, ``BODY_JOINTS``, ``BODY_DOF`` and
``STATE_NAMES`` were module constants in the shared file until it had to serve a
second robot; they are re-exported from here so the hundred-odd places that use
them keep reading the way they did.
"""

from actoris_harena.recording.features import RobotSchema

# The five actuated body joints per SO-101 arm, in URDF order.
BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SIDES = ["left", "right"]

#: The one description of this rig's vectors. Everything below is derived from
#: it, so there is no second place for the two to disagree.
SCHEMA = RobotSchema(limbs=tuple(SIDES), body_joints=tuple(BODY_JOINTS))

BODY_DOF = SCHEMA.body_dof  # 5
#: The 12 state/action channel names: per side the five body joints then the
#: gripper, left arm first -- "left_shoulder_pan.pos" ... "right_gripper.pos".
STATE_NAMES = SCHEMA.state_names
EE_NAMES = SCHEMA.ee_names
EE_DOF = SCHEMA.ee_dim  # 14
#: Columns 5 and 11. DERIVED, where the analyses used to write them down.
GRIPPER_COLUMNS = SCHEMA.gripper_columns
