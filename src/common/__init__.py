"""Common modules for the dual SO-101 rig.

Teleoperation, this rig's hardware boundaries, and the console routes that
compose its tabs. The collection / visualisation / training / deployment
pipeline itself lives in ``actoris_harena``; what is left here is what is
actually about these arms.

Importing this package declares this rig's camera profile to the shared view
builder -- see ``common.rig_profile``. It is a side effect on purpose: the
tables have to be installed before anything builds a camera view, and every
entry point in this repo reaches ``common`` long before it does.
"""

from common import rig_profile  # noqa: F401  (imported for its side effect)
