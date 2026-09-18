"""This rig's shape, on the shared loopback monitor.

THE SERVER MOVED. It is `actoris_harena.recording.monitor_server` now, because
the console that consumes it serves every robot on the machine and the server
had this rig's schema compiled into it -- `SIDES`, `BODY_JOINTS` and a
`DualDataManager`'s method names, imported at module scope. A single UR3e could
not have used a line of it.

WHAT IS LEFT HERE is the part that is genuinely about two SO-101 arms: the
adapter that answers the three questions a frame store cannot. Everything else
-- the routes, the framing, the key allow-list, the MJPEG pacing -- is shared and
unmodified.

`MonitorServer(data_manager, ...)` still takes a data manager, so no caller
changed: the adapter is built here rather than at every call site.
"""

from __future__ import annotations

from typing import Any

from actoris_harena.recording.monitor_server import MonitorSource  # noqa: F401
from actoris_harena.recording.monitor_server import (  # noqa: F401
    DEFAULT_MAX_WIDTH,
    DEFAULT_QUALITY,
    DEFAULT_VIEW_FPS,
)
from actoris_harena.recording.monitor_server import MonitorServer as _MonitorServer
from actoris_harena.recording.monitor_server import joint_snapshot as _joint_snapshot

from common.robot_schema import SCHEMA, SIDES


def joint_snapshot(measured, grippers, commands, now, fresh_s=None):
    """This rig's schema, curried onto the shared pure function.

    Kept at the old signature because `common/web/session.py` and the unit tests
    call it that way, and because the schema is not a parameter anyone on this
    rig would ever want to vary.
    """
    if fresh_s is None:
        return _joint_snapshot(SCHEMA, measured, grippers, commands, now)
    return _joint_snapshot(SCHEMA, measured, grippers, commands, now, fresh_s)


class DualArmMonitorSource:
    """A `DualDataManager`, as the shared monitor wants to ask it questions.

    Delegation rather than inheritance: the data manager is the session's whole
    blackboard and the monitor should only be able to READ the handful of things
    it needs. `__getattr__` forwards the frame half, which the data manager
    already satisfies by being a `FramePublisher`.
    """

    def __init__(self, data_manager: Any) -> None:
        self.data_manager = data_manager

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not defined below -- the frame-store half.
        return getattr(self.data_manager, name)

    def monitor_joints(self, now: float) -> "dict[str, Any]":
        dm = self.data_manager
        return joint_snapshot(
            dm.get_current_joint_angles(),
            {side: dm.get_current_gripper_open_value(side) for side in SIDES},
            {side: dm.get_last_sent_command(side) for side in SIDES},
            now,
        )

    def monitor_activity(self) -> str:
        return str(self.data_manager.get_robot_activity_state().value)

    def monitor_joint_drift_s(self, now: float) -> "float | None":
        interpolated = self.data_manager.get_current_joint_angles_at(now)
        return None if interpolated is None else round(abs(interpolated[1]), 4)


class MonitorServer(_MonitorServer):
    """The shared server, wired to this rig's schema and adapter."""

    def __init__(self, data_manager: Any, **kwargs: Any) -> None:
        super().__init__(DualArmMonitorSource(data_manager), SCHEMA, **kwargs)

    @property
    def data_manager(self) -> Any:
        """The blackboard, for callers that had it before the adapter existed."""
        return self.source.data_manager
