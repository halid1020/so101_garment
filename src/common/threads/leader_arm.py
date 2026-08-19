"""Leader-arm reader thread for joint-space dual-arm teleoperation.

Polls two passive SO-101 leader arms (LeRobot ``SOLeader``) at the joint
streaming rate and publishes 10-DOF URDF-degree targets plus synthesised
trigger values into the shared ``DualDataManager``. The existing per-arm
joint-state threads remain the sole motor writers, so the recorder's
``action`` feature and the sidecar command columns keep exactly the Quest
semantics.

Unit mapping: a leader in DEGREES normalisation reads degrees about its
calibrated mid-range — the same convention as the follower's hardware
frame — so the leader reading is treated as the follower's HW reading and
converted with the follower's own hw→URDF signs/offsets.

Engage safety: whenever teleoperation (re)activates, the published command
seeds from the followers' measured joints and slews toward the leader pose
at a bounded joint velocity. The clamp stays on permanently, so it doubles as a
joint-velocity limit during tracking. Its per-tick budget comes from the time the
tick actually took, not from the nominal period, so the limit stays a velocity in
degrees per second however fast the loop happens to run — see
:func:`velocity_step_deg`, which explains why the difference is what an operator
feels as follower lag.
"""

import math
import time
import traceback
from typing import Any, Mapping

import numpy as np

from common.configs import (
    GRIPPER_OPEN_MAX_FRAC,
    HANDLE_ROLL_OFFSET_DEG,
    JOINT_STATE_STREAMING_RATE,
    LEFT_ARM_HW_TO_URDF_OFFSETS_DEG,
    LEFT_ARM_HW_TO_URDF_SIGNS,
    MAX_JOINT_VEL_HW_RAD_S,
    RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG,
    RIGHT_ARM_HW_TO_URDF_SIGNS,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.device_faults import bus_gone

_BODY_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# Slew budget: the longest single tick that may be spent as movement, in nominal
# periods (see velocity_step_deg).
_MAX_CATCHUP_TICKS = 5.0
# Starvation reporting: a tick this many times over budget counts as slow, and a
# run of them lasting this long is reported, at most this often.
_SLOW_TICK_FACTOR = 2.0
_SLOW_RUN_S = 1.0
_SLOW_WARN_PERIOD_S = 5.0

_HW_TO_URDF = {
    "left": (
        np.array(LEFT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
        np.array(LEFT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    ),
    "right": (
        np.array(RIGHT_ARM_HW_TO_URDF_SIGNS, dtype=np.float64),
        np.array(RIGHT_ARM_HW_TO_URDF_OFFSETS_DEG, dtype=np.float64),
    ),
}


def leader_action_to_urdf(action: Mapping[str, float], side: str) -> np.ndarray:
    """Map one leader's ``get_action()`` body joints to follower URDF degrees.

    The leader's DEGREES reading shares the follower's hardware-frame
    convention (zero at the calibrated mid-range), so the follower's own
    conversion applies: ``urdf = sign * hw + offset``.

    A fixed ``HANDLE_ROLL_OFFSET_DEG`` is added to the wrist_roll joint so the
    follower's wrist camera sits on top of the gripper at the neutral pose —
    the joint-space counterpart of the Quest-mode tip-roll bias (roll about
    the tip is the wrist_roll joint), keeping the two input modes' framing
    consistent.
    """
    signs, offsets = _HW_TO_URDF[side]
    hw = np.array([action[f"{j}.pos"] for j in _BODY_JOINTS], dtype=np.float64)
    urdf = signs * hw + offsets
    urdf[_BODY_JOINTS.index("wrist_roll")] += HANDLE_ROLL_OFFSET_DEG
    return urdf


def leader_gripper_to_trigger(gripper_0_100: float) -> float:
    """Map the leader jaw (0–100, 0 = closed) to Quest trigger semantics.

    The joint-state threads compute the follower opening as
    ``(1 - trigger) * GRIPPER_OPEN_MAX_FRAC``, so a fully open leader jaw
    yields the capped opening and a closed jaw closes fully — identical
    command semantics to Quest recordings.
    """
    frac = min(max(gripper_0_100 / 100.0, 0.0), 1.0)
    return 1.0 - frac


def slew_toward(cmd: np.ndarray, target: np.ndarray, max_step_deg: float) -> np.ndarray:
    """Advance ``cmd`` toward ``target`` by at most ``max_step_deg`` per joint."""
    return cmd + np.clip(target - cmd, -max_step_deg, max_step_deg)


def velocity_step_deg(
    elapsed_s: float, max_joint_vel_rad_s: float, max_catchup_s: float
) -> float:
    """How far a joint may move this tick, from the time the tick actually took.

    The slew limit exists to bound joint VELOCITY, so its per-tick budget has to
    be derived from real elapsed time. Deriving it from the nominal period
    instead turns it into a per-tick allowance: when the thread is starved --
    which it is during recording, competing with the camera and video-encoder
    threads -- the loop runs slower but each tick still advances by the 10 ms
    budget, so the followers track at a fraction of the intended speed and visibly
    trail the leaders.

    ``max_catchup_s`` caps the budget so a long stall cannot be cashed in as one
    large jump: the limit stays a velocity limit rather than becoming a licence to
    teleport after a hiccup. Pure.
    """
    budget_s = min(max(elapsed_s, 0.0), max_catchup_s)
    return math.degrees(max_joint_vel_rad_s) * budget_s


def leader_arm_thread(
    data_manager: DualDataManager,
    leaders: Mapping[str, Any],  # SOLeader teleoperators (get_action())
    rate_hz: float = JOINT_STATE_STREAMING_RATE,
    max_joint_vel_rad_s: float = MAX_JOINT_VEL_HW_RAD_S,
    gripper_open_max_frac: float = GRIPPER_OPEN_MAX_FRAC,
) -> None:
    """Poll both leaders and publish joint targets while teleop is active.

    Args:
        data_manager: Shared dual-arm DataManager.
        leaders: {"left": SOLeader, "right": SOLeader} — sides follow the
                 tool's bus wiring (left = bus_0). Each leader owns its own
                 serial port, so no bus lock is shared with the followers.
        rate_hz: Poll/publish rate (default: the joint streaming rate).
        max_joint_vel_rad_s: Per-joint slew limit — bounds the engage
                 catch-up and doubles as a permanent velocity limit.
    """
    print("🕹️  Leader-arm thread started")
    dt = 1.0 / rate_hz
    # Longest tick that may be spent as slew budget in one go (see
    # velocity_step_deg). A few nominal periods absorbs ordinary scheduling
    # jitter without letting a long stall become a jump.
    max_catchup_s = _MAX_CATCHUP_TICKS * dt
    cmd: np.ndarray | None = None  # slewed 10-DOF URDF-degree command
    last_tick: float | None = None  # when the previous command was published
    slow_since: float | None = None  # start of the current run of slow ticks
    last_slow_warn = 0.0
    # Same serial-robustness policy as the joint-state threads: a dropped
    # Feetech status packet surfaces as ConnectionError; skip the tick and
    # only give up after a full second of consecutive failures.
    _MAX_CONSECUTIVE_FAILURES = 100
    consecutive_failures = 0

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            try:
                urdf: dict[str, np.ndarray] = {}
                trigger: dict[str, float] = {}
                for side, leader in leaders.items():
                    action = leader.get_action()
                    urdf[side] = leader_action_to_urdf(action, side)
                    trigger[side] = leader_gripper_to_trigger(action["gripper.pos"])
                    # Observability publish (no consumer in the write path).
                    data_manager.set_leader_mapped_state(
                        side,
                        urdf[side],
                        (1.0 - trigger[side]) * gripper_open_max_frac,
                    )
            except ConnectionError as e:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise
                if consecutive_failures in (1, 10) or consecutive_failures % 50 == 0:
                    print(
                        f"⚠️  leader read failed ({consecutive_failures} in a "
                        f"row, tolerating up to {_MAX_CONSECUTIVE_FAILURES}): {e}"
                    )
                time.sleep(dt)
                continue
            consecutive_failures = 0
            leader_10 = np.concatenate([urdf["left"], urdf["right"]])

            active = (
                data_manager.get_teleop_active()
                and data_manager.get_robot_activity_state()
                == RobotActivityState.ENABLED
            )
            now = time.time()
            if active:
                if cmd is None:
                    # Engage: start from where the FOLLOWERS are, then slew
                    # toward the leader pose — no snap on the first tick.
                    measured = data_manager.get_current_joint_angles()
                    if measured is not None and len(measured) == 10:
                        cmd = np.array(measured, dtype=np.float64)
                    else:
                        cmd = leader_10.copy()
                    last_tick = None
                # First tick after engaging gets one nominal period, not the gap
                # since the thread started.
                elapsed_since_cmd = dt if last_tick is None else now - last_tick
                max_step_deg = velocity_step_deg(
                    elapsed_since_cmd, max_joint_vel_rad_s, max_catchup_s
                )
                cmd = slew_toward(cmd, leader_10, max_step_deg)
                last_tick = now
                data_manager.set_target_joint_angles(cmd)
                for side in ("left", "right"):
                    # transform=None: no IK thread reads controller transforms
                    # in leader mode; only the trigger feeds the gripper path.
                    data_manager.set_controller_state(side, None, 0.0, trigger[side])
            else:
                cmd = None  # force a fresh engage slew next activation
                last_tick = None

            elapsed = time.time() - iteration_start
            # Surface starvation: the followers cannot track a leader faster than
            # this thread runs, so a sustained slow loop is felt directly as lag
            # and the operator should be told which of the two it is.
            if elapsed > _SLOW_TICK_FACTOR * dt:
                slow_since = iteration_start if slow_since is None else slow_since
                if (
                    iteration_start - slow_since > _SLOW_RUN_S
                    and iteration_start - last_slow_warn > _SLOW_WARN_PERIOD_S
                ):
                    last_slow_warn = iteration_start
                    print(
                        f"⚠️  leader loop starved: {1.0 / max(elapsed, 1e-6):.0f} Hz "
                        f"(want {rate_hz:.0f} Hz) — the followers will trail the "
                        "leaders while this lasts"
                    )
            else:
                slow_since = None
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        if bus_gone(e):
            print("❌ a leader arm's serial bus disappeared — ending the session")
            data_manager.note_bus_lost()
        else:
            print(f"❌ Leader-arm thread error: {e}")
            traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🕹️  Leader-arm thread stopped")
