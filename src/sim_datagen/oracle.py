"""Scripted-oracle experts for the sim-VLA contact-grasp pick-and-place tasks.

Two tasks share one grasp model:

* ``SinglePickPlaceScript`` -- one arm picks the bar off the table and places
  it on the marked target; the other arm idles, holding its initial pose.
* ``HandoverContactScript`` -- the picker lifts the bar, hands it to the
  placer at the midline, and the placer sets it on the target.

Both drive the arms through minimum-jerk EE tracks (reused from
``sim_benchmark.handover``) and command per-side gripper open fractions with
short linear ramps rather than the benchmark's boolean mock grasp: the bar is
held by genuine finger contact, so the schedule squeezes to a bounded pinch
and dwells while the contact settles.

Frames: every position here is in the IK frame (the dual-URDF frame the teleop
methods solve in). The environment converts to the twin world with the
measured IK->world offset. The grasp offset ``GRASP_OFFSET_EE`` maps the
eef_link control point to the pinch centre between the fingertips, so a target
grasp point ``p`` is reached by commanding the EE to ``p - R @ GRASP_OFFSET_EE``
for the held orientation ``R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sim_benchmark.constants import SIDES
from sim_benchmark.handover import _ReachChecker, _Track

# ---------------------------------------------------------------------------
# Grasp geometry (measured from the compiled twin at the neutral pose).
# ---------------------------------------------------------------------------
# The gripper closes along the EE y-axis; the pinch centre sits ~1 cm behind
# and level with the eef_link control point. Tunable by the contact-grasp
# tuning loop (tool/collect_sim_dataset.py --dry-run).
GRASP_OFFSET_EE = np.array([0.007, -0.015, 0.0])

# Neutral EE rotation (world<-EE), identical for both arms, measured from the
# twin. Used to convert the grasp offset for MuJoCo-free feasibility checks.
NEUTRAL_EE_ROT = np.array(
    [
        [0.8192, 0.0277, 0.5729],
        [0.0008, 0.9988, -0.0495],
        [-0.5735, 0.0410, 0.8182],
    ]
)
GRASP_OFFSET_WORLD = NEUTRAL_EE_ROT @ GRASP_OFFSET_EE

# Fixed grasp attitude (spec risk #3: jaws square to the prism). Position-only
# IK lets the wrist pitch drift with position, so a bar gripped rigidly at one
# attitude is rotated past its ~12 deg topple margin during the lift. Instead
# every script commands a CONSTANT grasp attitude — pitch-down at the value the
# IK naturally settles to at table-level grasps (measured 62-70 deg across
# seeds, mean ~66) with the pinch axis horizontal — and the collector raises
# the IK orientation cost so the attitude actually holds through the carry.
GRASP_PITCH_DEG = 66.0


def grasp_rotation(
    azimuth: float, flip: bool = False, pitch_deg: float | None = None
) -> np.ndarray:
    """EE rotation for a top grasp: x tip-down at ``pitch_deg``, yaw=azimuth.

    Columns: x = gripper axis (wrist->tip) pitched down; y = horizontal pinch
    axis; z completes the right-handed frame. ``azimuth`` is the horizontal
    heading of the gripper axis (radians, world frame). ``flip`` rolls the
    gripper half a turn about its axis — jaw-equivalent for the pinch, but it
    swaps which physical side the moving jaw and wrist camera stick out on
    (used to deconflict the two grippers during the aerial handover).
    ``pitch_deg`` defaults to GRASP_PITCH_DEG.
    """
    p = np.deg2rad(GRASP_PITCH_DEG if pitch_deg is None else pitch_deg)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    cp, sp = np.cos(p), np.sin(p)
    x = np.array([ca * cp, sa * cp, -sp])
    y = np.array([-sa, ca, 0.0])
    if flip:
        y = -y
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


# Horizontal positions of the two shoulder-pan axes in the IK frame (measured
# with pinocchio FK on the dual URDF: the arms mount 0.30 m apart and the pan
# axis sits 38.8 mm ahead of the base origin). The gripper azimuth follows the
# arm, so grasp attitudes yaw toward each target.
BASE_XY = {
    "left": np.array([0.0388, 0.15]),
    "right": np.array([0.0388, -0.15]),
}


def _azimuth(side: str, xy: np.ndarray) -> float:
    d = np.asarray(xy, dtype=float) - BASE_XY[side]
    return float(np.arctan2(d[1], d[0]))


# Cube geometry in the IK frame. The twin table top rests at IK z = table_z
# (measured ~ -0.0364; the environment supplies the runtime value). The cube
# rests with its centre CUBE_HALF above the table; both tasks grasp it there
# with the pitched top-grasp attitude (grasp_rotation).
CUBE_HALF = 0.011
NOMINAL_TABLE_Z_IK = -0.0364  # scenario feasibility uses this nominal value

# Grasp height relative to the table top. Commanded BELOW the cube centre
# (which sits at CUBE_HALF): the differential IK undershoots the descent by
# ~8 mm at these strained low poses, so aiming at the centre closes the jaws
# above the 2.2 cm cube. Aiming low lets the undershoot land the pinch on the
# cube (swept: 0.004 gave 100%, cube-centre 0.011 only 67%).
GRASP_DZ = 0.004

# Relay (handover) midline: the left arm lays the cube here, the right arm
# picks it up and carries it on. Reachable by BOTH arms (verified with the
# reach checker across x in 0.24-0.28, y ~ 0).
RELAY_MIDDLE_X = 0.26

# Placement: set the cube gently onto the table (a hair high so it drops a
# couple of mm rather than driving into the surface).
PLACE_Z_MARGIN = 0.006
PLACE_DWELL = 0.6  # s the cube rests on the table, still gripped, before release

# Vertical clearances (m) relative to the grasp height.
HOVER_DZ = 0.080  # approach height above the grasp
LIFT_DZ = 0.070  # carry height above the grasp
ATTITUDE_SETTLE = 0.4  # s dwell at hover so the wrist attitude converges
# Commanding the grasp attitude swings the EE downward while the wrist
# reorients (the gripper is a lever on the wrist joint), so each acting arm
# first climbs a staging rise at its start XY — absorbing the transient well
# above the cube — before traversing.
STAGING_RISE = 0.060  # m vertical climb before the first traverse
STAGING_SETTLE = 0.8  # s hold at the staging point while the attitude converges
# Attitude blend: snapping the commanded attitude to the grasp constant makes
# the wrist rotation outpace the position task and the EE dives (the gripper is
# a lever on the wrist). Slerping the command over this window keeps the
# reorientation quasi-static, so the position holds.
ATTITUDE_BLEND_S = 1.5


def _blend_rot(r_from: np.ndarray, r_to: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation between two rotation matrices (alpha in [0, 1])."""
    if alpha >= 1.0:
        return r_to
    if alpha <= 0.0:
        return r_from.copy()
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([r_from, r_to]))
    return slerp([alpha]).as_matrix()[0]


# Grip schedule.
GRIP_OPEN = 1.0
GRIP_SQUEEZE = 0.0  # full-squeeze setpoint (pinch force bounded by forcerange)
GRIP_RAMP = 0.3  # s linear ramp when closing
OPEN_RAMP = 0.5  # s slower ramp when releasing (avoid nudging the bar over)
POST_GRASP_SETTLE = 0.5  # s dwell after a squeeze before lifting
SETTLE_DWELL = 0.6  # s dwell after an open before moving on
# Dwell at the grasp pose BEFORE squeezing: the low cube grasp is a strained,
# slow-converging IK pose and the collector's pre-grasp alignment servo needs
# a stationary window to centre the pinch on the 2.2 cm cube before the jaws
# close (see tool/collect_sim_dataset.run_episode_direct).
GRASP_CONVERGE = 0.9

# Speeds (m/s). Deliberately slow so the cube stays seated in the pinch and
# a gentle carry places it on the mark.
CARRY_SPEED = 0.05
LIFT_SPEED = 0.03
RETREAT_SPEED = 0.03  # slow straight-up retreat after releasing
SWING_SETTLE = 0.5  # s dwell after lifting / above the target to damp the swing

SETTLE_TIME = 1.6  # s tail after the final release before scoring


def _ee_target(
    point: np.ndarray, rot: np.ndarray, offset: np.ndarray = GRASP_OFFSET_EE
) -> np.ndarray:
    """EE command that places the pinch centre at grasp point ``point``."""
    return np.asarray(point, dtype=float) - rot @ offset


class _GripSchedule:
    """Per-side piecewise-linear gripper open-fraction schedule."""

    def __init__(self) -> None:
        self._bp: dict[str, list[tuple[float, float]]] = {s: [] for s in SIDES}

    def add(self, side: str, t: float, frac: float) -> None:
        self._bp[side].append((t, frac))

    def frac(self, side: str, t: float) -> float:
        bps = self._bp[side]
        if not bps:
            return GRIP_OPEN
        if t <= bps[0][0]:
            return bps[0][1]
        if t >= bps[-1][0]:
            return bps[-1][1]
        for (t0, f0), (t1, f1) in zip(bps, bps[1:]):
            if t0 <= t <= t1:
                s = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return float(f0 + s * (f1 - f0))
        return bps[-1][1]


# ---------------------------------------------------------------------------
# Single-arm pick and place
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SingleScenario:
    """One single-arm pick-and-place instance (payload + target on one side)."""

    index: int
    side: str  # acting arm
    payload_xy: np.ndarray
    target_xy: np.ndarray


def generate_single_scenarios(n: int = 30, seed: int = 0) -> list[SingleScenario]:
    """Sample ``n`` feasible single-arm scenarios (side alternates), seeded."""
    rng = np.random.default_rng(seed)
    checker = _ReachChecker()
    gz = NOMINAL_TABLE_Z_IK + GRASP_DZ
    scenarios: list[SingleScenario] = []
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError("Single-arm scenario sampling failed to converge")
        # Side alternates within a batch AND with the seed, so per-seed
        # generation (batch of one) still yields both sides across seeds.
        side = "left" if (seed + len(scenarios)) % 2 == 0 else "right"
        y_sign = 1.0 if side == "left" else -1.0
        payload = np.array([rng.uniform(0.24, 0.30), y_sign * rng.uniform(0.09, 0.17)])
        target = np.array([rng.uniform(0.24, 0.30), y_sign * rng.uniform(0.09, 0.17)])
        if np.linalg.norm(payload - target) < 0.06:
            continue

        def ee(xy: np.ndarray, z: float) -> np.ndarray:
            return np.array([xy[0], xy[1], z]) - GRASP_OFFSET_WORLD

        checks = [
            (side, ee(payload, gz)),
            (side, ee(payload, gz + HOVER_DZ)),
            (side, ee(payload, gz + LIFT_DZ)),
            (side, ee(target, gz + LIFT_DZ)),
            (side, ee(target, gz)),
        ]
        if checker.keyposes_feasible(checks):
            scenarios.append(SingleScenario(len(scenarios), side, payload, target))
    return scenarios


class SinglePickPlaceScript:
    """Single-arm EE targets + grip fractions implementing one scenario."""

    def __init__(
        self,
        scenario: SingleScenario,
        initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
        table_z: float,
    ) -> None:
        self.scenario = scenario
        self.side = scenario.side
        self.idle = "right" if self.side == "left" else "left"
        self._idle_rot = initial_poses[self.idle][1].copy()
        self._init_rot = initial_poses[self.side][1].copy()
        px, py = scenario.payload_xy
        tx, ty = scenario.target_xy
        gz = table_z + GRASP_DZ

        def ee(xy: tuple[float, float], z: float) -> np.ndarray:
            rot = grasp_rotation(_azimuth(self.side, np.asarray(xy)))
            return _ee_target(np.array([xy[0], xy[1], z]), rot)

        track = _Track()
        track.append(initial_poses[self.side][0], 0.0)
        # staging rise (attitude transient) -> hover (dwell) -> slow descend
        t_stage = track.move_to(
            initial_poses[self.side][0] + np.array([0.0, 0.0, STAGING_RISE])
        )
        track.hold(t_stage + STAGING_SETTLE)
        t_hover = track.move_to(ee((px, py), gz + HOVER_DZ))
        track.hold(t_hover + ATTITUDE_SETTLE)
        t_at_grasp = track.move_to(ee((px, py), gz), speed=LIFT_SPEED)
        # settle + align at the grasp pose, then close (ramp) before lifting
        t_close_start = t_at_grasp + GRASP_CONVERGE
        t_closed = t_close_start + GRIP_RAMP
        t_lift = t_closed + POST_GRASP_SETTLE
        track.hold(t_lift)
        # slow lift, dwell to damp the swing, transport, dwell again above the
        # target, then descend to place (a hair high, undershoot compensated)
        # and let the bar rest on the table before releasing.
        t_lifted = track.move_to(ee((px, py), gz + LIFT_DZ), speed=LIFT_SPEED)
        track.hold(t_lifted + SWING_SETTLE)
        t_over = track.move_to(ee((tx, ty), gz + LIFT_DZ), speed=CARRY_SPEED)
        track.hold(t_over + SWING_SETTLE)
        t_at_place = track.move_to(ee((tx, ty), gz + PLACE_Z_MARGIN), speed=LIFT_SPEED)
        track.hold(t_at_place + PLACE_DWELL)
        # open slowly, hold still while the bar steadies, then retreat up
        t_open_start = t_at_place + PLACE_DWELL
        t_opened = t_open_start + OPEN_RAMP
        track.hold(t_opened + SETTLE_DWELL)
        track.move_to(ee((tx, ty), gz + HOVER_DZ), speed=RETREAT_SPEED)
        self._track = track
        self._idle_pos = initial_poses[self.idle][0].copy()

        self._grip = _GripSchedule()
        self._grip.add(self.side, 0.0, GRIP_OPEN)
        self._grip.add(self.side, t_close_start, GRIP_OPEN)
        self._grip.add(self.side, t_closed, GRIP_SQUEEZE)
        self._grip.add(self.side, t_open_start, GRIP_SQUEEZE)
        self._grip.add(self.side, t_opened, GRIP_OPEN)
        self._grip.add(self.idle, 0.0, GRIP_OPEN)

        self.t_close_start = t_close_start
        self.t_open_start = t_open_start
        self.duration = track.times[-1] + SETTLE_TIME

    def baked_offset(self, side: str, ee_xy: np.ndarray) -> np.ndarray:
        """Expected (prism centre - EE) XY for a settled grasp (upright prism)."""
        rot = grasp_rotation(_azimuth(side, ee_xy))
        return (rot @ GRASP_OFFSET_EE)[:2]

    def targets(self, t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        pos = self._track.at(t)
        rot = _blend_rot(
            self._init_rot,
            grasp_rotation(_azimuth(self.side, pos[:2])),
            t / ATTITUDE_BLEND_S,
        )
        return {
            self.side: (pos, rot),
            self.idle: (self._idle_pos.copy(), self._idle_rot.copy()),
        }

    def grip_fractions(self, t: float) -> dict[str, float]:
        return {s: self._grip.frac(s, t) for s in SIDES}


# ---------------------------------------------------------------------------
# Bimanual relay: left picks on the left and lays the cube at the midline;
# right picks it up there and places it on the right target. A table-mediated
# hand-off — two chained single-arm cube pick-places sharing the midline — so
# nothing is ever held in mid-air by one arm and there is no unconstrained
# yaw to fight (the failure mode of the earlier aerial prism handover).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelayScenario:
    """One relay instance: cube on the left, target on the right, shared midline."""

    index: int
    payload_xy: np.ndarray  # left-side spawn (left arm reaches it)
    middle_xy: np.ndarray  # midline hand-off (both arms reach it)
    target_xy: np.ndarray  # right-side target (right arm reaches it)

    @property
    def pick_side(self) -> str:
        return "left"

    @property
    def place_side(self) -> str:
        return "right"


def generate_relay_scenarios(n: int = 30, seed: int = 0) -> list[RelayScenario]:
    """Sample ``n`` feasible relay scenarios (cube left, target right), seeded."""
    rng = np.random.default_rng(seed)
    checker = _ReachChecker()
    gz = NOMINAL_TABLE_Z_IK + GRASP_DZ

    def ee(xy: np.ndarray, z: float) -> np.ndarray:
        return np.array([xy[0], xy[1], z]) - GRASP_OFFSET_WORLD

    scenarios: list[RelayScenario] = []
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError("Relay scenario sampling failed to converge")
        payload = np.array([rng.uniform(0.24, 0.31), rng.uniform(0.09, 0.17)])
        middle = np.array([RELAY_MIDDLE_X, rng.uniform(-0.02, 0.02)])
        target = np.array([rng.uniform(0.24, 0.31), -rng.uniform(0.09, 0.17)])
        checks = [
            ("left", ee(payload, gz)),
            ("left", ee(payload, gz + HOVER_DZ)),
            ("left", ee(payload, gz + LIFT_DZ)),
            ("left", ee(middle, gz)),
            ("left", ee(middle, gz + LIFT_DZ)),
            ("right", ee(middle, gz)),
            ("right", ee(middle, gz + HOVER_DZ)),
            ("right", ee(middle, gz + LIFT_DZ)),
            ("right", ee(target, gz)),
            ("right", ee(target, gz + LIFT_DZ)),
        ]
        if checker.keyposes_feasible(checks):
            scenarios.append(RelayScenario(len(scenarios), payload, middle, target))
    return scenarios


class HandoverContactScript:
    """Dual-arm EE targets + grip fractions for one cube relay.

    Phase 1 (left arm): stage, hover over the cube on the left, descend,
    squeeze, lift, carry to the midline, lay the cube down, release, and
    retreat back to its start so it is clear of the midline. Phase 2 (right
    arm): once the left arm has cleared, stage, hover over the cube at the
    midline, descend, squeeze, lift, carry to the right target, place, and
    release. The idle arm holds its start pose (gripper open) throughout the
    other arm's phase.
    """

    def __init__(
        self,
        scenario: RelayScenario,
        initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
        table_z: float,
    ) -> None:
        self.scenario = scenario
        self.pick, self.place = scenario.pick_side, scenario.place_side
        self._init_rot = {s: initial_poses[s][1].copy() for s in SIDES}
        payload = np.asarray(scenario.payload_xy, dtype=float)
        middle = np.asarray(scenario.middle_xy, dtype=float)
        target = np.asarray(scenario.target_xy, dtype=float)
        gz = table_z + GRASP_DZ

        def ee(side: str, xy: np.ndarray, z: float) -> np.ndarray:
            rot = grasp_rotation(_azimuth(side, np.asarray(xy)))
            return _ee_target(np.array([xy[0], xy[1], z]), rot)

        left = _Track()
        right = _Track()
        left.append(initial_poses["left"][0], 0.0)
        right.append(initial_poses["right"][0], 0.0)

        grip = _GripSchedule()
        grip.add("left", 0.0, GRIP_OPEN)
        grip.add("right", 0.0, GRIP_OPEN)

        # -- helper: append a full pick(src) -> place(dst) segment on one arm's
        # track starting from its current end time, scheduling the grip too.
        def pick_place(track: _Track, side: str, src: np.ndarray, dst: np.ndarray):
            t = track.move_to(track.points[-1] + np.array([0.0, 0.0, STAGING_RISE]))
            track.hold(t + STAGING_SETTLE)
            t = track.move_to(ee(side, src, gz + HOVER_DZ))
            track.hold(t + ATTITUDE_SETTLE)
            t_grasp = track.move_to(ee(side, src, gz), speed=LIFT_SPEED)
            t_close = t_grasp + GRASP_CONVERGE
            t_closed = t_close + GRIP_RAMP
            t_lift = t_closed + POST_GRASP_SETTLE
            track.hold(t_lift)
            t = track.move_to(ee(side, src, gz + LIFT_DZ), speed=LIFT_SPEED)
            track.hold(t + SWING_SETTLE)
            t = track.move_to(ee(side, dst, gz + LIFT_DZ), speed=CARRY_SPEED)
            track.hold(t + SWING_SETTLE)
            t_place = track.move_to(
                ee(side, dst, gz + PLACE_Z_MARGIN), speed=LIFT_SPEED
            )
            track.hold(t_place + PLACE_DWELL)
            t_open = t_place + PLACE_DWELL
            t_opened = t_open + OPEN_RAMP
            track.hold(t_opened + SETTLE_DWELL)
            grip.add(side, t_close, GRIP_OPEN)
            grip.add(side, t_closed, GRIP_SQUEEZE)
            grip.add(side, t_open, GRIP_SQUEEZE)
            grip.add(side, t_opened, GRIP_OPEN)
            return t_close, t_open, t_opened + SETTLE_DWELL

        # Phase 1: left picks the cube and lays it at the midline, then retreats
        # up and BACK to its start so it never sits above the midline while the
        # right arm works there.
        self.t_pick_close, _, t_left_done = pick_place(left, "left", payload, middle)
        left.move_to(ee("left", middle, gz + HOVER_DZ), speed=RETREAT_SPEED)
        t_left_clear = left.move_to(initial_poses["left"][0], speed=CARRY_SPEED)

        # Phase 2: right waits until the left arm is clear, then picks the cube
        # at the midline and places it on the right target.
        right.hold(t_left_clear)
        self.t_place_close, self.t_place_open, _ = pick_place(
            right, "right", middle, target
        )
        right.move_to(ee("right", target, gz + HOVER_DZ), speed=RETREAT_SPEED)

        self._tracks = {"left": left, "right": right}
        # Attitude activation: each arm holds its initial rotation until it
        # begins moving (left at t=0, right once the left arm has cleared), so
        # the idle arm does not pre-rotate its wrist and dip.
        self._t_act = {"left": 0.0, "right": t_left_clear}
        self._grip = grip
        self.duration = max(left.times[-1], right.times[-1]) + SETTLE_TIME

    def baked_offset(self, side: str, ee_xy: np.ndarray) -> np.ndarray:
        """Expected (cube centre - EE) XY for a settled top-grasp by ``side``."""
        rot = grasp_rotation(_azimuth(side, ee_xy))
        return (rot @ GRASP_OFFSET_EE)[:2]

    def targets(self, t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in SIDES:
            pos = self._tracks[side].at(t)
            rot = _blend_rot(
                self._init_rot[side],
                grasp_rotation(_azimuth(side, pos[:2])),
                (t - self._t_act[side]) / ATTITUDE_BLEND_S,
            )
            out[side] = (pos, rot)
        return out

    def grip_fractions(self, t: float) -> dict[str, float]:
        return {s: self._grip.frac(s, t) for s in SIDES}
