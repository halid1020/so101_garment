"""Pick–handover–place scenarios and the mocked human motion script.

A scenario places the payload on one arm's side of the table and the goal
position on the other arm's side, so the task *requires* a bimanual
handover: the picker arm grasps and carries the object to a midline
handover point, the placer arm takes it and places it on the target.

Scenarios are sampled with a fixed seed and kept only if position-only IK
confirms every keypose is reachable by the arm that must reach it, which
is what "physically feasible for the two arms" means here.

The script mimics a human teleoperator: minimum-jerk segments between
keyposes, dwell times around grasp/release, orientation held at the pose
latched at grip-press (same clutch semantics as the tracking benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from sim_benchmark.constants import ARM_JOINT_SUFFIXES, DUAL_URDF_PATH, EE_FRAMES, SIDES
from sim_benchmark.scene import PAYLOAD_HALF

# Motion parameters of the mocked human.
CARRY_SPEED = 0.08  # m/s between keyposes
DWELL = 0.5  # s pause at grasp/transfer/release events
GRASP_HEIGHT = 0.03  # EE z when grasping the payload on the table
TRANSPORT_HEIGHT = 0.11  # EE z while carrying
HOVER_HEIGHT = 0.08  # EE z before descending onto the payload
PLACE_HEIGHT = 0.045  # EE z when releasing above the target
HANDOVER_POINT = np.array([0.26, 0.0, TRANSPORT_HEIGHT])
# Placer EE offset from the handover point (it "reaches into" the grasp
# radius from its own side; sign flipped for a left-side placer). The -z
# component aims at where the payload actually hangs: it is attached
# ~GRASP_HEIGHT - PAYLOAD_HALF below the picker's EE.
HANDOVER_APPROACH = np.array([-0.01, -0.015, -(GRASP_HEIGHT - PAYLOAD_HALF)])
SETTLE_TIME = 1.2  # s tail after release before scoring

FEASIBILITY_TOL = 0.006  # m position residual accepted as "reachable"


@dataclass(frozen=True)
class Scenario:
    """One pick–handover–place instance."""

    index: int
    pick_side: str  # arm that picks the payload
    payload_pos: np.ndarray  # initial payload center (on table)
    target_pos: np.ndarray  # desired final payload center (on table)

    @property
    def place_side(self) -> str:
        return "right" if self.pick_side == "left" else "left"


class _ReachChecker:
    """Position-only IK feasibility test on the Pinocchio reduced model."""

    def __init__(self) -> None:
        full = pin.buildModelFromUrdf(str(DUAL_URDF_PATH))
        gripper_ids = [i for i in range(1, full.njoints) if "gripper" in full.names[i]]
        self.model = pin.buildReducedModel(full, gripper_ids, pin.neutral(full))
        self.data = self.model.createData()
        self._frame_id = {
            side: self.model.getFrameId(EE_FRAMES[side]) for side in SIDES
        }
        self._q_idx = {
            side: np.array(
                [
                    self.model.joints[self.model.getJointId(f"{side}_{sfx}")].idx_q
                    for sfx in ARM_JOINT_SUFFIXES
                ]
            )
            for side in SIDES
        }

    def reach_error(self, side: str, pos: np.ndarray) -> float:
        """Best achievable EE-position residual (m) for one arm."""
        from scipy.optimize import least_squares

        idx = self._q_idx[side]
        q_full = pin.neutral(self.model)

        def residual(q_arm: np.ndarray) -> np.ndarray:
            q = q_full.copy()
            q[idx] = q_arm
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            return self.data.oMf[self._frame_id[side]].translation - pos

        result = least_squares(
            residual,
            q_full[idx],
            bounds=(
                self.model.lowerPositionLimit[idx],
                self.model.upperPositionLimit[idx],
            ),
            method="trf",
            max_nfev=60,
        )
        return float(np.linalg.norm(result.fun))

    def keyposes_feasible(self, checks: list[tuple[str, np.ndarray]]) -> bool:
        """True if every (side, EE position) keypose is reachable by that arm.

        Factored out of :meth:`scenario_feasible` so other tasks (the
        contact-grasp single-arm and handover oracles) can supply their own
        keypose lists to the same reach test.
        """
        return all(
            self.reach_error(side, pos) < FEASIBILITY_TOL for side, pos in checks
        )

    def scenario_feasible(self, scenario: Scenario) -> bool:
        """Every keypose reachable by the arm that must reach it."""
        pick, place = scenario.pick_side, scenario.place_side
        approach = HANDOVER_APPROACH.copy()
        if place == "left":
            approach[1] = -approach[1]
        checks = [
            (pick, np.array([*scenario.payload_pos[:2], GRASP_HEIGHT])),
            (pick, np.array([*scenario.payload_pos[:2], HOVER_HEIGHT])),
            (pick, HANDOVER_POINT),
            (place, HANDOVER_POINT + approach),
            (place, np.array([*scenario.target_pos[:2], PLACE_HEIGHT])),
            (place, np.array([*scenario.target_pos[:2], HOVER_HEIGHT])),
        ]
        return self.keyposes_feasible(checks)


def generate_scenarios(n: int = 30, seed: int = 0) -> list[Scenario]:
    """Sample n feasible scenarios (alternating pick side), seeded."""
    rng = np.random.default_rng(seed)
    checker = _ReachChecker()
    scenarios: list[Scenario] = []
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > 100 * n:
            raise RuntimeError("Scenario sampling failed to converge")
        pick_side = "left" if (seed + len(scenarios)) % 2 == 0 else "right"
        y_sign = 1.0 if pick_side == "left" else -1.0
        payload = np.array(
            [
                rng.uniform(0.22, 0.33),
                y_sign * rng.uniform(0.08, 0.20),
                PAYLOAD_HALF,
            ]
        )
        target = np.array(
            [
                rng.uniform(0.22, 0.33),
                -y_sign * rng.uniform(0.08, 0.20),
                PAYLOAD_HALF,
            ]
        )
        candidate = Scenario(len(scenarios), pick_side, payload, target)
        if checker.scenario_feasible(candidate):
            scenarios.append(candidate)
    return scenarios


def _min_jerk(x: float) -> float:
    """Quintic minimum-jerk profile on [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


@dataclass
class _Track:
    """Piecewise minimum-jerk EE position track for one arm."""

    times: list[float] = field(default_factory=list)  # keypose arrival times
    points: list[np.ndarray] = field(default_factory=list)

    def append(self, point: np.ndarray, arrive_at: float) -> None:
        self.times.append(arrive_at)
        self.points.append(point.copy())

    def move_to(self, point: np.ndarray, speed: float = CARRY_SPEED) -> float:
        """Append a keypose reached at ``speed`` m/s; return arrival time.

        ``speed`` defaults to CARRY_SPEED so existing callers are unaffected;
        the contact-grasp oracle passes a slower LIFT_SPEED for lifts.
        """
        dist = float(np.linalg.norm(point - self.points[-1]))
        arrival = self.times[-1] + max(dist / speed, 0.4)
        self.append(point, arrival)
        return arrival

    def hold(self, until: float) -> None:
        if until > self.times[-1]:
            self.append(self.points[-1], until)

    def at(self, t: float) -> np.ndarray:
        if t <= self.times[0]:
            return self.points[0].copy()
        if t >= self.times[-1]:
            return self.points[-1].copy()
        k = int(np.searchsorted(self.times, t) - 1)
        t0, t1 = self.times[k], self.times[k + 1]
        p0, p1 = self.points[k], self.points[k + 1]
        s = _min_jerk((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
        return p0 + s * (p1 - p0)


class HandoverScript:
    """Dual-arm EE targets + grip signals implementing one scenario."""

    def __init__(
        self,
        scenario: Scenario,
        initial_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.scenario = scenario
        self._rot = {side: initial_poses[side][1].copy() for side in SIDES}
        pick, place = scenario.pick_side, scenario.place_side
        obj_xy = scenario.payload_pos[:2]
        tgt_xy = scenario.target_pos[:2]
        approach = HANDOVER_APPROACH.copy()
        if place == "left":
            approach[1] = -approach[1]

        picker = _Track()
        placer = _Track()
        picker.append(initial_poses[pick][0], 0.0)
        placer.append(initial_poses[place][0], 0.0)

        # -- pick
        picker.move_to(np.array([*obj_xy, HOVER_HEIGHT]))
        t_grasp_arrive = picker.move_to(np.array([*obj_xy, GRASP_HEIGHT]))
        self.t_pick_close = t_grasp_arrive + 0.5 * DWELL
        picker.hold(t_grasp_arrive + DWELL)

        # -- carry to handover; placer approaches meanwhile
        picker.move_to(np.array([*obj_xy, TRANSPORT_HEIGHT]))
        t_pick_at_handover = picker.move_to(HANDOVER_POINT)
        placer.hold(self.t_pick_close)  # placer waits until pick succeeds
        t_place_at_handover = placer.move_to(HANDOVER_POINT + approach)

        # -- transfer: placer closes, picker opens shortly after
        t_transfer = max(t_pick_at_handover, t_place_at_handover) + 0.5 * DWELL
        self.t_place_close = t_transfer
        self.t_pick_open = t_transfer + DWELL
        picker.hold(self.t_pick_open + 0.5 * DWELL)
        placer.hold(self.t_pick_open + 0.5 * DWELL)

        # -- picker retreats; placer carries to target and releases
        picker.move_to(
            HANDOVER_POINT + np.array([0, 0.06 if pick == "left" else -0.06, 0.02])
        )
        placer.move_to(np.array([*tgt_xy, TRANSPORT_HEIGHT]))
        t_place_arrive = placer.move_to(np.array([*tgt_xy, PLACE_HEIGHT]))
        self.t_place_open = t_place_arrive + 0.5 * DWELL
        placer.hold(t_place_arrive + DWELL)
        placer.move_to(np.array([*tgt_xy, HOVER_HEIGHT]))

        self._tracks = {pick: picker, place: placer}
        self.duration = max(picker.times[-1], placer.times[-1]) + SETTLE_TIME

    def targets(self, t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            side: (self._tracks[side].at(t), self._rot[side].copy()) for side in SIDES
        }

    def grips(self, t: float) -> dict[str, bool]:
        """Desired gripper-closed state per side at time t."""
        pick, place = self.scenario.pick_side, self.scenario.place_side
        return {
            pick: self.t_pick_close <= t < self.t_pick_open,
            place: self.t_place_close <= t < self.t_place_open,
        }
