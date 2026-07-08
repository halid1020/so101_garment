"""Analytic workspace envelope + out-of-envelope (OOE) target policies.

The reachable position set of one SO-101 arm is approximated per arm as an
annulus around its shoulder-lift pivot plus a table clearance floor:

    r_min <= || p - pivot(azimuth) || <= r_max   and   p.z >= z_floor

where ``pivot(azimuth)`` is the shoulder-lift joint center. The pivot is not
on the pan axis — it orbits it at ~35.5 mm — so it is recomputed from the
*target's* azimuth about the pan axis (pan tracks the target azimuth in
steady state; the transient error is absorbed by WORKSPACE_SAFETY_MARGIN).

The radii are NOT hand-derived from link lengths (the wrist_roll URDF rpy
makes that treacherous); they come from ``derive_workspace_radii`` — a FK
grid sweep over elbow_flex x wrist_flex (the pivot->EE distance is invariant
to shoulder_lift, which rotates the distal chain rigidly). The swept values
are hardcoded in configs (WORKSPACE_R_MIN/WORKSPACE_R_MAX) and a unit test
re-derives them so URDF edits cannot silently stale the constants.

Four selectable policies decide what happens when the operator's hand drags
the target outside the envelope (``WORKSPACE_OOB_MODE`` / ``--oob-mode``):

- ``warn``    — passthrough + throttled warning (legacy behavior + telemetry).
- ``project`` — clamp the target to the nearest envelope point every frame;
                the arm slides along the boundary and re-entry is seamless.
- ``freeze``  — hold the last feasible target while outside ("leash").
- ``slow``    — scale the outward motion component down as the boundary
                approaches (within WORKSPACE_SOFT_MARGIN); tangential motion
                stays full-rate; degrades to projection if still outside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from common.configs import (
    WORKSPACE_R_MAX,
    WORKSPACE_R_MIN,
    WORKSPACE_SAFETY_MARGIN,
    WORKSPACE_SOFT_MARGIN,
    WORKSPACE_Z_FLOOR,
)

_WARN_PERIOD_S = 1.0


@dataclass(frozen=True)
class ArmEnvelope:
    """Annulus-plus-floor reachable-set approximation for one arm."""

    pan_xy: np.ndarray  # world xy of the pan axis
    pivot_offset_xy: np.ndarray  # lift-pivot offset in the arm frame
    # (x = toward the target azimuth)
    pivot_z: float  # world z of the lift pivot
    r_min: float
    r_max: float
    z_floor: float

    def pivot(self, p: np.ndarray) -> np.ndarray:
        """Lift-pivot position for the pan angle implied by target ``p``."""
        vx = p[0] - self.pan_xy[0]
        vy = p[1] - self.pan_xy[1]
        n = np.hypot(vx, vy)
        if n < 1e-9:
            cx, sx = 1.0, 0.0
        else:
            cx, sx = vx / n, vy / n
        ox, oy = self.pivot_offset_xy
        return np.array(
            [
                self.pan_xy[0] + cx * ox - sx * oy,
                self.pan_xy[1] + sx * ox + cx * oy,
                self.pivot_z,
            ]
        )

    def margin(self, p: np.ndarray) -> float:
        """Signed distance to the nearest boundary (>0 inside)."""
        r = float(np.linalg.norm(p - self.pivot(p)))
        return min(p[2] - self.z_floor, self.r_max - r, r - self.r_min)

    def is_inside(self, p: np.ndarray) -> bool:
        return self.margin(p) >= 0.0

    def outward_normal(self, p: np.ndarray) -> np.ndarray:
        """Unit direction of *decreasing* margin (toward the active boundary)."""
        piv = self.pivot(p)
        v = p - piv
        r = float(np.linalg.norm(v))
        v_hat = v / r if r > 1e-9 else np.array([1.0, 0.0, 0.0])
        terms = (p[2] - self.z_floor, self.r_max - r, r - self.r_min)
        idx = int(np.argmin(terms))
        if idx == 0:  # floor: outward = down
            return np.array([0.0, 0.0, -1.0])
        if idx == 1:  # outer sphere: outward = radially out
            return v_hat
        return -v_hat  # inner sphere: outward = toward the pivot

    def project(self, p: np.ndarray) -> np.ndarray:
        """Nearest point on/inside the envelope.

        z-clamp then radial clamp is exact for this geometry: the pivot sits
        above the floor by more than r_min (0.1166 vs 0.0837 + 0.01 m), so
        the r_min inflation can never push a point below the floor, and the
        r_max contraction moves toward the (above-floor) pivot, never down
        through it. A final defensive z-clamp guards regressions.

        Iterated: the radial clamp slightly shifts the target's azimuth
        about the pan axis, which moves the pivot; a few fixed-point passes
        reconverge (worst observed: 2 extra passes for points near the pan
        axis). Terminates early once the update falls under 1 um.
        """
        q = p.copy()
        for _ in range(8):
            q_prev = q.copy()
            if q[2] < self.z_floor:
                q[2] = self.z_floor
            piv = self.pivot(q)
            v = q - piv
            r = float(np.linalg.norm(v))
            if r < 1e-9:
                v = np.array([1.0, 0.0, 0.0])
                r = 1.0
            r_c = min(max(r, self.r_min), self.r_max)
            if r_c != r:
                q = piv + v * (r_c / r)
            if float(np.linalg.norm(q - q_prev)) < 1e-6:
                break
        if q[2] < self.z_floor:  # defensive; unreachable for current geometry
            q[2] = self.z_floor
        return q


@dataclass
class OOEStatus:
    """Per-apply telemetry of the envelope check."""

    inside: bool
    margin_m: float
    clamped: bool


class OOEPolicy(ABC):
    """Strategy for handling out-of-envelope target positions."""

    name = ""

    def __init__(self, side: str, envelope: ArmEnvelope) -> None:
        self.side = side
        self.envelope = envelope
        self._last_warn_t = -np.inf

    @abstractmethod
    def apply(self, p_target: np.ndarray, t: float) -> tuple[np.ndarray, OOEStatus]:
        """Map a raw target position to the position handed to the IK."""

    def reset(self) -> None:
        """Clear per-episode state (called on teleop deactivation)."""

    def _warn(self, t: float, margin_m: float, p: np.ndarray) -> None:
        if t - self._last_warn_t >= _WARN_PERIOD_S:
            self._last_warn_t = t
            print(
                f"⚠️  {self.side} target out of envelope "
                f"(margin {margin_m * 1000:+.0f} mm at {np.round(p, 3)}, "
                f"policy={self.name})"
            )


class WarnOnlyPolicy(OOEPolicy):
    """Passthrough (current behavior) + throttled warning + telemetry."""

    name = "warn"

    def apply(self, p_target: np.ndarray, t: float) -> tuple[np.ndarray, OOEStatus]:
        margin = self.envelope.margin(p_target)
        inside = margin >= 0.0
        if not inside:
            self._warn(t, margin, p_target)
        return p_target, OOEStatus(inside=inside, margin_m=margin, clamped=False)


class ProjectPolicy(OOEPolicy):
    """Clamp the target to the nearest envelope point every frame."""

    name = "project"

    def apply(self, p_target: np.ndarray, t: float) -> tuple[np.ndarray, OOEStatus]:
        margin = self.envelope.margin(p_target)
        inside = margin >= 0.0
        if inside:
            return p_target, OOEStatus(inside=True, margin_m=margin, clamped=False)
        self._warn(t, margin, p_target)
        return self.envelope.project(p_target), OOEStatus(
            inside=False, margin_m=margin, clamped=True
        )


class FreezePolicy(OOEPolicy):
    """Hold the last feasible target while outside; release on re-entry."""

    name = "freeze"

    def __init__(self, side: str, envelope: ArmEnvelope) -> None:
        super().__init__(side, envelope)
        self._last_feasible: np.ndarray | None = None

    def reset(self) -> None:
        self._last_feasible = None

    def apply(self, p_target: np.ndarray, t: float) -> tuple[np.ndarray, OOEStatus]:
        margin = self.envelope.margin(p_target)
        inside = margin >= 0.0
        if inside:
            self._last_feasible = p_target.copy()
            return p_target, OOEStatus(inside=True, margin_m=margin, clamped=False)
        self._warn(t, margin, p_target)
        if self._last_feasible is None:
            # Never seen a feasible target (e.g. activation outside): project.
            self._last_feasible = self.envelope.project(p_target)
        return self._last_feasible.copy(), OOEStatus(
            inside=False, margin_m=margin, clamped=True
        )


class SlowdownPolicy(OOEPolicy):
    """Scale outward motion down near the boundary; slide tangentially freely.

    The emitted target integrates the operator's per-tick step with its
    outward-normal component scaled by smoothstep(margin / soft_margin)
    (1 far inside, 0 at the boundary). If the result still escapes (fast
    hand), it degrades to projection.
    """

    name = "slow"

    def __init__(
        self,
        side: str,
        envelope: ArmEnvelope,
        soft_margin: float = WORKSPACE_SOFT_MARGIN,
    ) -> None:
        super().__init__(side, envelope)
        self.soft_margin = soft_margin
        self._p_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._p_prev = None

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def apply(self, p_target: np.ndarray, t: float) -> tuple[np.ndarray, OOEStatus]:
        margin = self.envelope.margin(p_target)
        if self._p_prev is None:
            self._p_prev = self.envelope.project(p_target)
        prev_margin = self.envelope.margin(self._p_prev)
        step = p_target - self._p_prev
        clamped = False
        if prev_margin < self.soft_margin:
            n_out = self.envelope.outward_normal(self._p_prev)
            d_n = float(step @ n_out)
            if d_n > 0.0:  # only outward motion is attenuated
                scale = self._smoothstep(prev_margin / self.soft_margin)
                step = step + (scale - 1.0) * d_n * n_out
                clamped = True
            if prev_margin < 0.0:
                self._warn(t, prev_margin, self._p_prev)
        p_out = self._p_prev + step
        if not self.envelope.is_inside(p_out):
            p_out = self.envelope.project(p_out)
            clamped = True
        self._p_prev = p_out
        return p_out, OOEStatus(inside=margin >= 0.0, margin_m=margin, clamped=clamped)


OOE_POLICIES: dict[str, type[OOEPolicy]] = {
    WarnOnlyPolicy.name: WarnOnlyPolicy,
    ProjectPolicy.name: ProjectPolicy,
    FreezePolicy.name: FreezePolicy,
    SlowdownPolicy.name: SlowdownPolicy,
}


def derive_workspace_radii(
    urdf_model: pin.Model, grid: int = 61, side: str = "left"
) -> tuple[float, float]:
    """(r_min, r_max) of the EE-to-lift-pivot distance by FK grid sweep.

    Sweeps elbow_flex x wrist_flex within URDF limits; the distance is
    invariant to shoulder_lift (verified), so it is left at neutral.
    """
    data = urdf_model.createData()
    eef_id = urdf_model.getFrameId(f"{side}_eef_link")
    lift_id = urdf_model.getJointId(f"{side}_shoulder_lift")
    elbow_iq = urdf_model.joints[urdf_model.getJointId(f"{side}_elbow_flex")].idx_q
    wrist_iq = urdf_model.joints[urdf_model.getJointId(f"{side}_wrist_flex")].idx_q
    lo, hi = urdf_model.lowerPositionLimit, urdf_model.upperPositionLimit
    r_min, r_max = np.inf, 0.0
    q = pin.neutral(urdf_model)
    for q_elbow in np.linspace(lo[elbow_iq], hi[elbow_iq], grid):
        for q_wrist in np.linspace(lo[wrist_iq], hi[wrist_iq], grid):
            q[elbow_iq], q[wrist_iq] = q_elbow, q_wrist
            pin.forwardKinematics(urdf_model, data, q)
            pin.updateFramePlacements(urdf_model, data)
            r = float(
                np.linalg.norm(
                    data.oMf[eef_id].translation - data.oMi[lift_id].translation
                )
            )
            r_min, r_max = min(r_min, r), max(r_max, r)
    return r_min, r_max


def build_envelopes(urdf_model: pin.Model) -> dict[str, ArmEnvelope]:
    """Per-side envelopes from FK at neutral + WORKSPACE_* constants.

    The safety margin is applied here: the usable annulus is shrunk by
    WORKSPACE_SAFETY_MARGIN on both radii relative to the swept values.
    """
    data = urdf_model.createData()
    q0 = pin.neutral(urdf_model)
    pin.forwardKinematics(urdf_model, data, q0)
    pin.updateFramePlacements(urdf_model, data)
    envelopes: dict[str, ArmEnvelope] = {}
    for side in ("left", "right"):
        p_pan = data.oMi[urdf_model.getJointId(f"{side}_shoulder_pan")].translation
        p_piv = data.oMi[urdf_model.getJointId(f"{side}_shoulder_lift")].translation
        # At neutral the arm reaches along +x, so the world-frame offset IS
        # the arm-frame offset.
        offset_xy = (p_piv - p_pan)[:2].copy()
        envelopes[side] = ArmEnvelope(
            pan_xy=p_pan[:2].copy(),
            pivot_offset_xy=offset_xy,
            pivot_z=float(p_piv[2]),
            r_min=WORKSPACE_R_MIN + WORKSPACE_SAFETY_MARGIN,
            r_max=WORKSPACE_R_MAX - WORKSPACE_SAFETY_MARGIN,
            z_floor=WORKSPACE_Z_FLOOR,
        )
    return envelopes


def make_policies(mode: str, envelopes: dict[str, ArmEnvelope]) -> dict[str, OOEPolicy]:
    """One policy instance per side; raises on an unknown mode."""
    if mode not in OOE_POLICIES:
        raise ValueError(
            f"Unknown out-of-envelope mode {mode!r}; "
            f"choose from {sorted(OOE_POLICIES)}"
        )
    return {side: OOE_POLICIES[mode](side, env) for side, env in envelopes.items()}
