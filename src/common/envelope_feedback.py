"""Operator feedback for out-of-envelope (OOE) teleop targets.

When the operator drags a target past the reachable workspace envelope
(``src/common/workspace_envelope.py``), the OOE policy quietly clamps/holds
the motion — but the operator, wearing a headset, has no visual cue that the
arm has stopped tracking their hand. This module turns each per-frame
``OOEStatus`` into a debounced, intensity-shaped *signal* the operator can
feel or hear, decoupled from the throttled diagnostic telemetry that already
prints in the IK thread.

The abstract :class:`RateLimitedFeedback` owns the shared debounce/shaping
logic (edge detection, repeat throttling, intensity curve); concrete backends
only implement ``_emit``. Today the shipping backend is a terminal bell
(:class:`TerminalBellFeedback`); the headset haptic backend
(:class:`QuestHapticFeedback`) is a documented stub whose transport protocol
is specified below so it can be finished later without re-deciding anything.

FUTURE Quest haptic channel — protocol specification (implemented later)
========================================================================

The Quest link is strictly one-way today: the host reads controller poses by
parsing ``adb logcat`` from the ``meta_quest_teleop`` reader. Adding an
operator-facing *out* channel needs a host->headset path. The chosen transport
is an Android broadcast Intent sent over the SAME adb session the reader
already owns::

    adb shell am broadcast \\
        -a com.neuracore.metaquestteleop.HAPTIC \\
        --es side <left|right> \\
        --ei intensity <0..100> \\
        --ei duration_ms <40..500>

Intent spec
-----------
* action: ``com.neuracore.metaquestteleop.HAPTIC``
* ``--es side``        — ``left`` or ``right`` (which controller to buzz).
* ``--ei intensity``   — integer 0..100 (0 = stop; the cue's strength).
* ``--ei duration_ms`` — integer 40..500 (single-pulse length).

Why ``am broadcast`` over adb (and not an adb-reverse socket)
-------------------------------------------------------------
* The adb session is already the reader's transport, so this adds zero new
  plumbing and inherits its reconnect behaviour (survives cable re-plugs and
  device re-pairing without extra lifecycle code).
* OOE cues are debounced to a few Hz with only coarse intensity, so the
  50-200 ms latency of an ``am broadcast`` shell round-trip is acceptable.
* An adb-reverse socket would need an APK-side server thread, port lifecycle
  and its own reconnect logic — disproportionate for a few-Hz cue.

Upgrade path
------------
If continuous (>20 Hz) force-feel is ever wanted, the broadcast latency stops
being acceptable and the reverse-socket path (APK server thread + framed
binary protocol) becomes the right transport; the ``EnvelopeFeedback``
interface here does not change, only the ``_emit`` backend behind it.

APK side (future work)
----------------------
A ``BroadcastReceiver`` registered for the action above decodes the extras and
feeds the requested ``(side, intensity, duration_ms)`` into the existing local
haptics machinery (``ovrInputDeviceHandBase::UpdateHaptics`` in the APK C++).

Host-side rule
--------------
The ``adb shell`` round-trip is blocking I/O and MUST run on a worker thread —
never inline in the 100 Hz IK loop. The base-class ``notify`` path here is
pure book-keeping; only ``_emit`` may touch I/O, and a real Quest backend must
hand the shell call off to a worker rather than block.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.configs import FEEDBACK_REPEAT_PERIOD_S, WORKSPACE_SOFT_MARGIN
from common.workspace_envelope import OOEStatus, smoothstep


class EnvelopeFeedback(ABC):
    """Turn per-frame envelope status into operator-facing signalling."""

    @abstractmethod
    def notify(self, side: str, status: OOEStatus, t: float) -> None:
        """Consume one arm's envelope status for frame time ``t`` (seconds)."""

    def reset(self) -> None:
        """Clear per-episode state (called on teleop deactivation)."""


class RateLimitedFeedback(EnvelopeFeedback):
    """Shared debounce + intensity shaping; backends implement ``_emit``.

    Signalling policy, per side:

    * inside->outside edge — emit immediately (the operator has just left the
      reachable set; cue without delay);
    * sustained outside    — re-emit at most every ``repeat_period_s`` (a
      gentle "still outside" reminder, not a continuous buzz);
    * outside->inside edge — emit exactly once with intensity 0.0 (the "stop"
      event, so a backend can cancel any ongoing cue);
    * sustained inside     — emit nothing.

    Intensity ramps ``smoothstep(clip(-margin_m / soft_margin, 0, 1))``: 0 at
    the boundary, saturating to 1 one soft-margin outside it.
    """

    def __init__(
        self,
        repeat_period_s: float = FEEDBACK_REPEAT_PERIOD_S,
        soft_margin: float = WORKSPACE_SOFT_MARGIN,
    ) -> None:
        self.repeat_period_s = repeat_period_s
        self.soft_margin = soft_margin
        # Per-side book-keeping: was the target outside last frame, and when
        # did we last emit while it stayed outside.
        self._was_outside: dict[str, bool] = {}
        self._last_emit_t: dict[str, float] = {}

    def _intensity(self, margin_m: float) -> float:
        """Cue strength in [0, 1]: 0 at the boundary, 1 a soft-margin out."""
        depth = -margin_m / self.soft_margin
        return smoothstep(min(max(depth, 0.0), 1.0))

    def notify(self, side: str, status: OOEStatus, t: float) -> None:
        outside = not status.inside
        was_outside = self._was_outside.get(side, False)
        if outside:
            if not was_outside:
                # Rising edge: cue immediately.
                self._was_outside[side] = True
                self._last_emit_t[side] = t
                self._emit(side, self._intensity(status.margin_m), t)
            elif t - self._last_emit_t.get(side, t) >= self.repeat_period_s:
                # Still outside: throttled reminder.
                self._last_emit_t[side] = t
                self._emit(side, self._intensity(status.margin_m), t)
        elif was_outside:
            # Falling edge: single stop event.
            self._was_outside[side] = False
            self._last_emit_t.pop(side, None)
            self._emit(side, 0.0, t)

    @abstractmethod
    def _emit(self, side: str, intensity: float, t: float) -> None:
        """Deliver one cue for ``side`` at ``intensity`` in [0, 1]."""

    def reset(self) -> None:
        self._was_outside.clear()
        self._last_emit_t.clear()


class TerminalBellFeedback(RateLimitedFeedback):
    """Audible terminal-bell cue — the default operator signalling backend.

    Mirrors the style of the throttled envelope warning in
    ``workspace_envelope.py`` so the two read as a family, but this is the
    operator *signal* (debounced, shaped) rather than the raw telemetry print.
    """

    def _emit(self, side: str, intensity: float, t: float) -> None:
        if intensity <= 0.0:
            # Re-entry: no bell, just a quiet note that tracking resumed.
            print(f"   {side} target back inside envelope")
            return
        # ``\a`` is the ASCII bell; the message carries the depth and strength.
        print(f"\a🔔 {side} target out of envelope (intensity {intensity:.2f})")


class NullFeedback(EnvelopeFeedback):
    """No-op backend — disables operator signalling entirely."""

    def notify(self, side: str, status: OOEStatus, t: float) -> None:
        pass

    def reset(self) -> None:
        pass


class QuestHapticFeedback(RateLimitedFeedback):
    """Headset haptic backend — a documented stub (see the module docstring).

    Holds the adb device handle the cue would be broadcast on, but does not
    yet send anything: the APK-side ``BroadcastReceiver`` and the shell worker
    are future work. Constructing it is fine; emitting raises so a caller that
    wires it up before the transport exists fails loudly rather than silently
    dropping cues.
    """

    def __init__(
        self,
        device: Any,
        repeat_period_s: float = FEEDBACK_REPEAT_PERIOD_S,
        soft_margin: float = WORKSPACE_SOFT_MARGIN,
    ) -> None:
        super().__init__(repeat_period_s=repeat_period_s, soft_margin=soft_margin)
        # ppadb device handle the HAPTIC broadcast will be sent on (see the
        # transport spec in the module docstring). Unused until implemented.
        self._device = device

    def _emit(self, side: str, intensity: float, t: float) -> None:
        raise NotImplementedError(
            "Quest haptic feedback is not implemented yet; see the transport "
            "protocol specified in the envelope_feedback module docstring "
            "(adb 'am broadcast' of com.neuracore.metaquestteleop.HAPTIC)."
        )
