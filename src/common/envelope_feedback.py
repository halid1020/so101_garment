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

import os
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from common.configs import FEEDBACK_REPEAT_PERIOD_S, WORKSPACE_SOFT_MARGIN
from common.workspace_envelope import OOEStatus, smoothstep

# Speaker beep synthesis — cosmetic, no YAML/schema hook.
_BEEP_DURATION_S = 0.12  # single-pulse length; short enough to not mask motion.
_BEEP_SAMPLE_RATE = 44100  # standard CD rate; every CLI player accepts it.
# Two pitches so the operator can tell which arm hit the boundary by ear alone.
_BEEP_HZ_BY_SIDE = {"left": 660.0, "right": 990.0}  # lower = LEFT, higher = RIGHT.


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


def _synthesise_beep_wav(path: str, freq_hz: float) -> None:
    """Write a short 16-bit mono sine beep to ``path`` (once, at construction).

    A brief raised-cosine attack/decay envelope tops and tails the tone so the
    speaker does not click on the hard sample edges.
    """
    n = int(_BEEP_SAMPLE_RATE * _BEEP_DURATION_S)
    t = np.arange(n) / _BEEP_SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * freq_hz * t)
    # 8 ms raised-cosine ramp on each end (a click-free attack/decay envelope).
    ramp_n = max(1, int(_BEEP_SAMPLE_RATE * 0.008))
    env = np.ones(n)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp_n)))
    env[:ramp_n] = ramp
    env[-ramp_n:] = ramp[::-1]
    samples = (0.6 * tone * env * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_BEEP_SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


# CLI players tried in order: PulseAudio/PipeWire, then ALSA, then ffplay.
# Each entry maps a probe binary to the argv template that plays one WAV path.
_BEEP_PLAYERS: tuple[tuple[str, Callable[[str, str], list[str]]], ...] = (
    ("paplay", lambda exe, wav: [exe, wav]),
    ("aplay", lambda exe, wav: [exe, "-q", wav]),
    (
        "ffplay",
        lambda exe, wav: [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", wav],
    ),
)


def _make_subprocess_player(
    exe: str, build_argv: Callable[[str, str], list[str]]
) -> Callable[[str], None]:
    """Return a fire-and-forget player that spawns ``exe`` on a WAV path.

    ``subprocess.Popen`` returns immediately, so this never blocks the IK loop.
    """

    def play(wav_path: str) -> None:
        subprocess.Popen(
            build_argv(exe, wav_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return play


class SpeakerBeepFeedback(RateLimitedFeedback):
    """Play an audible beep through the control laptop's speaker on OOE edges.

    The terminal bell (:class:`TerminalBellFeedback`) is often muted or routed
    away from the operator; this backend adds a real tone on the default audio
    device. Two pitches distinguish the arms (see ``_BEEP_HZ_BY_SIDE``). The
    beep WAVs are synthesised once at construction; playback is fire-and-forget
    ``subprocess.Popen`` so it never blocks the ~100 Hz IK thread.

    If no CLI player is found, a single warning prints at construction and the
    backend degrades to a no-op (the terminal bell still covers cueing). A
    ``player`` callable can be injected to bypass real subprocess playback (the
    unit tests use this).
    """

    def __init__(
        self,
        player: Callable[[str], None] | None = None,
        repeat_period_s: float = FEEDBACK_REPEAT_PERIOD_S,
        soft_margin: float = WORKSPACE_SOFT_MARGIN,
    ) -> None:
        super().__init__(repeat_period_s=repeat_period_s, soft_margin=soft_margin)
        # Detect the audio player once (unless one is injected for tests).
        self._play = player if player is not None else self._resolve_subprocess_player()
        # Synthesise a beep WAV per side up front; skip if there is no player.
        self._wav_paths: dict[str, str] = {}
        if self._play is None:
            return
        tmp_dir = tempfile.mkdtemp(prefix="so101_beep_")
        for side, freq in _BEEP_HZ_BY_SIDE.items():
            path = os.path.join(tmp_dir, f"beep_{side}.wav")
            _synthesise_beep_wav(path, freq)
            self._wav_paths[side] = path

    def _resolve_subprocess_player(self) -> Callable[[str], None] | None:
        """Pick the first available CLI player; None (with a warning) if none."""
        for probe, build_argv in _BEEP_PLAYERS:
            exe = shutil.which(probe)
            if exe is None:
                continue
            print(f"🔊 Speaker OOE cue using '{probe}'")
            return _make_subprocess_player(exe, build_argv)
        print(
            "⚠️  No audio player found (paplay/aplay/ffplay); speaker OOE cue "
            "disabled — the terminal bell still cues out-of-envelope targets."
        )
        return None

    def _emit(self, side: str, intensity: float, t: float) -> None:
        if self._play is None or intensity <= 0.0:
            # Re-entry (intensity 0) or no player: nothing to play.
            return
        default = next(iter(self._wav_paths.values()), "")
        self._play(self._wav_paths.get(side, default))


class CompositeFeedback(EnvelopeFeedback):
    """Fan ``notify``/``reset`` out to several feedback backends at once.

    Used to run the terminal bell and the speaker beep together (the ``audio``
    CLI choice) without either backend knowing about the other.
    """

    def __init__(self, backends: Iterable[EnvelopeFeedback]) -> None:
        self._backends = list(backends)

    def notify(self, side: str, status: OOEStatus, t: float) -> None:
        for backend in self._backends:
            backend.notify(side, status, t)

    def reset(self) -> None:
        for backend in self._backends:
            backend.reset()


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
