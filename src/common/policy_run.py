"""What a rollout is doing, and the one switch that gates its motion.

``tool/run_policy_real.py`` drives the arms from a policy's action chunks. Two
parties need to agree about that run: the control loop, which writes goals at
the control rate, and a watcher (``common.web.policy_view``), which reads what
is happening and may ask for less motion. This module is what they share, and
it is deliberately pure -- no camera, no bus, no HTTP -- so the arithmetic can
be tested without either of them.

THE THROTTLE. A rollout is in one of three modes:

  * ``run``  -- serve an action every tick, which is what an evaluation does.
  * ``hold`` -- serve nothing; the servos keep their last commanded goal.
  * ``step`` -- serve exactly one chunk, then fall back to ``hold``.

``step`` is how a failure is examined: one plan is executed, the arms stop, and
the plan that produced the motion is still on the screen beside it.

Leaving ``hold`` DROPS whatever is queued (``queue_stale``), because a chunk was
planned from an observation taken before the pause -- possibly minutes before --
and executing it afterwards would drive the arms from a stale picture of the
world. The cost is one round trip on resume; the alternative is a surprise.

The modes only ever gate motion the operator already authorised at the terminal,
where torque is enabled after a confirmation. Nothing here can enable torque.
"""

from __future__ import annotations

import math
import threading
import time

#: What a rollout may be doing. ``stop`` is a request, not a mode: it ends the
#: run through the same path Ctrl+C takes, which disables torque.
MODES = ("run", "step", "hold")
REQUESTS = MODES + ("stop",)


def gate(mode: str, depth: int, budget: int) -> "tuple[str, int]":
    """What this tick may do, and what is left of a step's budget. Pure.

    Returns ``serve`` (take an action and write it), ``hold`` (write nothing) or
    ``wait`` (write nothing, but a chunk has been asked for and is on its way).
    ``wait`` differs from ``hold`` only in what an operator is told.

    In ``step`` the budget is set from the queue the moment a chunk lands, so
    one press executes one chunk however long that policy's chunk happens to be.
    """
    if mode == "run":
        return ("serve" if depth > 0 else "wait"), 0
    if mode == "hold":
        return "hold", 0
    if budget > 0:
        return ("serve", budget - 1) if depth > 0 else ("wait", budget)
    if depth > 0:
        return "serve", depth - 1
    return "wait", 0


def prefetch_threshold(
    static: int,
    explicit: "int | None",
    round_trip_s: float,
    hz: float,
    chunk: int,
    margin: float = 1.5,
) -> int:
    """How many actions may remain when the next chunk is requested. Pure.

    The queue drains at the control rate while a request is in flight, so the
    threshold has to cover *round trip x hz* or the arms run dry every chunk --
    which is exactly what a 32-action diffusion chunk does at 30 Hz against half
    a second of inference: a third of a chunk is 0.33 s of runway against a
    0.6 s round trip. The margin is for the jitter around that measurement.

    An operator who passes ``--prefetch`` means it, so ``explicit`` wins. The
    threshold never reaches the chunk length, or a request would be in flight
    permanently.
    """
    if explicit is not None:
        return max(0, int(explicit))
    if round_trip_s <= 0.0 or hz <= 0.0:
        return static
    need = int(math.ceil(round_trip_s * hz * margin))
    return max(static, min(need, max(1, chunk - 1)))


class RunControl:
    """The mode, the step budget and the latest snapshot, behind one lock.

    The control loop calls :meth:`decide` once per tick and :meth:`publish` with
    what it did; the view calls :meth:`request` and :meth:`snapshot`. Neither
    blocks the other for longer than a dictionary swap.
    """

    def __init__(self, mode: str = "run") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self._lock = threading.Lock()
        self._mode = mode
        self._budget = 0
        self._stop = False
        self._queue_stale = False
        self._snapshot: "dict" = {"mode": mode, "started": time.time()}

    # -- the watcher's side --------------------------------------------
    def request(self, mode: str) -> str:
        """Ask for a mode (or ``stop``). Returns the mode now in force."""
        if mode not in REQUESTS:
            raise ValueError(
                f"unknown mode: {mode} (want one of {', '.join(REQUESTS)})"
            )
        with self._lock:
            if mode == "stop":
                self._stop = True
                return self._mode
            if mode != self._mode:
                # Anything queued was planned before this decision was taken.
                self._queue_stale = True
                self._budget = 0
                self._mode = mode
            return self._mode

    def snapshot(self) -> "dict":
        with self._lock:
            return dict(self._snapshot)

    # -- the control loop's side ---------------------------------------
    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stop

    def queue_stale(self) -> bool:
        """True once, when the queue must be dropped before the next tick."""
        with self._lock:
            stale, self._queue_stale = self._queue_stale, False
            return stale

    def decide(self, depth: int) -> str:
        """``serve`` / ``hold`` / ``wait`` for this tick, advancing a step."""
        with self._lock:
            what, budget = gate(self._mode, depth, self._budget)
            self._budget = budget
            if self._mode == "step" and what == "serve" and budget == 0:
                self._mode = "hold"  # that was the last action of the chunk
            return what

    def publish(self, **fields) -> None:
        with self._lock:
            self._snapshot.update(fields, mode=self._mode, stopping=self._stop)
