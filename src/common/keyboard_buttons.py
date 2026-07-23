"""Raw-terminal keyboard listener mapping single keys to callbacks.

Replaces the Meta Quest buttons in leader-arm teleoperation
(``tool/meta_quest_teleopration.py --input leader``): the tool already runs
in a terminal, so a cbreak-mode stdin reader needs no extra dependency and
— unlike pynput — works on Wayland and over SSH.

Dispatch model matches the Quest reader: callbacks run on the listener
thread with no except clause here, so callers must wrap handlers in a
crash-proof wrapper (the tool's ``_safe_button``). ``tty.setcbreak`` keeps
ISIG, so Ctrl+C still delivers KeyboardInterrupt to the main thread.
``stop()`` restores the saved terminal attributes and is safe to call from
a ``finally`` block even if ``start()`` never ran.
"""

import os
import select
import sys
import termios
import threading
import tty
from typing import Callable


class KeyboardButtons:
    """cbreak stdin reader thread; maps single keys to callbacks."""

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_attrs: list | None = None

    def on(self, key: str, callback: Callable[[], None]) -> None:
        """Register a callback for a single character (case-insensitive)."""
        if len(key) != 1:
            raise ValueError("key must be a single character")
        self._callbacks[key.lower()] = callback

    def _dispatch(self, ch: str) -> None:
        """Fire the callback bound to ``ch`` (lowercased); unknown = no-op."""
        cb = self._callbacks.get(ch.lower())
        if cb is not None:
            cb()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.2)
            if readable:
                ch = os.read(self._fd, 1).decode(errors="ignore")
                if ch:
                    self._dispatch(ch)

    def start(self) -> None:
        """Switch the terminal to cbreak mode and start the reader thread."""
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader and restore the terminal (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old_attrs is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None
