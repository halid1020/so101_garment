"""Generic UVC (OpenCV) camera capture thread for data collection.

Generalised from the retired single-camera thread (git 185df55^:
src/common/threads/camera_usb.py): one :class:`CameraCapture` per stream, each
opening its own ``/dev/videoN`` device, forcing the configured resolution
(warn-once resize if the device negotiates a different size), converting BGR to
RGB, optionally rotating 180 deg, and publishing into the shared
``DualDataManager`` via ``set_rgb_image(rgb, name)``. On device loss the loop
retries the open every 2 s so a bumped USB cable does not kill the session.

The recorder polls :meth:`seconds_since_last_frame` to detect stale streams.
"""

from __future__ import annotations

import threading
import time
import traceback

import cv2  # type: ignore[import]
import numpy as np

from common.data_manager_dual import DualDataManager

# How often (seconds) to retry opening a device that failed or was lost.
_REOPEN_INTERVAL_S = 2.0


class CameraCapture:
    """Capture thread for a single named UVC camera stream."""

    def __init__(
        self,
        name: str,
        device: int,
        width: int,
        height: int,
        fps: int,
        rotate180: bool,
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate180 = rotate180

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame_mono: float | None = None
        self._warned_resize = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure(self, cap: "cv2.VideoCapture") -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

    def open(self) -> bool:
        """Attempt to open the device once. Returns whether it opened.

        Used at startup for a fail-fast check BEFORE the dataset is created.
        """
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap.release()
            return False
        self._configure(cap)
        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"  📷 camera '{self.name}' opened on device {self.device}: "
            f"{actual_w}x{actual_h} (forced to {self.width}x{self.height})"
        )
        return True

    def start(self, data_manager: DualDataManager) -> None:
        """Spawn the capture thread (opens the device if not already open)."""
        self._thread = threading.Thread(
            target=self._loop, args=(data_manager,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Freshness ────────────────────────────────────────────────────────────

    def seconds_since_last_frame(self, now_mono: float) -> float | None:
        """Age (seconds) of the most recent published frame, or None if never."""
        with self._lock:
            if self._last_frame_mono is None:
                return None
            return now_mono - self._last_frame_mono

    # ── Capture loop ─────────────────────────────────────────────────────────

    def _force_size(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            if not self._warned_resize:
                print(
                    f"⚠️  camera '{self.name}' delivered "
                    f"{rgb.shape[1]}x{rgb.shape[0]}, resizing to "
                    f"{self.width}x{self.height} (warned once)"
                )
                self._warned_resize = True
            rgb = cv2.resize(rgb, (self.width, self.height))
        return rgb

    def _loop(self, data_manager: DualDataManager) -> None:
        dt = 1.0 / float(self.fps)
        last_reopen = 0.0
        try:
            while not self._stop.is_set() and not data_manager.is_shutdown_requested():
                iteration_start = time.time()

                if self._cap is None:
                    now = time.time()
                    if now - last_reopen >= _REOPEN_INTERVAL_S:
                        last_reopen = now
                        if self.open():
                            print(f"  ✓ camera '{self.name}' reopened")
                    time.sleep(dt)
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    print(
                        f"⚠️  camera '{self.name}' read failed; "
                        "will retry-reopen every 2 s"
                    )
                    self._cap.release()
                    self._cap = None
                    last_reopen = time.time()
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._force_size(rgb)
                if self.rotate180:
                    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
                data_manager.set_rgb_image(rgb, self.name)
                with self._lock:
                    self._last_frame_mono = time.monotonic()

                sleep_time = dt - (time.time() - iteration_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:  # pragma: no cover - hardware failure path
            print(f"❌ camera '{self.name}' thread error: {e}")
            traceback.print_exc()
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            print(f"📷 camera '{self.name}' thread stopped")
