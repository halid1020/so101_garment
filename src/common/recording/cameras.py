"""Generic UVC (OpenCV) camera capture thread for data collection.

Generalised from the retired single-camera thread (git 185df55^:
src/common/threads/camera_usb.py): one :class:`CameraCapture` per stream, each
opening its own ``/dev/videoN`` device, forcing the configured resolution
(warn-once resize if the device negotiates a different size), converting BGR to
RGB, optionally rotating 180 deg, and publishing into the shared
``DualDataManager`` via ``set_rgb_image(rgb, name)``. On device loss the loop
retries the open every 2 s so a bumped USB cable does not kill the session.

Freshness is a design constraint, not an accident: every stream is opened with a
compressed pixel format (``fourcc``, MJPG by default — several uncompressed
640x480@30 streams do not fit through the shared USB controllers) and with the
shortest driver queue, and the loop is paced by the blocking read alone so no
queue of stale frames can build up. See :meth:`_configure`.

The recorder polls :meth:`seconds_since_last_frame` to detect stale streams.
"""

from __future__ import annotations

import threading
import time
import traceback

import cv2  # type: ignore[import]
import numpy as np

from common.camera_controls import apply_controls
from common.data_manager_dual import DualDataManager

# How long to wait between FAILED attempts to reopen a lost device. A device that
# drops off the USB bus is usually back within a few hundred milliseconds, and
# every millisecond spent waiting is a millisecond of the episode that is not
# being recorded, so the first attempt is immediate and this only paces the
# retries after that.
_REOPEN_INTERVAL_S = 0.5

# Minimum time one capture iteration may take. A blocking V4L2 read paces the
# loop by itself; this only stops a device that returns frames instantly from
# spinning a core.
_BUSY_SPIN_S = 0.001

# V4L2 buffers per stream. The smallest value that still sustains the device's
# full frame rate (1 halves it — see _configure).
_CAPTURE_BUFFERS = 2


class CameraCapture:
    """Capture thread for a single named UVC camera stream."""

    def __init__(
        self,
        name: str,
        device: "int | str",
        width: int,
        height: int,
        fps: int,
        rotate180: bool,
        fourcc: str = "",
        controls: "dict[str, float | None] | None" = None,
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate180 = rotate180
        self.fourcc = fourcc
        # Per-camera image controls; None values are left to the camera. See
        # common.camera_controls -- exposure among them caps frame rate.
        self.controls = dict(controls or {})

        # How many times this device stopped delivering and had to be reopened.
        # Read at the end of the session by common.recording.fault_report, which
        # weighs it against the other devices' counts: several failures on one hub
        # mean the hub, one device's failures mean that device.
        self.disconnects = 0

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame_mono: float | None = None
        self._warned_resize = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure(self, cap: "cv2.VideoCapture") -> None:
        # Pixel format first: switching FOURCC can reset the mode.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Short driver queue, to bound how stale a delivered frame can be if this
        # thread is momentarily late. Two is deliberate and MEASURED: a single
        # buffer starves the driver (it has nowhere to put the next frame while
        # we hold the only one, so it drops every other frame and the stream
        # halves to ~15 fps), while 2 and above all sustain the device's full
        # rate. Do not "optimise" this to 1.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, _CAPTURE_BUFFERS)
        # Image controls last, since some cameras reset them when the format or
        # size changes. Exposure among them caps the frame rate, so this is not
        # only about how the picture looks -- see camera_controls.
        apply_controls(cap, self.controls)

    def open(self) -> bool:
        """Attempt to open the device once. Returns whether it opened.

        Used at startup for a fail-fast check BEFORE the dataset is created.
        """
        # Pin the V4L2 backend: the default fallback chain can silently
        # open a DIFFERENT physical camera when given a metadata node or a
        # /dev/v4l/by-path alias (verified with the tactile cameras).
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
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
                # Stamp the capture instant immediately, before colour convert /
                # resize, so downstream alignment reflects when the sensor was
                # actually read rather than when the processed frame is published.
                t_capture = time.monotonic()
                if not ret or frame is None:
                    self.disconnects += 1
                    print(
                        f"⚠️  camera '{self.name}' read failed "
                        f"({self.disconnects} so far); reopening"
                    )
                    self._cap.release()
                    self._cap = None
                    # Zero, not now: the first attempt has to happen on the very
                    # next iteration. Waiting a full interval before even trying
                    # once made the recorded gap as long as the backoff, which is
                    # the wrong way round -- the backoff exists to stop a dead
                    # device from being hammered, not to delay a live one.
                    last_reopen = 0.0
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._force_size(rgb)
                if self.rotate180:
                    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
                data_manager.set_rgb_image(rgb, self.name, t_capture=t_capture)
                with self._lock:
                    self._last_frame_mono = t_capture

                # No fps sleep here: the V4L2 ``read`` above already blocks until
                # the next frame, so it paces this loop at the device rate. An
                # extra sleep on top would hold the loop below that rate and let
                # the driver queue build up — the very drift this avoids. Only
                # guard against a device that returns instantly (busy-spin).
                if time.time() - iteration_start < _BUSY_SPIN_S:
                    time.sleep(_BUSY_SPIN_S)
        except Exception as e:  # pragma: no cover - hardware failure path
            print(f"❌ camera '{self.name}' thread error: {e}")
            traceback.print_exc()
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            print(f"📷 camera '{self.name}' thread stopped")
