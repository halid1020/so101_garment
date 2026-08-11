"""Composite per-episode video frames for policy evaluation.

Each frame stacks two blocks:

* a row of the four camera panels a human wants while judging a rollout —
  the three policy cameras (``scene``, ``wrist_camera_left``,
  ``wrist_camera_right``, the
  exact views the policy consumes) plus a free third-person ``overview``
  (``TwinSim.render_overview``); and
* a matplotlib panel of the joint signals: the measured ``observation.state``
  (solid) against the commanded ``action`` (dashed), per arm in degrees with
  the two gripper channels below, scrolling over a trailing time window.

The 12-channel state/action layout is the recorder's
(``common.recording.features``): per side the five body joints then the
gripper, left arm first. Kept import-light (matplotlib Agg + OpenCV) so it
unit-tests without a sim or a GPU.

The signal panel reuses persistent line artists (``set_data`` each tick, no
per-frame ``clear``/``tight_layout``) and rasterises at half resolution, then
upscales — a rollout is ~900 ticks and an eval sweep is dozens of episodes,
so a full figure rebuild every frame would dominate the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2  # type: ignore[import]
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from common.recording.features import BODY_JOINTS

# Camera panels, left to right. The first three are policy inputs; "overview"
# is the free third-person view rendered separately by the caller.
CAMERA_ORDER = ("scene", "wrist_camera_left", "wrist_camera_right", "overview")

# 12-D state/action channel indices for each side (5 body joints + gripper),
# matching common.recording.features.STATE_NAMES (side outer, gripper last).
_BODY_IDX = {"left": [0, 1, 2, 3, 4], "right": [6, 7, 8, 9, 10]}
_GRIP_IDX = {"left": 5, "right": 11}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Rasterise the plot at this fraction of its final pixel size, then upscale.
_PLOT_SCALE = 0.5


class EvalVideoComposer:
    """Accumulate composite RGB frames for one episode, then write an mp4."""

    def __init__(
        self,
        fps: int,
        tile_wh: tuple[int, int] = (320, 240),
        window_s: float = 6.0,
        plot_h: int = 368,  # tile_h + plot_h stays a multiple of 16 (mp4 codec)
        plot_every: int = 3,  # redraw the signal panel every N ticks (cache between)
    ) -> None:
        self.fps = fps
        self.tile_w, self.tile_h = tile_wh
        self.window_s = window_s
        self.plot_h = plot_h
        self.plot_every = max(1, plot_every)
        self.row_w = self.tile_w * len(CAMERA_ORDER)
        self._last_panel: np.ndarray | None = None

        self._t: list[float] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []

        # One reusable Agg figure at half the final size (upscaled on capture).
        self.fig = Figure(
            figsize=(self.row_w / 100, self.plot_h / 100), dpi=100 * _PLOT_SCALE
        )
        self.canvas = FigureCanvasAgg(self.fig)
        axes = self.fig.subplots(2, 2)
        self._axes = axes
        colors = [f"C{i}" for i in range(len(BODY_JOINTS))]

        # Persistent line artists: per side, 5 measured + 5 target joint lines
        # and a measured/target gripper pair. Updated via set_data each tick.
        self._joint_lines: dict[str, list[tuple]] = {}
        self._grip_lines: dict[str, tuple] = {}
        for col, side in enumerate(("left", "right")):
            ax_j = axes[0][col]
            pairs: list[tuple] = []
            for j in range(len(BODY_JOINTS)):
                (meas,) = ax_j.plot([], [], "-", color=colors[j], lw=1.0)
                (tgt,) = ax_j.plot([], [], "--", color=colors[j], lw=1.0)
                pairs.append((meas, tgt))
            self._joint_lines[side] = pairs
            ax_j.set_title(f"{side} arm joints (deg)", fontsize=8)
            ax_j.tick_params(labelsize=6)

            ax_g = axes[1][col]
            (gm,) = ax_g.plot([], [], "-", color="k", lw=1.0, label="measured")
            (gt,) = ax_g.plot([], [], "--", color="k", lw=1.0, label="target")
            self._grip_lines[side] = (gm, gt)
            ax_g.set_title(f"{side} gripper (open frac)", fontsize=8)
            ax_g.set_xlabel("time (s)", fontsize=7)
            ax_g.tick_params(labelsize=6)
            if col == 0:
                ax_g.legend(fontsize=6, loc="upper right")
        self.fig.tight_layout(pad=0.5)

    # ------------------------------------------------------------------
    def add(
        self,
        cameras: dict[str, np.ndarray],
        overview: np.ndarray,
        state12: Sequence[float],
        action12: Sequence[float],
    ) -> None:
        """Append one composite frame for the current tick."""
        self._t.append(len(self._t) / self.fps)
        self._state.append(np.asarray(state12, dtype=float))
        self._action.append(np.asarray(action12, dtype=float))
        cam_row = self._camera_row({**cameras, "overview": overview})
        # Cameras refresh every tick (cheap); the matplotlib panel — the
        # expensive part — is redrawn every plot_every ticks and reused between.
        if self._last_panel is None or (len(self._t) - 1) % self.plot_every == 0:
            self._last_panel = self._signal_panel()
        self._frames.append(np.vstack([cam_row, self._last_panel]))

    # ------------------------------------------------------------------
    def _camera_row(self, images: dict[str, np.ndarray]) -> np.ndarray:
        tiles = []
        for name in CAMERA_ORDER:
            img = images.get(name)
            if img is None:
                tile = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(np.ascontiguousarray(img), (self.tile_w, self.tile_h))
            cv2.putText(tile, name, (6, 22), _FONT, 0.6, (0, 255, 0), 2)
            tiles.append(tile)
        return np.hstack(tiles)

    def _signal_panel(self) -> np.ndarray:
        t = np.asarray(self._t)
        state = np.asarray(self._state)
        action = np.asarray(self._action)
        lo = max(0.0, t[-1] - self.window_s)
        mask = t >= lo
        tw = t[mask]
        xlim = (lo, max(lo + self.window_s, t[-1]))

        for col, side in enumerate(("left", "right")):
            ax_j = self._axes[0][col]
            for j, (meas, tgt) in enumerate(self._joint_lines[side]):
                idx = _BODY_IDX[side][j]
                meas.set_data(tw, state[mask, idx])
                tgt.set_data(tw, action[mask, idx])
            ax_j.set_xlim(*xlim)
            ax_j.relim()
            ax_j.autoscale_view(scalex=False, scaley=True)

            ax_g = self._axes[1][col]
            gm, gt = self._grip_lines[side]
            gidx = _GRIP_IDX[side]
            gm.set_data(tw, state[mask, gidx])
            gt.set_data(tw, action[mask, gidx])
            ax_g.set_xlim(*xlim)
            ax_g.relim()
            ax_g.autoscale_view(scalex=False, scaley=True)

        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())[..., :3]
        if buf.shape[:2] != (self.plot_h, self.row_w):
            buf = cv2.resize(buf, (self.row_w, self.plot_h))
        return np.ascontiguousarray(buf)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import imageio.v2 as imageio

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(p, self._frames, fps=self.fps)

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)
