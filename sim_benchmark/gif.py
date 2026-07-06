"""Offscreen GIF recording of benchmark episodes (headless, via EGL)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class GifRecorder:
    """Renders sim frames at a fixed FPS and saves them as an animated GIF."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 480,
        height: int = 360,
        fps: float = 10.0,
        label: str = "",
    ) -> None:
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = [0.25, 0.0, 0.06]
        self.camera.distance = 0.95
        self.camera.azimuth = 160
        self.camera.elevation = -30
        self.fps = fps
        self.label = label
        self.frames: list[Image.Image] = []
        self._next_t = 0.0

    def maybe_capture(self, data: mujoco.MjData, t: float) -> None:
        """Capture a frame if sim time t has reached the next FPS slot."""
        if t + 1e-9 < self._next_t:
            return
        self._next_t += 1.0 / self.fps
        self.renderer.update_scene(data, self.camera)
        frame = Image.fromarray(np.asarray(self.renderer.render()))
        if self.label:
            draw = ImageDraw.Draw(frame)
            draw.text((8, 6), self.label, fill=(255, 255, 255))
        self.frames.append(frame)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
        )

    def close(self) -> None:
        self.renderer.close()
