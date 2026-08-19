#!/usr/bin/env python3
"""Live per-camera image tuning, with the frame-rate cost shown while you tune.

Camera settings on this rig cannot be judged from a number. Whether a wrist view
is too dark or blown out depends on the lamp above the bench and on what the arm
is shadowing, and the two wrist cameras are lit differently from each other, so
the values that suit them differ. This tool shows the streams side by side, lets
each control be nudged with a key, and writes the result back into
``src/conf/recording.yaml`` so the next collection uses it.

The reason it reports frames per second next to each tile is that one of these
controls is not cosmetic. Exposure bounds frame rate --- a camera cannot deliver
frames faster than it exposes them --- so lengthening it to brighten a dark image
silently slows the stream, which is how the wrist cameras came to be recording at
18 fps instead of 30. Measured on those cameras, every exposure up to about 300
sustains the full rate and 500 costs a third of it. Below that ceiling brightness
is free, so the tile's rate readout tells you immediately whether the adjustment
you just made cost anything.

    venv/bin/python tool/tune_cameras.py                 # every enabled camera
    venv/bin/python tool/tune_cameras.py --camera wrist_camera_left

Keys: 1-9 select a camera; e/E exposure; g/G gain; b/B brightness; c/C contrast;
t/T saturation (lower/raise); a returns the selected camera to automatic
exposure; r reloads the saved values; s writes them to recording.yaml; q quits.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np

from common.camera_controls import (
    CONTROL_NAMES,
    EXPOSURE_AUTO,
    apply_controls,
    read_controls,
)
from common.sensor_view import quiet_qt_warnings

_root = Path(__file__).resolve().parent.parent
_RECORDING_YAML = _root / "src" / "conf" / "recording.yaml"

# How much one key press moves each control. Exposure moves in larger steps
# because its useful range is wide and its effect on rate is what matters.
_STEPS: "dict[str, int]" = {
    "exposure": 25,
    "gain": 5,
    "brightness": 5,
    "contrast": 5,
    "saturation": 5,
}
# Lower/raise keys per control.
_KEYS: "dict[str, tuple[str, str]]" = {
    "exposure": ("e", "E"),
    "gain": ("g", "G"),
    "brightness": ("b", "B"),
    "contrast": ("c", "C"),
    "saturation": ("t", "T"),
}
_WINDOW = "camera tuning"


class TunedCamera:
    """One open camera plus the control values being tried on it."""

    def __init__(self, name: str, device, width: int, height: int, fourcc: str):
        self.name = name
        self.device = device
        self.width, self.height = width, height
        self.controls: dict = {}
        self.frame = None
        self._stamps: list[float] = []
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if self.cap.isOpened():
            if fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    @property
    def opened(self) -> bool:
        return bool(self.cap.isOpened())

    def apply(self) -> None:
        apply_controls(self.cap, self.controls)

    def auto_exposure(self) -> None:
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_AUTO)
        self.controls["exposure"] = None

    def read(self) -> None:
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
            self._stamps.append(time.monotonic())
            del self._stamps[:-60]

    def fps(self) -> float:
        if len(self._stamps) < 3:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def exposure_meter(self) -> "tuple[float, float]":
        """Mean brightness and the percentage of pixels blown out to white.

        The pair is what the eye cannot judge reliably on a small tile: an image
        can look acceptable while its highlights are already clipped, and a
        clipped highlight is unrecoverable detail rather than a matter of taste.
        """
        if self.frame is None:
            return 0.0, 0.0
        grey = self.frame.mean(axis=2)
        return float(grey.mean()), float((grey > 250).mean() * 100.0)

    def release(self) -> None:
        self.cap.release()


def parse_yaml_controls(text: str, camera: str) -> "dict[str, float | None]":
    """The control values recorded for ``camera`` in a recording.yaml. Pure."""
    block = _camera_block(text, camera)
    out: dict = {}
    for name in CONTROL_NAMES:
        m = re.search(rf"^\s+{name}:\s*(\S+)", block, re.M)
        if m:
            raw = m.group(1)
            out[name] = None if raw in ("null", "~", "None") else float(raw)
    return out


def _camera_block(text: str, camera: str) -> str:
    """The lines belonging to one camera's entry. Pure."""
    m = re.search(rf"^  {re.escape(camera)}:\s*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end() :]
    end = re.search(r"^  \S", rest, re.M)
    return rest[: end.start()] if end else rest


def write_yaml_controls(text: str, camera: str, controls: "dict") -> str:
    """Return ``text`` with ``camera``'s controls set to ``controls``. Pure.

    Edits the individual lines rather than re-serialising the document, because
    recording.yaml carries the explanation of every setting in its comments and a
    YAML round-trip through the parser available here would silently delete all
    of them. A control not already present is appended to the camera's block.
    """
    block = _camera_block(text, camera)
    if not block:
        return text
    updated = block
    for name in CONTROL_NAMES:
        value = controls.get(name)
        shown = "null" if value is None else str(int(round(float(value))))
        pattern = rf"^(\s+){name}:\s*\S+(.*)$"
        m = re.search(pattern, updated, re.M)
        if m:
            updated = re.sub(
                pattern,
                lambda mm: f"{mm.group(1)}{name}: {shown}{mm.group(2)}",
                updated,
                count=1,
                flags=re.M,
            )
        elif value is not None:
            indent = re.search(r"^(\s+)\S", updated, re.M)
            pad = indent.group(1) if indent else "    "
            updated = updated.rstrip("\n") + f"\n{pad}{name}: {shown}\n"
    return text.replace(block, updated, 1)


def _tile(cam: TunedCamera, selected: bool) -> np.ndarray:
    """One camera's frame with its live readout drawn on it."""
    if cam.frame is None:
        img = np.zeros((cam.height, cam.width, 3), np.uint8)
    else:
        img = cam.frame.copy()
    mean, blown = cam.exposure_meter()
    fps = cam.fps()
    # Red once the rate has fallen below the dataset rate: that is the cost the
    # operator most needs to see, and it is invisible in the picture itself.
    rate_colour = (
        (0, 220, 0) if fps >= 25 else (0, 165, 255) if fps >= 20 else (0, 0, 255)
    )
    lines = [
        (f"{'>' if selected else ' '} {cam.name}", (255, 255, 255)),
        (f"{fps:.1f} fps", rate_colour),
        (f"brightness {mean:.0f}  blown {blown:.1f}%", (200, 200, 200)),
    ]
    for name in CONTROL_NAMES:
        value = cam.controls.get(name)
        lines.append(
            (f"{name} {'auto' if value is None else int(value)}", (180, 220, 255))
        )
    y = 18
    for text, colour in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y += 17
    if selected:
        cv2.rectangle(
            img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 200, 255), 3
        )
    return img


def _resolve_cameras(only: "list[str]") -> list:
    """Enabled cameras from recording.yaml, resolved to their assigned devices."""
    from common.config_parser import load_recording_config
    from tool.meta_quest_teleopration import overlay_sensor_map_devices
    from tool.test_sensor_rates import SENSOR_MAP_PATH, load_sensor_map

    cfg = load_recording_config()["cameras"]
    if SENSOR_MAP_PATH.exists():
        cfg = overlay_sensor_map_devices(cfg, load_sensor_map(SENSOR_MAP_PATH))
    out = []
    for name, entry in cfg.items():
        if only and name not in only:
            continue
        if not only and not entry.get("enabled"):
            continue
        cam = TunedCamera(
            name, entry["device"], entry["width"], entry["height"], entry["fourcc"]
        )
        if not cam.opened:
            print(f"⚠️  {name}: could not open {entry['device']} — skipping")
            continue
        cam.controls = {k: entry.get(k) for k in CONTROL_NAMES}
        cam.apply()
        out.append(cam)
    return out


def main() -> int:
    quiet_qt_warnings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Tune only this camera (repeatable); default is every enabled one",
    )
    args = parser.parse_args()

    cams = _resolve_cameras(args.camera)
    if not cams:
        print("❌ no camera could be opened — is a collection session running?")
        return 1
    print(
        f"🎛️  tuning {len(cams)} camera(s). Keys: "
        + ", ".join(f"{lo}/{up} {name}" for name, (lo, up) in _KEYS.items())
        + "; 1-9 select, a auto-exposure, r reload, s save, q quit"
    )

    selected = 0
    while True:
        for cam in cams:
            cam.read()
        tiles = [_tile(c, i == selected) for i, c in enumerate(cams)]
        from tool.test_sensor_rates import _fit_to_screen, grid_tiles

        cv2.imshow(_WINDOW, _fit_to_screen(grid_tiles(tiles), 1600, 900))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(cams):
                selected = index
        cam = cams[selected]
        for name, (lower, upper) in _KEYS.items():
            if key in (ord(lower), ord(upper)):
                step = _STEPS[name] * (1 if key == ord(upper) else -1)
                base = cam.controls.get(name)
                if base is None:
                    base = read_controls(cam.cap).get(name, 0.0)
                cam.controls[name] = max(0.0, float(base) + step)
                cam.apply()
        if key == ord("a"):
            cam.auto_exposure()
        if key == ord("r"):
            text = _RECORDING_YAML.read_text()
            for c in cams:
                c.controls = {**c.controls, **parse_yaml_controls(text, c.name)}
                c.apply()
            print("↺ reloaded from recording.yaml")
        if key == ord("s"):
            text = _RECORDING_YAML.read_text()
            for c in cams:
                text = write_yaml_controls(text, c.name, c.controls)
            _RECORDING_YAML.write_text(text)
            print(f"💾 saved to {_RECORDING_YAML}")
            for c in cams:
                shown = {k: c.controls.get(k) for k in CONTROL_NAMES}
                print(f"   {c.name}: {shown}  ({c.fps():.1f} fps)")

    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
