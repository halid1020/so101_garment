"""Resolved digital-twin parameters, derived from ``config.scad``.

World frame convention (shared by the MuJoCo scene, the generated URDF
and the Isaac package):

* ``z = 0`` is the **table top**. The rig stacks on it: nut plate,
  board, adapters — so the arms' ``base_link`` origins sit at
  ``arm_base_height`` (the URDF bakes this in).
* Arms face **+X**; the two bases sit at ``y = +/- arm_spacing/2``;
  the camera tower is centered between them.
* ``config.scad`` values are mm; everything exposed here is meters,
  radians and kilograms unless the name says otherwise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from sim_benchmark.constants import (
    ACTUATOR_KP,
    ACTUATOR_KV,
    JOINT_ARMATURE,
    JOINT_DAMPING,
    JOINT_FRICTIONLOSS,
    NEUTRAL_ARM_ANGLES_DEG,
)
from sim_twin.scad_params import parse_scad_params

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SCAD = REPO_ROOT / "src" / "platform" / "config.scad"
PLATFORM_DIR = CONFIG_SCAD.parent
BUILD_DIR = REPO_ROOT / "build" / "twin"

# Estimated printed-part masses (kg) for the URDF camera links; PLA at
# ~25% infill. Only wrist dynamics care, and only coarsely.
WRIST_MOUNT_MASS = 0.015
C310_CABLE_FUDGE = 1.0  # mass multiplier hook, dims already in config

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def _mm(v: float) -> float:
    return v / 1000.0


@dataclass(frozen=True)
class Pose:
    """Position (m) + fixed-axis rpy (rad), URDF-style."""

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TwinParams:
    scad: dict[str, float] = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | str = CONFIG_SCAD) -> "TwinParams":
        params = cls(scad=parse_scad_params(config_path))
        params.validate()
        return params

    def __getattr__(self, name: str) -> float:
        try:
            return self.scad[name]
        except KeyError:
            raise AttributeError(name) from None

    def validate(self) -> None:
        s = self.scad
        spacing, pitch = s["arm_spacing"], s["grid_pitch"]
        if spacing % (2 * pitch) != 0:
            raise ValueError(
                f"arm_spacing={spacing} must be a multiple of 2*grid_pitch"
                f"={2 * pitch} so both adapters land on grid holes"
            )
        if not s["min_arm_spacing"] <= spacing <= s["max_arm_spacing"]:
            raise ValueError(
                f"arm_spacing={spacing} outside [{s['min_arm_spacing']},"
                f" {s['max_arm_spacing']}]"
            )
        if s["tower_y_offset"] % pitch != 0:
            raise ValueError("tower_y_offset must be a grid_pitch multiple")

    # ------------------------------------------------------------------
    # Vertical stack (printed board sits directly on the table; its
    # captive-nut pockets replaced the old separate nut plate)
    # ------------------------------------------------------------------

    @property
    def board_top_z(self) -> float:
        return _mm(self.board_thickness)

    @property
    def adapter_top_z(self) -> float:
        return self.board_top_z + _mm(self.adapter_thick)

    @property
    def arm_base_height(self) -> float:
        """World z of the arms' base_link origins (base underside sits on
        the adapter; the URDF origin is base_bottom_z above it)."""
        return self.adapter_top_z + _mm(-self.base_bottom_z)

    @property
    def arm_spacing_m(self) -> float:
        return _mm(self.arm_spacing)

    # ------------------------------------------------------------------
    # Horizontal placement (x of world 0 = the base_link origins)
    # ------------------------------------------------------------------

    @property
    def adapter_center_x(self) -> float:
        """Adapter/board-row center: the base-hole trapezoid center is
        base_holes_x ahead of base_link along the arm's +X."""
        return _mm(self.base_holes_x)

    @property
    def board_center(self) -> tuple[float, float]:
        return (self.adapter_center_x, 0.0)

    @property
    def board_size(self) -> tuple[float, float]:
        """(x, y) world extents: board depth runs along X."""
        return (_mm(self.board_depth), _mm(self.board_width))

    @property
    def tower_center(self) -> tuple[float, float]:
        return (self.adapter_center_x, _mm(self.tower_y_offset))

    @property
    def tower_yaw_rad(self) -> float:
        """Tower rotation on its bolt square (default: one triangle
        corner pointing at the front, +X)."""
        return math.radians(self.tower_yaw_deg)

    @property
    def tower_platform_top_z(self) -> float:
        return self.board_top_z + _mm(
            self.tower_base_thick
            + self.tower_height_total
            + self.tower_spigot_h
            + self.camera_platform_thick
        )

    @property
    def tower_total_height(self) -> float:
        """Tower extent above the board top (for collision proxies)."""
        return self.tower_platform_top_z - self.board_top_z

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    @property
    def fovy_deg(self) -> float:
        """Vertical FOV for a 4:3 sensor from the diagonal FOV."""
        half_diag = math.tan(math.radians(self.cam_dfov_deg) / 2)
        return math.degrees(2 * math.atan(half_diag * 3 / 5))

    def _lens_in_part(self, fwd_mm: float, floor_z_mm: float, tilt_deg: float):
        """C310 optical-center pose in a mount part's frame.

        Mirrors cam_tray_lib.scad: the tray floor-top origin sits at
        (0, fwd, floor_z), tilted by ``rotate([-tilt, 0, 0])``; the lens
        is on the body's front face, half a body height up.
        """
        t = math.radians(tilt_deg)
        lens_tray = (
            _mm(self.cam_lens_x_offset),
            _mm(self.cam_body_d / 2),
            _mm(self.cam_body_h / 2 + self.cam_lens_z_offset),
        )
        # Rx(-t)
        y = lens_tray[1] * math.cos(t) + lens_tray[2] * math.sin(t)
        z = -lens_tray[1] * math.sin(t) + lens_tray[2] * math.cos(t)
        return Pose(
            xyz=(lens_tray[0], _mm(fwd_mm) + y, _mm(floor_z_mm) + z),
            rpy=(-t, 0.0, 0.0),
        )

    @property
    def wrist_mount_pose_in_gripper(self) -> Pose:
        """Mount part frame in gripper_link: origin at the measured screw
        midpoint on the wrist face; part +Y (fingertips) = gripper -Z,
        part +Z (off the face) = gripper +Y  ->  Rx(-pi/2)."""
        return Pose(
            xyz=(
                _mm(self.wrist_iface_x),
                _mm(self.wrist_iface_y),
                _mm(self.wrist_iface_z),
            ),
            rpy=(-math.pi / 2, 0.0, 0.0),
        )

    @property
    def wrist_cam_pose_in_mount(self) -> Pose:
        """C310 body (optical-center origin) in the wrist-mount frame."""
        return self._lens_in_part(
            self.wrist_cam_fwd,
            self.wrist_plate_thick + self.wrist_cam_rise,
            self.wrist_cam_tilt_deg,
        )

    @property
    def tower_cradle_pose_world(self) -> Pose:
        """Cradle frame on the platform top, look direction (+Y) aimed
        along world +X  ->  Rz(-pi/2)."""
        cx, cy = self.tower_center
        return Pose(
            xyz=(cx, cy, self.tower_platform_top_z),
            rpy=(0.0, 0.0, -math.pi / 2),
        )

    @property
    def tower_cam_pose_in_cradle(self) -> Pose:
        return self._lens_in_part(
            0.0,
            self.tower_cradle_thick + self.tower_cam_rise,
            self.tower_cam_tilt_deg,
        )

    # Optical frame relative to the C310 body frame (body: +Y look,
    # +Z up). MuJoCo cameras look along -Z with +Y up -> Rx(+pi/2).
    OPTICAL_RPY_MUJOCO = (math.pi / 2, 0.0, 0.0)

    @property
    def cam_mass(self) -> float:
        return self.cam_mass_g / 1000.0 * C310_CABLE_FUDGE

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Resolved snapshot consumed by the portable Isaac package."""

        def pose(p: Pose) -> dict:
            return {"xyz": list(p.xyz), "rpy": list(p.rpy)}

        return {
            "source": str(CONFIG_SCAD.relative_to(REPO_ROOT)),
            "world": {
                "arm_spacing_m": self.arm_spacing_m,
                "arm_base_height_m": self.arm_base_height,
                "table_size_m": [
                    _mm(self.table_size_x),
                    _mm(self.table_size_y),
                    _mm(self.table_thick),
                ],
                "table_height_m": _mm(self.table_height),
                "table_x_offset_m": _mm(self.table_x_offset),
                "board_thickness_m": self.board_top_z,
                "tower_yaw_rad": self.tower_yaw_rad,
                "board_top_z": self.board_top_z,
                "board_center_xy": list(self.board_center),
                "board_size_xy": list(self.board_size),
                "adapter_center_x": self.adapter_center_x,
                "adapter_size_m": [
                    _mm(self.adapter_w),
                    _mm(self.adapter_d),
                    _mm(self.adapter_thick),
                ],
                "tower_center_xy": list(self.tower_center),
                "tower_platform_top_z": self.tower_platform_top_z,
                "tower_base_width_m": _mm(self.tower_base_width),
                "tower_top_width_m": _mm(self.tower_top_width),
                "tower_base_plate_m": _mm(self.tower_base_plate),
            },
            "cameras": {
                "width": CAMERA_WIDTH,
                "height": CAMERA_HEIGHT,
                "fovy_deg": self.fovy_deg,
                "dfov_deg": self.cam_dfov_deg,
                "wrist_mount_in_gripper": pose(self.wrist_mount_pose_in_gripper),
                "wrist_cam_in_mount": pose(self.wrist_cam_pose_in_mount),
                "tower_cradle_world": pose(self.tower_cradle_pose_world),
                "tower_cam_in_cradle": pose(self.tower_cam_pose_in_cradle),
                "cam_mass_kg": self.cam_mass,
            },
            "control": {
                "actuator_kp": ACTUATOR_KP,
                "actuator_kv": ACTUATOR_KV,
                "joint_damping": JOINT_DAMPING,
                "joint_armature": JOINT_ARMATURE,
                "joint_frictionloss": JOINT_FRICTIONLOSS,
                "neutral_arm_angles_deg": list(NEUTRAL_ARM_ANGLES_DEG),
            },
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
