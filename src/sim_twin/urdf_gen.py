"""Generate the twin's dual-arm URDF from the checked-in description.

``src/so101_dual_description/robot.urdf`` stays untouched (the existing
IK/benchmark stack owns it). This module rewrites two things into a copy
under ``build/twin/``:

* the ``{left,right}_base_joint`` origins — real arm spacing and the
  mounting-stack height from ``config.scad`` are baked in, so plain FK
  in any consumer (MuJoCo, Pinocchio, Isaac) works in world coordinates;
* three fixed links per arm for the wrist camera: printed mount (visual
  mesh + box collision), C310 body (visual mesh), and an ``_optical``
  frame at the lens, oriented for MuJoCo's camera convention (-Z look,
  +Y up).

Mesh references stay relative (``meshes/...``); the asset build copies
all robot + printed-part meshes next to the generated file so
``build/twin/`` is self-contained and portable.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from sim_benchmark.constants import DUAL_URDF_PATH
from sim_twin.params import WRIST_MOUNT_MASS, Pose, TwinParams

SIDES = ("left", "right")


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _box_inertia(mass: float, x: float, y: float, z: float) -> str:
    ixx = mass / 12 * (y * y + z * z)
    iyy = mass / 12 * (x * x + z * z)
    izz = mass / 12 * (x * x + y * y)
    return (
        f'<inertia ixx="{ixx:.3e}" ixy="0" ixz="0" '
        f'iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}" />'
    )


def _link(
    name: str, mass: float, size, visual_mesh: str | None, collision_box=None
) -> str:
    sx, sy, sz = size
    parts = [f'  <link name="{name}">']
    parts.append(
        "    <inertial>\n"
        f'      <origin xyz="0 0 0" rpy="0 0 0" />\n'
        f'      <mass value="{mass}" />\n'
        f"      {_box_inertia(mass, sx, sy, sz)}\n"
        "    </inertial>"
    )
    if visual_mesh:
        parts.append(
            "    <visual>\n"
            '      <origin xyz="0 0 0" rpy="0 0 0" />\n'
            f'      <geometry><mesh filename="meshes/{visual_mesh}" /></geometry>\n'
            '      <material name="3d_printed" />\n'
            "    </visual>"
        )
    if collision_box is not None:
        cx, cy, cz, bx, by, bz = collision_box
        parts.append(
            "    <collision>\n"
            f'      <origin xyz="{_fmt((cx, cy, cz))}" rpy="0 0 0" />\n'
            f'      <geometry><box size="{_fmt((bx, by, bz))}" /></geometry>\n'
            "    </collision>"
        )
    parts.append("  </link>")
    return "\n".join(parts)


def _fixed_joint(name: str, parent: str, child: str, pose: Pose) -> str:
    return (
        f'  <joint name="{name}" type="fixed">\n'
        f'    <origin xyz="{_fmt(pose.xyz)}" rpy="{_fmt(pose.rpy)}" />\n'
        f'    <parent link="{parent}" />\n'
        f'    <child link="{child}" />\n'
        "  </joint>"
    )


def _camera_links(params: TwinParams, side: str) -> str:
    p = params
    mm = 1e-3
    # rough bounding boxes for inertia / collision (meters)
    mount_box = (
        p.scad["tray_len"] * mm,
        (p.scad["cam_body_d"] + 2 * p.scad["tray_wall"]) * mm,
        (p.scad["wrist_cam_rise"] + p.scad["tray_lip_back"]) * mm,
    )
    cam_box = (
        p.scad["cam_body_w"] * mm,
        p.scad["cam_body_d"] * mm,
        p.scad["cam_body_h"] * mm,
    )
    mount = f"{side}_wrist_cam_mount_link"
    cam = f"{side}_wrist_cam_link"
    optical = f"{side}_wrist_cam_optical"
    blocks = [
        _link(
            mount,
            WRIST_MOUNT_MASS,
            mount_box,
            "wrist_camera_mount.stl",
            # box centered over the pedestal/tray region
            (0, p.scad["wrist_cam_fwd"] * mm, mount_box[2] / 2 + 0.004, *mount_box),
        ),
        _fixed_joint(
            f"{side}_wrist_cam_mount_joint",
            f"{side}_gripper_link",
            mount,
            p.wrist_mount_pose_in_gripper,
        ),
        _link(
            cam, p.cam_mass, cam_box, "cam_body.stl", (0, -cam_box[1] / 2, 0, *cam_box)
        ),
        _fixed_joint(f"{side}_wrist_cam_joint", mount, cam, p.wrist_cam_pose_in_mount),
        _link(optical, 1e-6, (0.001, 0.001, 0.001), None),
        _fixed_joint(
            f"{side}_wrist_cam_optical_joint",
            cam,
            optical,
            Pose(xyz=(0, 0, 0), rpy=(math.pi / 2, 0, 0)),
        ),
    ]
    return "\n\n".join(blocks)


def generate(
    params: TwinParams, out_path: Path, template_path: Path = DUAL_URDF_PATH
) -> Path:
    text = template_path.read_text()

    half = params.arm_spacing_m / 2
    z = params.arm_base_height
    for side, sign in (("left", 1), ("right", -1)):
        pattern = (
            rf'(<joint name="{side}_base_joint" type="fixed">\s*<origin xyz=")'
            r'[^"]*(")'
        )
        text, n = re.subn(pattern, rf"\g<1>0 {sign * half:.6g} {z:.6g}\g<2>", text)
        if n != 1:
            raise RuntimeError(f"{side}_base_joint origin not found in template")

    camera_xml = "\n\n".join(_camera_links(params, side) for side in SIDES)
    text = text.replace("</robot>", f"{camera_xml}\n\n</robot>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path
