"""MuJoCo digital twin of the dual SO-101 board rig.

Loads the generated ``build/twin/robot.urdf`` (real arm spacing, real
mounting-stack height, wrist C310s included) and decorates it with the
physical rig: floor, table, nut plate, board, adapter plates, camera
tower + cradle + tower C310, and the three camera sensors named after
the software streams (``rgb_wrist_left``, ``rgb_wrist_right``,
``rgb_scene``).

Collision model: robot geoms collide with the world (table, board,
platform parts, any payload) but not with each other — self-collision
of the raw URDF meshes is noisy, and the real rig relies on IK limits
anyway. World geoms use contype=2/conaffinity=1, robot geoms 1/2.

Rebuild assets first (``python -m sim_twin.assets``) or call
``build_model()`` which does it for you.
"""

from __future__ import annotations

import math
import re
import tempfile

import mujoco
import numpy as np

from sim_benchmark.constants import (
    ACTUATOR_KP,
    ACTUATOR_KV,
    ARM_JOINTS,
    EE_FRAMES,
    GRIPPER_JOINTS,
    JOINT_ARMATURE,
    JOINT_DAMPING,
    JOINT_FRICTIONLOSS,
    NEUTRAL_ARM_ANGLES_DEG,
    SIDES,
)
from sim_twin.params import BUILD_DIR, CAMERA_HEIGHT, CAMERA_WIDTH, TwinParams

ROBOT_CONTYPE, ROBOT_CONAFFINITY = 1, 2
WORLD_CONTYPE, WORLD_CONAFFINITY = 2, 1

# Render the sensors just ahead of the C310 lens ring so the camera's own
# body mesh doesn't straddle the near plane (MuJoCo-only detail).
LENS_STANDOFF = 0.004

WOOD = [0.62, 0.48, 0.32, 1.0]
PRINT_BLUE = [0.25, 0.35, 0.55, 1.0]
PRINT_DARK = [0.22, 0.22, 0.24, 1.0]


def _quat_from_rpy(rpy) -> list[float]:
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _patched_urdf_path() -> str:
    """Temp copy of the twin URDF with a MuJoCo compiler block."""
    urdf = BUILD_DIR / "robot.urdf"
    if not urdf.exists():
        raise FileNotFoundError(
            f"{urdf} missing — run `python -m sim_twin.assets` first"
        )
    mujoco_block = (
        f'<mujoco><compiler meshdir="{BUILD_DIR}" '
        'balanceinertia="true" discardvisual="false" fusestatic="false"/>'
        "</mujoco>"
    )
    text = re.sub(
        r"(<robot[^>]*>)", rf"\1\n  {mujoco_block}", urdf.read_text(), count=1
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="so101_twin_", delete=False
    )
    tmp.write(text)
    tmp.close()
    return tmp.name


def _add_mesh(spec: mujoco.MjSpec, name: str) -> str:
    # the URDF import may have registered it already (e.g. cam_body)
    if all(mesh.name != name for mesh in spec.meshes):
        spec.add_mesh(name=name, file=f"meshes/{name}.stl")
    return name


def _world_box(body, name, pos, half, rgba, collide=True):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        pos=list(pos),
        rgba=list(rgba),
        contype=WORLD_CONTYPE if collide else 0,
        conaffinity=WORLD_CONAFFINITY if collide else 0,
    )


def _visual_mesh(body, name, mesh, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=PRINT_BLUE):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh,
        pos=list(pos),
        quat=list(quat),
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
    )


def build_spec(
    params: TwinParams | None = None, all_collisions: bool = False
) -> mujoco.MjSpec:
    """Assemble the twin scene.

    all_collisions=True puts every geom in one collision group (robot vs
    robot included, parent-child pairs excluded by MuJoCo as usual) — used
    by the teleop rehearsal tool so arm-arm and arm-rig contact is felt.
    Default False keeps the original robot-vs-world-only model.
    """
    p = params or TwinParams.load()
    mm = 1e-3

    spec = mujoco.MjSpec.from_file(_patched_urdf_path())
    spec.modelname = "so101_dual_twin"
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # Robot geoms: collide with the world group only (see module doc).
    # (When all_collisions is set, a final pass below flattens every
    # colliding geom into one group instead.)
    for geom in spec.geoms:
        geom.contype = ROBOT_CONTYPE
        geom.conaffinity = ROBOT_CONAFFINITY

    for joint in spec.joints:
        joint.damping[0] = JOINT_DAMPING
        joint.armature = JOINT_ARMATURE
        joint.frictionloss = JOINT_FRICTIONLOSS

    world = spec.worldbody
    world.add_light(pos=[0.4, 0, 1.2], dir=[-0.2, 0, -1], castshadow=False)
    world.add_light(pos=[-0.6, 0.5, 0.9], dir=[0.5, -0.4, -0.8], castshadow=False)

    # ---- Furniture stack (z=0 is the table top) ----
    table_h = p.table_height * mm
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -table_h],
        rgba=[0.35, 0.35, 0.35, 1.0],
        contype=WORLD_CONTYPE,
        conaffinity=WORLD_CONAFFINITY,
    )
    _world_box(
        world,
        "table",
        pos=[p.table_x_offset * mm, 0, -p.table_thick * mm / 2],
        half=[p.table_size_x * mm / 2, p.table_size_y * mm / 2, p.table_thick * mm / 2],
        rgba=[0.55, 0.42, 0.30, 1.0],
    )

    # ---- Printed perforated board (tiles + splice bars, one visual
    # mesh; box collision). Mesh origin is the board corner: width
    # along part X -> world Y, so it gets the same Rz(+90) as the
    # adapters. ----
    _add_mesh(spec, "board_assembled")
    bx, by = p.board_center
    bsx, bsy = p.board_size
    board = world.add_body(
        name="board",
        pos=[bx, by, 0],
        quat=_quat_from_rpy((0, 0, math.pi / 2)),
    )
    _visual_mesh(
        board,
        "board_visual",
        "board_assembled",
        pos=[-p.board_width * mm / 2, -p.board_depth * mm / 2, 0],
        rgba=PRINT_BLUE,
    )
    _world_box(
        board,
        "board_col",
        pos=[0, 0, p.board_top_z / 2],
        half=[p.board_width * mm / 2, p.board_depth * mm / 2, p.board_top_z / 2],
        rgba=[0, 0, 0, 0],
    )

    # ---- Adapter plates (mesh origin at the plate corner; the part's
    # front (-Y) faces the arm's forward +X -> Rz(+90)) ----
    _add_mesh(spec, "adapter")
    adapter_quat = _quat_from_rpy((0, 0, math.pi / 2))
    half_w = p.adapter_w * mm / 2
    half_d = p.adapter_d * mm / 2
    for side, sign in (("left", 1), ("right", -1)):
        body = world.add_body(
            name=f"{side}_adapter",
            pos=[p.adapter_center_x, sign * p.arm_spacing_m / 2, p.board_top_z],
            quat=adapter_quat,
        )
        _visual_mesh(
            body, f"{side}_adapter_visual", "adapter", pos=[-half_w, -half_d, 0]
        )
        _world_box(
            body,
            f"{side}_adapter_col",
            pos=[0, 0, p.adapter_thick * mm / 2],
            half=[half_w, half_d, p.adapter_thick * mm / 2],
            rgba=[0, 0, 0, 0],
        )

    # ---- Camera tower (mesh centered on its base) ----
    _add_mesh(spec, "tower_assembled")
    tcx, tcy = p.tower_center
    tower = world.add_body(
        name="camera_tower",
        pos=[tcx, tcy, p.board_top_z],
        quat=_quat_from_rpy((0, 0, p.tower_yaw_rad)),
    )
    _visual_mesh(tower, "tower_visual", "tower_assembled")
    # collision proxies: base plate, three mast boxes, platform
    _world_box(
        tower,
        "tower_col_base",
        pos=[0, 0, p.tower_base_thick * mm / 2],
        half=[p.tower_base_plate * mm / 2] * 2 + [p.tower_base_thick * mm / 2],
        rgba=[0, 0, 0, 0],
    )
    mast_h = p.tower_height_total * mm
    z0 = p.tower_base_thick * mm
    for i in range(3):
        frac_lo, frac_hi = i / 3, (i + 1) / 3
        width = (
            (
                p.tower_base_width
                + (p.tower_top_width - p.tower_base_width) * (frac_lo + frac_hi) / 2
            )
            * mm
            * 0.72
        )  # equilateral triangle -> box side fudge
        _world_box(
            tower,
            f"tower_col_mast{i}",
            pos=[0, 0, z0 + mast_h * (frac_lo + frac_hi) / 2],
            half=[width / 2, width / 2, mast_h / 6],
            rgba=[0, 0, 0, 0],
        )
    plat_top = p.tower_platform_top_z - p.board_top_z
    _world_box(
        tower,
        "tower_col_platform",
        pos=[0, 0, plat_top - p.camera_platform_thick * mm / 2],
        half=[p.camera_plate * mm / 2] * 2 + [p.camera_platform_thick * mm / 2],
        rgba=[0, 0, 0, 0],
    )

    # ---- Tower cradle + scene C310 + rgb_scene camera ----
    _add_mesh(spec, "tower_camera_cradle")
    _add_mesh(spec, "cam_body")
    cradle_pose = p.tower_cradle_pose_world
    cradle = world.add_body(
        name="tower_cradle",
        pos=list(cradle_pose.xyz),
        quat=_quat_from_rpy(cradle_pose.rpy),
    )
    _visual_mesh(cradle, "cradle_visual", "tower_camera_cradle")
    cam_pose = p.tower_cam_pose_in_cradle
    _visual_mesh(
        cradle,
        "scene_cam",
        "cam_body",
        pos=list(cam_pose.xyz),
        quat=_quat_from_rpy(cam_pose.rpy),
        rgba=[0.1, 0.1, 0.1, 1.0],
    )
    tilt = math.radians(p.tower_cam_tilt_deg)
    look = (0.0, math.cos(tilt), -math.sin(tilt))  # optical +Y tilted down
    cradle.add_camera(
        name="rgb_scene",
        pos=[c + LENS_STANDOFF * a for c, a in zip(cam_pose.xyz, look)],
        quat=_quat_from_rpy((math.pi / 2 - tilt, 0.0, 0.0)),
        fovy=p.fovy_deg,
        resolution=[CAMERA_WIDTH, CAMERA_HEIGHT],
    )

    # ---- Wrist cameras (optical frames come from the URDF) ----
    for side in SIDES:
        optical = spec.body(f"{side}_wrist_cam_optical")
        # optical frame looks along its -Z
        optical.add_camera(
            name=f"rgb_wrist_{side}",
            pos=[0, 0, -LENS_STANDOFF],
            fovy=p.fovy_deg,
            resolution=[CAMERA_WIDTH, CAMERA_HEIGHT],
        )
        # tint the wrist C310 visuals dark like the real housing
        for geom in spec.body(f"{side}_wrist_cam_link").geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.rgba = [0.1, 0.1, 0.1, 1.0]

    # ---- EE sites + actuators (same servo model as the benchmark) ----
    for side in SIDES:
        spec.body(EE_FRAMES[side]).add_site(
            name=f"{side}_eef_site",
            pos=[0, 0, 0],
            size=[0.008, 0.008, 0.008],
            rgba=[0, 1, 0, 0.4],
        )

    # ---- Teleop visualization: target spheres, headset-center marker,
    # and coordinate-frame triads (sim world frame at the origin; the VR
    # operator frame rides on the headset marker — robot-aligned axes by
    # construction of the operator-frame mapping) ----
    target_rgba = {"left": (0.9, 0.2, 0.2, 0.5), "right": (0.2, 0.4, 0.9, 0.5)}
    for side in SIDES:
        target = world.add_body(name=f"{side}_target", mocap=True)
        target.add_geom(
            name=f"{side}_target_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            rgba=list(target_rgba[side]),
            contype=0,
            conaffinity=0,
        )

    def _axis_triad(body, prefix: str, length: float = 0.06) -> None:
        colors = {
            "x": ([1.0, 0.1, 0.1, 0.9], [length / 2, 0, 0], [length / 2, 0.003, 0.003]),
            "y": ([0.1, 0.8, 0.1, 0.9], [0, length / 2, 0], [0.003, length / 2, 0.003]),
            "z": (
                [0.15, 0.3, 1.0, 0.9],
                [0, 0, length / 2],
                [0.003, 0.003, length / 2],
            ),
        }
        for axis, (rgba, pos, half) in colors.items():
            body.add_geom(
                name=f"{prefix}_{axis}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half,
                pos=pos,
                rgba=rgba,
                contype=0,
                conaffinity=0,
            )

    _axis_triad(world, "world_frame", length=0.10)
    headset = world.add_body(name="headset_marker", mocap=True)
    headset.add_geom(
        name="headset_marker_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03, 0, 0],
        rgba=[0.15, 0.15, 0.18, 0.7],
        contype=0,
        conaffinity=0,
    )
    _axis_triad(headset, "vr_frame", length=0.08)
    for joint_name in (*ARM_JOINTS, *GRIPPER_JOINTS):
        act = spec.add_actuator(
            name=f"act_{joint_name}",
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = ACTUATOR_KP
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0] = 0.0
        act.biasprm[1] = -ACTUATOR_KP
        act.biasprm[2] = -ACTUATOR_KV

    if all_collisions:
        # Flatten every colliding geom into one group: arm-arm, arm-rig,
        # arm-table contacts all become live. Pure-visual geoms (0/0)
        # stay visual.
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = 1
                geom.conaffinity = 1
        # The base shell and the rotating shoulder NEST on real hardware —
        # their raw meshes interpenetrate by design. MuJoCo's automatic
        # parent-child contact exclusion does not cover them because the
        # base is static (welded to the world), so exclude explicitly.
        for side in SIDES:
            spec.add_exclude(
                bodyname1=f"{side}_base_link", bodyname2=f"{side}_shoulder_link"
            )

    return spec


def build_model(
    params: TwinParams | None = None, all_collisions: bool = False
) -> mujoco.MjModel:
    return build_spec(params, all_collisions=all_collisions).compile()


class TwinSim:
    """Compiled twin with the same driving interface as DualArmSim."""

    def __init__(
        self, params: TwinParams | None = None, all_collisions: bool = False
    ) -> None:
        self.params = params or TwinParams.load()
        self.model = build_model(self.params, all_collisions=all_collisions)
        self.data = mujoco.MjData(self.model)
        self.arm_qpos_idx = np.array(
            [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        )
        self.arm_ctrl_idx = np.array(
            [self.model.actuator(f"act_{j}").id for j in ARM_JOINTS]
        )
        self.gripper_ctrl_idx = np.array(
            [self.model.actuator(f"act_{j}").id for j in GRIPPER_JOINTS]
        )
        self.eef_site_id = {
            side: self.model.site(f"{side}_eef_site").id for side in SIDES
        }
        self.target_mocap_id = {
            side: self.model.body(f"{side}_target").mocapid[0] for side in SIDES
        }
        self.headset_mocap_id = self.model.body("headset_marker").mocapid[0]

    def neutral_q(self) -> np.ndarray:
        return np.deg2rad(np.array(NEUTRAL_ARM_ANGLES_DEG * 2))

    def reset(self, q: np.ndarray | None = None) -> None:
        if q is None:
            q = self.neutral_q()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_idx] = q
        self.data.ctrl[self.arm_ctrl_idx] = q
        # Park the headset marker behind the rig until calibration places it.
        self.data.mocap_pos[self.headset_mocap_id] = [-0.45, 0.0, 0.45]
        mujoco.mj_forward(self.model, self.data)

    def set_arm_targets(self, q: np.ndarray) -> None:
        self.data.ctrl[self.arm_ctrl_idx] = q

    def arm_q(self) -> np.ndarray:
        """Measured arm joint angles (radians, ARM_JOINTS order)."""
        return self.data.qpos[self.arm_qpos_idx].copy()

    def set_target_markers(
        self, targets: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Move the target mocap spheres (positions in twin world coords)."""
        for side, (pos, _rot) in targets.items():
            self.data.mocap_pos[self.target_mocap_id[side]] = pos

    def set_headset_marker(self, pos: np.ndarray) -> None:
        """Place the notional headset-center marker (twin world coords)."""
        self.data.mocap_pos[self.headset_mocap_id] = pos

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def eef_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self.eef_site_id[side]
        return (
            self.data.site_xpos[sid].copy(),
            self.data.site_xmat[sid].reshape(3, 3).copy(),
        )

    def render_camera(
        self, camera: str, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT
    ) -> np.ndarray:
        """Offscreen RGB render of one of the twin's C310 cameras."""
        renderer = getattr(self, "_renderer", None)
        if renderer is None or renderer._width != width:
            renderer = self._renderer = mujoco.Renderer(
                self.model, height=height, width=width
            )
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()
