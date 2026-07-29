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

# ---------------------------------------------------------------------------
# Contact-graspable payload (payload=True only). An upright rectangular bar
# the arms pick up with REAL contact (no kinematic weld). The parameters are
# the anti-slip grasp recipe from the sim-VLA design: a stiff-but-forgiving
# contact (short solref time-constant, high solimp) with condim 4 so the
# tangential+torsional friction resists the bar twisting out of the pinch.
# ---------------------------------------------------------------------------
PAYLOAD_HALF_EXTENTS = [0.011, 0.011, 0.011]  # m (2.2 cm cube)
PAYLOAD_MASS = 0.040  # kg
PAYLOAD_FRICTION = [1.5, 0.02, 0.0002]  # slide, torsion, roll
PAYLOAD_CONDIM = 4
PAYLOAD_PRIORITY = 1
PAYLOAD_SOLREF = [0.004, 1.0]
PAYLOAD_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]  # (dmin, dmax, width, mid, power)
PAYLOAD_REST_Z = PAYLOAD_HALF_EXTENTS[2]  # resting centre z above table top (world 0)
PAYLOAD_RGBA = [0.95, 0.45, 0.10, 1.0]

# Fingertip pads: small high-friction boxes on each jaw, priority above the
# bar so their friction/solref win the contact. The local placements were
# measured from the compiled twin — the grasp axis is the EE y-axis and the
# fingertips sit near EE x=0, so a fixed-jaw pad (on {side}_gripper_link) and
# a moving-jaw pad (on {side}_moving_jaw_so101_v1_link) close onto the bar
# from opposite sides. Symmetric between arms (mirrored bodies share locals).
PAD_HALF = [0.010, 0.006, 0.012]  # box half-extents (m); ~cube-height contact face
PAD_FRICTION = [3.0, 0.1, 0.005]
PAD_PRIORITY = 2
PAD_SOLREF = [0.004, 1.0]
PAD_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
# Pads are invisible in normal payload mode (alpha 0 — the policy cameras must
# see the real gripper, not green tuning aids); payload_debug turns them green.
PAD_RGBA_DEBUG = [0.15, 0.60, 0.20, 1.0]
PAD_RGBA_HIDDEN = [0.15, 0.60, 0.20, 0.0]
FIXED_PAD_LOCAL = [-0.0139, -0.0002, -0.0881]  # on {side}_gripper_link
MOVING_PAD_LOCAL = [-0.0081, -0.0647, 0.019]  # on {side}_moving_jaw_so101_v1_link

# Visual-only goal cue: a translucent green disc on the table the policy must
# read from rgb_scene (mocap so the collector/eval can move it per scenario).
TARGET_ZONE_RADIUS = 0.02
TARGET_ZONE_HALF_H = 0.001
TARGET_ZONE_RGBA = [0.10, 0.85, 0.25, 0.40]

# Pinch-force clamp: an unclamped closed command against a jaw blocked by the
# bar explodes the contact solver, so bound the gripper actuator force.
GRIPPER_FORCERANGE = 12.0

# Payload-mode solver options (finer step both divides 1/30 s into 20 whole
# substeps and stabilises the contacts; elliptic cone + impratio + noslip are
# the standard anti-slip grasp recipe).
PAYLOAD_TIMESTEP = 1.0 / 600.0
PAYLOAD_IMPRATIO = 10.0
PAYLOAD_NOSLIP_ITERATIONS = 2


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


def _add_payload(spec: mujoco.MjSpec, world, debug: bool = False) -> None:
    """Add the graspable bar, fingertip pads and target-zone disc (payload mode).

    debug=True renders the fingertip pads green (contact-tuning aid); default
    keeps them fully transparent so the policy cameras see the bare gripper.
    Physics is identical either way.
    """
    # -- Free-jointed cube, spawned on the table beyond the board's front edge.
    bar = world.add_body(name="payload", pos=[0.30, 0.0, PAYLOAD_REST_Z])
    bar.add_freejoint(name="payload_free")
    geom = bar.add_geom(
        name="payload_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(PAYLOAD_HALF_EXTENTS),
        rgba=list(PAYLOAD_RGBA),
        contype=1,
        conaffinity=1,
    )
    geom.mass = PAYLOAD_MASS
    geom.friction = list(PAYLOAD_FRICTION)
    geom.condim = PAYLOAD_CONDIM
    geom.priority = PAYLOAD_PRIORITY
    geom.solref = list(PAYLOAD_SOLREF)
    geom.solimp = list(PAYLOAD_SOLIMP)

    # -- Fingertip pads on both jaws of each gripper.
    for side in SIDES:
        for jaw, local in (
            (f"{side}_gripper_link", FIXED_PAD_LOCAL),
            (f"{side}_moving_jaw_so101_v1_link", MOVING_PAD_LOCAL),
        ):
            tag = "fixed" if "gripper_link" in jaw else "moving"
            pad = spec.body(jaw).add_geom(
                name=f"{side}_{tag}_pad",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(PAD_HALF),
                pos=list(local),
                rgba=list(PAD_RGBA_DEBUG if debug else PAD_RGBA_HIDDEN),
                contype=1,
                conaffinity=1,
            )
            pad.friction = list(PAD_FRICTION)
            pad.condim = PAYLOAD_CONDIM
            pad.priority = PAD_PRIORITY
            pad.solref = list(PAD_SOLREF)
            pad.solimp = list(PAD_SOLIMP)

    # -- Visual-only target-zone disc (mocap; the policy's sole goal cue).
    zone = world.add_body(name="target_zone", mocap=True)
    zone.add_geom(
        name="target_zone_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[TARGET_ZONE_RADIUS, TARGET_ZONE_HALF_H, 0.0],
        rgba=list(TARGET_ZONE_RGBA),
        contype=0,
        conaffinity=0,
    )


def build_spec(
    params: TwinParams | None = None,
    all_collisions: bool = False,
    payload: bool = False,
    payload_debug: bool = False,
) -> mujoco.MjSpec:
    """Assemble the twin scene.

    all_collisions=True puts every geom in one collision group (robot vs
    robot included, parent-child pairs excluded by MuJoCo as usual) — used
    by the teleop rehearsal tool so arm-arm and arm-rig contact is felt.
    Default False keeps the original robot-vs-world-only model.

    payload=True adds the contact-graspable bar, the fingertip pads, the
    visual target-zone disc, a pinch-force clamp and the anti-slip contact
    solver options used by the sim-VLA pick-and-place tasks. It implies
    all_collisions (the bar must feel every colliding geom). payload_debug
    renders the (normally invisible) fingertip pads green for tuning.
    """
    p = params or TwinParams.load()
    mm = 1e-3
    if payload:
        all_collisions = True

    spec = mujoco.MjSpec.from_file(_patched_urdf_path())
    spec.modelname = "so101_dual_twin"
    spec.option.timestep = PAYLOAD_TIMESTEP if payload else 0.002
    if payload:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = PAYLOAD_IMPRATIO
        spec.option.noslip_iterations = PAYLOAD_NOSLIP_ITERATIONS
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
        if payload and joint_name in GRIPPER_JOINTS:
            act.forcerange = [-GRIPPER_FORCERANGE, GRIPPER_FORCERANGE]

    if payload:
        _add_payload(spec, world, debug=payload_debug)
        _setup_payload_collisions(spec)
    elif all_collisions:
        # Flatten every colliding geom into one group: arm-arm, arm-rig,
        # arm-table contacts all become live. Pure-visual geoms (0/0)
        # stay visual.
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = 1
                geom.conaffinity = 1

    if all_collisions or payload:
        # The base shell and the rotating shoulder NEST on real hardware —
        # their raw meshes interpenetrate by design. MuJoCo's automatic
        # parent-child contact exclusion does not cover them because the
        # base is static (welded to the world), so exclude explicitly.
        for side in SIDES:
            spec.add_exclude(
                bodyname1=f"{side}_base_link", bodyname2=f"{side}_shoulder_link"
            )

    return spec


# Collision groups for payload mode (contype, conaffinity) bitmasks. Two geoms
# collide iff (c1 & a2) or (c2 & a1). The scheme keeps every arm-arm/arm-rig
# contact live (as all_collisions does) BUT lets the bar touch only the rig and
# the fingertip pads — never the raw jaw/arm meshes — so a clean two-pad pinch
# holds it, instead of a dozen mesh corners tilting it.
_BIT_ARM = 1  # arm/jaw meshes
_BIT_BAR = 2  # the payload bar
_BIT_RIG = 4  # world/rig collidable by the bar
_BIT_PAD = 8  # fingertip pads
_GRP_ARM = (_BIT_ARM, _BIT_ARM)
_GRP_RIG = (_BIT_ARM | _BIT_RIG, _BIT_ARM | _BIT_BAR)
_GRP_BAR = (_BIT_BAR, _BIT_RIG | _BIT_PAD)
_GRP_PAD = (_BIT_PAD, _BIT_BAR)


def _setup_payload_collisions(spec: mujoco.MjSpec) -> None:
    """Assign the payload-mode collision groups (see the bitmask table above)."""
    for geom in spec.geoms:
        if not (geom.contype or geom.conaffinity):
            continue  # pure-visual geoms stay visual
        if geom.name == "payload_geom":
            geom.contype, geom.conaffinity = _GRP_BAR
        elif geom.name.endswith("_pad"):
            geom.contype, geom.conaffinity = _GRP_PAD
        elif geom.contype == WORLD_CONTYPE:  # table / board / adapters / tower
            geom.contype, geom.conaffinity = _GRP_RIG
        else:  # arm / jaw meshes
            geom.contype, geom.conaffinity = _GRP_ARM


def build_model(
    params: TwinParams | None = None,
    all_collisions: bool = False,
    payload: bool = False,
    payload_debug: bool = False,
) -> mujoco.MjModel:
    return build_spec(
        params,
        all_collisions=all_collisions,
        payload=payload,
        payload_debug=payload_debug,
    ).compile()


class TwinSim:
    """Compiled twin with the same driving interface as DualArmSim."""

    def __init__(
        self,
        params: TwinParams | None = None,
        all_collisions: bool = False,
        payload: bool = False,
        payload_debug: bool = False,
    ) -> None:
        self.params = params or TwinParams.load()
        self.payload = payload
        self.model = build_model(
            self.params,
            all_collisions=all_collisions,
            payload=payload,
            payload_debug=payload_debug,
        )
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
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        if payload:
            free = self.model.joint("payload_free")
            qadr, vadr = free.qposadr[0], free.dofadr[0]
            self._payload_pos_sl = slice(qadr, qadr + 3)
            self._payload_quat_sl = slice(qadr + 3, qadr + 7)
            self._payload_vel_sl = slice(vadr, vadr + 6)
            self.target_zone_mocap_id = self.model.body("target_zone").mocapid[0]
            self._gripper_ctrl_by_side = {
                side: int(self.model.actuator(f"act_{side}_gripper").id)
                for side in SIDES
            }
            self._gripper_qpos_idx = {
                side: int(self.model.joint(f"{side}_gripper").qposadr[0])
                for side in SIDES
            }
            self._gripper_range = {
                side: tuple(self.model.joint(f"{side}_gripper").range) for side in SIDES
            }

    def neutral_q(self) -> np.ndarray:
        return np.deg2rad(np.array(NEUTRAL_ARM_ANGLES_DEG * 2))

    def reset(self, q: np.ndarray | None = None) -> None:
        if q is None:
            q = self.neutral_q()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_idx] = q
        self.data.ctrl[self.arm_ctrl_idx] = q
        if self.payload:
            # Start with the grippers commanded fully open so the bar can be
            # placed between the jaws before any squeeze.
            for side in SIDES:
                self.set_gripper_frac(side, 1.0)
        # Park the headset marker behind the rig until calibration places it.
        self.data.mocap_pos[self.headset_mocap_id] = [-0.45, 0.0, 0.45]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Payload / grasp helpers (payload mode only)
    # ------------------------------------------------------------------

    def set_payload_pose(self, pos: np.ndarray, quat: np.ndarray | None = None) -> None:
        """Place the bar (settled, zero velocity); default upright attitude."""
        self.data.qpos[self._payload_pos_sl] = pos
        self.data.qpos[self._payload_quat_sl] = (
            [1.0, 0.0, 0.0, 0.0] if quat is None else quat
        )
        self.data.qvel[self._payload_vel_sl] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def payload_pos(self) -> np.ndarray:
        return self.data.qpos[self._payload_pos_sl].copy()

    def payload_vel(self) -> np.ndarray:
        return self.data.qvel[self._payload_vel_sl].copy()

    def set_target_zone(self, pos: np.ndarray) -> None:
        """Move the visual target-zone disc (twin world coords)."""
        self.data.mocap_pos[self.target_zone_mocap_id] = pos

    def gripper_open_frac(self, side: str) -> float:
        """Measured gripper open fraction (0 closed, 1 open)."""
        q = float(self.data.qpos[self._gripper_qpos_idx[side]])
        lo, hi = self._gripper_range[side]
        return (q - lo) / (hi - lo)

    def set_gripper_frac(self, side: str, frac: float) -> None:
        """Command the gripper by open fraction (0 closed, 1 open)."""
        lo, hi = self._gripper_range[side]
        self.data.ctrl[self._gripper_ctrl_by_side[side]] = lo + float(frac) * (hi - lo)

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
        """Offscreen RGB render of one of the twin's C310 cameras.

        Renderers are cached per (width, height) so the collector/eval can
        serve several resolutions from one sim without clobbering a
        single-resolution cache.
        """
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self._renderers[key] = mujoco.Renderer(
                self.model, height=height, width=width
            )
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()
