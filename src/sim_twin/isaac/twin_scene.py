"""Isaac Lab InteractiveScene config mirroring the MuJoCo twin.

Import this from an Isaac Lab script (see run_demo.py). Geometry and
gains come from twin_params.json; run convert_assets.py first so the
USD files exist.

Written against Isaac Lab 2.x. Camera conventions: the URDF's
``*_wrist_cam_optical`` links and the computed tower-camera pose are in
the OpenGL convention (-Z look, +Y up), so every CameraCfg here uses
``convention="opengl"`` with an identity offset.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaaclab.sim as sim_utils  # noqa: E402
import params as twin_params  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

TWIN = twin_params.twin_dir()
P = twin_params.load_params()
_WORLD = P["world"]
_CAMS = P["cameras"]
_CTRL = P["control"]

_USD = TWIN / "usd"

_ARM_JOINT_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
_NEUTRAL = {
    f"{side}_{suffix}": math.radians(_CTRL["neutral_arm_angles_deg"][i])
    for side in ("left", "right")
    for i, suffix in enumerate(_ARM_JOINT_SUFFIXES)
}
_NEUTRAL.update({"left_gripper": 0.0, "right_gripper": 0.0})

_FOCAL = twin_params.focal_length_from_dfov(_CAMS["dfov_deg"])


def _camera(prim_path: str, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
    return CameraCfg(
        prim_path=prim_path,
        update_period=1 / 30,
        width=_CAMS["width"],
        height=_CAMS["height"],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


def _static_box(prim_path: str, size, pos, color):
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=tuple(size),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos)),
    )


def _static_usd(prim_path: str, usd: str, pos, rot=(1.0, 0.0, 0.0, 0.0)):
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(_USD / usd)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(pos), rot=rot),
    )


def _rig_assets() -> dict:
    """Furniture + rig, z=0 at the table top (same as the MuJoCo twin)."""
    table_h = _WORLD["table_height_m"]
    tx, ty, tt = _WORLD["table_size_m"]
    bx, by = _WORLD["board_center_xy"]
    bsx, bsy = _WORLD["board_size_xy"]
    board_top = _WORLD["board_top_z"]
    tower_yaw = _WORLD["tower_yaw_rad"]
    acx = _WORLD["adapter_center_x"]
    aw, ad, at = _WORLD["adapter_size_m"]
    spacing = _WORLD["arm_spacing_m"]
    tcx, tcy = _WORLD["tower_center_xy"]

    assets = {
        "ground": AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -table_h)),
        ),
        "dome_light": AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0),
        ),
        "table": _static_box(
            "{ENV_REGEX_NS}/table",
            (tx, ty, tt),
            (P["world"].get("table_x_offset_m", 0.15), 0, -tt / 2),
            (0.55, 0.42, 0.30),
        ),
        # printed perforated board (tiles + splice bars, one USD).
        # Mesh origin at the board corner, width along part X -> world
        # Y: same +90deg yaw as the adapters.
        "board": _static_usd(
            "{ENV_REGEX_NS}/board",
            "board_assembled.usd",
            (bx + bsx / 2, by - bsy / 2, 0.0),
            rot=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),
        ),
        "tower": _static_usd(
            "{ENV_REGEX_NS}/tower",
            "tower_assembled.usd",
            (tcx, tcy, board_top),
            rot=(math.cos(tower_yaw / 2), 0.0, 0.0, math.sin(tower_yaw / 2)),
        ),
    }
    # adapter plates: mesh origin at the plate corner, front (-Y) faces
    # the arm's +X  ->  yaw +90deg, corner offset rotated accordingly
    yaw90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    for side, sign in (("left", 1), ("right", -1)):
        assets[f"{side}_adapter"] = _static_usd(
            f"{{ENV_REGEX_NS}}/{side}_adapter",
            "adapter.usd",
            (acx + ad / 2, sign * spacing / 2 - aw / 2, board_top),
            rot=yaw90,
        )
    # cradle + tower C310 body visuals
    cradle = _CAMS["tower_cradle_world"]
    cradle_rot = twin_params.mat_to_quat(twin_params.rpy_to_mat(cradle["rpy"]))
    assets["tower_cradle"] = _static_usd(
        "{ENV_REGEX_NS}/tower_cradle",
        "tower_camera_cradle.usd",
        tuple(cradle["xyz"]),
        rot=cradle_rot,
    )
    cam_xyz, cam_rot = twin_params.compose(cradle, _CAMS["tower_cam_in_cradle"])
    assets["scene_cam"] = _static_usd(
        "{ENV_REGEX_NS}/scene_cam",
        "cam_body.usd",
        tuple(cam_xyz),
        rot=twin_params.mat_to_quat(cam_rot),
    )
    return assets


ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD / "robot.usd"),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # spacing + base height are baked in the URDF
        joint_pos=_NEUTRAL,
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=_CTRL["actuator_kp"],
            damping=_CTRL["actuator_kv"],
        )
    },
)


@configclass
class TwinSceneCfg(InteractiveSceneCfg):
    """Dual SO-101 rig + three C310 cameras."""

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    rgb_wrist_left: CameraCfg = _camera(
        "{ENV_REGEX_NS}/Robot/left_wrist_cam_optical/cam"
    )
    rgb_wrist_right: CameraCfg = _camera(
        "{ENV_REGEX_NS}/Robot/right_wrist_cam_optical/cam"
    )
    rgb_scene: CameraCfg = _camera(
        "{ENV_REGEX_NS}/rgb_scene",
        pos=tuple(twin_params.scene_camera_world(P)[0]),
        rot=twin_params.scene_camera_world(P)[1],
    )

    def __post_init__(self):
        super().__post_init__()
        for name, asset in _rig_assets().items():
            setattr(self, name, asset)
