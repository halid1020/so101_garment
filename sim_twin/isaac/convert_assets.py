"""Convert the twin's URDF + meshes to USD for Isaac Lab.

Run INSIDE the Isaac Lab python environment on the Isaac machine:

    ./isaaclab.sh -p convert_assets.py            # writes twin/usd/
    python convert_assets.py --dry-run            # input validation only
                                                  # (no isaaclab needed)

Written against Isaac Lab 2.x (`isaaclab.sim.converters`). If the Cfg
field names drifted in your version, the converter classes are the only
thing to touch — geometry all comes from twin_params.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import params as twin_params  # noqa: E402

STATIC_MESHES = (
    "tower_assembled",
    "tower_camera_cradle",
    "cam_body",
    "adapter",
    "board_assembled",
)


def validate_inputs() -> Path:
    twin = twin_params.twin_dir()
    missing = [
        str(p)
        for p in (
            twin / "robot.urdf",
            twin / "twin_params.json",
            *(twin / "meshes" / f"{m}.stl" for m in STATIC_MESHES),
        )
        if not p.exists()
    ]
    if missing:
        raise SystemExit("missing twin assets: " + ", ".join(missing))
    data = twin_params.load_params()
    for key in ("world", "cameras", "control"):
        if key not in data:
            raise SystemExit(f"twin_params.json missing '{key}' section")
    print(f"twin assets ok: {twin}")
    return twin


def convert(twin: Path) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841 — keeps Kit alive

    from isaaclab.sim.converters import (
        MeshConverter,
        MeshConverterCfg,
        UrdfConverter,
        UrdfConverterCfg,
    )

    control = twin_params.load_params()["control"]
    usd_dir = twin / "usd"
    usd_dir.mkdir(exist_ok=True)

    print("converting robot.urdf -> robot.usd ...")
    urdf_cfg = UrdfConverterCfg(
        asset_path=str(twin / "robot.urdf"),
        usd_dir=str(usd_dir),
        usd_file_name="robot.usd",
        fix_base=True,
        # keep the wrist camera links as prims for CameraCfg attachment
        merge_fixed_joints=False,
        force_usd_conversion=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=control["actuator_kp"],
                damping=control["actuator_kv"],
            ),
        ),
    )
    UrdfConverter(urdf_cfg)

    for name in STATIC_MESHES:
        print(f"converting {name}.stl -> {name}.usd ...")
        MeshConverter(
            MeshConverterCfg(
                asset_path=str(twin / "meshes" / f"{name}.stl"),
                usd_dir=str(usd_dir),
                usd_file_name=f"{name}.usd",
                make_instanceable=False,
            )
        )
    print(f"USD assets in {usd_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs without importing isaaclab",
    )
    args, _ = parser.parse_known_args()
    twin = validate_inputs()
    if args.dry_run:
        print("dry run: skipping USD conversion")
        return 0
    convert(twin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
