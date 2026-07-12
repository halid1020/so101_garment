"""End-to-end verification of the digital twin (local, MuJoCo side).

    python -m sim_twin.verify

Checks, in order:
  1. asset export volume counts (CGAL floating-solid guard)
  2. the generated URDF loads in Pinocchio and keeps the EE frames the
     production Pink IK stack expects
  3. compiled MuJoCo model geometry: base height/spacing, camera fovy,
     tower platform height — all against config.scad-derived values
  4. no penetrating robot contacts at the neutral pose
  5. offscreen renders of the three C310 cameras + an overview to
     outputs/twin_verify/ for eyeball review

Exits non-zero on the first hard failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from sim_twin import assets
from sim_twin.params import BUILD_DIR, REPO_ROOT, TwinParams

OUT_DIR = REPO_ROOT / "outputs" / "twin_verify"
CAMERAS = ("rgb_scene", "rgb_wrist_left", "rgb_wrist_right")


def _ok(label: str) -> None:
    print(f"  ok  {label}")


def check_assets() -> TwinParams:
    params = assets.build()
    _ok("assets built (meshes, robot.urdf, twin_params.json)")
    return params


def check_pinocchio(params: TwinParams) -> None:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(BUILD_DIR / "robot.urdf"))
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    for name in (
        "left_eef_link",
        "right_eef_link",
        "left_wrist_cam_optical",
        "right_wrist_cam_optical",
    ):
        if not model.existFrame(name):
            raise AssertionError(f"frame {name} missing from generated URDF")
    for side, sign in (("left", 1), ("right", -1)):
        t = data.oMf[model.getFrameId(f"{side}_base_link")].translation
        expected = np.array(
            [0, sign * params.arm_spacing_m / 2, params.arm_base_height]
        )
        if not np.allclose(t, expected, atol=1e-6):
            raise AssertionError(f"{side}_base_link at {t}, expected {expected}")
    _ok("Pinocchio: URDF loads, EE/camera frames present, bases placed")


def check_mujoco(params: TwinParams):
    from sim_twin.scene import TwinSim

    sim = TwinSim(params)
    sim.reset()
    m, d = sim.model, sim.data

    for side, sign in (("left", 1), ("right", -1)):
        pos = d.xpos[m.body(f"{side}_base_link").id]
        expected = [0, sign * params.arm_spacing_m / 2, params.arm_base_height]
        if not np.allclose(pos, expected, atol=1e-6):
            raise AssertionError(f"{side}_base_link at {pos} != {expected}")
    _ok(
        f"arm bases at ±{params.arm_spacing_m / 2:.3f} m, "
        f"z={params.arm_base_height:.4f} m"
    )

    for cam in CAMERAS:
        fovy = m.camera(cam).fovy[0]
        if abs(fovy - params.fovy_deg) > 1e-6:
            raise AssertionError(f"{cam} fovy {fovy} != {params.fovy_deg}")
    _ok(
        f"3 cameras present, fovy={params.fovy_deg:.2f}° "
        f"(dFOV {params.cam_dfov_deg}°)"
    )

    scene_cam_z = d.cam_xpos[m.camera("rgb_scene").id][2]
    lens_z_min = params.tower_platform_top_z
    if not (lens_z_min < scene_cam_z < lens_z_min + 0.1):
        raise AssertionError(
            f"rgb_scene at z={scene_cam_z:.3f}, expected just above the "
            f"platform top {lens_z_min:.3f}"
        )
    _ok(
        f"tower platform top at {params.tower_platform_top_z:.3f} m, "
        f"scene camera above it at {scene_cam_z:.3f} m"
    )

    penetrating = [
        (
            m.geom(c.geom1).name or str(c.geom1),
            m.geom(c.geom2).name or str(c.geom2),
            c.dist,
        )
        for c in d.contact[: d.ncon]
        if c.dist < -1e-5
    ]
    if penetrating:
        for g1, g2, dist in penetrating:
            print(f"  PENETRATION {g1} <-> {g2}: {dist:.5f}")
        raise AssertionError(
            "penetrating contacts at neutral pose — add explicit excludes "
            "in src/sim_twin/scene.py"
        )
    _ok(f"no penetrating contacts at neutral ({d.ncon} total contacts)")

    # servo sanity: hold neutral for 1 s, arms must stay put (gravity sag
    # under kp=25 stays well under 3 deg)
    sim.step(500)
    drift = np.rad2deg(np.abs(sim.data.qpos[sim.arm_qpos_idx] - sim.neutral_q()))
    if drift.max() > 3.0:
        raise AssertionError(f"neutral-pose drift {drift.max():.2f}° > 3°")
    _ok(f"servo hold: max drift {drift.max():.2f}° after 1 s")
    return sim


def render_cameras(sim) -> None:
    import imageio.v2 as iio
    import mujoco

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        iio.imwrite(OUT_DIR / f"{cam}.png", sim.render_camera(cam))
    renderer = mujoco.Renderer(sim.model, height=720, width=960)
    view = mujoco.MjvCamera()
    view.azimuth, view.elevation, view.distance = 140, -25, 1.3
    view.lookat = [0.15, 0, 0.15]
    renderer.update_scene(sim.data, camera=view)
    iio.imwrite(OUT_DIR / "overview.png", renderer.render())
    # second overview with the camera view-frustums drawn (shows what
    # the tower camera actually sees)
    frustums = mujoco.MjvOption()
    frustums.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    view.azimuth, view.elevation, view.distance = 155, -18, 1.6
    view.lookat = [0.2, 0, 0.2]
    renderer.update_scene(sim.data, camera=view, scene_option=frustums)
    iio.imwrite(OUT_DIR / "overview_frustum.png", renderer.render())
    renderer.close()
    _ok(
        f"camera renders in {OUT_DIR.relative_to(REPO_ROOT)}/ "
        "(incl. overview_frustum.png)"
    )


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("verifying digital twin ...")
    try:
        params = check_assets()
        check_pinocchio(params)
        sim = check_mujoco(params)
        render_cameras(sim)
    except (AssertionError, RuntimeError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print("twin verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
