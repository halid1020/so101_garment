"""Interactive viewer for the digital twin, with live SCAD reload.

    python tool/view_twin.py            # build assets (cached) + view
    python tool/view_twin.py --watch    # rebuild + reload the scene
                                        # whenever src/platform/*.scad
                                        # changes (edit -> save -> the
                                        # viewer relaunches itself)
    python tool/view_twin.py --spacing 350   # what-if spacing override
                                             # (config.scad stays truth)

The arms hold their neutral pose under the position servos; drag with
double-click + Ctrl to perturb them, like any MuJoCo passive viewer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer

from sim_twin import assets
from sim_twin.params import PLATFORM_DIR, TwinParams
from sim_twin.scene import TwinSim

POLL_S = 0.5


def _scad_state() -> dict[str, float]:
    return {str(p): p.stat().st_mtime for p in sorted(PLATFORM_DIR.glob("*.scad"))}


def _load(spacing_mm: float | None) -> TwinSim:
    assets.build()
    params = TwinParams.load()
    if spacing_mm is not None:
        print(
            f"NOTE: --spacing {spacing_mm} overrides config.scad for this "
            "session only; edit config.scad to make it real"
        )
        params.scad["arm_spacing"] = float(spacing_mm)
        params.validate()
        # spacing lives in the generated URDF too
        from sim_twin import urdf_gen
        from sim_twin.params import BUILD_DIR

        urdf_gen.generate(params, BUILD_DIR / "robot.urdf")
    sim = TwinSim(params)
    sim.reset()
    return sim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="rebuild + reload when src/platform/*.scad changes",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="arm spacing override in mm (what-if only)",
    )
    args = parser.parse_args()

    cam_state = None
    while True:
        state = _scad_state()
        try:
            sim = _load(args.spacing)
        except Exception as exc:
            if not args.watch:
                raise
            # bad SCAD edit mid-save: report, wait for the next change
            print(f"rebuild failed ({exc}); waiting for the next edit ...")
            while _scad_state() == state:
                time.sleep(POLL_S)
            continue

        reload_requested = False
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            if cam_state is not None:
                viewer.cam.lookat[:] = cam_state["lookat"]
                viewer.cam.azimuth = cam_state["azimuth"]
                viewer.cam.elevation = cam_state["elevation"]
                viewer.cam.distance = cam_state["distance"]
            last_poll = time.time()
            while viewer.is_running():
                step_start = time.time()
                sim.step(1)
                viewer.sync()
                if args.watch and time.time() - last_poll > POLL_S:
                    last_poll = time.time()
                    if _scad_state() != state:
                        print("SCAD change detected — rebuilding twin ...")
                        reload_requested = True
                        cam_state = {
                            "lookat": viewer.cam.lookat.copy(),
                            "azimuth": viewer.cam.azimuth,
                            "elevation": viewer.cam.elevation,
                            "distance": viewer.cam.distance,
                        }
                        break
                leftover = sim.model.opt.timestep - (time.time() - step_start)
                if leftover > 0:
                    time.sleep(leftover)

        if not reload_requested:
            return 0


if __name__ == "__main__":
    sys.exit(main())
