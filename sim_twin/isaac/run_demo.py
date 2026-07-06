"""Spawn the twin in Isaac Lab, hold neutral, dump the camera views.

Run INSIDE the Isaac Lab environment (after convert_assets.py):

    ./isaaclab.sh -p run_demo.py --headless   # 300 steps, PNGs to out/
    ./isaaclab.sh -p run_demo.py              # interactive viewport

Verification checklist while it runs (mirrors the MuJoCo twin):
  * arm bases 0.0444 m above the board-top plane, arm_spacing apart
  * rgb_scene sees the workspace with both wrists at the frame top
  * rgb_wrist_* look down each gripper at the grasp point
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from twin_scene import TwinSceneCfg  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.002, device=args.device)
    )
    scene = InteractiveScene(TwinSceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()

    robot = scene["robot"]
    neutral = robot.data.default_joint_pos.clone()
    decimation = 10  # 50 Hz targets over 2 ms physics

    for step in range(args.steps):
        if step % decimation == 0:
            robot.set_joint_position_target(neutral)
            scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    OUT.mkdir(exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for cam in ("rgb_scene", "rgb_wrist_left", "rgb_wrist_right"):
        rgb = scene[cam].data.output["rgb"][0]
        array = rgb.to(torch.uint8).cpu().numpy()[..., :3]
        if Image is not None:
            Image.fromarray(array).save(OUT / f"{cam}.png")
        else:
            import numpy as np

            np.save(OUT / f"{cam}.npy", array)
    print(f"camera dumps in {OUT}")

    base = robot.data.body_pos_w[0, robot.data.body_names.index("left_base_link")]
    print(f"left_base_link world position: {base.tolist()}")


if __name__ == "__main__":
    main()
    app.close()
