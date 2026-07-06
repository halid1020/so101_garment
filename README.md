# Vision-Tactile World Model with Dual SO-101 Garment Manipulation

This repository provides an independent, industrial-grade pipeline for dexterous garment manipulation using dual SO-101 robotic arms equipped with tactile sensors. The framework facilitates Meta-Quest teleoperation, autonomous data collection, and direct integration with Vision-Language-Action (VLA) models via the LeRobot ecosystem.

**Author:** Abudureyimu Halite
**Framework Status:** Active Development
**Dependencies:** LeRobot 0.5.1, Pink IK, pyrealsense2

## 🚀 System Architecture

1. **Hardware Interface:** Direct `sts3215` serial bus control utilizing `lerobot.motors` for SO-101 leader/follower synchronization.
2. **Teleoperation:** Meta-Quest VR tracking mapped to dual-arm Cartesian poses via Pink IK, filtered through adaptive 1€ smoothing.
3. **Data Pipeline:** Synchronized, offline collection of RGB feeds, joint states, and tactile data formatted natively for Hugging Face LeRobot dataset standards.
4. **Digital Twin:** The 3D-printed rig (board, adapters, camera tower, C310 holders) is parametrized in OpenSCAD and mirrored 1:1 in MuJoCo and Isaac Lab from the same config — see [Digital twin & printed rig](#-digital-twin--printed-rig-openscad--mujoco--isaac-lab).

## ⚙️ Installation

1. Clone the repository and navigate to the root directory.
2. Create and activate a Python 3.10+ virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install 'lerobot[dataset,pi]==0.5.1'
   pip install pre-commit
   pre-commit install
   ```
4. For the simulation tools and teleop-method benchmark, additionally:
   ```bash
   pip install mujoco mink matplotlib
   ```
5. For the digital twin / printed-part pipeline, additionally:
   ```bash
   pip install trimesh
   sudo apt install openscad   # any recent version; must be on PATH
   ```

## 🎮 Teleoperation quick start (real Meta Quest)

Five IK methods are available behind the same production pipeline
(One-Euro filtering, grip clutch, handle calibration, armplane orientation
mapping): `production` (the tuned Pink solver), `pink_full`,
`pink_relaxed`, `dls`, `mink`, `scipy_ls`. Benchmark results and method
details: [`markdowns/teleop_benchmark_results.md`](markdowns/teleop_benchmark_results.md)
and [`sim_benchmark/README.md`](sim_benchmark/README.md).

**Prerequisites**

- Meta Quest with the teleop app installed, connected via USB (adb) or on
  the same network (then pass `--ip-address <QUEST_IP>`).
- For the real arms: both SO-101 buses connected and LeRobot-calibrated
  (ports/IDs in `src/conf/robot.yaml`).

**Controls** (both tools): hold **both grips** to activate teleop — at the
first grip of a session point both handles straight down (this calibrates
the handle axes). Triggers close the grippers. Release grips to pause.
`Ctrl+C` exits. On the real tool: `A` enables/disables the arms, `B` moves
to the middle pose, `Y` toggles height lock.

**Step 1 — rehearse in simulation** (headset drives the MuJoCo arms in a
live viewer; no robot hardware needed):

```bash
python tool/quest_sim_teleop.py --method pink_relaxed
python tool/quest_sim_teleop.py --method scipy_ls --ip-address <QUEST_IP>
```

No headset at hand? A scripted mock device exercises the full pipeline:

```bash
python tool/quest_sim_teleop.py --method dls --mock --duration 15
```

**Step 2 — run on the real arms** (same tool as always; the default
`--method production` is the unchanged production solver):

```bash
python tool/meta_quest_teleopration.py                          # production
python tool/meta_quest_teleopration.py --method pink_relaxed    # recommended first
python tool/meta_quest_teleopration.py --method scipy_ls --max-joint-vel 1.5
```

Benchmark methods run through a joint-space rate limiter
(`--max-joint-vel`, default 2 rad/s on the real arms). Based on the
simulation benchmark: try `pink_relaxed` first (best accuracy/smoothness
balance), `scipy_ls` for the tightest tracking; keep a hand near the
power switch the first time any new method crosses the edge of the
workspace.

## 🧪 Teleop method benchmark (simulation)

Offline benchmark of the five methods on mocked hand trajectories
(circles/lines) and a 30-scenario bimanual pick–handover–place task,
with plots and animated GIF comparisons:

```bash
python sim_benchmark/run_benchmark.py --plot outputs/teleop_benchmark_plots
python sim_benchmark/run_handover.py --plot outputs/teleop_benchmark_plots
python sim_benchmark/package_report.py   # shareable zip of all results
```

See [`sim_benchmark/README.md`](sim_benchmark/README.md) for all options.

## 🤖 Digital twin & printed rig (OpenSCAD + MuJoCo + Isaac Lab)

Faithful simulation of the physical rig — the two arms on their printed
adapter plates, the fully 3D-printed perforated board (3 tiles + splice
bars, with clamp wings at both ends), the triangular camera tower (one
corner facing the front), and all three Logitech C310s (both wrist
mounts + tower cradle) with matching camera sensors (`rgb_wrist_left`,
`rgb_wrist_right`, `rgb_scene`).

The printed-part design in
[`src/platform/config.scad`](src/platform/config.scad) is the **single
source of truth**: edit a dimension there and the meshes, URDF, and both
simulators follow. Printing and assembly instructions (hardware
shopping list, bolt stack-ups, camera-holder installation):
[`src/platform/README.md`](src/platform/README.md).

**View it** — pick one:

```bash
python tool/view_twin.py                 # whole rig in the MuJoCo viewer
python tool/view_twin.py --watch         # same, auto-reloads on every .scad save
python tool/view_twin.py --spacing 350   # what-if arm spacing (config.scad untouched)
openscad src/platform/board.scad         # one printed part in the OpenSCAD GUI
```

**Check it** — geometry sanity + what each camera actually sees:

```bash
python -m sim_twin.verify   # -> outputs/twin_verify/*.png
                            #    rgb_wrist_left/right + rgb_scene renders,
                            #    overview.png, and overview_frustum.png
                            #    (tower camera's view cone drawn in 3D)
```

**Print it** — STLs to slice + dimensioned drawings to check against:

```bash
python -m sim_twin.assets --print-parts   # every printable part -> build/print/*.stl
python tool/part_drawings.py              # engineering drawing sheets (multi-view +
                                          # dimensions + hardware callouts) ->
                                          # outputs/drawings/*.png + so101_rig_drawings.pdf
```

**Ship it to Isaac Lab** — self-contained package for a GPU machine:

```bash
python -m sim_twin.assets --package-isaac   # -> build/so101_twin_isaac.zip
```

Unzip there, run `convert_assets.py` (URDF/STL → USD), then
`run_demo.py` — see [`sim_twin/isaac/`](sim_twin/isaac/README.md).

The asset pipeline (`python -m sim_twin.assets`, SCAD → meshes → URDF →
`twin_params.json`, content-hash cached) runs automatically inside the
tools above; call it directly only with `--force` (rebuild everything)
or `--check` (CGAL volume checks, CI-style).
