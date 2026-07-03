# Vision-Tactile World Model with Dual SO-101 Garment Manipulation

This repository provides an independent, industrial-grade pipeline for dexterous garment manipulation using dual SO-101 robotic arms equipped with tactile sensors. The framework facilitates Meta-Quest teleoperation, autonomous data collection, and direct integration with Vision-Language-Action (VLA) models via the LeRobot ecosystem.

**Author:** Abudureyimu Halite
**Framework Status:** Active Development
**Dependencies:** LeRobot 0.5.1, Pink IK, pyrealsense2

## 🚀 System Architecture

1. **Hardware Interface:** Direct `sts3215` serial bus control utilizing `lerobot.motors` for SO-101 leader/follower synchronization.
2. **Teleoperation:** Meta-Quest VR tracking mapped to dual-arm Cartesian poses via Pink IK, filtered through adaptive 1€ smoothing.
3. **Data Pipeline:** Synchronized, offline collection of RGB feeds, joint states, and tactile data formatted natively for Hugging Face LeRobot dataset standards.

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
