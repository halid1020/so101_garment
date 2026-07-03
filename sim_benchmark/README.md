# Teleop Method Benchmark (Simulation)

Verifies candidate Meta-Quest → dual SO-101 teleoperation pipelines in
MuJoCo with **mocked hand movements** before running the real headset on the
real robots.

## What it does

- Loads the dual-arm URDF (`src/so101_dual_description/robot.urdf`) into
  MuJoCo (patched in-memory with servo dynamics matching the STS3215),
  adds a table, position actuators, EE sites, and mocap spheres showing the
  commanded targets.
- Mocks the Quest clutch behavior: controller *deltas* relative to the
  grip-press pose are applied to the latched EE pose. The mock emits
  - **circles** of radius 3/5/8 cm in the horizontal plane, and
  - **line strokes** on the table along 0°/45°/90°/135°,
  with smooth ramp-in, mirrored across hands.
- Sweeps the trajectories over 5 teleop/IK methods and reports tracking
  metrics per (trajectory, method).

## Methods

| name | strategy | origin |
|---|---|---|
| `pink_full` | Pink QP, full 6D pose task (pos 1.0 / ori 0.75) | production pipeline (`tool/meta_quest_teleopration.py`) |
| `pink_relaxed` | Pink QP, relaxed orientation (pos 1.0 / ori 0.05) | TeleopXR's "relaxed 5-DoF IK" for SO-101 |
| `dls` | damped least-squares differential IK on the MuJoCo Jacobian | Dream-Machines vr-teleop-kit |
| `mink` | weighted-task QP IK on the MuJoCo model | mink-based LeRobot teleop stacks |
| `scipy_ls` | bounded nonlinear least-squares absolute pose IK, warm-started | telegrip-style position-first IK |

All methods consume the same absolute EE targets and emit joint commands in
a fixed order (5 left + 5 right joints), so they are compared purely on
target-following quality.

## Metrics

- `ik_err` — FK of the commanded joints vs target: pure IK quality.
- `err_mean/p95/max` — measured (physics-stepped) EE vs target: includes
  servo lag and gravity sag.
- `jerk_rms`, `qd_max` — command smoothness / peak joint velocity.
- `lim_margin` — worst-case distance to a joint limit (deg).
- `solve` — per-tick compute cost (ms).

## Usage

```bash
# full sweep, print table
venv/bin/python sim_benchmark/run_benchmark.py

# save metrics + top-view path plots (visual verification, headless)
venv/bin/python sim_benchmark/run_benchmark.py \
    --save outputs/teleop_benchmark/metrics.json \
    --plot outputs/teleop_benchmark_plots

# watch one method live in the MuJoCo viewer
venv/bin/python sim_benchmark/run_benchmark.py \
    --view --methods pink_relaxed --trajectories circle_r5cm
```

## Key findings so far

1. On a 5-DoF arm, **holding full 6D orientation while translating sideways
   is kinematically infeasible** (no wrist yaw to compensate shoulder pan):
   full-orientation methods (`pink_full`, `mink`, `dls`) squash sideways
   motion into an ellipse and rack up 30–50 mm IK error, while
   position-first methods track at 2–6 mm (`scipy_ls`) / 5–20 mm
   (`pink_relaxed`).
2. At the **workspace edge** (the 0° line stroke exceeds the arm's reach)
   QP IK without a velocity limit commands unbounded joint speeds — mink
   needed an explicit `VelocityLimit` (now included) to stay below
   3 rad/s. `scipy_ls` stays accurate but jumps between solutions there
   (high jerk); it would need rate limiting before running on hardware.

Caveat: the production pipeline works around this by rotating the
orientation *target* with the arm azimuth
(`hand_to_gripper_orientation_armplane` in `src/common/utils.py`), which
this benchmark's fixed-orientation targets deliberately do not do — the
sweep isolates the raw orientation-vs-position tradeoff of each IK layer.
The practical implication holds either way: for table-plane garment
strokes, down-weight or restructure orientation tracking rather than
fighting it.
