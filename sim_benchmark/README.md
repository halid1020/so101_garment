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

## Experiment 2: pick–handover–place (`run_handover.py`)

A bimanual coordination task on top of the same scene and methods: a 2.2 cm
payload cube starts on one arm's side of the table and must end on the
other arm's side, so the picker arm grasps and carries it to a midline
handover point, the placer arm takes it and places it on the target.

- **Scenarios:** N (default 30) seeded samples of (payload position,
  target position), alternating pick side; each is kept only if
  position-only IK confirms every keypose is reachable by the arm that
  must reach it ("physically feasible for the two arms").
- **Mock human:** minimum-jerk segments between keyposes at 8 cm/s, dwell
  at grasp/transfer/release, orientation latched at start (clutch
  semantics). After the placer acquires the object, its targets are
  shifted by the measured EE↔object offset — a human aligns the *object*
  over the target, not their hand.
- **Mock grasp:** kinematic attach when the gripper is commanded closed
  within 4.5 cm of the payload (stands in for grasp physics *and* the
  visual servoing a human does; the script is open-loop). Released
  payloads fall under physics and must land on the target.
- **Success:** payload within 2 cm (XY) of the target after release +
  settle. Also reported: handover rate, place error, tracking error.

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

# pick-handover-place: full 30-scenario sweep
venv/bin/python sim_benchmark/run_handover.py \
    --save outputs/teleop_benchmark/handover.json \
    --plot outputs/teleop_benchmark_plots

# watch one handover scenario live
venv/bin/python sim_benchmark/run_handover.py \
    --view --methods scipy_ls --scenarios 0
```

## Driving the methods with the real Meta Quest

`sim_benchmark/method_adapter.py` wraps any registered method in the
PinkIKSolver interface the production Quest pipeline expects, so all five
methods run behind the *unchanged* production stack (One-Euro filtering,
grip clutch, handle-axis calibration, armplane orientation mapping), with
a joint-space rate limiter for safety.

```bash
# headset -> MuJoCo sim (rehearse here first; viewer shows the arms)
venv/bin/python tool/quest_sim_teleop.py --method scipy_ls
venv/bin/python tool/quest_sim_teleop.py --method dls --ip-address <QUEST_IP>

# no headset: scripted mock device, pipeline smoke test
venv/bin/python tool/quest_sim_teleop.py --method mink --mock --duration 15

# headset -> REAL dual arms (same tool as always; default is the
# unchanged production solver)
venv/bin/python tool/meta_quest_teleopration.py --method pink_relaxed
venv/bin/python tool/meta_quest_teleopration.py            # production
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
