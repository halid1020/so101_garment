# Meta-Quest Teleop Method Benchmark — Simulation Results

**Branch:** `teleop-benchmark` · **Date:** 2026-07-03 · **Code:** `sim_benchmark/`

Goal: verify candidate Meta-Quest → dual SO-101 teleoperation pipelines in
MuJoCo with mocked hand movements, *before* running the real headset on the
real robots.

---

## 1. Experiment 1: tracking benchmark — design

### 1.1 Simulation environment

- **Simulator:** MuJoCo 3.10 (chosen over Isaac Lab: pip-installable,
  loads our existing URDF directly, milliseconds to reset).
- **Robot:** the repo's dual-arm URDF (`src/so101_dual_description/robot.urdf`),
  two 5-DoF SO-101 arms (grippers locked for IK) with bases 0.30 m apart
  (y = ±0.15 m), mounted on a table whose surface is the z = 0 plane.
- **Servo model:** the URDF carries no joint dynamics, so STS3215-like
  values were added (damping 0.60, armature 0.028, frictionloss 0.05 —
  matching the official SO-ARM100 MJCF) plus position actuators
  (kp = 25, kv = 1.5). Without these the arms oscillate unboundedly.
- **Collisions disabled** (tracking benchmark, not contact benchmark).
- **Rates:** physics 500 Hz (2 ms), teleop/IK layer 50 Hz — matching the
  real pipeline's `CONTROLLER_DATA_RATE`.
- Green sites mark the measured EE frames; red/blue mocap spheres show the
  commanded left/right targets in the viewer.

### 1.2 Mocked Quest input

The real pipeline latches the controller pose and the robot EE pose at
grip-press, then applies controller *deltas* to the latched EE pose
(clutch). The mock reproduces exactly that: each trajectory is a delta
curve applied to the EE pose captured at episode start. Orientation targets
are held at the latched orientation; both hands get mirrored deltas
(left hand's y negated), as a human drawing symmetric figures would.

Trajectories (smooth ramp-in, 2 cycles each):

| family | parameters | duration |
|---|---|---|
| circle (horizontal plane, at latched EE height) | radius 3 / 5 / 8 cm, period 6 s | 12 s |
| line stroke on table plane (back-and-forth, cosine profile) | direction 0° / 45° / 90° / 135°, length 12 cm, period 4 s | 8 s |

Directions: 0° = +x (away from the robots), 90° = +y (sideways).
Note the 0° line and the 8 cm circle intentionally push the EE toward the
edge of the SO-101's reachable workspace — they act as stress cases.

### 1.3 Methods under test

All methods receive identical absolute EE pose targets and output the ten
arm-joint commands; they differ only in the IK layer. Each is modeled on a
published/community solution for Quest → SO-101 teleop:

| method | strategy | modeled on |
|---|---|---|
| `pink_full` | Pink QP, full 6D pose task (pos cost 1.0 / ori 0.75, production gains) | our production pipeline (`tool/meta_quest_teleopration.py`) |
| `pink_relaxed` | same QP, orientation cost 0.05 | TeleopXR's "relaxed 5-DoF IK" for SO-101 |
| `dls` | damped least-squares differential IK on the MuJoCo Jacobian, 3 rad/s clamp | Dream-Machines vr-teleop-kit |
| `mink` | weighted-task QP on the MuJoCo model (pos 1.0 / ori 0.75, 3 rad/s velocity limit) | mink-based LeRobot teleop stacks |
| `scipy_ls` | bounded nonlinear least-squares absolute pose IK per tick, warm-started, ori weight 0.1 | telegrip-style position-first IK |

### 1.4 Metrics

- **ik_err** — ‖FK(commanded q) − target‖: pure IK quality, no physics.
- **err_mean / err_p95** — ‖measured EE − target‖ after stepping physics:
  includes servo lag and gravity sag (~5–9 mm floor at these gains).
- **jerk_rms** — RMS jerk of the joint commands (rad/s³): smoothness.
- **qd_max** — peak commanded joint velocity (rad/s): safety.
- **lim_margin** — worst-case distance to a joint limit (deg).
- **solve** — mean per-tick compute time (ms); budget at 50 Hz is 20 ms.

Errors pool both arms; each (method, trajectory) episode starts from the
same settled neutral pose `[0, −10, 20, 25, 0]°` per arm.

---

## 2. Experiment 1: results

### 2.1 Summary (mean measured error across all 7 trajectories)

| method | mean err (mm) | verdict |
|---|---|---|
| `scipy_ls` | **12.0** | most accurate; jerk spikes at workspace edge — needs rate limiting before hardware |
| `pink_relaxed` | **20.3** | best accuracy/smoothness balance; drop-in change to production config |
| `dls` | 30.3 | cheapest (0.10 ms/tick); mid-pack accuracy |
| `mink` | 33.7 | comparable to dls after adding a velocity limit |
| `pink_full` (production) | 41.0 | orientation task fights position on 5-DoF arms |

### 2.2 Per-trajectory tables

ik_err / err_mean / err_p95 in mm; jerk in rad/s³; qd in rad/s; lim in deg; solve in ms.

#### circle_r3cm

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 18.00 | 21.22 | 31.55 | 0.18 | 0.27 | 52.4 | 0.19 |
| pink_relaxed | 6.79 | 12.45 | 15.51 | 0.20 | 0.21 | 53.0 | 0.18 |
| dls | 12.74 | 16.36 | 23.74 | 0.36 | 0.34 | 51.0 | 0.10 |
| mink | 15.31 | 18.54 | 27.34 | 0.40 | 0.35 | 51.0 | 0.17 |
| scipy_ls | 1.53 | 9.18 | 10.62 | 0.48 | 0.34 | 51.0 | 2.08 |

#### circle_r5cm

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 30.81 | 33.71 | 51.67 | 0.33 | 0.48 | 39.0 | 0.19 |
| pink_relaxed | 11.83 | 17.38 | 23.09 | 0.35 | 0.36 | 39.1 | 0.19 |
| dls | 22.18 | 24.77 | 38.56 | 0.61 | 0.58 | 34.6 | 0.11 |
| mink | 26.11 | 28.35 | 44.27 | 0.68 | 0.60 | 34.5 | 0.17 |
| scipy_ls | 2.95 | 9.55 | 11.38 | 0.76 | 0.55 | 34.5 | 2.32 |

#### circle_r8cm (workspace stress case)

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 52.16 | 55.35 | 85.92 | 0.67 | 0.86 | 6.7 | 0.18 |
| pink_relaxed | 20.40 | 27.04 | 36.96 | 0.63 | 0.62 | 14.8 | 0.18 |
| dls | 37.71 | 39.77 | 64.28 | 6.24 | 0.96 | 0.0 | 0.10 |
| mink | 43.11 | 44.71 | 71.79 | 7.11 | 1.01 | 0.0 | 0.16 |
| scipy_ls | 6.05 | 11.68 | 15.80 | 10.68 | 0.94 | 0.0 | 2.56 |

#### line_0deg (+x, partially beyond reach — workspace stress case)

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 31.12 | 37.26 | 59.80 | 0.81 | 0.68 | 51.5 | 0.19 |
| pink_relaxed | 20.25 | 29.18 | 41.75 | 1.04 | 0.59 | 54.6 | 0.18 |
| dls | 18.55 | 26.92 | 59.55 | 49.77 | 3.00 | 23.0 | 0.10 |
| mink | 7.40 | 19.27 | 36.92 | 219.94 | 3.00 | 21.6 | 0.17 |
| scipy_ls | 2.52 | 16.03 | 24.69 | 440.87 | 6.53 | 23.0 | 2.99 |

#### line_45deg

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 40.99 | 44.10 | 75.11 | 0.63 | 0.51 | 58.6 | 0.18 |
| pink_relaxed | 15.11 | 24.11 | 33.66 | 0.83 | 0.47 | 59.6 | 0.18 |
| dls | 23.69 | 28.96 | 49.58 | 5.14 | 1.84 | 37.0 | 0.10 |
| mink | 30.96 | 35.34 | 63.20 | 7.65 | 1.23 | 30.6 | 0.16 |
| scipy_ls | 2.01 | 14.18 | 17.66 | 326.01 | 5.08 | 22.9 | 2.56 |

#### line_90deg (pure sideways)

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 47.72 | 50.16 | 97.05 | 0.16 | 0.08 | 69.4 | 0.19 |
| pink_relaxed | 5.32 | 14.86 | 19.58 | 0.59 | 0.37 | 67.8 | 0.18 |
| dls | 37.81 | 40.85 | 77.22 | 0.40 | 0.15 | 68.5 | 0.10 |
| mink | 47.57 | 50.24 | 96.59 | 0.24 | 0.09 | 69.4 | 0.16 |
| scipy_ls | 3.71 | 12.43 | 16.19 | 1.18 | 0.37 | 64.3 | 2.32 |

#### line_135deg

| method | ik_err | err_mean | err_p95 | jerk_rms | qd_max | lim_margin | solve |
|---|---|---|---|---|---|---|---|
| pink_full | 41.97 | 45.29 | 76.46 | 0.69 | 0.50 | 47.4 | 0.18 |
| pink_relaxed | 10.90 | 17.24 | 25.40 | 0.75 | 0.38 | 49.0 | 0.19 |
| dls | 31.31 | 34.75 | 65.82 | 1.51 | 0.72 | 45.4 | 0.10 |
| mink | 36.48 | 39.59 | 75.35 | 1.66 | 0.74 | 44.4 | 0.15 |
| scipy_ls | 4.16 | 11.16 | 13.83 | 1.97 | 0.69 | 50.2 | 2.17 |

Top-view path plots (target vs IK command vs measured, per method and arm):
`outputs/teleop_benchmark_plots/*.png`. Raw metrics:
`outputs/teleop_benchmark/metrics.json`.

---

## 3. Experiment 2: bimanual pick–handover–place

### 3.1 Design

A coordination task requiring both arms: a 2.2 cm payload cube starts on
one arm's side of the table and must end up on the *other* arm's side, so
the picker arm grasps it and carries it to a midline handover point
(x = 0.26, y = 0), the placer arm takes it there, carries it to the target
and releases it. The mocked human motion is minimum-jerk between keyposes
at 8 cm/s with 0.5 s dwells around grasp/transfer/release; orientation is
held at the latched pose (clutch semantics, as in Experiment 1).

- **Scenarios:** 30 seeded samples (`seed=0`) of payload position
  (picker's side, x ∈ [0.22, 0.33], |y| ∈ [0.08, 0.20]) and target
  position (mirrored range on the placer's side), alternating which arm
  picks (15 left / 15 right). Every scenario is verified **physically
  feasible** before use: position-only IK must reach all six keyposes
  (grasp, hover, handover for both arms, place, retreat) within 6 mm for
  the arm that must reach them.
- **Mock grasp:** kinematic attach when a gripper is commanded closed
  within 4.5 cm of the payload; released payloads fall under physics and
  must *land* on the target. The generous radius and the object-alignment
  correction (after the placer acquires the payload its targets are
  shifted by the measured EE↔object offset) stand in for the visual
  servoing a human teleoperator performs — the script is open-loop.
- **Success:** payload within 2 cm (XY) of the target after release and a
  1.2 s settle.

### 3.2 Results (30 scenarios × 5 methods)

| method | success | handover | place err mean (mm) | place err p95 (mm) | track err mean (mm) | solve (ms) |
|---|---|---|---|---|---|---|
| `scipy_ls` | **30/30 (100%)** | 100% | 12.7 | 16.5 | 11.7 | 1.55 |
| `pink_relaxed` | **29/30 (96.7%)** | 100% | 16.4 | 19.7 | 13.3 | 0.19 |
| `pink_full` (production) | 0/30 | 0% | 278.2 | 351.0 | 65.5 | 0.19 |
| `dls` | 0/30 | 0% | 254.0 | 322.8 | 54.6 | 0.11 |
| `mink` | 0/30 | 0% | 276.2 | 350.4 | 62.7 | 0.18 |

The failure mode of the full-orientation methods is total: their 55–65 mm
tracking error exceeds the 4.5 cm grasp radius, so the **pick never
succeeds** — the payload simply never moves (place error ≈ the raw
pick-to-target distance). This is Experiment 1's orientation-vs-position
finding expressed as task outcome instead of tracking error: handover
scenarios inherently demand sideways (y) reaches across the midline,
exactly the direction a 5-DoF arm cannot serve while holding full 6D
orientation.

Visualizations:

- `outputs/teleop_benchmark_plots/handover_success_map.png` — pick→target
  arrows for all 30 scenarios, per method, green/red by success.
- `outputs/teleop_benchmark_plots/handover_payload_paths.png` — payload
  trajectory (top + side view) for scenario 0, all methods.
- `outputs/teleop_benchmark_gifs/*.gif` — animated renders of every
  (trajectory, method) episode of Experiment 1 for visual comparison
  (`<trajectory>_<method>.gif`).
- Raw per-episode results: `outputs/teleop_benchmark/handover.json`.

Reproduce with:

```bash
venv/bin/python sim_benchmark/run_handover.py \
    --save outputs/teleop_benchmark/handover.json \
    --plot outputs/teleop_benchmark_plots

# watch a single handover live
venv/bin/python sim_benchmark/run_handover.py --view --methods scipy_ls --scenarios 0

# regenerate the GIFs
venv/bin/python sim_benchmark/run_benchmark.py --gif outputs/teleop_benchmark_gifs
```

---

## 4. Findings

1. **Full 6D orientation tracking is over-constrained on a 5-DoF arm.**
   The SO-101 has no wrist yaw, so any sideways (y) translation requires
   shoulder pan, which necessarily rotates the EE. Methods that fight this
   (`pink_full`, `mink`, `dls` at ori weight 0.5) squash circles into
   ellipses and hit 30–50 mm IK error on sideways strokes — worst exactly
   on `line_90deg`. Position-first methods (`scipy_ls` ik_err 2–6 mm,
   `pink_relaxed` 5–20 mm) track faithfully.
   *Caveat:* the production pipeline partially works around this by
   rotating the orientation target with the arm azimuth
   (`hand_to_gripper_orientation_armplane`); the benchmark's fixed
   orientation targets deliberately isolate the raw tradeoff.

2. **Workspace-edge behavior separates the methods on safety.** On
   `line_0deg` (target partially unreachable) and `circle_r8cm`:
   - `mink` without a velocity limit commanded **80 rad/s** joint
     velocities (jerk_rms 9037). Adding `mink.VelocityLimit` (3 rad/s)
     fixed it — QP IK must never run on hardware without one.
   - `scipy_ls` stays accurate but jumps between solutions
     (jerk_rms 441 at qd 6.5 rad/s) — it needs a joint-space rate limiter
     before hardware use.
   - The Pink variants degrade gracefully (velocity-damped QP), at the
     cost of larger steady error.

3. **Compute cost is a non-issue at 50 Hz** for the differential methods
   (0.1–0.2 ms/tick). `scipy_ls` costs ~2.5 ms/tick — still 8× under
   budget, but the only method where cost would grow with tighter
   tolerances.

4. **Physics adds a ~5–9 mm error floor** (servo lag + gravity sag at
   kp = 25) on top of IK error — visible as the gap between `ik_err` and
   `err_mean` for the accurate methods.

5. **The handover task turns the tracking gap into a pass/fail cliff.**
   Methods above ~45 mm tracking error score 0/30 (cannot even pick);
   methods below ~15 mm score 29–30/30. Task-level success is far less
   forgiving than average tracking error suggests — worth remembering
   when reading Experiment 1's tables.

## 5. Recommendation

For garment-folding strokes on the table plane:

- **First choice: `pink_relaxed`** — a one-line config change to the
  production stack (orientation cost 0.75 → ~0.05), 2× accuracy gain,
  smooth everywhere, inherits the existing safety behavior, and 29/30 on
  the bimanual handover task.
- **If tighter tracking is needed:** `scipy_ls` (30/30 on handover, best
  place accuracy) wrapped in a joint-space rate limiter (clamp Δq per
  tick), or a hybrid: `scipy_ls` targets fed through the Pink
  velocity-damped QP.
- Keep a **velocity limit in any QP-based method** and a workspace clamp
  on targets before they reach the IK layer.
- **Do not deploy the current production config (`pink_full`) for
  bimanual handovers** — it scored 0/30 in simulation.

## 6. Reproducing

```bash
# full sweep + report table
venv/bin/python sim_benchmark/run_benchmark.py \
    --save outputs/teleop_benchmark/metrics.json \
    --plot outputs/teleop_benchmark_plots

# watch a single method live in the MuJoCo viewer
venv/bin/python sim_benchmark/run_benchmark.py \
    --view --methods scipy_ls --trajectories circle_r5cm
```
