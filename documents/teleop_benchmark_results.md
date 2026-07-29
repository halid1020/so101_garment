# Meta-Quest Teleop Method Benchmark — Simulation Results

**First run:** 2026-07-03 (branch `teleop-benchmark`) · **Last updated:** 2026-07-15 (branch `teleop-feel-fixes`) · **Code:** `src/sim_benchmark/`

Goal: verify candidate Meta-Quest → dual SO-101 teleoperation pipelines in
MuJoCo with mocked hand movements, *before* running the real headset on the
real robots.


[TODO: also add the results of the telegrip method]
[TODO: in simulation visualisation, please also draw the axsi for ee pos, target ee pos, teleop world pos, and robot world pos]
[TODO: my wrist canno roll 360, but I would like to control the gripper to roll through multiple wrist movement by gripping and realsing the teleop holder. so that I can adjust the gripper orientation to grap and hand-over objects.]
---

## 1. Experiment 1: tracking benchmark — design

### 1.1 Simulation environment

- **Simulator:** MuJoCo 3.10 (chosen over Isaac Lab: pip-installable,
  loads our existing URDF directly, milliseconds to reset).
- **Robot:** the repo's dual-arm URDF (`src/so101_dual_description/robot.urdf`),
  two 5-DoF SO-101 arms (grippers locked for IK) with bases 0.30 m apart
  (y = ±0.15 m), mounted on a table whose surface is the z = 0 plane. [TODO: use the platform rig setup.]
- **Servo model:** the URDF carries no joint dynamics, so STS3215-like
  values were added (damping 0.60, armature 0.028, frictionloss 0.05 —
  matching the official SO-ARM100 MJCF) plus position actuators
  (kp = 25, kv = 1.5). Without these the arms oscillate unboundedly.
- **Collisions disabled** (tracking benchmark, not contact benchmark). [TODO: enable all the Collision]
- **Rates:** physics 500 Hz (2 ms), teleop/IK layer 50 Hz — matching the
  real pipeline's `CONTROLLER_DATA_RATE`. [TODO: these hyper-parameters should be saved in a yaml file with clear explanation]
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
| `telegrip` | split IK: analytic wrist (elevation/roll → wrist_flex/wrist_roll) + 3-joint position-only DLS | faithful port of DipFlip/telegrip's actual algorithm (added later; see §8) |

### 1.4 Metrics

Each metric isolates one link of the chain *target → IK command → servo →
measured pose*, so together they separate "the solver is wrong" from "the
solver is right but the robot cannot follow" from "the command is
unfollowable by design":

- **ik_err** (mm) — ‖FK(commanded q) − target‖ per tick, averaged over the
  episode: run the *commanded* joints through forward kinematics and
  measure the distance to the commanded target. No physics is involved,
  so this is pure solver quality — how close the IK's own answer lands.
- **err_mean / err_p95** (mm) — ‖measured EE − target‖ after stepping
  physics: the mean and 95th percentile over all ticks of the distance
  between the *simulated* end-effector and the target. This adds what
  ik_err excludes — servo lag, gravity sag, and dynamics (~5–9 mm floor
  at these gains even for a perfect solver). The p95 catches transient
  spikes that a mean hides. The difference (err − ik_err) is therefore
  the tracking cost of the *hardware*, not the solver.
- **jerk_rms** (rad/s³) — root-mean-square of the third time-derivative
  of the joint *commands*, computed by finite differences over the
  command sequence and pooled across joints. It measures command
  smoothness: high jerk means the solver asks the servos for abrupt
  acceleration changes, which excites oscillation and wears gears even
  when the position error looks fine.
- **qd_max** (rad/s) — the peak commanded joint velocity over the
  episode: a safety metric; values at the rate-limiter clamp mean the
  method is being saturated.
- **lim_margin** (deg) — the worst-case distance of any commanded joint
  from its position limit: small margins warn that a method solves by
  parking joints at their limits.
- **solve** (ms) — mean per-tick compute time; the budget at 50 Hz is
  20 ms.

Errors pool both arms; each (method, trajectory) episode starts from the
same settled neutral pose `[0, −10, 20, 25, 0]°` per arm.

---

## 2. Experiment 1: results

### 2.1 Summary (mean measured error across all 7 trajectories)

| method | mean err (mm) | verdict |
|---|---|---|
| `scipy_ls` | **12.0** | most accurate; jerk spikes at workspace edge — needs rate limiting before hardware |
| `telegrip` | **13.1** | added later (§8): near-`scipy_ls` accuracy at 13× lower solve cost (0.19 ms); jerky at the 3 rad/s clamp |
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

[TODO: use the true contact model for this test. Do not use anchoring. Instead of grasping cubes try to grasp a longer rectangular cube, as the wrist of the robot hand is not flexible.]

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

Column definitions: **success** — episodes where the payload ends within
2 cm (XY) of the target after release and settling; **handover** —
episodes where the receiving gripper acquired the payload at the
mid-air exchange (a success prerequisite); **place err mean/p95** (mm) —
distance from the payload's final resting position to the target centre,
mean and 95th percentile over the 30 scenarios (measures end-to-end
placement accuracy, payload physics included); **track err mean** (mm) —
the same measured-EE-vs-target tracking error as §1.4, averaged over the
episode (measures how well the arm followed the script, independent of
what the payload did); **solve** (ms) — mean per-tick compute time.

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
  (`<trajectory>_<method>.gif`), plus one handover trial per method
  (`handover_s00_<method>.gif`).
- Raw per-episode results: `outputs/teleop_benchmark/handover.json`.

### 3.3 Head-to-head GIF gallery

[TODO: update the visusaliseation and results.]

Panel order in every comparison strip, left → right:
**pink_full · pink_relaxed · dls · mink · scipy_ls**.
(Files live in `outputs/teleop_benchmark_gifs/`; regenerate with the
commands in §3.2 and stitch with `src/sim_benchmark/combine_gifs.py`.)

**Pick–handover–place, scenario 0** — watch `pink_full`, `dls` and `mink`
descend *next to* the cube but never reach it (tracking error > grasp
radius), while `pink_relaxed` and `scipy_ls` complete the pick, midline
transfer, and placement:

![handover head-to-head](../outputs/teleop_benchmark_gifs/compare_handover_s00.gif)

**Circle, r = 5 cm** — the full-orientation methods trace visibly
squashed ellipses; the position-first methods draw round circles:

![circle head-to-head](../outputs/teleop_benchmark_gifs/compare_circle_r5cm.gif)

**Line, 90° (pure sideways)** — the starkest case: `pink_full` and `mink`
barely move sideways at all:

![line head-to-head](../outputs/teleop_benchmark_gifs/compare_line_90deg.gif)

Single-method trials for the handover task:

| pink_full | pink_relaxed | scipy_ls |
|---|---|---|
| <img src="../outputs/teleop_benchmark_gifs/handover_s00_pink_full.gif" width="260"/> | <img src="../outputs/teleop_benchmark_gifs/handover_s00_pink_relaxed.gif" width="260"/> | <img src="../outputs/teleop_benchmark_gifs/handover_s00_scipy_ls.gif" width="260"/> |

| dls | mink |
|---|---|
| <img src="../outputs/teleop_benchmark_gifs/handover_s00_dls.gif" width="260"/> | <img src="../outputs/teleop_benchmark_gifs/handover_s00_mink.gif" width="260"/> |

Reproduce with:

```bash
venv/bin/python src/sim_benchmark/run_handover.py \
    --save outputs/teleop_benchmark/handover.json \
    --plot outputs/teleop_benchmark_plots

# watch a single handover live
venv/bin/python src/sim_benchmark/run_handover.py --view --methods scipy_ls --scenarios 0

# regenerate the GIFs (tracking suite + one handover trial per method)
venv/bin/python src/sim_benchmark/run_benchmark.py --gif outputs/teleop_benchmark_gifs
venv/bin/python src/sim_benchmark/run_handover.py --scenarios 0 --gif outputs/teleop_benchmark_gifs

# stitch a head-to-head strip (any set of episode GIFs)
venv/bin/python src/sim_benchmark/combine_gifs.py \
    outputs/teleop_benchmark_gifs/handover_s00_{pink_full,pink_relaxed,dls,mink,scipy_ls}.gif \
    -o outputs/teleop_benchmark_gifs/compare_handover_s00.gif --panel-width 320
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
  on targets before they reach the IK layer. → **Now implemented**: an
  analytic workspace envelope with four selectable out-of-envelope
  policies (`--oob-mode`, `WORKSPACE_OOB_MODE` in `configs.py`) sits in
  the IK thread ahead of every solver; benchmark in §9.
- **Do not deploy the current production config (`pink_full`) for
  bimanual handovers** — it scored 0/30 in simulation.

## 6. Trying the methods with the real Meta Quest

All five methods are wired into the production Quest pipeline via
`src/sim_benchmark/method_adapter.py` (PinkIKSolver-compatible facade + a
joint-space rate limiter, default 2–3 rad/s). Two entry points:

```bash
# 1. Headset -> MuJoCo sim: full production stack (One-Euro filter, grip
#    clutch, handle calibration, armplane orientation) driving the
#    simulated arms in a live viewer. Rehearse every method here first.
venv/bin/python tool/quest_sim_teleop.py --method scipy_ls
#    (--mock replaces the headset with a scripted device for smoke tests)

# 2. Headset -> REAL arms: the standard teleop tool, IK layer selectable.
#    Default remains the unchanged production solver.
venv/bin/python tool/meta_quest_teleopration.py --method pink_relaxed
```

Pipeline smoke test (mock device, 10 s circles, sim tracking error):

| method | mean (mm) | p95 (mm) |
|---|---|---|
| scipy_ls | 5.9 | 9.3 |
| mink | 6.1 | 9.7 |
| dls | 6.2 | 10.1 |
| pink_relaxed | 8.2 | 13.0 |
| pink_full | 11.0 | 19.8 |

Note how much closer the methods are here than in the open-loop benchmark:
the production pipeline's **armplane orientation mapping** builds
orientation targets that are reachable by construction, largely neutralizing
the 5-DoF orientation conflict that dominated Experiments 1–2. The
open-loop benchmark ranks the raw IK layers; with the production mapping on
top, method choice becomes a smoothness/robustness trade rather than a
pass/fail one — which is exactly what to evaluate hands-on with the
headset.

## 7. Reproducing

```bash
# full sweep + report table
venv/bin/python src/sim_benchmark/run_benchmark.py \
    --save outputs/teleop_benchmark/metrics.json \
    --plot outputs/teleop_benchmark_plots

# watch a single method live in the MuJoCo viewer
venv/bin/python src/sim_benchmark/run_benchmark.py \
    --view --methods scipy_ls --trajectories circle_r5cm
```

---

## 8. Experiment 3: wrist-agility benchmark

Motivation: the reported "wrist is not agile" feel of the production
teleop. Seven new trajectories hold the EE position at the latch point
while oscillating the target orientation: roll ±45° and flex ±30° at
0.5/1/2 Hz, plus a combined slow-circle + 1 Hz-roll case
(`src/sim_benchmark/mock_quest.py::default_suite`). Two new metrics
(`src/sim_benchmark/metrics.py`): **ori err** — geodesic angle between target
and measured EE rotation — and **roll lag** — the cross-correlation peak
between commanded and measured roll about the tip axis (flex tables show
"—": no roll content). The suite also motivated a sixth method,
**`telegrip`** (`src/sim_benchmark/methods/telegrip_split.py`): a faithful
port of DipFlip/telegrip's split IK — the wrist joints are set
*analytically* from the target orientation (tip elevation → wrist_flex,
roll about tip → wrist_roll, via FD-calibrated affine models), and only
the 3 proximal joints run position-DLS.

### 8.1 Results — 1 Hz roll (the headline case)

| method | err (mm) | ori mean/p95 (deg) | lag (ms) | jerk | qd max | solve (ms) |
|---|---|---|---|---|---|---|
| `scipy_ls` | 9.9 | 11.9 / 19.6 | 80 | 61 | 4.92 | 2.45 |
| `pink_full` | 10.7 | 16.9 / 28.1 | 100 | 58 | 4.72 | 0.21 |
| `telegrip` | 9.9 | 19.5 / 33.9 | 120 | 373 | 3.00 | 0.19 |
| `mink` | 24.7 | 20.7 / 34.1 | 140 | 241 | 3.00 | 0.21 |
| `dls` | 9.9 | 22.0 / 37.4 | 140 | 211 | 3.00 | 0.11 |
| `pink_relaxed` | 9.8 | 27.8 / 46.4 | 300 | 7 | 0.61 | 0.22 |

Full tables for all seven trajectories are in
`outputs/teleop_bench_full.json` (`export_latex_tables.py` renders them
into `documents/paper/teleoperation/tables/`).

### 8.2 Findings

1. **The complaint is real and quantified: `pink_relaxed` is the wrist
   bottleneck.** Its low orientation cost (0.05) — exactly what makes its
   *position* tracking good — caps wrist joint velocity at ~0.6 rad/s, so
   a ±45° 1 Hz roll lags by ~300 ms (a third of the period) with 28° mean
   error. The current production config `pink_full` is actually fine here
   (100 ms), confirming the wrist/position trade is a single scalar knob
   in the QP cost.
2. **Direct wrist mapping works.** `telegrip` cuts lag to 120 ms and
   error to 19.5° while keeping `pink_relaxed`-class position accuracy
   (§2.1: 13.1 mm) — but slams the 3 rad/s clamp (jerk 373 rad/s³): the
   ±45°@1 Hz profile peaks at 4.9 rad/s, beyond the clamp by design.
3. **`scipy_ls` again leads raw tracking** (80 ms lag, 11.9°) but needs
   its usual rate limiter, and costs 13× more compute than `telegrip`
   (2.45 vs 0.19 ms/tick).
4. At 2 Hz every method saturates — that regime needs faster servos, not
   better IK.

Reproduce: `venv/bin/python src/sim_benchmark/run_benchmark.py --trajectories
wrist_roll_f1hz wrist_flex_f1hz wrist_combo_f1hz`.

---

## 9. Experiment 4: out-of-envelope target handling

The workspace clamp recommended in §5 is now implemented and benchmarked.
`src/common/workspace_envelope.py` gives each arm an analytic annulus
envelope — `r_min ≤ ‖p − pivot(azimuth)‖ ≤ r_max` plus a z-floor, radii
derived numerically from the URDF (re-verified by
`test/test_workspace_envelope.py`) — and four selectable policies for
targets outside it:
[TODO: please draw this envolop from different views in simulation.]
- **`warn`** — legacy behavior: pass through, throttled console warning.
- **`project`** — closest feasible point on the boundary.
- **`freeze`** — hold the last feasible target until the hand re-enters.
- **`slow`** — outward motion damped smoothly to zero near the boundary
  (tangential motion unaffected); degrades to `project` outside.

The policy runs in the IK thread (`dual_ik_solver.py`) ahead of *every*
solver, after the height lock; selectable via `--oob-mode` on both teleop
tools, default `WORKSPACE_OOB_MODE` in `configs.py`. Three deliberately
infeasible trajectories (`mock_quest.envelope_suite`): `envelope_radial`
(0.30 m forward push + dwell), `envelope_swoop` (0.20 m below the floor),
`envelope_slide` (exit radially, slide laterally while outside, return —
the case that separates `freeze` from the tracking policies).

### 9.1 Results — `envelope_slide` / `pink_relaxed`

| policy | oob (s) | err vs emitted (mm) | err vs raw (mm) | qd_oob | jerk |
|---|---|---|---|---|---|
| `warn` | 6.2 | 129.9 | 129.9 | 0.99 | 2 |
| `project` | 6.2 | 26.9 | 147.7 | 0.87 | 3 |
| `freeze` | 6.2 | 24.9 | 156.7 | 0.82 | 3 |
| `slow` | 6.2 | 26.8 | 148.0 | 0.77 | 2 |

(Full sweep — 3 trajectories × 3 methods × 4 policies — in
`outputs/teleop_envelope_bench.json`; plots in
`outputs/teleop_envelope_plots/`.)

### 9.2 Findings
[Produce visualisations for these experiments.]
1. **`warn` (the old behavior) grinds the arm 130–184 mm from its own
   commanded target** while the hand is outside — the solver chases an
   impossible point and parks at joint limits. Every active policy keeps
   the *emitted*-target error at ~25–61 mm with lower peak velocity.
2. **`project` is the recommended default for data collection**: on the
   slide trajectory it keeps tracking the lateral hand motion (raw-target
   error lower than `freeze`'s 156.7 mm) and re-enters seamlessly
   (recovery ≈ 1.74 s on `envelope_radial`, measured to raw-error
   < 10 mm sustained 0.2 s).
3. **`freeze` minimizes motion but loses the operator** during lateral
   OOB movement; good for safety-critical demos, worse for teleop feel.
4. **`slow` behaves like `project` with slightly lower peak velocities** —
   the soft-boundary shaping matters most for slow approaches, not the
   fast excursions tested here.
5. The policy layer is method-agnostic: `telegrip`'s higher OOB jerk
   (~85 rad/s³) is the method's clamp behavior, not the policy's.

Reproduce: `venv/bin/python src/sim_benchmark/run_envelope.py --save
outputs/teleop_envelope_bench.json --plot outputs/teleop_envelope_plots`.

### 9.3 Related fix: gravity-sag ratchet in the sim rehearsal tool

While validating with the mock device, circles reported an impossible
−23 mm envelope margin: with teleop *idle*, `quest_sim_teleop.py` streamed
the IK thread's idle targets (synced from *measured* joints) back to the
position servos, so gravity sag ratcheted the arms ~13 cm downward in 2 s
before the activation anchor latched. The tool now streams targets only
while teleop is active and holds the last active command otherwise —
mirroring what the real-robot thread already did. Sim EE tracking with the
production stack: 11.6 mm mean (`pink_relaxed`), 7.0 mm (`telegrip`).

---

## 10. Planned: bimanual user study (subjective evaluation)

Everything above is objective and open-loop. A within-subjects user study
(~5 participants, pick–handover–place + towel half-fold, conditions
`pink_full`/`pink_relaxed`/`telegrip` with `project` OOE + the upstream
Telegrip stack via `tool/telegrip_native.py`) adds the subjective side:
NASA-TLX workload, SUS, custom teleop-feel Likert items, and interviews,
alongside per-trial objective logs. Design in
`documents/paper/teleoperation/` §"Bimanual user study (protocol)"; step-by-step
runbook in `documents/user_study_protocol.md`. Results land here and in
the paper once sessions are run.

---

## 11. Rehearsal sweep — thumbstick wrist interface (branch `teleop-feel-fixes`, commit 08d4446, 2026-07-15)

**Provenance (read this first — these are rehearsal numbers, not benchmark
numbers).** These are 20 s *headless* MuJoCo rehearsals of the full teleop
stack driven by the scripted **mock Quest device** (the same device the
smoke tests in §6 use), *not* runs of the controlled benchmark suite in
§1–§9. They exercise the new `mymethod` thumbstick wrist interface and its
attitude modes end-to-end through the real IK thread, so the numbers reflect
the integrated pipeline (filter, clutch, envelope, IK, joint-space wrist
trims) rather than an isolated solver. Treat them as rehearsal-level
sanity/feel evidence, not as suite-grade metrics.

`mymethod` variants: `-hold` (attitude held from grip, sticks the only
change — the default), `-wrist` (incremental wrist mapping + trims), `-soft`
(weak absolute follow, trims decay toward the hand). `pos` = EE position
tracking error vs command, mean/p95 (mm); `ori` = attitude error vs command,
mean/p95 (deg).

| condition | pos (mm) | ori (deg) |
|---|---|---|
| circle armplane | 54.4 / 79.6 | 1.7 / 2.1 |
| circle pink_relaxed | 15.4 / 20.4 | 15.4 / 30.7 |
| circle mymethod-hold | 13.0 / 16.9 | 5.5 / 10.1 |
| circle mymethod-wrist | 14.4 / 20.3 | 2.9 / 3.8 |
| wrist armplane | 35.1 / 72.0 | 15.1 / 30.0 |
| wrist pink_relaxed | 15.2 / 19.8 | 32.1 / 54.4 |
| wrist mymethod-hold | 13.3 / 14.7 | 3.0 / 3.3 |
| wrist mymethod-wrist | 24.9 / 47.7 | 6.9 / 14.0 |
| wrist mymethod-soft | 15.2 / 19.8 | 32.1 / 54.5 |
| excursion pink_relaxed warn | 95.6 / 225.7 | 53.0 / 74.6 |
| excursion pink_relaxed project | 31.4 / 42.5 | 43.1 / 58.0 |
| excursion pink_relaxed slow | 31.4 / 42.6 | 43.2 / 57.9 |
| excursion pink_relaxed freeze | 31.3 / 42.1 | 42.9 / 57.5 |
| excursion mymethod-hold slow | 24.9 / 35.5 | 11.6 / 23.8 |
| excursion mymethod-hold project | 24.9 / 36.0 | 11.7 / 23.8 |
| joystick mymethod-hold | 15.4 / 22.5 | 3.6 / 5.5 |
| joystick mymethod-wrist | 15.4 / 22.4 | 3.6 / 5.5 |
| joystick mymethod-soft | 22.6 / 36.7 | 37.4 / 86.6 |

### Conclusions

1. **`mymethod-hold` gives the best position tracking in every pattern** and
   a steady attitude — position error stays ~13–15 mm across circle, wrist,
   and joystick patterns, and it more than halves the excursion-pattern error
   of `pink_relaxed` (≈25 vs ≈31 mm). Holding the attitude from grip is what
   lets the position solver stay fully relaxed.
2. **`-soft` is eliminated.** Its trims decay back toward the hand, so it
   inherits `pink_relaxed`'s large attitude errors (wrist 32°/54°, joystick
   37°/87°) with no position benefit. Drop it from the recommended set.
3. **Recommended combination: `--method mymethod --wrist-mode hold
   --oob-mode slow`.** Honest caveat: `slow` and `project` were *numerically
   identical* in this sweep (excursion 24.9 mm both, ori ~11.6–11.7°), so the
   `slow` choice rests on the operator's real-world preference for a soft
   boundary, **not** on any simulation evidence here.
4. **Read the orientation metric with care.** It measures how well the
   commanded attitude is *followed*, not operator *intent*. For `-hold` the
   command is "stay put" by design, so a low `ori` there means "held steady",
   not "did what the operator wanted" — only the planned user study can score
   intent-following.
