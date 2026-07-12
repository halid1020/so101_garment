# Vision-Tactile World Model with Dual SO-101 Garment Manipulation

This repository provides an independent, industrial-grade pipeline for dexterous garment manipulation using dual SO-101 robotic arms equipped with tactile sensors. The framework facilitates Meta-Quest teleoperation, autonomous data collection, and direct integration with Vision-Language-Action (VLA) models via the LeRobot ecosystem.

**Author:** Abudureyimu Halite
**Framework Status:** Active Development
**Dependencies:** LeRobot 0.5.x (from source), Pink IK, pyrealsense2, MuJoCo

## 🚀 System Architecture

1. **Hardware Interface:** Direct `sts3215` serial bus control utilizing `lerobot.motors` for SO-101 leader/follower synchronization.
2. **Teleoperation:** Meta-Quest VR tracking mapped to dual-arm Cartesian poses via Pink IK, filtered through adaptive 1€ smoothing.
3. **Data Pipeline:** Synchronized, offline collection of RGB feeds, joint states, and tactile data formatted natively for Hugging Face LeRobot dataset standards.
4. **Digital Twin:** The 3D-printed rig (board, adapters, camera tower, C310 holders) is parametrized in OpenSCAD and mirrored 1:1 in MuJoCo and Isaac Lab from the same config — see [Digital twin & printed rig](#-digital-twin--printed-rig-openscad--mujoco--isaac-lab).

## 📚 Documentation

- [`documents/teleop_benchmark_results.md`](documents/teleop_benchmark_results.md) — teleop IK-method benchmark results (tracking, wrist, handover, out-of-envelope sweeps).
- [`documents/user_study_protocol.md`](documents/user_study_protocol.md) — bimanual teleoperation user-study runbook.
- [`documents/telegrip_native.md`](documents/telegrip_native.md) — driving the arms with the unmodified upstream Telegrip stack.
- [`documents/paper/teleoperation/`](documents/paper/teleoperation/) — the living teleoperation paper (LaTeX). Build it with `make paper`.
- [`src/platform/README.md`](src/platform/README.md) — printed-rig design, hardware shopping list, assembly.
- [`src/sim_benchmark/README.md`](src/sim_benchmark/README.md) — IK-method benchmark harness and options.
- [`src/sim_twin/isaac/README.md`](src/sim_twin/isaac/README.md) — portable Isaac Lab package (asset conversion + demo).

## ⚙️ Installation

Everything installs with **one script**. From the repo root:

```bash
bash install.sh
```

It is idempotent (safe to re-run) and does all of the following:

1. Creates the `venv/` virtual environment (Python 3.12+, per LeRobot's
   `requires-python`) and upgrades pip.
2. Clones **LeRobot** into a parallel `../lerobot/`, checks out the pinned
   commit, and installs it editable with the extras this project needs:
   `feetech` (SO-101 bus), `dataset`, `pi` (pi0 / **pi0.5** policies),
   `libero` + `pusht` (the simulation benchmarks), `training`
   (`accelerate` + `wandb`, required by `lerobot-train` for any policy),
   `diffusion` (`diffusers`, for the small policy used by the smoke test),
   and `peft` (LoRA fine-tuning — see the pi0.5 section below).
3. Installs this repo's own deps from [`requirements.txt`](requirements.txt)
   (MuJoCo twin, Pink IK, cameras, plotting).
4. Clones + installs the Meta Quest teleop reader and `adb`.
5. Installs OpenSCAD (digital-twin part export) and the pre-commit hooks.

> **Layout:** `install.sh` expects `so101_garment/` and `lerobot/` to sit
> side by side under a common parent (e.g. `~/Projects/lerobot` and
> `~/Projects/so101_garment`). LeRobot is a source checkout so you can pin
> and patch it; LIBERO requires Linux.

**Before every working session**, prepare the environment:

```bash
source setup.sh
```

This activates the venv, sets `PYTHONPATH`, picks a MuJoCo render backend
(`egl` when headless), points `HF_LEROBOT_HOME` at the dataset cache and
`SO101_OUTPUT_DIR` at `outputs/`, grants SO-101 serial access, checks the
Meta Quest link, and prints your GPU / free-disk readout.

## 🎮 Teleoperation quick start (real Meta Quest)

Five IK methods are available behind the same production pipeline
(One-Euro filtering, grip clutch, handle calibration, armplane orientation
mapping): `production` (the tuned Pink solver), `pink_full`,
`pink_relaxed`, `dls`, `mink`, `scipy_ls`. Benchmark results and method
details: [`documents/teleop_benchmark_results.md`](documents/teleop_benchmark_results.md)
and [`src/sim_benchmark/README.md`](src/sim_benchmark/README.md).

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
python src/sim_benchmark/run_benchmark.py --plot outputs/teleop_benchmark_plots
python src/sim_benchmark/run_handover.py --plot outputs/teleop_benchmark_plots
python src/sim_benchmark/package_report.py   # shareable zip of all results
```

See [`src/sim_benchmark/README.md`](src/sim_benchmark/README.md) for all options.

## 🧠 Policy training & evaluation (LeRobot · pi0.5 · LIBERO)

Train and evaluate VLA policies with LeRobot, using **pi0.5** on the
**LIBERO** manipulation benchmark. Always `source setup.sh` first.

### Smoke-test the pipeline first (minutes, any machine)

Before committing GPU hours to pi0.5, verify the train → checkpoint →
eval plumbing on your machine. This uses a **small diffusion policy** on a
few episodes of the lightweight PushT dataset — it runs in minutes and
will not freeze a laptop:

```bash
bash test/system/smoke_test_pipeline.sh                 # auto device, tiny run
bash test/system/smoke_test_pipeline.sh --device cpu    # force CPU
bash test/system/smoke_test_pipeline.sh --steps 100 --eval-episodes 3
```

It **auto-selects the device**: CUDA only if the GPU has ≥8 GB VRAM,
otherwise CPU — so a small laptop GPU (e.g. 4 GB) won't OOM. On CPU it's
slower but safe. Typical wall time: **~3–6 min on GPU, ~8–20 min on CPU**,
plus a one-time ~1 GB PushT download on the first run (cached under
`$HF_LEROBOT_HOME`).

It prints an up-front time estimate, shows progress bars (dataset load,
training steps, eval episodes), writes everything into one run directory,
and ends with `✅ PIPELINE SMOKE TEST PASSED`. The success rate it reports
is **not** meaningful (only a few training steps) — the point is that the
train → checkpoint → eval wiring works before you commit GPU hours to
pi0.5.

### The real run: pi0.5 on LIBERO (needs a big GPU)

pi0.5 is a ~4B-parameter VLA. **Full-parameter finetuning needs a
datacenter GPU** (the reference model was finetuned on 8×H100 — full
AdamW's optimizer state alone needs >30 GB just for the params being
tuned, on top of everything else), and even evaluation wants ≳24 GB VRAM.
Run full finetuning on a capable machine — the LIBERO dataset is ~35 GB
(stream it with `--dataset.streaming=true` to avoid the full download).

**Train** (finetune pi0.5 on LIBERO-Long):

```bash
lerobot-train \
  --policy.type=pi05 \
  --policy.repo_id="${HF_USER}/pi05_libero" \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.video_backend=pyav \
  --env.type=libero --env.task=libero_10 \
  --output_dir="$SO101_OUTPUT_DIR/pi05_libero/run1" \
  --steps=30000 --batch_size=32 \
  --save_freq=5000 --wandb.enable=true
```

#### Single-GPU option: LoRA finetuning (VERIFIED, ~24 GB VRAM)

No datacenter box handy? `--peft.r=16` LoRA-adapts pi0.5 instead of
full-parameter tuning — pi0.5 ships default target modules (the action
expert's attention projections + the small state/action heads), so
trainable params drop from ~4.1B to ~1.3M and the whole run fits in
**~9 GB VRAM**. Finetuning from the published checkpoint (rather than
from scratch) keeps it a real, useful finetune despite the short run:

```bash
lerobot-train \
  --policy.path=lerobot/pi05_libero_finetuned \
  --peft.r=16 \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.streaming=true \
  --dataset.video_backend=pyav \
  --output_dir="$SO101_OUTPUT_DIR/pi05_libero/run1" \
  --steps=1000 --batch_size=1 \
  --save_freq=500 --wandb.enable=false
```

This finetunes from `lerobot/pi05_libero_finetuned` rather than from
scratch (`--policy.type=pi05` for that instead). VERIFIED on an RTX 3090
Ti: 10 steps ran in ~3.5 min at ~9 GB VRAM, and evaluating the resulting
checkpoint (see below, pointed at
`<run>/checkpoints/last/pretrained_model`) got 2/2 successful rollouts on
LIBERO-Spatial task 0. Data loading (streaming from the Hub) dominates
wall time far more than the LoRA backward pass does, so more steps scale
roughly linearly, not per-parameter.

**Evaluate** — either your own checkpoint or the published
[`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
across the four standard suites (10 episodes/task = 400 episodes):

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir="$SO101_OUTPUT_DIR/pi05_libero/eval"
```

LIBERO suites: `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10` (long), `libero_90`. Restrict tasks with
`--env.task_ids='[0,1]'`. Reference results and details:
[`../lerobot/docs/source/libero.mdx`](../lerobot/docs/source/libero.mdx).

Rollout videos always land at
`<output_dir>/videos/<suite>_<task_id>/eval_episode_<n>.mp4` — for a
quick single-task check instead of the full 400-episode sweep:

```bash
lerobot-eval \
  --policy.path="$SO101_OUTPUT_DIR/pi05_libero/run1/checkpoints/last/pretrained_model" \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=2 \
  --env.max_parallel_tasks=1 \
  --output_dir="$SO101_OUTPUT_DIR/pi05_libero/run1/eval"
```

### Pipeline output structure

Everything for one run lands in a **single directory** under
`$SO101_OUTPUT_DIR` (default `outputs/`). `lerobot-train` owns the
`checkpoints/` layout; point `lerobot-eval --output_dir` at the same run
folder so results sit alongside the weights:

```
outputs/pi05_libero/run1/          # one run = one directory
├── train_config.json              # the exact resolved training config
├── checkpoints/
│   ├── 005000/                     # one folder per saved step
│   │   └── pretrained_model/       # <- load this with --policy.path=…
│   │       ├── config.json         #    policy architecture/config
│   │       ├── model.safetensors   #    weights
│   │       └── *_processor/        #    pre/post-processors + norm stats
│   ├── 030000/pretrained_model/
│   └── last -> 030000              # symlink to the newest checkpoint
├── eval/                           # from lerobot-eval --output_dir
│   ├── eval_info.json              # per-task + aggregated success rates
│   └── videos/                     # rendered rollout videos
└── logs/                           # train.log / eval.log (smoke test)
```

Resume or evaluate any checkpoint by pointing at its `pretrained_model/`
folder, e.g. `--policy.path=outputs/pi05_libero/run1/checkpoints/last/pretrained_model`.

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
`run_demo.py` — see [`src/sim_twin/isaac/`](src/sim_twin/isaac/README.md).

The asset pipeline (`python -m sim_twin.assets`, SCAD → meshes → URDF →
`twin_params.json`, content-hash cached) runs automatically inside the
tools above; call it directly only with `--force` (rebuild everything)
or `--check` (CGAL volume checks, CI-style).
