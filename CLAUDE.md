# CLAUDE.md — working guide for this repo

Background for Claude Code (and humans) working in `so101_garment`. Read
this before making changes; keep it current when structure or workflow
changes.

## What this project is

A dual **SO-101** LeRobot-arm rig for garment manipulation: Meta-Quest
teleoperation, data collection, and VLA policy training/eval (LeRobot,
**pi0.5**, LIBERO). Plus a fully 3D-printed rig with a MuJoCo/Isaac
**digital twin** generated from OpenSCAD.

## Environment & how to run things

- **venv** lives at `venv/`. There is **no system `python`** — always use
  `venv/bin/python` (or `source setup.sh` / `source venv/bin/activate`).
- **`source setup.sh` before every session**: sets `PYTHONPATH`
  (`.:src`), MuJoCo render backend, `HF_LEROBOT_HOME`, `SO101_OUTPUT_DIR`,
  serial access, Quest/adb check, and a GPU/disk readout.
- **`bash install.sh`** is the one-shot installer (idempotent).
- **LeRobot is a source checkout** at `../lerobot` (parallel to this
  repo), installed editable at a pinned commit with extras
  `feetech,dataset,pi,libero,pusht`. To inspect the real train/eval API,
  read `../lerobot/src/lerobot/...` — do not guess CLI flags.
- **Hardware here:** RTX 3050 Laptop, **4 GB VRAM** — too small for pi0.5
  train/eval. Do heavy VLA work on a bigger machine; keep local runs to
  the diffusion smoke test.

## Layout

- `src/common/` — teleop pipeline: `configs.py` (all tuning constants),
  `threads/dual_ik_solver.py` (the production IK loop), `arm_poses.py`,
  `teleop_buttons.py` (shared A/B/Y state machine), `pink_ik_solver.py`.
- `tool/` — runnable entry points: `meta_quest_teleopration.py` (real
  arms), `quest_sim_teleop.py` (sim rehearsal, same stack + rig +
  cameras), `identify_arms.py`, `view_twin.py`, `part_drawings.py`.
- `sim_benchmark/` — MuJoCo IK-method benchmark + `sim_arms.py`,
  `method_adapter.py`, `mock_quest_device.py`.
- `sim_twin/` — OpenSCAD→MuJoCo/Isaac digital-twin pipeline. `config.scad`
  is the single source of truth (see the memory note / `src/platform/`).
- `src/platform/` — OpenSCAD rig design (`config.scad`, `board.scad`, …).
- `test/` — `smoke_test_pipeline.sh` (train→eval plumbing check) + unit
  tests.
- `markdowns/` — design docs & worklogs (teleop benchmark, teleop-v2).
- Outputs go under `outputs/` (`$SO101_OUTPUT_DIR`, gitignored).

## Training / eval pipeline (LeRobot)

- CLIs: `lerobot-train`, `lerobot-eval` (installed with LeRobot).
- Registered names used here: policy `pi05` / `diffusion`; env `libero`
  (tasks `libero_spatial|object|goal|10|90`) / `pusht`.
- **One run = one directory**: `lerobot-train --output_dir=<run>` writes
  `train_config.json` + `checkpoints/<step>/pretrained_model/` +
  `checkpoints/last`. Point `lerobot-eval --output_dir=<run>/eval` at the
  same folder. Load a checkpoint with `--policy.path=<…>/pretrained_model`.
- Gotchas (all verified while wiring the smoke test):
  - `--output_dir` must **not** already exist (train aborts) — the smoke
    script lets lerobot-train create it and moves logs in after.
  - `--policy.push_to_hub=false` (default true → needs a repo_id).
  - `--wandb.enable=false` for quiet runs.
  - LIBERO dataset is ~35 GB → use `--dataset.streaming=true`.
  - `MUJOCO_GL=egl` for headless LIBERO render.
  - Diffusion policy pulls torchvision ImageNet weights (flaky CDN hash) —
    the smoke test passes `--policy.pretrained_backbone_weights=null`.
  - The default diffusion policy is **~263M params**: it OOMs a 4 GB GPU,
    so the smoke script auto-selects CPU unless VRAM ≥ 8 GB.
  - `gym_pusht` needs **pymunk < 7** (7.x dropped `add_collision_handler`);
    LeRobot's `pusht` extra pins it, and requirements.txt re-pins it.
- The **smoke test** (`test/smoke_test_pipeline.sh`, VERIFIED end-to-end on
  CPU: train→checkpoint→eval all pass) uses a small `diffusion` policy on a
  few PushT episodes — validates plumbing, not skill. Real pi0.5+LIBERO
  belongs on a big GPU.

## Conventions & gotchas

- **Pre-commit runs black, isort, flake8, mypy.** Match them or the commit
  hook fails. flake8 is pinned ≥7.1 (older pycodestyle false-positives
  inside f-strings on Python 3.12); `E203/E501/E231/E402` are ignored.
- Teleop **arm/side mapping is load-bearing and easy to get wrong**:
  `follower_0`=RIGHT arm/handle, `follower_1`=LEFT. HW→URDF offsets are
  keyed per follower in `configs.py`. Confirm with `tool/identify_arms.py`.
- All teleop tuning is centralized in `src/common/configs.py`
  (`GRIPPER_MAX_OPEN`, `FRAME_TASK_GAIN`, `JOINT_VEL_SCALE`, `WORKSPACE_*`,
  `NEUTRAL_JOINT_ANGLES`). Change there, not inline.
- Git: work on a branch, never commit straight to `main`; end commit
  messages with the `Co-Authored-By: Claude …` trailer. The user pushes.
- Don't hardcode rig geometry — edit `src/platform/config.scad`.

## Verifying changes

- Teleop/sim: `python tool/quest_sim_teleop.py --mock --headless --no-rig
  --duration 14` prints EE tracking error + gripper cap.
- Digital twin: `python -m sim_twin.verify`.
- Training plumbing: `bash test/smoke_test_pipeline.sh`.
