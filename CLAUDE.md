# CLAUDE.md — working guide for this repo

Background for Claude Code (and humans) working in `so101_garment`. Read
this before making changes; keep it current when structure or workflow
changes.

## What this project is

A dual **SO-101** LeRobot-arm rig for garment manipulation: Meta-Quest
teleoperation, data collection, and VLA policy training/eval (LeRobot,
**pi0.5**, LIBERO). Plus a fully 3D-printed rig with a MuJoCo/Isaac
**digital twin** generated from OpenSCAD.

## Working style

- Keep documentation and the README at an industrial level — accurate
  enough to install and run the project from scratch.
- Reduce redundancy between files; keep good modularisation and a
  well-organised file tree.
- Be surgical: make only the changes the task needs, and don't change
  things that don't need changing.

## Environment & how to run things

- **venv** lives at `venv/`. There is **no system `python`** — always use
  `venv/bin/python` (or `source setup.sh` / `source venv/bin/activate`).
- **`source setup.sh` before every session**: sets `PYTHONPATH`
  (`.:src`), MuJoCo render backend, `HF_LEROBOT_HOME`, `SO101_OUTPUT_DIR`,
  serial access, Quest/adb check, and a GPU/disk readout.
- **`bash install.sh`** is the one-shot installer (idempotent).
- **LeRobot is a source checkout** at `../lerobot` (parallel to this
  repo), installed editable at a pinned commit with extras
  `feetech,dataset,pi,libero,pusht,training,diffusion,peft`. To inspect the real
  train/eval API, read `../lerobot/src/lerobot/...` — do not guess CLI
  flags.
- **Hardware varies by machine — check, don't assume.** `source setup.sh`
  prints the live GPU/VRAM readout; trust that over any note here. Two
  machines seen so far: a laptop with an RTX 3050 (4 GB VRAM, too small
  for pi0.5 train/eval — keep local runs to the diffusion smoke test) and
  a remote box with an RTX 3090 Ti (~24 GB VRAM, no sudo access — enough
  headroom for real pi0.5 LIBERO *evaluation*, though finetuning still
  wants datacenter-scale GPUs).

## Layout

- `src/common/` — teleop pipeline: `configs.py` (all tuning constants),
  `threads/dual_ik_solver.py` (the production IK loop),
  `workspace_envelope.py` (analytic reach envelope + out-of-envelope
  policies), `pink_ik_solver.py`, `one_euro_filter.py`,
  `data_manager_dual.py`, `utils.py` (operator control frame),
  `sync.py` (pure timestamped-history buffers + nearest/interpolated
  sample selection used to align every stream to one reference time per
  recorded frame), and `recording/` (LeRobot episode recorder behind
  `--record` on the real teleop tool: 30 fps dataset + ~100 Hz sidecar
  parquet + UVC camera threads; `realsense_camera.py` adds a central
  RGB-D camera (`--central-depth`: RGB video feature + aligned 16-bit
  depth written by `depth.py` as PNG16 under `<root>/extra/depth/`, with
  intrinsics/scale in `<root>/meta/realsense.json`); `drift.py` logs
  per-frame per-stream temporal drift to `<root>/extra/`; EE-space
  features (`ee_pose` measured + `ee_target` projected+constrained,
  neutral keys the LeRobot classifier ignores so the joint policy is
  untouched) let either a joint- or EE-space policy train from the same
  episodes (quest mode only; `--no-record-ee` opts out), with the
  action-definition constants in `<root>/meta/action_space.json`; the
  `--sensor-view` monitor shows live per-stream drift + drop counts;
  `monitor_server.py` serves the same frames + recorder status + both
  arms' measured-vs-last-sent joints over loopback for the rig console
  when the recorder is given `--monitor-port` (off by default; its
  control surface is a per-mode allow-list — `allowed_keys_for`: the episode
  and quit keys with a headset on, plus ENABLE for a leader session, whose
  keys are otherwise read from a terminal the console-started session does
  not have; park and home stay physical in both);
  `controls.py` is the ONE list of operator steps, printed by the teleop
  tool and shown by the console; config in `src/conf/recording.yaml`,
  device indices are per-machine placeholders).
- `tool/` — runnable entry points: `meta_quest_teleopration.py` (real
  arms), `quest_sim_teleop.py` (sim rehearsal, same stack + rig +
  cameras), `telegrip_native.py` (drive the arms with the *unmodified
  upstream* Telegrip checkout — see `documents/telegrip_native.md`),
  `check_mirror.py` / `fit_joint_offsets.py` (arm-side/offset checks),
  `collect_preflight.py` (green/red rig-readiness table before data
  collection: sensor map, calibrations, poses, cameras/RealSense, disk,
  CPU governor, USB autosuspend; `--no-hardware` for config-only),
  `view_twin.py` (`--payload` shows the collection scene), `part_drawings.py`,
  the sim-VLA pair `collect_sim_dataset.py` (oracle demonstrations in the
  twin; only verified successes are saved) / `eval_sim_policy.py` (policy
  rollouts in the same env — see `documents/long_vla_sim_guide.md`), plus
  policy train/eval helpers (`sim_pipeline_pi05.py`, `train_vla_lerobot.py`,
  `send_middle_and_rest.py`), and the policy-deployment pair
  `run_policy.py` (cameras + buses + safety; `--server` sends observation
  windows out and executes the action chunks that come back, spliced by
  `--strategy`, default `receding` at `--execute-ratio 0.5`; `--sim [task]`
  puts the twin behind the same `common/policy_rig.py` seam so the WHOLE
  procedure — page, arming, throttle, splice, log — can be rehearsed with no
  robot. Not to be confused with `run_policy_sim.py`, which is the BATCH
  chunking-strategy sweep over many scored seeds — its `--grid` expands a
  strategy x hyper-parameter grid via `common/chunk_sweep.py`) /
  `policy_server.py` (loads a checkpoint on a GPU box and answers with chunks;
  wire format in `src/common/policy_wire.py`, runbook in
  `documents/remote_policy_inference.md`), and `rig_web.py` (the browser
  console — see below).
- `src/common/recording/dataset_check.py` — is a dataset whole? The counted
  episodes against the ones in `meta/episodes/`, `data/` and `extra/`, the
  offset invariant, and the repair for an episode nobody wrote. Pure parquet +
  JSON, no LeRobot import, so it can describe a dataset LeRobot refuses to open.
- `src/common/recording/dataset_view.py` — camera-ablation **views**: a dataset
  directory naming only some cameras, with the video files symlinked from the
  source (a few MB, not a copy). `meta/info.json` drives
  `LeRobotDatasetMetadata.video_keys`, so an unnamed camera is never decoded AND
  never reaches the policy — no `--policy.input_features` override to keep in
  step. Also collapses a dataset's task strings onto the one covering the most
  frames (position is no guide: on `cube-pnp-new` the typo is registered first),
  and maps cameras onto pi0.5's pretrained slots (`PI05_SLOTS`). Built by
  `tool/make_camera_view.py`; the cluster job builds one per `cameras` row.
- `src/common/joint_frames.py` — the servo↔URDF sign/offset tables (values
  in `configs.py`) and the conversion, shared by the joint-state thread, the
  sidecar writer and the console's idle arm reader.
- `src/common/web/` — the rig console served by `tool/rig_web.py` on
  loopback: `datasets_api.py` (browsing, playback and episode curation —
  this is the former `tool/dataset_web.py`, moved unchanged),
  `lifecycle.py` (whole-dataset create-name checks, rename, delete and
  merge; pure/filesystem, unit-tested) + `lifecycle_api.py` (its routes;
  a merge may delete its sources once it has succeeded), `jobs.py` (the
  one worker thread and the records the page's dock polls: merge,
  compaction and the freeing of a deleted dataset all outlive their
  request),
  `roots.py` (which collection directory the console works on: name/target
  rules, the sshfs command, `/proc/mounts` parsing, the remembered list —
  pure, unit-tested) + `roots_api.py` (its routes, the `root_required`
  middleware, and the refusal to switch under a running session or job),
  `session.py` (supervises ONE collection
  session as a subprocess — the same stream resolvers as the CLI, the
  teleop command, the quit→SIGINT→SIGTERM stop ladder, and the console's
  own idle previews of the cameras and of the follower arms, both released
  before a session starts) + `session_api.py` (Collect routes; live frames
  and the two allowed keys are PROXIED to the session's monitor, never
  taken from a device), `sensors.py` (binding devices to stream names:
  pure map operations + the wiggle-test arithmetic + an uncalibrated,
  torque-off `ArmProbe`) + `sensors_api.py` (its routes; all refused while
  a session runs, and the map path is injectable so a test never rewrites
  the machine's real `sensor_map.yaml`), `util.py`, and the front-end
  under `static/`
  (`index.html` + one script per tab, no build step). The same package also
  holds the ROLLOUT view, which is a separate page served by
  `tool/run_policy.py --web` and not a console tab: `policy_view.py`
  (what the policy was shown / planned / did, and the hold·step·run
  throttle; reads a snapshot, owns no device. Stop and Reset scene end a
  TRIAL and release the arms — torque off, run still up, page/cameras/buses/
  session all kept; only Reset puts the scene back and numbers the next trial,
  and only Ctrl+C ends the process. A page-armed run is disarmed by either, so
  consent is asked again per trial) + `policy_ghost.py` (the
  returned chunk drawn as two URDF ghosts — measured blue inside planned
  orange — in a viser scene the operator can orbit; served on its OWN port and
  embedded in the page as an iframe, so the page only ever aims it,
  `/twin/at?seq&i`) + `static/policy.{html,css,js}`. Its non-web halves are `common/policy_run.py`
  (the throttle's pure state machine, the per-trial consent and the prefetch
  arithmetic, shared with the control loop) and `common/policy_log.py` (the per-run log under
  `outputs/policy_runs/`). Runbooks: `documents/rig_web.md`,
  `documents/remote_policy_inference.md`. Deletion is always available; every irreversible
  one asks in the browser first, and marking an episode (reversible) does
  not. The third tab is called **Signals** in the UI while the module,
  routes and `sensor_map.yaml` keep the older `sensor` name.
- `src/sim_datagen/` — the simulated tasks and their scripted demonstrators.
  `env.py` holds `TASKS` (`single`, `handover`, and `handover_split`) and the
  tick rate: `PHYSICS_HZ` 600, `DEFAULT_FPS` **25**, and `substeps_for(fps)`,
  which REFUSES a rate that does not divide the physics rate rather than
  rounding it — every sim tool takes `--fps` and threads it into
  `PickPlaceTwinEnv(task, fps=...)`. The existing 30 fps datasets and the
  handover checkpoint stay valid; anything measuring one of them must pin
  `--fps 30`. `handover_split` is the relay whose plate lies OUTSIDE the left
  arm's reach, so the hand-off is forced by geometry and not merely
  demonstrated; its cube spawn varies (the plain `simple` mode repeated ONE
  spawn 100 times, per-channel spread exactly zero). MEASURED while setting
  its sampling boxes: excluding the cube from the right arm as well breaks the
  pick — a 22 mm cube at that extension exceeds the oracle's open-loop grasp
  accuracy (direct oracle failed 3 of 4 probes there, 90.9 % inside the box
  actually shipped), while excluding the PLATE from the left arm costs nothing
  because a release only has to land inside the 20 mm success radius.
- `src/sim_benchmark/` — MuJoCo IK-method benchmark: `scene.py`,
  `method_adapter.py`, `methods/` (pluggable registry incl.
  `telegrip_split.py`), `mock_quest.py` / `mock_quest_device.py`,
  `run_benchmark.py`, `run_envelope.py` (OOE policy sweep),
  `export_latex_tables.py` (JSON → paper tables).
- `src/sim_twin/` — OpenSCAD→MuJoCo/Isaac digital-twin pipeline.
  `config.scad` is the single source of truth (see the memory note /
  `src/platform/`).
- `src/platform/` — OpenSCAD rig design (`config.scad`, `board.scad`, …).
- `test/` — tiered: `test/unit/` (fast, pure-python/pinocchio, no MuJoCo),
  `test/integration/` (MuJoCo scenes, plus the console↔session two-process
  check `test_console_session.py`), `test/system/`
  (`smoke_test_pipeline.sh`, train→eval plumbing check;
  `smoke_vla_sim.sh`, sim-VLA collect→train→eval plumbing check).
  `test/__init__.py` is load-bearing (keeps the stdlib `test` package from
  shadowing it).
- `documents/` — design docs & worklogs (teleop benchmark results, user
  study protocol, telegrip-native, remote policy inference, rig console)
  plus the living paper under `documents/paper/`.
- `hpc/` — the Slurm cell on KCL CREATE and the traffic in both directions:
  `provision_create.sh` (login node, once; `SO101_STAGE_PI05=1` also caches the
  ~14.5 GB `lerobot/pi05_base`, which is NOT licence-gated), `stage_datasets.sh`
  (collected datasets up), `runs.tsv` + `submit_real.sh` +
  `create_real_vla.sbatch` (one array task per dataset/policy/**camera set**;
  policies `act|diffusion|pi05`), `create_sim_vla.sbatch`, and
  `fetch_policies.sh` (the finished checkpoints back down, into the layout
  `tool/policy_server.py` and `tool/run_policy.py` expect; `--datasets`
  matches a run name or the dataset before its `__<cameras>` suffix). Runbook:
  `hpc/README.md`.
- `Makefile` — test tiers (`test-unit`, `test-integration`, `test`,
  `test-system`, `test-system-vla`), `paper`, and `lint` targets.
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
  - LeRobot's `requires-python = ">=3.12"`. `install.sh` runs `python3 -m
    venv venv` — if `python3` on `$PATH` resolves to something older
    (e.g. an Anaconda `python3.9` ahead of `/usr/bin` in `PATH`), the venv
    silently builds on the wrong interpreter and the LeRobot editable
    install fails its Python-version check. Force it if needed:
    `/usr/bin/python3.12 -m venv venv`.
  - `lerobot-train` needs the `training` extra (`accelerate`+`wandb`) and,
    for the `diffusion` policy the smoke test uses, the `diffusion` extra
    (`diffusers`) — neither is pulled in by `feetech,dataset,pi,libero,pusht`
    alone. `install.sh`'s `LEROBOT_EXTRAS` includes both now.
  - **A merged dataset cannot be curated again, unrepaired:**
    `aggregate_datasets` copies each source's episode-metadata rows and
    merely OFFSETS their `meta/episodes/file_index`, while writing every
    row into the destination's first file — so the merged dataset names
    metadata files that were never written, and the next
    `delete_episodes`/aggregation on it dies with `FileNotFoundError`
    (MEASURED: 62 rows in `file-000.parquet` claiming indices 0..55).
    `dataset_edit.repair_episode_metadata` rewrites those two
    self-referential columns from each file's own path; the console runs
    it before a merge reads its sources, after a merge writes its output,
    and before any compaction.
  - **An episode counted but never written breaks the whole dataset, and the
    error blames the network:** `DatasetReader._check_cached_episodes_sufficient`
    needs `set(range(total_episodes))` to be a subset of the episodes actually
    present, so ONE missing index makes `LeRobotDataset.__init__` judge the
    local copy incomplete and go to the Hub for a version tag — which offline
    raises `OfflineModeIsEnabled: Cannot reach https://huggingface.co/...` for
    a dataset that never left the drive (MEASURED on `cube-pnp-new`: 88 counted,
    87 written, episode 4 absent from `meta/episodes/`, `data/` and `extra/`
    alike, and `total_frames` 19 too high). `common/recording/dataset_check.py`
    is the guard: `ensure_loadable` runs before anything constructs a
    `LeRobotDataset`, and `repair_phantom_episodes` drops the empty slots and
    renumbers the survivors. The console offers it as a Repair button. How such
    a slot appears is not proven; the recorder no longer counts an episode whose
    save raised, which is the one path we own.
  - **Video decoding / no sudo:** LeRobotDataset videos (PushT, LIBERO) are
    AV1-encoded. The default `torchcodec` backend dlopen's the *system*
    FFmpeg shared libs — on a box with no `ffmpeg` installed (or no sudo to
    install one) this fails, and even a stray old FFmpeg (e.g. bundled with
    an Anaconda install) is usually too old to decode AV1. Fix: pass
    `--dataset.video_backend=pyav` to every `lerobot-train` call — PyAV's
    wheel bundles its own modern, AV1-capable FFmpeg (`libdav1d`), so it
    needs nothing from the system. (`lerobot-eval` doesn't load a dataset,
    so it never needs this flag.)
  - `hf_libero` prompts on stdin ("custom dataset path? Y/N") the *first*
    time anything imports `libero.libero`, if `~/.libero/config.yaml`
    doesn't exist yet — hangs any non-interactive script with `EOFError`.
    `setup.sh` pre-writes that config (via `importlib.util.find_spec`, to
    avoid importing the package before the file exists) so this never
    blocks. LIBERO's object/asset meshes aren't in the pip package either;
    they auto-download from HF Hub (~586 files) the first time a
    `LiberoEnv` is actually built.
  - **pi0.5 full finetuning OOMs a 24 GB GPU** — it's a ~4.1B-param model;
    plain AdamW's optimizer state (exp_avg + exp_avg_sq, same dtype/size as
    the params) alone needs >16 GB on top of the ~17 GB params+grads+
    activations already in use. Fix: LoRA via the `peft` extra —
    `--peft.r=16` (pi0.5 has built-in default target modules: the action
    expert's q/v attention projections + the small state/action projection
    heads) cuts trainable params to ~1.3M, and the whole run fits in
    ~9 GB. VERIFIED: `--policy.path=lerobot/pi05_libero_finetuned
    --peft.r=16 --dataset.repo_id=HuggingFaceVLA/libero
    --dataset.streaming=true --dataset.video_backend=pyav` finetuned for
    10 steps in ~3.5 min, then `lerobot-eval` on the resulting checkpoint
    got 2/2 (100%) success on LIBERO-Spatial task 0 with real rollout
    videos written to `<run>/eval/videos/`.
  - **pi0.5 finetuning must RENAME cameras, not re-derive them:** `pi05_base`
    ships a populated `input_features` (three openpi slots — `base_0_rgb`,
    `left_wrist_0_rgb`, `right_wrist_0_rgb` — and 32-D state/action), and
    `make_policy` only derives features from the dataset `if not
    cfg.input_features`. So a rig dataset's camera keys never reach it and the
    forward pass raises "All image features are missing from the batch". The fix
    is `--rename_map` (train.py: it requires a pretrained checkpoint), mapping
    each rig camera onto the slot that means the same viewpoint. Overriding
    `--policy.input_features` does NOT work as a way to drop a camera: draccus
    MERGES the dict, so a slot left out of the override survives. It does not
    need dropping anyway — pi0.5 pads a slot with no camera behind it to -1 and
    gives it a zero attention mask, which is what an ablated camera should be.
  - **AV1 is not the slow part** (measured, contrary to the obvious guess): our
    recordings decode through `libdav1d` at ~2x the speed of the same clips
    transcoded to H.264, so do not transcode a dataset to "speed up" training.
    The dataloader floor is cameras x workers: 127 ms/batch for three cameras
    and 56 ms for one, at batch 8. `long_vla_real.sh` now follows
    `SLURM_CPUS_PER_TASK` instead of a hardcoded 4 worker processes.
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
  keyed per follower in `configs.py`. Confirm with `tool/check_mirror.py`
  / `tool/fit_joint_offsets.py`.
- All teleop tuning is centralized in `src/common/configs.py`
  (`FRAME_TASK_GAIN`, `ORIENTATION_COST`, `CONTROLLER_*` One-Euro params,
  `WORKSPACE_*` envelope radii/margins, `WORKSPACE_OOB_MODE`,
  `NEUTRAL_JOINT_ANGLES`). Change there, not inline.
- **Git: every change set starts on a fresh branch off `main`** (`git
  checkout -b <topic>`), even for docs-only changes; never commit straight
  to `main`. Finish with a commit on that branch ending in the
  `Co-Authored-By: Claude …` trailer. The user pushes/merges.
- **Living paper rule:** there are THREE living papers, and each guards its
  domain in the same branch as the change:
  - `documents/paper/teleoperation/` — any change on the teleoperation
    side: methods, orientation mapping, envelope/OOE handling,
    calibration/control-frame behavior, benchmark results (also update
    `documents/teleop_benchmark_results.md` when results change);
    regenerate tables with `src/sim_benchmark/export_latex_tables.py`.
  - `documents/paper/sim_training/` — any change to the sim-VLA side:
    simulated tasks/payload/contacts, oracle demonstrators, collection
    gating/seed protocol, or the experiment protocol and its results.
  - `documents/paper/real_training/` — any change to the real-world
    data-collection platform: the recording system (two-rate dataset +
    sidecar, stamp-on-read + reference-time alignment + drift budget),
    the recorded stream set (RGB / RGB-D / tactile / proprio / joint +
    EE targets), the teleoperation-to-autonomy contract, or the
    replicability/readiness procedure.
  All build with `make paper` (or `latexmk -pdf main.tex` in the paper
  dir). All paper writing follows
  `documents/academic_writing_guideline.md` (flow diagram before LaTeX,
  British English, active voice, no numbers in the abstract, no code
  paths in prose, `\unjustified{}` flags).
- Don't hardcode rig geometry — edit `src/platform/config.scad`.

## Verifying changes

- Teleop/sim rehearsal: `python tool/quest_sim_teleop.py --mock --headless
  --duration 14` prints EE tracking error (add `--mock-pattern
  wrist|excursion`, `--oob-mode project|freeze|slow|warn`, `--scene plain`
  as needed).
- Tests are tiered and run via the `Makefile` (all need `PYTHONPATH=.:src`
  and `MUJOCO_GL=egl`, which the targets set):
  - `make test-unit` — fast pure-python/pinocchio tests (`test/unit/`).
  - `make test-integration` — MuJoCo-backed tests (`test/integration/`).
  - `make test` — unit + integration.
  - `make test-system` — the train→eval plumbing smoke test
    (`test/system/smoke_test_pipeline.sh`; network + time).
  - `make test-system-vla` — the sim-VLA collect→train→eval plumbing
    smoke test (`test/system/smoke_vla_sim.sh`; ~15–45 min on CPU).
  Discover manually with e.g. `PYTHONPATH=.:src MUJOCO_GL=egl venv/bin/python
  -m unittest discover -s test/unit -t .`. `test/__init__.py` must exist or
  the stdlib `test` package shadows the directory.
- Benchmarks: `python src/sim_benchmark/run_benchmark.py` (tracking + wrist
  suites), `python src/sim_benchmark/run_envelope.py` (OOE policies).
- Digital twin: `python -m sim_twin.verify`.
- Lint (black/isort/flake8/mypy + unit-test hook): `make lint`.
- Paper: `make paper` (or `cd documents/paper/teleoperation && latexmk -pdf
  main.tex`).
