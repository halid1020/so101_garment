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
- **Ring the bell when you are blocked on the user.** Anything only they can
  do — a cluster login, a VPN, a drive to plug in, a `git push` — stops the
  work, so say so immediately rather than at the end of a long report: send a
  `PushNotification` (desktop, and phone if Remote Control is on) AND print a
  clearly marked block in the terminal, naming what is blocked and the one
  action that unblocks it. The notification can be disabled in the host config,
  so the terminal block is the half that always has to be there. Then carry on
  with whatever does not depend on the answer.
- **Analysis results are dated.** Anything `tool/analyse_policy_inputs.py`,
  `tool/analysis_slides.py` or a future analysis writes goes to
  `outputs/analysis/<YYYY-MM-DD>/<content>/`, where `<content>` is
  `<policy>-<camera slug>` and the deck sits in `<content>/slides/`. The rules
  are `src/common/analysis/paths.py`; keep new analyses on that helper rather
  than composing a path by hand.

## Environment & how to run things

- **venv** lives at `venv/`. There is **no system `python`** — always use
  `venv/bin/python` (or `source setup.sh` / `source venv/bin/activate`).
- **`source setup.sh` before every session**: sets `PYTHONPATH`
  (`.:src`), MuJoCo render backend, `HF_LEROBOT_HOME`, `SO101_OUTPUT_DIR`,
  serial access, Quest/adb check, and a GPU/disk readout.
- **`bash install.sh`** is the one-shot installer (idempotent).
- **TWO SIBLING CHECKOUTS are required**, both shared with `../ur3e_raven`:
  - `../lerobot` — a source checkout, installed editable at a pinned commit with
    extras `feetech,dataset,pi,libero,pusht,training,diffusion,peft`. To inspect
    the real train/eval API, read `../lerobot/src/lerobot/...` — do not guess CLI
    flags.
  - `../actoris_harena` — the shared pipeline, installed `[rig]` and never
    `[sim]`. The two extras cannot coexist: LeRobot pins `numpy>=2.0,<2.3` and
    the simulation stack pins `numpy<2.0`.

  INSTALL ORDER MATTERS, and the reason is easy to trip over: actoris_harena is
  installed EDITABLE, so whatever branch is checked out there is what this repo
  imports. A `git checkout` in that repo changes this one's behaviour with no
  warning.
- **Hardware varies by machine — check, don't assume.** `source setup.sh`
  prints the live GPU/VRAM readout; trust that over any note here. Two
  machines seen so far: a laptop with an RTX 3050 (4 GB VRAM, too small
  for pi0.5 train/eval — keep local runs to the diffusion smoke test) and
  a remote box with an RTX 3090 Ti (~24 GB VRAM, no sudo access — enough
  headroom for real pi0.5 LIBERO *evaluation*, though finetuning still
  wants datacenter-scale GPUs).

## Layout

**THE PIPELINE IS NOT IN THIS REPO.** Data collection, dataset curation,
training, deployment, the browser console and every policy this project trains
live in `actoris_harena` (`../actoris_harena`, installed `[rig]`), shared with
the single-arm UR3e rig in `../ur3e_raven`. What is here is what is actually
about THESE ARMS.

Where to look for what:

| In `actoris_harena` | In this repo |
|---|---|
| `recording/` — the recorder, the captures, dataset integrity, camera views | `common/recording/{observations,sidecar,episode_motion,monitor_server,controls,usb_budget}.py` |
| `training/` — destinations, the run matrix, the log parser | `hpc/`, the run matrix's rows, the drivers under `test/system/` |
| `analysis/` — the whole attribution study | `tool/analyse_policy_inputs.py`, `tool/analysis_slides.py` |
| `deploy/` — chunking, the wire, the client, the Rig protocol | `BenchRig` in `tool/run_policy.py`, `tool/replay_on_robot.py` |
| `web/` — the console and its rig-independent tabs | `common/web/{datasets_api,session,sensors,policy_view,policy_ghost}.py`, `tool/rig_web.py` |
| `policies/` — all nine, ports and originals | nothing; `src/so101_policies/` is a SHIM (see below) |

- `common/rig_profile.py` — **the one place this rig hands the shared pipeline
  its facts**, run as a side effect of importing `common`: the camera profile
  (pi0.5 slots, the `tactile_quad` composite, the slug elisions), the recording
  config path, the destinations file and this checkout, and the gripper columns.
  Load-bearing, and its failure mode is not always loud — a tool that reaches
  `actoris_harena` without importing `common` gets an unconfigured pipeline, and
  an empty camera profile produces a DIFFERENT run-directory slug rather than an
  error, so a run trains fine and lands where nothing looks for it.
  `test/unit/test_migration_seams.py` guards it.
- `common/robot_schema.py` — twelve channels: five body joints then a gripper,
  per arm. `GRIPPER_COLUMNS` is `(5, 11)` and is DERIVED from the schema, never
  written down, so the layout and the columns cannot drift apart.
- `common/recording/observations.py` — `DualArmObservations`, this rig's answer
  to the shared recorder's three questions (state and action, EE, armed). Its
  world-to-base transforms are computed lazily: they build a pinocchio model,
  which costs a second, and a session not recording EE should not pay for it.
- `src/so101_policies/` — **a shim**. The policies are
  `actoris_harena.policies`, registered under `harena_*` with every legacy
  `so101_*` name kept as an alias. The shim exists because
  `--policy.discover_packages_path=so101_policies` is written into Slurm scripts
  already submitted and into `train_config.json` files a resuming job reads. Its
  docstring says what must be true before it can be deleted.
- `src/common/` — teleop: `configs.py` (all tuning constants),
  `threads/dual_ik_solver.py` (the production IK loop), `workspace_envelope.py`,
  `pink_ik_solver.py`, `one_euro_filter.py`, `data_manager_dual.py`, `utils.py`
  (the two-handed operator control frame), and `recording/` (this rig's half of
  collection: the sidecar, the EE motion model, the monitor's joint rows, the
  operator control list, the measured USB budget).
- `tool/` — runnable entry points: `meta_quest_teleopration.py` (real arms, with
  `--record`), `quest_sim_teleop.py` (sim rehearsal), `rig_agent.py` (this rig's
  devices, for the shared console, in this venv), `rig_web.py` (this repo's own
  way into the console), `telegrip_native.py`, `check_mirror.py` /
  `fit_joint_offsets.py`, `collect_preflight.py`, `view_twin.py`,
  `collect_sim_dataset.py` / `eval_sim_policy.py`, `run_policy.py` /
  `policy_server.py`, `policy_report.py`, `analyse_policy_inputs.py`,
  `analysis_slides.py`, `train_launch.py`.
- `rig.yaml` — what the shared console needs to drive this rig WITHOUT importing
  it: the interpreter, the agent, the teleop entry point, the cameras and the
  schema. A unit test checks its schema block agrees with `robot_schema.py`.

- `src/common/recording/usb_budget.py` — how many camera streams fit on each USB
  controller, and which selection does not. Pure string work over the by-path
  aliases in `sensor_map.yaml` (no device is opened), so the console and the
  preflight can both warn before a session starts. MEASURED: every camera is USB
  2.0, so bandwidth is allocated per HOST CONTROLLER (a hub adds none — these
  hubs are already USB 3.0 and it changes nothing). What a controller carries is
  NOT a stream count: one cost model fits every trial — a **tactile camera costs
  2 units, an RGB camera 1, a controller carries 4** (so `central` + 2 tactile is
  refused, but 2 tactile alone, or `central` + a wrist + 1 tactile, both fit).
  Which stream is refused is **random**, and a refused one opens and then
  delivers nothing for ever. Neither a lower rate nor a smaller frame helps:
  each camera reports exactly ONE frame interval per format+size (tactile: 60 fps
  at 640x480, 30 fps at 320x240; wrists: 30 fps), so there is no slower mode to
  ask for — which is also why `fps:` in `recording.yaml` cannot slow a camera.
  The uvcvideo FIX_BANDWIDTH quirk was MEASURED not to help (uvcvideo appears to
  skip it for compressed formats); `quirks_active()` only reports its state. The
  fix was a third host controller, and it worked: the rig's current five cameras
  (both wrist cameras unplugged and disabled) run as `central` alone on
  `pci-0000:06:00.3` plus a tactile pair on each of `05:00.4` and `06:00.4` —
  see `documents/rig_web.md`.
- `tool/compare_port_training.py` — does a port TRAIN like its twin? Runs the
  real `lerobot-train` twice from one seed and compares the logged loss step for
  step, which is the only thing that covers what the trainer assembles AROUND
  the model (processors, optimiser preset, dataloader, plugin discovery).
  MEASURED on thanos: `act`/`so101_act` and `diffusion`/`so101_diffusion` agree
  at every logged point. `make test-port-parity DATASET_ROOT=<ds>` is the CPU
  version; `test/integration/test_policy_ports_checkpoints.py` is the other half
  — one batch, but loss and every gradient compared bit for bit.
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
  `training_api.py` (the Training tab's routes; the rules are
  `common.training` and the launch is `tool/train_launch.py`, so the page
  cannot start a run the terminal would refuse — refusals come back at 200
  inside the plan, and a launch is a `jobs.py` record because staging is
  minutes) + `projects.py`/`projects_api.py` (grouping DISCOVERED runs into
  named experiments: a run is keyed `machine|run|policy` and the membership
  lives apart from the launch records, because most runs have none. A member no
  machine answered for is reported as missing rather than dropped — a list that
  silently shrank is the one way this view could misreport what was run),
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
  under `static/` (`index.html` + one script per tab, no build step). The
  Training tab is the one pane split into TWO SUBVIEWS, so `app.js` grew a
  generic `showSubview` over `.subview[data-view]` and the hash addresses one
  (`#training/jobs`): `training.js` is Start Training (its policy list grouped
  LeRobot / this repo off `local` and `ported_from`), `training_jobs.js` is
  Training Jobs (projects, sortable table, config diff, panel-per-metric,
  smoothing, hover readout) and `chart.js` is the drawing they share. With no
  build step and no module system every script runs at top level against one
  page, so a typo'd element id kills that whole file silently —
  `test_rig_web_routes.py` cross-checks every `$('#id')` in every script
  against the markup. The same package also
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
  `outputs/policy_runs/`: ticks, chunks, and `trials.jsonl` — one VERDICT per
  attempt, given from the page, where a `discard` leaves the denominator rather
  than counting against the policy and an unjudged run reads as unscored, never
  as nought per cent; `--log-frames` additionally keeps the frames each plan was
  drawn from, off by default and required by `analyse_policy_inputs.py`. Read
  back by `tool/policy_report.py`). Runbooks: `documents/rig_web.md`,
  `documents/remote_policy_inference.md`. Deletion is always available; every irreversible
  one asks in the browser first, and marking an episode (reversible) does
  not. The third tab is called **Signals** in the UI while the module,
  routes and `sensor_map.yaml` keep the older `sensor` name; the fourth is
  **Training**.
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
- `hpc/` — the Slurm cell on KCL CREATE, the plain-GPU-box path, and the traffic
  in both directions. `tool/train_launch.py` is the ONE front door over both
  (stage → write the manifest → submit → record); `gpu_box_run.sh` is the
  non-Slurm executor, which holds a `flock` because that card is single-tenant
  and detaches under `setsid` so a closing SSH cannot end a 36-hour run. Then:
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
  - **FastWAM is not in the pinned LeRobot.** `3dd19d04` (2026-06-27) has no
    `src/lerobot/policies/fastwam`; it exists upstream. `matrix.policy_available`
    probes for the module rather than comparing versions, so a `fastwam` row is
    refused at submit time and the gate opens by itself when `LEROBOT_COMMIT`
    moves. `install.sh`/`provision_create.sh` already name the `fastwam` extra
    (MEASURED: pip ignores an extra the checkout does not define).
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
  - **A camera that stops mid-episode leaves a dataset every count agrees
    with.** The metadata row, the frames, the side files and the totals are all
    written and consistent; only that one camera's video is short. It opens, it
    trains, and it dies partway through the first pass with LeRobot's
    `FrameTimestampError`, naming a timestamp rather than a cause — on a rented
    GPU, hours in. MEASURED on `fold-short-from-flattend-tactile`: episode 63
    held 260 rows and 260 frames from each of the four fingertip cameras, and
    213 from `central`; every other check in `dataset_check.py` passed it.
    `camera_span_problems` is the guard (`to - from == length / fps`, per camera
    per episode) and it runs inside `dataset_integrity`, so `ensure_loadable`
    refuses such a dataset before staging. It is NOT repairable automatically:
    the frames are gone, and dropping the episode versus keeping the recorded
    part depends on what it shows.
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
  - **A box that reboots mid-run used to cost the WHOLE run.** `train_cell`
    reused a FINISHED checkpoint but `rm -rf`'d a partial one, while
    lerobot-train has supported `--resume=true --config_path=<ckpt>/
    train_config.json` all along. It now decides three ways — reuse, resume,
    delete only what has no checkpoint — so an interruption costs one
    `save_freq`. MEASURED: thanos rebooted at 20:08 on 2026-08-28 and again the
    day before; the log just stops mid-tqdm-frame with no traceback and `who -b`
    is the only record. A resume APPENDS to the same log, and lerobot builds its
    bar with `total = steps - step`, so the second attempt's frames count from
    zero and its metric lines restart their ordinal — `progress.py` reads them
    apart by the driver's `↻ resuming <run> at step N of M` line, NOT by
    comparing totals (that moves the first attempt's points too).
  - **The step in a training log is a label, not a number.** `format_big_number`
    (`lerobot/utils/utils.py`) divides by a thousand per suffix and rounds, so
    `step:`, `smpl:` and `ep:` are lossy above 1000 — steps 10 500 and 10 600
    both print `10K`. `loss`, `grdn`, `lr`, `updt_s`, `data_s`, `smp/s` and
    `mem_gb` are full precision. Anything plotting a curve must reconstruct the
    x-axis (`common/training/progress.py`); a plot against `step:` piles two
    thirds of an 80 000-step run onto eight x-values.
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
