# Long VLA simulation experiment — run guide

How to run the full-scale train/eval experiment
(`test/system/long_vla_sim.sh`) on a machine with a capable GPU
(validated target: RTX 3090 Ti, 24 GB, no sudo). The experiment
collects oracle-teleoperated demonstrations in the digital-twin
simulation, trains **ACT**, **Diffusion Policy**, and **pi0.5** on
them, and evaluates each on held-out scenario seeds.

> The script's `--help` output is authoritative if it ever disagrees
> with this guide. For a fast plumbing-only check of the same pipeline,
> run `make test-system-vla` (the smoke variant) instead.

## 1. What the experiment does

Two environment setups, run in order:

| Mode | Train demos/task | Train seeds | Val seeds | Eval seeds |
|---|---|---|---|---|
| `simple` | ~100 (seed-0 scenario only) | {0} | {0} | {0} × 30 trials |
| `full` | ~1000 (one demo per seed) | 0–999 | 10000–10009 | 20000–20029 |

The two tasks share one 2.2 cm cube. `single` is a one-arm
pick-and-place; `handover` is a **bimanual relay** — the left arm picks
the cube on the left and lays it at the midline, then the right arm
picks it up there and places it on the right target (a table-mediated
hand-off, so nothing is held mid-air by one arm).

For each mode × task (`single`, `handover`) ×
policy (`act`, `diffusion`, `pi05`):

0. **Oracle gate** — dry-runs both oracles (`teleop` = scripted
   operator through the full teleoperation pipeline, `direct` = IK
   fallback) and prints a success-rate table. Aborts if the teleop
   oracle is below 95 % (single) / 70 % (handover): do not train on a
   broken oracle. The handover bar sits well below the teleop relay's
   measured ~75–86 % per-attempt band so sampling noise on a 30-episode
   dry run never false-aborts, yet far above a genuinely broken (<50 %)
   oracle. Every saved episode is still a verified success (failures
   are discarded, not trained on); the `direct` oracle remains at 100 %
   for either task if a fuller dataset is wanted.
1. **Collect** — demonstrations through the teleop oracle; failed
   episodes are never saved (a failed seed retries up to 3×, then is
   skipped and recorded).
2. **Train** — ACT and Diffusion from scratch; pi0.5 finetuned from
   `lerobot/pi05_base` with LoRA (`--peft.r=16`).
3. **Validate** — every saved checkpoint is rolled out on the val
   seeds; the best checkpoint per cell is selected.
4. **Evaluate** — the selected checkpoint only, on **all 30 eval
   seeds** (one trial per seed).
5. **Report** — `results.md` + `results.json` (success-rate table with
   mode as a dimension, place-error stats, seed pools, per-seed
   outcomes, selected checkpoint steps).

The three seed pools are disjoint by construction; train/val/eval never
share a scenario in `full` mode. `simple` mode deliberately reuses the
seed-0 scenario everywhere — it is the overfit-one-scenario sanity
check: a policy that cannot master a single fixed scenario has a bug,
not a data problem.

## 2. Prerequisites (once per machine)

1. Repo cloned with the LeRobot source checkout beside it
   (`../lerobot`), then `bash install.sh`. If `python3` resolves to
   something older than 3.12, force it:
   `/usr/bin/python3.12 -m venv venv` first (see CLAUDE.md).
2. `source setup.sh` — check its GPU/disk readout. You want
   ≥ 20 GB free VRAM and **≥ 150 GB free disk** (two ~1000-episode
   three-camera video datasets plus checkpoints).
3. Digital-twin assets present: `PYTHONPATH=.:src MUJOCO_GL=egl
   venv/bin/python -m sim_twin.verify` must pass (regenerates
   `build/twin` from OpenSCAD if the pipeline is installed).
4. Network on first run only: HF Hub access for `lerobot/pi05_base`
   (~8 GB) and the PaliGemma tokenizer. The tokenizer repo
   (`google/paligemma-3b-pt-224`) is **gated**: log in once with
   `venv/bin/hf auth login` and accept the licence on the model page,
   or the pi0.5 cells abort in preflight. Everything else is local.
5. No sudo needed. Video decode/encode uses PyAV's bundled FFmpeg
   (`--dataset.video_backend=pyav` is passed everywhere by the script).

## 3. Running it

```bash
cd ~/Projects/so101_garment
source setup.sh
bash test/system/long_vla_sim.sh                 # both modes, all cells
```

Useful variants:

```bash
bash test/system/long_vla_sim.sh --modes simple            # sanity half only
bash test/system/long_vla_sim.sh --modes full --only pi05  # one policy
bash test/system/long_vla_sim.sh --skip-collect            # reuse datasets
bash test/system/long_vla_sim.sh --skip-train              # re-eval only
bash test/system/long_vla_sim.sh --pi05-steps 20000        # higher pi0.5 quality
bash test/system/long_vla_sim.sh --simple-episodes 50      # cheaper sanity half
bash test/system/long_vla_sim.sh --val-trials 5            # 5–10 supported
```

### Running in the background (tmux)

The full matrix is multi-day, so run it inside `tmux` — the script
keeps running on the server after you disconnect:

```bash
tmux new -s vla                  # start a session (skip if already inside one)
bash test/system/long_vla_sim.sh
# Ctrl+b d  — detach; the run continues on the server
tmux attach                      # reattach later (tmux ls lists sessions)
# Ctrl+b c  — new window alongside the run; Ctrl+b n/p switches
# Ctrl+b [  — scroll back through the run's output (q exits)
```

Monitor without attaching, from any other login:

```bash
tail -f $SO101_OUTPUT_DIR/vla_sim_long/<run-name>/logs/*.log
ls $SO101_OUTPUT_DIR/vla_sim_long/<run-name>/oracle_gate/
```

Nothing looks stuck for hours by design: Phase 0 and teleop-oracle
collection run in real time, and the pi0.5 cells print a measured
s/step projection after ~50 steps (see the wall-time table below).

### Resuming after a crash

Do **not** simply rerun the bare command — each bare invocation makes
a fresh timestamped run directory and starts from scratch. Rerun with
the SAME run name plus skip flags for the phases already done; within
a run dir the script also reuses per-cell artefacts it finds (gate
JSONs, datasets, checkpoints):

```bash
bash test/system/long_vla_sim.sh --run-name long_20260723_141954 \
    --skip-gate --skip-collect          # e.g. died during training
```

### Wall-time budget (3090 Ti, defaults)

| Phase | Estimate |
|---|---|
| Oracle gate | ~1 h (teleop episodes are real-time) |
| Collect, simple (2 × ~100 demos) | ~2 h |
| Collect, full (2 × ~1000 demos) | ~9–11 h **per task** |
| ACT (80k steps) / Diffusion (100k steps) | ~5–8 h / ~10–14 h per cell |
| pi0.5 LoRA (default 10k steps) | ~1–2 days per cell; the script prints a measured s/step projection after ~50 steps — decide then whether to trim `--pi05-steps` or drop cells |
| Val + final eval | ~2–4 h total |

The teleop oracle collects in real time (it drives the actual
asynchronous teleoperation stack). If collection time is the
bottleneck, `--oracle direct` collects faster than real time at the
cost of not passing through the teleoperation pipeline.

## 4. Outputs

Everything lands under `$SO101_OUTPUT_DIR/vla_sim_long/<timestamp>/`:

```
oracle_gate/            per-cell dry-run stats JSON
datasets → $HF_LEROBOT_HOME/so101_sim_<task>_<mode>/   (+ stats JSON)
<mode>/<task>/<policy>/ train run dir (checkpoints/, logs/)
<mode>/<task>/<policy>/val/  per-checkpoint val results
<mode>/<task>/<policy>/eval/ final 30-seed results.json (+ videos if enabled)
results.md, results.json     the aggregate report
```

Send back (or commit on the experiment branch): `results.md`,
`results.json`, the collection stats JSONs, and the final-eval
`videos/` directories. These feed the paper's experiments section.

### Viewing collected data

Every collection prints a "view your data" block with these commands
pre-filled. On a machine with a display:

```bash
venv/bin/lerobot-dataset-viz --repo-id local/so101_sim_single_full \
    --root "$HF_LEROBOT_HOME/local/so101_sim_single_full" --episode-index 0
```

On the headless remote box, either stream to your laptop:

```bash
remote$ venv/bin/lerobot-dataset-viz --repo-id <id> --root <root> \
            --episode-index 0 --mode distant --grpc-port 9876
laptop$ rerun rerun+http://<REMOTE_IP>:9876/proxy
```

or export and copy an `.rrd` file (`--save 1 --output-dir <dir>`, then
`scp` it and open with `rerun <file>.rrd`). The hub's web viewer
(https://huggingface.co/spaces/lerobot/visualize_dataset) only works
for datasets pushed to the HF hub — these datasets are local-only by
design and are never pushed automatically.

### Analysing the results

Open `notebooks/long_vla_analysis.ipynb` (VSCode's notebook support or
`venv/bin/pip install notebook && venv/bin/jupyter notebook`), point
its first cell at the run directory, and run all cells. It produces
the success-rate and place-error tables, per-seed heatmaps and
validation curves, and composes head-to-head GIFs — the same eval
scenario played side by side across ACT / Diffusion / pi0.5 (and the
oracle demonstration where available) — displayed inline and written
into the run directory.

## 5. Troubleshooting

- **Headless render fails** — `MUJOCO_GL=egl` must be set
  (`setup.sh` does this); on a driverless node try `osmesa` (slow).
- **`--output_dir` exists** — lerobot-train refuses to overwrite; the
  script creates fresh timestamped dirs, so this only happens when
  resuming by hand. Delete or rename the stale dir.
- **Hub timeouts mid-run** — after the first pi0.5 download everything
  is cached; re-run with `HF_HUB_OFFLINE=1` exported.
- **pi0.5 OOM** — confirm the run uses LoRA (`--peft.r=16` in the train
  log); full finetuning does not fit in 24 GB.
- **Oracle gate fails** — the contact-grasp tuning has regressed;
  re-run the tuning loop
  (`venv/bin/python tool/collect_sim_dataset.py --task single
  --episodes 30 --dry-run --gif outputs/oracle_gifs`) and inspect the
  GIFs before touching training.
- **Eval success is 0 for every policy** — check the eval camera
  resolution matches collection (the script passes identical values;
  a hand-run of `tool/eval_sim_policy.py` must repeat them).
