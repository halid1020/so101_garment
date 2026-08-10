# Running the sim-VLA pipeline on KCL CREATE (Slurm)

This directory provisions and submits the **minimal working sim-VLA cell**
— task `single`, mode `simple`, policies `act` + `diffusion` — on the
[KCL CREATE](https://docs.er.kcl.ac.uk/) cluster. It reproduces the recipe
that reaches 100 % eval success on the local box (diffusion 5/5, ~9 mm
placement), training from a pre-staged dataset (`train → validate → eval`)
and skipping the slow teleop-oracle collection.

Two files do the work:

| File | Where it runs | What it does |
|------|---------------|--------------|
| `provision_create.sh` | CREATE **login** node, once | Builds the venv + LeRobot (pinned) + requirements; warms the vision-backbone cache. Sim-only subset of `../install.sh`. |
| `create_sim_vla.sbatch` | CREATE **compute** node, via `sbatch` | Sets scratch paths + `MUJOCO_GL=egl`, checks EGL + dataset, runs `test/system/long_vla_sim.sh`. |

## Why a CREATE-specific path (not just `install.sh` + the driver)

- **Compute nodes have no internet.** Both ACT and Diffusion Policy build
  a torchvision ResNet18 whose default weights download from a CDN.
  `provision_create.sh` **warms** that cache on the login node
  (`~/.cache/torch/hub/checkpoints`, visible to compute nodes) so training
  never reaches for the network. The pre-staged dataset (below) means no
  Hub download either. (pi0.5 is intentionally out of scope here — it
  would need its licence-gated base pre-staged.)
- **`install.sh` isn't cluster-safe.** Its later steps `sudo apt install`
  adb/openscad and clone `meta_quest_teleop`; none are needed for sim
  training/eval and they abort without sudo. `provision_create.sh` runs
  only the venv + LeRobot + requirements + cache steps.
- **Storage.** `outputs/` runs to tens of GB; home quotas are small, so
  the job points `SO101_OUTPUT_DIR` and `HF_LEROBOT_HOME` at scratch.

## Steps

### 1. Get the code onto CREATE (login node)

```bash
# this repo and the LeRobot checkout live side by side (../lerobot)
git clone <this-repo-url> so101_garment
# provision_create.sh clones ../lerobot for you at the pinned commit
```

### 2. Provision the environment (login node, once)

```bash
cd so101_garment
module load <a Python >=3.12 module>     # verify the exact name: `module avail python`
bash hpc/provision_create.sh
```

Idempotent — safe to re-run. On success `venv/` is built and the ResNet18
weights are cached.

### 3. Stage the dataset to scratch (from the collection box)

Only the `single`/`simple` dataset (~758 MB) is needed for this cell:

```bash
rsync -avP \
  /home/ah390/.cache/huggingface/lerobot/local/so101_sim_single_simple \
  <user>@<create-login-host>:<scratch>/hf_lerobot/local/
```

`<scratch>/hf_lerobot` must match `HF_LEROBOT_HOME` in the sbatch script.

### 4. Fill the placeholders in `create_sim_vla.sbatch`

Every `<...>`: `--partition`, `--account` (delete the line if unused),
`REPO_ROOT`, `SCRATCH`, and the optional `module load cuda`.

### 5. Submit and monitor

```bash
sbatch hpc/create_sim_vla.sbatch
squeue --me
tail -f sim_vla_single-<jobid>.out
```

The job runs the EGL + dataset fail-fast checks, then trains ACT (80k
steps) and Diffusion (100k steps, cameras downsampled to 180×240 to fit
the encoders), validates every checkpoint on the VAL seeds, and evaluates
the best on all 30 EVAL seeds.

### 6. Collect results

```
<scratch>/so101_outputs/vla_sim_long/create_<jobid>/simple/single/
    act/eval/{results.json, videos/*.mp4}
    diffusion/eval/{results.json, videos/*.mp4}
    results.md            # summary table
```

`results.json` holds `success_rate` + placement-error stats; each
`videos/ep_*.mp4` is the enhanced composite (scene + both wrist cams +
third-person overview, over scrolling measured-vs-target joint traces).

## Assumptions to verify on CREATE

These I could not confirm remotely — check them before the first run:

- **Scheduler / partitions.** CREATE is Slurm; confirm the GPU partition
  name with `sinfo -s` and whether an `--account` is required.
- **Python module.** `module avail python` for a `>=3.12` module (LeRobot
  requires it; an older `python3` silently breaks the editable install).
- **Scratch path.** e.g. `/scratch/users/<k-number>` or
  `/scratch/prj/<project>`; must be readable from compute nodes.
- **Headless EGL.** The sbatch probe renders a dummy MuJoCo scene and
  exits non-zero if EGL is unavailable. If it fails on a multi-GPU node,
  uncomment `MUJOCO_EGL_DEVICE_ID` in the script (it pins MuJoCo to the
  Slurm-allocated GPU).
- **Wall-time.** `--time=18:00:00` is generous for one GPU; trim to your
  partition's limit. To resume a timed-out run, resubmit with the **same**
  `--run-name` (the driver reuses finished checkpoints and eval results).

## Scaling out later

- More cells: drop `--only`/`--tasks`/`--modes` (or widen them) to run the
  full matrix — but full mode collects ~1000 demos, which needs the oracle
  gate + collection phases (remove `--skip-collect`) and much more time.
- Parallelism: submit one job per policy (a Slurm array), each with its
  own `--only`, instead of training them sequentially in one job.
