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

## Step-by-step

A zero-to-results walkthrough. Each step says **where it runs**
— a CREATE *login node*, the *collection box* (this machine, which holds
the dataset), or a CREATE *GPU node* (via the submitted job). Fill every
`<...>` with a value you discover in step 2.

### 0. Prerequisites

A CREATE account with GPU access and SSH; the collection box still holds
`so101_sim_single_simple` under its `HF_LEROBOT_HOME`
(`/home/ah390/.cache/huggingface/lerobot/local/` on the box these scripts
were built on).

### 1. Log in — *login node*

```bash
ssh <user>@<create-login-host>      # <create-login-host>: from the CREATE docs / your onboarding
```

### 2. Discover your specifics — *login node*

Run each and record the value; these replace the guesswork the sbatch
placeholders stand in for.

```bash
# identity + scratch root  ->  <scratch>
echo "$USER  $HOME"
ls -d /scratch/users/$USER 2>/dev/null || ls /scratch   # e.g. /scratch/users/<k-number> or /scratch/prj/<project>

# GPU partition  ->  <gpu-partition>   (look for a gpu* row)
sinfo -s

# account, if the cluster enforces one  ->  <account>   (empty output => delete the sbatch --account line)
sacctmgr -np show assoc user=$USER format=account

# a Python >=3.12 module, IF one exists  ->  <python-module>
module avail python 2>&1 | grep -iE 'python[-/][._]*3\.1[2-9]'
```

Python must be **≥3.12** — LeRobot's editable install silently fails on an
older `python3` (see CLAUDE.md). **CREATE has no ≥3.12 module today** (the
command above comes back empty; its 3.10/3.11 modules are too old), so
`provision_create.sh` bootstraps a self-contained CPython 3.12 with
[`uv`](https://docs.astral.sh/uv/) automatically — nothing to `module
load`. If your cluster *does* expose a ≥3.12 module, note its name as
`<python-module>` and load it in step 4 to use it instead.

### 3. Clone the repo — *login node*

```bash
git clone <repo-url> so101_garment
cd so101_garment && pwd        # -> <REPO_ROOT>
```

`provision_create.sh` clones the sibling `../lerobot` at the pinned commit
for you, so there is nothing else to fetch.

### 4. Provision the environment — *login node, once*

The login node has internet; compute nodes do **not**, which is why this
(and the cache warm-up) happens here.

```bash
# module load <python-module>   # ONLY if step 2 found a >=3.12 module
bash hpc/provision_create.sh
```

On CREATE there is no ≥3.12 module, so leave the `module load` out: the
script finds no qualifying `python3` and bootstraps a standalone CPython
3.12 with `uv` (it installs `uv` to `~/.local/bin` on first run). It then
prints `=> Using interpreter: … (Python 3.12.x)` and continues. If step 2
*did* find a module, `module load` it first and the script uses that
instead.

Success signals: the venv builds and the script prints
`✓ ResNet18 weights cached …` then `✓ CREATE provisioning complete`.
Idempotent — safe to re-run.

### 5. Stage the dataset — *collection box (NOT CREATE)*

Only the `single`/`simple` dataset (~758 MB) is needed for this cell. Run
this **on the collection box**, pushing to your CREATE scratch:

```bash
rsync -avP \
  /home/ah390/.cache/huggingface/lerobot/local/so101_sim_single_simple \
  <user>@<create-login-host>:<scratch>/hf_lerobot/local/
```

Confirm arrival — *login node*:

```bash
ls <scratch>/hf_lerobot/local/so101_sim_single_simple/meta/info.json
```

`<scratch>/hf_lerobot` must equal `HF_LEROBOT_HOME` in the sbatch (step 6).

### 6. Fill the placeholders in `hpc/create_sim_vla.sbatch` — *login node*

| Placeholder | Value | Found by |
|-------------|-------|----------|
| `--partition=<gpu-partition>` | your GPU partition | `sinfo -s` (step 2) |
| `--account=<account-or-delete-line>` | your account, or **delete the line** | `sacctmgr …` (step 2) |
| `REPO_ROOT="<path-to>/so101_garment"` | the clone path | `pwd` (step 3) |
| `SCRATCH="<scratch>"` | your scratch root | step 2 |
| `# module load cuda/<version>` | usually leave commented | only if your node needs it |

### 7. Submit and monitor — *login node → GPU node*

```bash
sbatch hpc/create_sim_vla.sbatch
squeue --me                          # QUEUED -> RUNNING
tail -f sim_vla_single-<jobid>.out   # live log
```

A healthy start prints, near the top of the log,
`✓ MuJoCo EGL offscreen render OK` and passes the
`staged dataset present?` check — **before** any training. The job then
trains ACT (80k steps) and Diffusion (100k steps, cameras downsampled to
180×240 to fit the encoders), validates every checkpoint on the VAL seeds,
and evaluates the best on all 30 EVAL seeds, ending with `LONG RUN
COMPLETE`.

### 8. Collect results — *login node, then anywhere*

```
<scratch>/so101_outputs/vla_sim_long/create_<jobid>/simple/single/
    act/eval/{results.json, videos/*.mp4}
    diffusion/eval/{results.json, videos/*.mp4}
    results.md            # summary table
```

`results.json` holds `success_rate` + placement-error stats; each
`videos/ep_*.mp4` is the enhanced composite (scene + both wrist cams +
third-person overview, over scrolling measured-vs-target joint traces).
Pull them to a machine with a screen to watch:

```bash
rsync -avP \
  <user>@<create-login-host>:<scratch>/so101_outputs/vla_sim_long/create_<jobid> \
  ./create_<jobid>
```

### 9. Troubleshooting

- **EGL probe fails** (`EGLError` / no render) on a multi-GPU node — MuJoCo
  is likely on a GPU other than the one Slurm gave you: uncomment
  `MUJOCO_EGL_DEVICE_ID` in the sbatch (it pins MuJoCo to the allocated
  GPU).
- **Provisioning can't get Python ≥3.12** — normally the `uv` bootstrap
  handles this, but it needs the login node's internet. If `uv` itself
  fails to install (`❌ uv not found after install`), check `curl` and
  outbound HTTPS work on this node, or `module load` a ≥3.12 module if one
  exists (step 2) and re-run. Never run `provision_create.sh` on a compute
  node — they are offline.
- **Hit the wall-time** — resubmit with the **same** `--run-name` (edit the
  sbatch to hard-code it instead of `create_${SLURM_JOB_ID}`); the driver
  reuses finished checkpoints, val results and eval results.
- **`staged dataset missing`** — the path in step 5 must equal
  `HF_LEROBOT_HOME` (`<scratch>/hf_lerobot`) with the dataset under
  `local/so101_sim_single_simple`.

## Scaling out later

- More cells: drop `--only`/`--tasks`/`--modes` (or widen them) to run the
  full matrix — but full mode collects ~1000 demos, which needs the oracle
  gate + collection phases (remove `--skip-collect`) and much more time.
- Parallelism: submit one job per policy (a Slurm array), each with its
  own `--only`, instead of training them sequentially in one job.
