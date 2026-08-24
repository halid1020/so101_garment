# Running VLA training on KCL CREATE (Slurm)

This directory provisions and submits two training cells on the
[KCL CREATE](https://docs.er.kcl.ac.uk/) cluster, sharing one environment:

- the **sim-VLA cell** — task `single`, mode `simple`, policies `act` +
  `diffusion` — which reproduces the recipe that reaches 100 % eval success
  on the local box (diffusion 5/5, ~9 mm placement), training from a
  pre-staged dataset (`train → validate → eval`) and skipping the slow
  teleop-oracle collection;
- the **real-data cell**, which trains policies on datasets collected on the
  physical rig. It takes a list of datasets rather than one, and needs no
  edit to any tracked file to change what is trained.

The files:

| File | Where it runs | What it does |
|------|---------------|--------------|
| `provision_create.sh` | CREATE **login** node, once | Builds the venv + LeRobot (pinned) + requirements; warms the vision-backbone cache. Sim-only subset of `../install.sh`. Serves both cells. |
| `create_sim_vla.sbatch` | CREATE **compute** node, via `sbatch` | Sets scratch paths + `MUJOCO_GL=egl`, checks EGL + dataset, runs `test/system/long_vla_sim.sh`. |
| `stage_datasets.sh` | the **collection box** | Checks and rsyncs collected datasets to the cluster's scratch. |
| `runs.tsv` | — | The real-data run matrix: one row per (dataset, policy) run. |
| `submit_real.sh` | CREATE **login** node | Reads the matrix, checks staging, submits it as Slurm job arrays. |
| `create_real_vla.sbatch` | CREATE **compute** node, via `sbatch` | One array task = one row: runs `test/system/long_vla_real.sh` for that dataset and policy. |
| `fetch_policies.sh` | the **collection box** (or any machine) | Brings the finished checkpoints back out of scratch, into the layout the policy server and the on-robot runner expect. |
| `tool/make_camera_view.py` | wherever the dataset is | Builds a **camera view**: the same episodes with only some cameras named. The job calls it; you rarely do. |

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

### 9. Troubleshooting (sim cell)

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
  reuses finished checkpoints, val results and eval results. The real cell
  needs none of this: its run name is the dataset, so resubmitting resumes.
- **`staged dataset missing`** — the path in step 5 must equal
  `HF_LEROBOT_HOME` (`<scratch>/hf_lerobot`) with the dataset under
  `local/so101_sim_single_simple`.

## Real-data cell (`runs.tsv` + `submit_real.sh`)

Training policies on datasets **collected on the physical rig** reuses the same
environment (`provision_create.sh` is unchanged) but a different job. There is
no simulation here, so the job has no MuJoCo probe; and there is no in-loop
evaluation, because a real policy is evaluated **on the robot** with
`tool/run_policy_real.py` back at the rig, not on the cluster.

The unit of work is one **(dataset, policy)** pair. They are listed in
`hpc/runs.tsv`, and each becomes one Slurm array task. Nothing tracked has to be
edited to change which datasets are trained — edit the matrix, or filter it on
the command line.

### 1. Provision — *login node, once*

`bash hpc/provision_create.sh`, as for the sim cell. Both cells share the venv.

**Training any `pi05` row needs one more thing.** pi0.5 is a finetune, so the
base weights must be on disk before training starts, and a compute node cannot
fetch them. Pull them once, on the login node:

```bash
SO101_STAGE_PI05=1 bash hpc/provision_create.sh
```

That caches `lerobot/pi05_base` (~14.5 GB) under the shared HF cache, which
`HF_HUB_OFFLINE=1` then resolves by name. The repo is **not** licence-gated, so
no token and no terms acceptance are involved. Skip this and a `pi05` row fails
at startup; `act` and `diffusion` rows are unaffected either way.

### 2. Stage the datasets — *collection box (NOT CREATE)*

```bash
bash hpc/stage_datasets.sh --dir /mnt/seagate/so101 \
  --dest <user>@<create-login-host>:<scratch>/hf_lerobot/local \
  cube-pnp cube-dual-pnp-new fold-short
```

Every dataset is checked before anything is transferred: it must be a real
dataset, and it must have no episodes still **marked for deletion**. A marked
episode is only flagged until the dataset is compacted — it is still on disk, so
staging one would ship takes the operator threw away, and the training driver
would refuse the dataset on arrival anyway. Clear them in
`tool/rig_web.py` first.

`<scratch>/hf_lerobot` is the `HF_LEROBOT_HOME` the job will use.

### 3. Edit the run matrix — `hpc/runs.tsv`

One row per run: `dataset policy cameras steps batch hours extra`. `-` means
"the driver's default for this policy"; `extra` is the last column and is handed
to `lerobot-train` verbatim, so a flag the driver does not name is still
reachable. Rows sharing an `hours` value are submitted as one array with that
wall time, so a short cell does not queue behind a long reservation.

`policy` is `act`, `diffusion` or `pi05`. `cameras` is `all` or a comma list of
camera names **with no spaces** — a space there would shift every later column
one place left, so the wrapper refuses the row rather than training the wrong
size for the wrong time.

#### What the `cameras` column does

Ablating a camera has to change *only* which cameras the policy sees. The job
therefore builds a **view** of the dataset for each distinct camera set: a
directory LeRobot opens as an ordinary dataset, whose metadata names only those
cameras and whose video files are symlinks back to the source. Same episodes,
same frames, same actions.

That is cheap in both senses. On disk a view is a few MB against the source's
hundreds, because the videos — 99% of the bytes — are shared rather than copied.
In time it is a saving, because `video_keys` comes from the view's own
`info.json`, so a camera that is not named is never decoded: measured locally at
batch 8, three cameras cost 127 ms/batch and one costs 56 ms.

Building is idempotent and atomic, so array tasks that want the same view cannot
collide and a resubmission reuses what is there. A run directory is named after
the view (`cube-pnp-new__all`, `cube-pnp-new__central+wrist_left`,
`cube-pnp-new__wrist_left`), which is what keeps the arms of an ablation from
overwriting one another.

Views also settle the dataset's **task string**. `cube-pnp-new` carries two
spellings of one instruction because it was retyped partway through collection;
the view keeps whichever covers the most frames and rewrites the rest. `act` and
`diffusion` ignore language, but pi0.5 is conditioned on it. The source dataset
is never modified.

#### How pi0.5 differs

Three consequences of it being a finetune of a 4.1B-param base:

- it starts from `--policy.path` (the base staged in step 1), not a policy type;
- it trains through **LoRA** (`--peft.r=16`), because full finetuning needs more
  than 24 GB for AdamW's state alone. `--lora-r 0` turns that off for a very
  large GPU;
- its cameras are **renamed** onto the base's three pretrained slots
  (central → `base_0_rgb`, left wrist → `left_wrist_0_rgb`, right wrist →
  `right_wrist_0_rgb`) rather than derived from the dataset, because each slot
  carries what it learned about that viewpoint. A slot with no camera behind it
  is padded and masked by pi0.5 itself — which is exactly what an ablated camera
  should look like to the model.

### 4. Submit — *login node → GPU nodes*

```bash
bash hpc/submit_real.sh --dry-run        # see exactly what would be submitted
bash hpc/submit_real.sh                  # everything in the matrix
bash hpc/submit_real.sh --datasets cube-pnp-new --only act
bash hpc/submit_real.sh --cameras wrist_camera_left    # one arm of the ablation
bash hpc/submit_real.sh --partition <gpu-partition> --account <account> --concurrency 2
```

The wrapper discovers your scratch (`/scratch/users/$USER` unless `--scratch`
or `$SO101_SCRATCH` says otherwise), refuses to submit a dataset that is not
staged — printing the `stage_datasets.sh` line that would fix it — and
**snapshots the rows it submits** under
`<scratch>/so101_outputs/submissions/<stamp>/`, together with an `index_*.md`
mapping each array id to its dataset and policy. The array reads the snapshot,
so editing `runs.tsv` afterwards cannot shift the indices of a queued array.

Monitor with `squeue --me`; logs land in the submit directory as
`real_vla-<arrayjobid>_<taskid>.out`, and the `index_*.md` says which task is
which run.

### 5. Collect results — *login node, then anywhere*

```
<scratch>/so101_outputs/vla_real_long/<dataset>__<cameras>/
    train/{act,diffusion,pi05}/checkpoints/last/pretrained_model
    results_<policy>.md     # one per finished run
    results.md              # that camera set's summary table
    logs/
```

Every policy trained on one camera set shares that directory, because the run
name is the **view**, not the job id. That is also what makes a resubmission
cheap: a job that hit its wall time is simply submitted again, and every
finished checkpoint is reused instead of retrained.

The camera ablation on `cube-pnp-new` therefore lands as three directories —
`cube-pnp-new__all`, `cube-pnp-new__central+wrist_left`,
`cube-pnp-new__wrist_left` — with three policies inside each.

### 6. Evaluate — *at the rig*

Bring the finished checkpoints back with `fetch_policies.sh`, which copies only
`checkpoints/last/pretrained_model` (the run directory itself is tens of GB) into
one directory per (dataset, policy):

```bash
bash hpc/fetch_policies.sh --from <user>@<create-login-host> --list
bash hpc/fetch_policies.sh --from <user>@<create-login-host> --dest ~/outputs/policies

# the whole ablation, or one arm of it
bash hpc/fetch_policies.sh --from <host> --datasets cube-pnp-new --dest ~/outputs/policies
bash hpc/fetch_policies.sh --from <host> --datasets cube-pnp-new__wrist_left --dest ~/outputs/policies
```

Then `tool/run_policy_real.py --checkpoint <ckpt> --task "<task>"` — start with
`--dry-run`. A checkpoint too large for the rig's own machine can be served from
a GPU box instead (`--dest <gpu-host>:...`, then `tool/policy_server.py` +
`--server <url>`); see
[`../documents/remote_policy_inference.md`](../documents/remote_policy_inference.md).

**pi0.5 trains here now** (`--only pi05`), provided its base was staged in
step 1. It is not licence-gated, contrary to an earlier note here — checked
2026-08-24, `gated: false` — so staging is a plain download.

## Scaling out later

- More sim cells: drop `--only`/`--tasks`/`--modes` (or widen them) to run
  the full matrix — but full mode collects ~1000 demos, which needs the
  oracle gate + collection phases (remove `--skip-collect`) and much more
  time.
- More real cells: add rows to `runs.tsv`. Parallelism is already the
  default there — one array task per (dataset, policy, camera set) — and
  `--concurrency` caps how many run at once if the queue needs it.
- A different ablation: any camera subset the dataset actually has works in
  the `cameras` column. Adding a camera the rig does not carry is refused by
  name, with the ones that do exist listed.
