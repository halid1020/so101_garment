# Running VLA training

Most of this directory is about the [KCL CREATE](https://docs.er.kcl.ac.uk/)
Slurm cluster, which is where the long runs go. **A plain GPU box with an SSH
login is also a destination** — see [On a machine with no
queue](#on-a-machine-with-no-queue) — and either can be driven from one
command, `tool/train_launch.py`, or from the rig console's Training tab:

```bash
source setup.sh
venv/bin/python tool/train_launch.py --list-destinations
venv/bin/python tool/train_launch.py --dir /mnt/seagate/so101 \
    --dataset fold-short-from-flattend-tactile --dest create \
    --policies act,diffusion --dry-run
```

The launcher stages the dataset, writes the run matrix onto the machine and
submits it; the steps below are what it is doing, and what to do when it will
not. Everything after a row is picked is identical at both destinations: the
same camera view, the same `test/system/long_vla_real.sh`, the same run
directory.

This directory provisions and submits two training cells on the
cluster, sharing one environment:

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
| `runs.tsv` | — | The real-data run matrix: one row per (dataset, policy, camera set) run. |
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
  Hub download either. pi0.5 trains here too, but needs **two** more things
  pre-staged — its base weights and, separately, its tokenizer — and only the
  first is a plain download; see [What a `pi05` row
  needs](#what-a-pi05-row-needs).
- **`install.sh` isn't cluster-safe.** Its later steps `sudo apt install`
  adb/openscad and clone `meta_quest_teleop`; none are needed for sim
  training/eval and they abort without sudo. `provision_create.sh` runs
  only the venv + LeRobot + requirements + cache steps.
- **Storage.** `outputs/` runs to tens of GB; home quotas are small, so
  the job points `SO101_OUTPUT_DIR` and `HF_LEROBOT_HOME` at scratch. Scratch
  is not unlimited either: on CREATE it is a **hard 200 GB** quota
  (`getfattr -n ceph.quota.max_bytes /scratch/users/$USER`), enforced by
  failing the *write*. `lerobot-train` checkpoints every `--save_freq` steps
  and never deletes an old one, so a 100k-step diffusion run alone parks ten
  ~3.3 GB copies. In August 2026 that filled the quota and killed four array
  tasks 1.5–17 h in, inside `save_pretrained`, with `OSError: [Errno 122]
  Disk quota exceeded`. `long_vla_real.sh` now prunes as it goes:
  `--keep-checkpoints N` (**default 2**) keeps only the newest N step
  directories per policy, plus whatever `last` points at. Nothing downstream
  wants the intermediates — `fetch_policies.sh` copies only
  `checkpoints/last/pretrained_model`. Check the headroom before a big
  submission:

  ```bash
  getfattr -n ceph.quota.max_bytes -n ceph.dir.rbytes /scratch/users/$USER
  ```

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
`tool/run_policy.py` back at the rig, not on the cluster.

The unit of work is one **(dataset, policy)** pair. They are listed in
`hpc/runs.tsv`, and each becomes one Slurm array task. Nothing tracked has to be
edited to change which datasets are trained — edit the matrix, or filter it on
the command line.

### 1. Provision — *login node, once*

`bash hpc/provision_create.sh`, as for the sim cell. Both cells share the venv.

#### What a `pi05` row needs

pi0.5 is a finetune, so everything it loads by name must already be in the shared
HF cache: a compute node has no internet, and the job runs with
`HF_HUB_OFFLINE=1`. Stage it once, on the login node:

```bash
SO101_STAGE_PI05=1 bash hpc/provision_create.sh
```

That covers **two separate repos**, and they are not equally easy:

| Repo | What it is | Gated? |
|------|-----------|--------|
| `lerobot/pi05_base` | the ~14.5 GB base weights | **No** (checked 2026-08-24) |
| `google/paligemma-3b-pt-224` | pi0.5's tokenizer/processor (~22 MB of it) | **Yes — `gated: manual`** |

`lerobot/pi05_base` is a plain download needing no token — but it ships **no
tokenizer**, and LeRobot builds pi0.5's tokenizer from the PaliGemma repo *by
name*. That repo is manually gated: the Hub answers **401** for its files unless
the request carries a token whose account has been granted access. A CREATE login
node has internet but no token, so it cannot fetch it unaided. Two ways out:

- **accept the terms** at <https://huggingface.co/google/paligemma-3b-pt-224>
  with your HF account, then `hf auth login` on the login node and re-run; or
- **copy the tokenizer files in** from a machine whose HF account already holds
  the grant — only the non-weight files (~22 MB), preserving the hub cache
  layout (`blobs/` real files, `snapshots/<rev>/` symlinks, `refs/main`). No
  token travels, only the files. `provision_create.sh` prints the exact recipe
  when it cannot stage the repo itself.

`provision_create.sh` now **asserts** both before it exits: no `.incomplete`
blobs and a full-size `model.safetensors` for the base, and — the only test that
means anything — that `AutoProcessor.from_pretrained` for the tokenizer actually
builds with `HF_HUB_OFFLINE=1`. Both checks exist because the failure is
otherwise invisible until a GPU node hits it hours in. In August 2026 three
`pi05` array tasks died at startup with

```
ValueError: Failed to instantiate processor step 'tokenizer_processor' ...
tokenizer_name: 'google/paligemma-3b-pt-224' ... couldn't connect to
https://huggingface.co
```

and a fourth staging attempt had silently left a 1.1 GB fragment of the 14.5 GB
base behind: `huggingface_hub` picks a new temp filename per attempt, so a
half-download never resumes and never complains.

Verify by hand at any time, on the login node:

```bash
HF_HUB_OFFLINE=1 venv/bin/python -c \
  "from transformers import AutoProcessor; \
   AutoProcessor.from_pretrained('google/paligemma-3b-pt-224'); print('OK')"
```

Skip all this and a `pi05` row fails at startup; `act` and `diffusion` rows are
unaffected either way.

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

One row per run: `dataset policy cameras steps batch hours slots extra`. `-`
means "the driver's default for this policy"; `extra` is the last column and is
handed to `lerobot-train` verbatim, so a flag the driver does not name is still
reachable. Rows sharing an `hours` value are submitted as one array with that
wall time, so a short cell does not queue behind a long reservation.

`policy` is `act`, `diffusion`, `pi05` or `fastwam`. `cameras` is `all`, a comma
list of camera names **with no spaces**, or a composite (below) — a space there
would shift every later column one place left, so the wrapper refuses the row
rather than training the wrong size for the wrong time.

`slots` is for pi0.5 only and is usually `-`. pi0.5 was pretrained with exactly
**three** image slots, so a row naming more cameras than that is refused rather
than quietly trained on fewer. With `-`, a camera whose viewpoint pi0.5 knows
(the overhead one) takes that slot by name and a camera it has never seen —
every tactile one — takes the next free slot in order; the driver prints the map
it resolved. Pin it when the assignment is the experiment:
`central=base,left_arm_left_gripper=left_wrist` (the slots are `base`,
`left_wrist`, `right_wrist`).

A row written before the `slots` column existed has seven fields, and `read`
would put its `extra` into `slots` — so both the wrapper and the array task
detect that on the **empty `extra`** and say which column is missing, rather
than handing a string of `lerobot-train` flags to `--slots`.

#### Composites: several cameras in one view

A policy whose architecture fixes how many views it takes cannot be given more.
FastWAM concatenates its image features into a single frame of
`policy.image_size` — 224x448 by default, so exactly two square views — and this
rig records five cameras. `tactile_quad` names a **composite**: the four
fingertip cameras tiled 2x2 into one 224x224 feature, which sits beside the
overhead view inside that frame. One view spent on four cameras instead of three
of them thrown away.

Name it like a camera (`central,tactile_quad`). `all` never includes one:
spending a view this way is a choice about what the model sees. Unlike an
ordinary view a composite cannot share the source's video files, because its
frames do not exist until they are made — it decodes the four sources in
lockstep and encodes one, with PyAV rather than the `ffmpeg` command line,
because a compute node has neither an `ffmpeg` nor a way to install one.
MEASURED: 26 s for 9985 frames x 4 cameras.

#### FastWAM is gated

`fastwam` needs a LeRobot that ships `src/lerobot/policies/fastwam`. The commit
this repo pins (`3dd19d04`, 2026-06-27) does not; it exists upstream. A fastwam
row is therefore **refused before anything is reserved**, by a probe for that
module rather than a version comparison, so the gate opens by itself when
`LEROBOT_COMMIT` moves in `install.sh` and `provision_create.sh` (both already
list the `fastwam` extra, which pip ignores until the checkout defines it).

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

**It also needs a smaller batch than the other two.** MEASURED 2026-08-25 on
this dataset: `pi05` at **batch 8** dies on the first forward pass of a 40 GB
A100 (`torch.OutOfMemoryError`, 39.22 GiB already in use), LoRA and all —
`--peft.r=16` shrinks the *optimiser state*, not the activations of a 4.1B-param
model reading three camera streams. **Batch 4 fits**, at `mem_gb:34.02` and
~0.49 s/step, so a 30k-step row takes about 4.5 h. Keep the `batch` column at 4
for `pi05` rows on `a100_40g`, or ask for a bigger card:

```bash
# node features on CREATE: a100_40g, l40s (48 GB), h200 (141 GB), b200
sinfo -p gpu -o "%25N %8t %12G %30f"
SBATCH_CONSTRAINT=h200 bash hpc/submit_real.sh --only pi05   # then batch 8 is fine
```

Note that **ablating a camera does not buy the memory back**: pi0.5 pads and
masks an empty slot rather than skipping it, so the one-camera row costs what
the three-camera row costs.

On the other hand `pi05` rows are nearly free on disk — a checkpoint is the LoRA
adapter only, ~15 MB, against ~590 MB for ACT and ~3.3 GB for diffusion.

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

`squeue` says only that a job exists. For loss, step, throughput, memory and
time remaining, open the rig console's **Training** tab: it discovers the run
directories on this machine and draws the curve
(`documents/rig_web.md`, "Watching a run"). It reads
`<scratch>/so101_outputs/vla_real_long/<run>/logs/train_<policy>.log`, which is
**the only machine-readable metric record that exists** — wandb is disabled
unconditionally and nothing writes a jsonl or a tensorboard event. Two things
about that log matter to anything reading it by hand as well:

- `step:` is **rounded above a thousand** (`10K` is anywhere from 9 500 to
  10 499), so count metric lines and multiply by `log_freq` instead;
- tqdm is **disabled inside Slurm**, so a cluster log has no progress bar and
  no exact step, while the same run on a plain GPU box does.

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

> **Reused means SKIPPED, not resumed.** `long_vla_real.sh` treats the mere
> existence of `train/<policy>/checkpoints/last/pretrained_model` as "this
> policy is done" and returns without calling `lerobot-train` at all — it never
> passes `--resume`. That is right for a run that finished, and wrong for one
> that died mid-training: resubmitting it reports success at whatever partial
> step its last checkpoint reached (say 30k of 100k), quietly, and writes a
> `results.md` saying so. Before resubmitting a row that **crashed** rather than
> merely queued, check what its `last` actually points at, and delete that one
> policy's train directory if it is short of the target:
>
> ```bash
> readlink -f <run>/train/<policy>/checkpoints/last   # 030000 of a 100000-step row?
> rm -rf <run>/train/<policy>                         # then resubmit that row
> ```
>
> Deleting `train/<policy>` costs the steps already spent; there is no resume
> path today. A row that has no `train/<policy>` directory at all — one that
> died before its first checkpoint — needs no such care.

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

# one policy
bash hpc/fetch_policies.sh --from <host> --only pi05 --dest ~/outputs/policies
```

`--only` takes **`act`, `diffusion` or `pi05`** (its own `--help` text still says
only the first two — the filter is a plain match against the policy directories
found on disk, so it has never been limited to them).

Then `tool/run_policy.py --checkpoint <ckpt> --task "<task>"` — start with
`--dry-run`. A checkpoint too large for the rig's own machine can be served from
a GPU box instead (`--dest <gpu-host>:...`, then `tool/policy_server.py` +
`--server <url>`); see
[`../documents/remote_policy_inference.md`](../documents/remote_policy_inference.md).

**pi0.5 trains here now** (`--only pi05`), provided both its base *and* its
tokenizer were staged in step 1 — see [What a `pi05` row
needs](#what-a-pi05-row-needs). The base is not gated; the tokenizer repo is.

## On a machine with no queue

A GPU box reached by SSH is a destination like the cluster, described in
`src/conf/train_destinations.yaml`:

```yaml
thanos:
  ssh: thanos                          # an ~/.ssh/config alias
  kind: ssh                            # no queue manager
  repo: ~/project/so101_garment
  scratch: ~/.cache/huggingface/lerobot
  stage: "{scratch}/local"
  limits: {}                           # nothing measured yet — see below
```

`~` and `$USER` are left alone and expand in the **remote** login shell, which
is why they are written rather than a literal path. They therefore reach that
shell unquoted, so what may appear in them is checked when the file is read.

```bash
venv/bin/python tool/train_launch.py --dir /mnt/seagate/so101 \
    --dataset fold-short-from-flattend-tactile --dest thanos --policies act
venv/bin/python tool/train_launch.py --status
venv/bin/python tool/train_launch.py --stop <run id>
```

`hpc/gpu_box_run.sh` is what runs there, and it does the two things Slurm was
doing for us.

**It holds a lock, and runs one row at a time.** Nothing else reserves the card:
pi0.5 under LoRA already needs 21.5 of a 24 GB card's 23.5 GiB, so a second run
started alongside does not queue — it takes the first one down with an OOM hours
in. A second invocation is refused (exit 3) unless it passes `--wait`.

**It detaches.** `--detach` re-execs under `setsid` with the output on a log and
prints `pid=… log=…`, so the launcher's SSH connection closing cannot take a
36-hour run with it. One failing row does not abandon the rest of the matrix.

**Measure the ceiling before a long run.** `limits:` above is empty on purpose:
the one pi0.5 figure we have was taken with *three* cameras, and activations
scale with the camera count, so it does not carry over to a five-camera dataset.
Run one short row per policy, watch `nvidia-smi`, and write the number in — a
row over a recorded ceiling is then refused in the console and the terminal
instead of by the GPU.

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
