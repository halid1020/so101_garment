# Running a trained policy on the rig with inference on another machine

A checkpoint trained on the cluster is tested **on the robot**. Nothing about
that changes here: the cameras, the follower buses, the control rate and every
safety decision stay on the rig's computer. What can move is the forward pass.

This is worth doing when the rig's computer cannot hold the policy comfortably.
The collection laptop has 14 GB of RAM and a 4 GB RTX 3050; the diffusion policy
alone is 293M parameters, and running it there competes with the camera threads
that must not miss a frame. A machine with a real GPU can answer instead.

## Why this is nearly free

An action-chunking policy plans many steps from one observation — one hundred
for the ACT configuration trained here (3.3 s at 30 Hz), thirty-two for
diffusion (about 1.1 s). It only looks at an observation when it re-plans. So
the link carries **one observation per chunk**, not one per control tick: about
70 kB of JPEG for this rig's three cameras, roughly once a second. The client
asks for the next chunk while it is still executing the current one, so the
round trip overlaps motion and never sits inside the control loop.

## What the policy consumes

Both `act` and `diffusion` trained on the rig's datasets take:

| input | shape | meaning |
|---|---|---|
| `observation.state` | float32[12] | left 5 joints (URDF degrees) + left gripper (open fraction 0–1), then the same for the right arm |
| `observation.images.central` | uint8 480×640×3 RGB | overhead camera |
| `observation.images.wrist_camera_left` | uint8 480×640×3 RGB | left wrist |
| `observation.images.wrist_camera_right` | uint8 480×640×3 RGB | right wrist |
| task | string | carried in the batch; ACT and diffusion ignore it |

A policy trained on fewer cameras asks for fewer: the client reads the camera
set from the server's handshake and sends exactly those, so an ablated
checkpoint needs no flag at the rig. A checkpoint asking for a camera the rig
cannot produce is a training/collection mismatch, and the run stops there.

Frames go over the wire at the recorded resolution and are never pre-resized:
the checkpoint's own processors normalise, and diffusion resizes internally.
ACT is handed one observation; diffusion is handed the **last two consecutive**
ones, because that is the history it was trained on. The client does this for
you — it reads the requirement from the server's handshake.

The output is a 12-D action in the same units as the state, which reaches the
motors through the identical conversion a replayed recording takes.

## What has been trained

The real-data cell on the cluster (`hpc/README.md`) has produced these:

| dataset | policy | final train loss | task string |
|---|---|---|---|
| cube-pnp | act | 0.066 | pick the cube and place it on the plate |
| cube-pnp | diffusion | 0.002 | pick the cube and place it on the plate |
| cube-dual-pnp-new | act | 0.085 | pick the cube and place it on the plate |
| cube-dual-pnp-new | diffusion | 0.003 | pick the cube and place it on the plate |
| fold-short | act | 0.114 | flatten the short and fold it |
| fold-short | diffusion | 0.004 | flatten the short and fold it |

Training loss ranks nothing on the robot; it is here so that a checkpoint can be
told apart from a run that never converged. ACT and diffusion ignore the task
string, but pass the recorded one anyway — it is what the dataset froze, and a
policy that does read it will need it.

Two footnotes on cube-pnp. Its ACT weights come from an earlier submission,
whose run directory is named after the job id rather than the dataset, so it
appears under that name in a listing. The re-run under the dataset name was
cancelled at its 24-hour wall time, 8k of 80k steps in, at about ten seconds per
optimiser step — worth diagnosing before that cell is submitted again, since the
same recipe trains diffusion to completion on the same dataset.

## 1. Bring the checkpoint back from the cluster

A finished cluster run holds its final weights under
`train/<policy>/checkpoints/last/pretrained_model/`, beside tens of gigabytes of
intermediate checkpoints and optimiser state that inference has no use for.
`hpc/fetch_policies.sh` reads the run directory, works out which
(dataset, policy) pairs actually finished, and copies only what a server loads —
one directory per pair, named for both:

```bash
bash hpc/fetch_policies.sh --from <user>@<create-login-host> --list
bash hpc/fetch_policies.sh --from <user>@<create-login-host> \
    --dest <gpu-host>:project/so101_garment/outputs/policies
```

`--list` first. A run cancelled at its wall time still has checkpoints, just not
a `last`, and it reads `unfinished` there instead of installing something that
looks like a checkpoint and is not. `--datasets` and `--only` take a subset.

A camera-ablation run is named for the dataset **and** the cameras it was
trained on — `cube-pnp-new__all`, `cube-pnp-new__central+wrist_left`,
`cube-pnp-new__wrist_left`. `--datasets` accepts either, so
`--datasets cube-pnp-new` brings back the whole ablation and
`--datasets cube-pnp-new__wrist_left` brings back one arm. Each lands in its own
directory, so the three can be served and compared without being confused.

The copy runs on the machine the weights are going to, pulling from the cluster
over your forwarded SSH agent, so nothing lands on the rig's disk and no key is
ever copied to the GPU box. A cluster that accepts logins only from inside its
own network cannot be reached that way — the script checks before it moves
anything, and `--bridge` then routes the bytes through the machine you are
sitting at, in two hops. KCL CREATE, from outside its network, needs `--bridge`.

Each checkpoint is then read back where it landed and reported: policy type,
observation steps in, actions out, and the camera names it was trained on. Those
names must be the rig's. If they are not, stop there — that is a
training/collection mismatch, and no amount of networking will fix it.

ACT is roughly 200 MB, diffusion about 1.1 GB.

## 2. Start the server — on the GPU machine

```bash
cd ~/project/so101_garment && source setup.sh
venv/bin/python tool/policy_server.py \
    --checkpoint outputs/policies/cube-pnp-act/ --port 8765
```

It binds `127.0.0.1` on purpose. The endpoint is unauthenticated: anyone who can
reach it can spend the GPU and read nothing useful, but it should not be exposed.
`--host 0.0.0.0` exists for a trusted private network and warns when used.

The startup lines state what the checkpoint expects — policy type, observation
steps in, actions out, and the camera names. If those camera names are not the
rig's, stop here: the mismatch is a training/collection mismatch, not a network
problem.

Detached (`nohup`, `tmux`) it is the same command with `venv/bin/python -u`:
redirected to a file, Python buffers its output, and the startup lines you want
to read sit in that buffer for a long time. `GET /meta` answers the same
questions over the tunnel once step 3 is up, whichever way it was started.

## 3. Open the tunnel — on the rig

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<host>
```

Leave it running in its own terminal. Traffic is then encrypted, no port is
published, and no firewall change is needed at either site.

Mind the port: the rig console gives a collection session's live monitor
**8766** by default (`tool/rig_web.py --monitor-port`), so a policy server or a
tunnel put there collides with a session that is recording. 8765 is free of
that.

## 4. Run it — on the rig, dry first

```bash
source setup.sh
venv/bin/python tool/run_policy_real.py \
    --server http://127.0.0.1:8765 --task "pick up the cube" \
    --dry-run --seconds 20
```

`--dry-run` reads the cameras and the followers and prints the actions the
policy chooses, with the chunk-queue depth and the round-trip time, but never
enables torque. Watch two things: the queue depth should never reach zero, and
the round trip should be a small fraction of a chunk's wall time.

Then, with the workspace clear:

```bash
venv/bin/python tool/run_policy_real.py \
    --server http://127.0.0.1:8765 --task "pick up the cube" --seconds 30
```

The arms ramp slowly to the policy's first action, then run at `--hz`. Ctrl+C,
the end of `--seconds`, and any error all disable torque on the way out.

## Watching a rollout

A rollout that misbehaves is over in seconds and the terminal shows almost none
of it. `--web` serves a live view of the run itself on loopback:

```bash
venv/bin/python tool/run_policy_real.py \
    --server http://127.0.0.1:8765 --task "pick the cube and place it on the plate" \
    --seconds 60 --web
```

It comes up with the cameras and the buses, before the first ramp, and answers
the four questions a failed grasp raises:

- **What the policy was shown** — the camera frames of the window that was
  actually sent, beside the twelve joint values that went with them. Not the
  live view: by the time a chunk is executing, the frames it was planned from
  are half a second old, and those are the ones that explain the plan.
- **What it planned** — the returned chunk as joint traces, with the part
  already executed shaded, and the same chunk replayed in the rig's own twin.
  The twin is forward kinematics with a camera: it shows where the plan puts the
  arms, and nothing else. A toggle points it at the measured arms instead.
- **What the arms did** — measured against commanded, per joint, with the two
  **grippers on their own panel**. That is deliberate: across the collected
  datasets the gripper channels span a few tenths of open fraction and never
  reach either end, so a grasp is won or lost in a number that a table of twelve
  hides.
- **Whether it arrived in time** — round trip, how much of it was inference,
  the queue depth against the depth that triggers the next request, and how many
  ticks were spent holding.

### The throttle

Four buttons: **Hold**, **Step one chunk**, **Run**, **Stop**. Stepping is how a
failure gets examined — one plan is executed, the arms stop, and the plan that
produced the motion is still on the screen beside the motion it produced.

Leaving a hold **drops whatever was queued**. A chunk planned before the pause
was drawn from a picture of the world that is now minutes old, and executing it
afterwards would be a surprise; the cost is one round trip on resume.

The view can only ever ask for *less* motion than the terminal already
authorised, or end the run. Torque is enabled once, at the confirmation prompt,
and nothing in the browser can enable it — the page is unauthenticated and it
steers a robot, which is why it binds loopback like everything else here. A
local run (`--checkpoint`, no server) has no chunk to step through, and the
button says so.

### What it leaves behind

Every rollout writes `$SO101_OUTPUT_DIR/policy_runs/<stamp>/`: `chunks.jsonl`,
one line per plan, appended as it lands, so a run that dies mid-episode still
leaves its plans; and `ticks.parquet`, one row per control tick with the
measured joints, the goal written, the mode and the queue depth. `--no-log`
skips it.

## What happens when the network misbehaves

The arms are under torque, so a late chunk is a safety question rather than a
performance one:

- **A late chunk**: the last commanded goal is held. Nothing new is written.
- **`--stall-hold` (default 0.5 s)**: a warning is printed; the hold continues.
- **`--stall-abort` (default 2.0 s)**: the run stops, which disables torque.
- **A rejected request (4xx)**: fatal at once — a wrong camera set, a wrong
  window length or a stale session cannot be fixed by retrying.
- **A dropped connection or a server error (5xx)**: retried on the next tick,
  under the same stall budget.

## Tuning

- `--actions-per-chunk N` executes at most N actions per request. The default is
  whatever the policy plans, which is what it does when run locally. Lower it to
  make the policy re-observe sooner — 15 is half a second at 30 Hz — at the cost
  of more requests.
- `--prefetch N` asks for the next chunk once the queue falls to N actions. It
  has to cover the round trip: the queue drains at the control rate, so N must
  exceed *round trip × `--hz`*. **The default now measures that rather than
  guessing it** — a third of a chunk until a round trip has been timed, then
  whatever covers it with half again for jitter, never reaching the chunk
  length. This matters: a diffusion chunk is 32 actions, 1.07 s at 30 Hz, and
  half a second of inference makes the round trip about 0.6 s — 18 actions,
  against a static third-of-a-chunk threshold of 10. Those rollouts ran a second
  and held, ran a second and held. ACT never had the problem: 100 actions per
  chunk against 18 ms of inference. Pass N to override the measurement.
- `--hz` is the control rate; it should match the rate the dataset was recorded
  at (30 Hz here).

## Running without a server

Drop `--server` and pass `--checkpoint` instead: inference then happens in the
same process, exactly as before. This is the right choice when the rig's machine
can hold the policy, and it is the reference the remote path is checked against —
given the same pixels, both produce the same action.
