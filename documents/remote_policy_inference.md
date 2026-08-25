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
venv/bin/python tool/run_policy.py \
    --server http://127.0.0.1:8765 --task "pick up the cube" \
    --dry-run --seconds 20
```

(Or just `--server http://127.0.0.1:8765 --web` and do everything in the
browser — see *Watching a rollout* below. The task is optional there.)

`--dry-run` reads the cameras and the followers and prints the actions the
policy chooses, with the chunk-queue depth and the round-trip time, but never
enables torque. Watch two things: the queue depth should never reach zero, and
the round trip should be a small fraction of a chunk's wall time.

Then, with the workspace clear:

```bash
venv/bin/python tool/run_policy.py \
    --server http://127.0.0.1:8765 --task "pick up the cube" --seconds 30
```

The arms ramp slowly to the policy's first action, then run at `--hz`. Ctrl+C,
**Stop** on the page, and any error all disable torque on the way out. There is
**no default time limit** — pass `--seconds N` if you want one. A console
session spent stepping through plans and swapping splices cannot know in advance
how long it wants, and a default that quietly ended one at 900 ticks was a
worse guess than none.

## Rehearse it in the twin — no robot required

Everything above can be walked through with the digital twin standing where the
arms stand. `--sim` swaps the rig behind the same seam
(`src/common/policy_rig.py`) and changes nothing else: the same page, the same
arming handshake, the same preview → step → run cycle, the same splice switch,
the same stall ladder, the same run log. No camera is opened, no bus is touched,
and there is no motor anywhere. It is how the procedure gets checked before it
is trusted with torque.

There is a policy to rehearse against. On **thanos**:

```bash
source setup.sh
venv/bin/python tool/policy_server.py --port 8765 --checkpoint \
  outputs/vla_sim_long/diffusion_handover_20260805_112423/\
simple/handover/diffusion/checkpoints/030000/pretrained_model
```

That is the sim-trained `handover` diffusion policy — the bimanual relay of the
2.2 cm cube — selected at step 30 000 on 100 % validation success and scoring
73 % on the evaluation seeds with 37 mm mean place error. Then, on the laptop:

```bash
ssh -N -L 8765:127.0.0.1:8765 thanos          # in its own terminal

source setup.sh
venv/bin/python tool/run_policy.py --sim handover --web \
    --server http://127.0.0.1:8765 \
    --camera-map wrist_camera_left=wrist_left,wrist_camera_right=wrist_right
```

**Why the `--camera-map`.** That checkpoint's dataset was collected before the
wrist streams were renamed (`364dbf1`), so it asks for
`scene` / `wrist_left` / `wrist_right` where the twin now produces
`scene` / `wrist_camera_left` / `wrist_camera_right`. Renaming on the way out is
the whole fix; retraining a policy because a stream changed name is not. Without
it the run refuses to start and prints both camera sets, before anything moves.

`--sim` alone means `handover`; `--sim single` is the one-arm task. `--sim-seed`
picks the scenario (default 0 for `single`, 14 for `handover` — the seeds the
`simple` collection mode uses). The per-second line gains the payload's distance
from its target, so the terminal says whether the rehearsal is getting anywhere.

The twin renders all three cameras in about 12 ms per tick on an RTX 3050, well
inside a 30 Hz tick, so this runs comfortably on the rig laptop.

**What this rehearses and what it does not.** It rehearses the procedure — every
button, every wait, every refusal, in order. It does not rehearse the hardware:
the arm-side mapping, the calibration, the workspace, and whether the blue ghost
really tracks the measured joints are all still only checkable on the rig. For
many scored episodes on chosen seeds instead of one interactive rollout, use
`tool/run_policy_sim.py`, which is the batch harness.

## Watching a rollout

A rollout that misbehaves is over in seconds and the terminal shows almost none
of it. `--web` serves a live view of the run itself on loopback, and it is meant
to be the **whole interface** — this is the entire command:

```bash
venv/bin/python tool/run_policy.py --server http://127.0.0.1:8765 --web
```

Everything else happens in the browser: the **task** is typed in the header (the
run waits for one before it infers anything, so `--task` is optional with
`--web`), the arms are armed there, the throttle and the splice are there, and
the run lasts until **Stop**. `--web` therefore implies three things, each of
which can still be overridden: no time limit (`--seconds N`), a throttle that
starts in **preview** (`--start-mode run`), and consent taken on the page
(`--arm-at-terminal`, or `--yes` to skip it).

It comes up with the cameras and the buses, before the first ramp, and answers
the four questions a failed grasp raises:

- **What the policy was shown** — the camera frames of the window that was
  actually sent, beside the twelve joint values that went with them. Not the
  live view: by the time a chunk is executing, the frames it was planned from
  are half a second old, and those are the ones that explain the plan.
- **What it planned** — the returned chunk as twelve small trace plots, one per
  joint on its own scale, and the same chunk walked through in the rig's own
  twin: a 3D scene you can turn, with the arms as they are now (blue) drawn
  inside where the plan sends them (orange).
- **What the arms did** — measured against commanded, per joint, with the two
  **grippers on their own panel**. That is deliberate: across the collected
  datasets the gripper channels span a few tenths of open fraction and never
  reach either end, so a grasp is won or lost in a number that a table of twelve
  hides.
- **Whether it arrived in time** — round trip, how much of it was inference,
  the queue depth against the depth that triggers the next request, and how many
  ticks were spent holding.

### The throttle

Five buttons: **Preview**, **Execute one chunk**, **Run**, **Hold**, **Stop**.

**Preview and Execute are a cycle**, and it is how a rollout is walked one plan
at a time. Preview freezes the arms but KEEPS what is queued, so the twin shows
the motion that is about to happen; Execute runs exactly that plan and returns
to Preview, with the next chunk already queued behind it. Nothing moves between
the two, so there is as long as you like to look.

**Hold is not Preview**, and the difference is why both exist: leaving a Hold
**drops whatever was queued**, because a plan made before a pause was drawn from
a picture of the world that may now be minutes old. Preview keeps it, because a
plan you are looking at is worth the same a moment later. Use Hold to stop; use
Preview to inspect.

### The twin

One scene, two ghosts: **the arms as they are now, in transparent blue, inside
where the selected action of the plan sends them, in transparent orange**. The
gap between them is the motion still to come, which is the thing worth looking
at and which neither pose shows on its own. It is forward kinematics only — no
physics, no contact, nothing grasped.

**Drag it.** The scene is a [viser](https://viser.studio) view, the same one
`tool/check_mirror.py` uses, rendered by your own browser: drag to orbit, scroll
to zoom. That is the point of it. A gripper that clears the block from the front
may be through it from the side, and a single fixed camera angle — which is what
this panel used to be — cannot tell you which.

It runs as its own small server, on `--web-port + 1` by default
(`--twin-port` to move it), and the page embeds it. If it cannot start, the page
says so in the panel and everything else carries on: the numbers, the cameras
and the throttle are all still true without it.

Underneath it is a transport: **play/pause, ◀ ▶ and a scrub bar** over the
actions of the current plan. Pause and step to action 17 and look at it, from
whatever angle you have turned to; the blue ghost keeps tracking the real arms
while the orange one holds still, so a plan can be examined against the pose it
will act from. A new plan restarts the walk at its first action.

The upright line in the plan panel marks the action the twin is showing, so the
two panels read as one picture.

**How the page keeps the two in step.** Every frame the page asks for one
action by name — `/twin/at?seq=…&i=…` — and the server answers `204` and moves
the ghosts, or `409` if that plan has been replaced, in which case nothing moves
and the scene holds the pose it already has. The page, not the server, decides
what is on screen; a subject the server chose afresh each frame is what made the
old rendered twin flicker, and there is no longer a server-chosen subject to
get wrong. *What is still to execute* is deliberately not offered as a subject
either: the queue drains every tick, and a list that changes length underneath
an animation is exactly that bug in its purest form.

**If the twin looks erratic, check that only one rollout is running.** A second
rollout cannot bind the view's port, and it used to carry on regardless, leaving
an older run's cameras, plan and throttle on the screen while the arms in front
of you moved to something else entirely. It now refuses to start instead, and
says so.

### Arming, and where consent is taken

**With `--web`, consent is taken in the browser.** There is no terminal prompt;
the page shows an **⚠️ Enable arms** button which asks the same question the
terminal asked, before it starts anything. The wait happens before the first
inference, so the plan the arms ramp to was drawn from the workspace as it is
when you consent, not as it was while you were still clearing it.

**This is a real change in who can start the arms.** The port is loopback and
unauthenticated, so anyone who can reach it can begin the motion. Pass
`--arm-at-terminal` to put the prompt back — the page then refuses an arm
request rather than offering a second, unguarded door to the same torque.
Arming is one-way either way: `Stop` is how a run ends, and it releases torque.

### Reading the page

**The live cameras are not connected until you ask.** Every MJPEG stream is a
connection that never closes, and a browser allows about six per origin; with
three cameras running, the short requests — the status poll, the frames the
policy was shown, the twin's one-action-per-frame aim — queue behind streams
that never finish, get dropped and retried. Press **connect** on the live panel
when you want them. (The twin itself costs one websocket to a *different*
origin, so it is outside that budget.)

The strip under the buttons is the health of the rollout, and it stays on
screen because a rollout is over in seconds and scrolling loses it: **queue**
against the depth that triggers the next request (red at zero — the arms are
waiting for a plan), **round trip** and how much of it was inference, **held**
ticks as a percentage, the size of the **queued** plan, the **splice** in force,
and progress. Below it, what the policy was shown sits beside what it planned
and what the arms did, because every question worth asking of a failed grasp
spans the three. The live cameras are last on purpose: by the time a chunk is
executing, the frames it was planned from are half a second old, and those are
the ones that explain the plan. A
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

## Choosing how a chunk joins the one already executing

A chunk is planned from one observation and comes back after the arms have
moved on. `--strategy` decides what to do about that, and it is the one knob
here that changes what the arms do rather than merely when they ask.

| strategy | what it does | use it to |
|---|---|---|
| `append` | queues the new chunk behind the leftovers, discarding nothing | reproduce every rollout recorded before this existed |
| `replace` | drops the leftovers and the rows whose moment has passed | execute each action at the tick it was planned for |
| `blend` | `replace`, then cross-fades out of the old plan over `--blend-window` ticks | remove the step change at the join |
| `ensemble` | averages the overlap, `--new-weight` on the newer plan | hedge between two plans that disagree |
| `sync` | blocks for the reply, then executes the WHOLE chunk | a baseline, and only that — see the warning below |
| `receding` | blocks, then executes only `--execute-ratio` of the chunk and throws the rest away | re-observe early; the classic receding horizon |
| `rtc` | guidance inside the denoiser, on the host | nothing yet; the host refuses it |

`append` is the default, so a run without the flag behaves exactly as it did
before. It is also the one to beat: because it discards nothing, the arriving
chunk's first action waits for every leftover to drain, and the request fires
when the queue falls to the prefetch threshold — which is sized to cover the
round trip. Every plan is therefore executed a whole threshold behind the
observation it was drawn from: about 0.9 s for a 32-action diffusion chunk at
30 Hz, about 1.1 s for ACT. The prefetch threshold ends up controlling two
unrelated things, the network runway and the staleness of every action, and the
aligning strategies exist to separate them.

**Alignment is not free.** Dropping the stale rows shortens every chunk by the
round trip, so the queue empties sooner and the arms hold more often. Measured
in the twin against a 32-action diffusion chunk and an 18-tick delay:
`replace` held on 28 % of ticks where `append` held on 6 %. Raise
`--actions-per-chunk`, or accept the holds, or use `blend`, which pays the same
price but does not step at the join.

**`receding` is the one you were promised, and it blocks.** Send an
observation, wait, execute a fraction of what comes back, stop, ask again. The
fraction is of *whatever length arrived*, so it means the same thing against
ACT's hundred actions and diffusion's thirty-two: `--execute-ratio 0.5` on a
12-action chunk executes 6. Lower it to re-observe sooner, at the cost of a
round trip for every fraction of a chunk of motion. Nothing about the name is a
claim of concurrency — the arms hold still for every round trip, exactly as
`sync` does. It is `sync --actions-per-chunk N` with N chosen as a fraction, and
it earns its own name because that fraction is the quantity being studied and
because it can be changed while the rig runs.

**`sync` blocks the control loop.** It waits for the reply inside the tick, so
at a 700 ms round trip the loop runs at about 1.3 Hz rather than 30. The arms
are safe — the servos hold their last goal and nothing is written from a stale
plan — but the motion is a series of pauses, and it is a baseline to measure
against rather than a way to run the rig.

**`rtc` is refused, on purpose.** Its smoothing happens inside the flow-matching
denoiser on the host, and `tool/policy_server.py` does not do that yet, so the
answer would be plain `replace` under another name. The client checks the
handshake and stops rather than giving you a result labelled `rtc` that is not.
It applies to pi0.5 and other flow-matching policies only; ACT and diffusion
have no such hook.

### Changing the splice without stopping

Four strategies compared across four runs means four ramps, four workspace
resets, and whatever drifted between them landing in the comparison. The page's
**splice** row changes it mid-run instead: pick a strategy, and the one number
it actually uses appears beside it, with a sentence saying what the strategy
does and whether it blocks. The three numbers are:

- **execute … of each chunk** (`receding`) — run this fraction of each plan,
  then stop and look again.
- **ease in over N ticks** (`blend`) — the first N actions are mixed out of the
  old plan into the new one, so the arms do not jump at the join. Shown in
  seconds too, since ticks mean nothing without the rate.
- **trust the new plan** (`ensemble`) — where the old and new plans overlap,
  0 keeps the old and 1 takes the new.

Switching **drops the queue**. What is in there was spliced under the old rule —
under `append` it may be two plans deep, under `blend` its leading rows are a
cross-fade into a plan the new rule would not have chosen — so carrying it over
would make the first chunk after every switch belong to neither. The cost is one
round trip, the same as resuming from a pause.

Each switch closes a **measurement segment**. The run ends with one row per
splice that was actually in force, so a single session on a single scene fills
the comparison table rather than producing one average over settings that were
never in force at the same time.

`--seconds 0` runs until you press **Stop** (or Ctrl+C), which is what that
session wants: one ramp, one workspace, as long as it takes.

### What the run tells you afterwards

A remote run now ends with one line:

```
📐 replace: seam ratio 1.04  held 28%  path 573.1  chunks 12  rtt median 740 ms
```

The **seam ratio** compares the step in commanded joints at a chunk boundary
with the step everywhere else: 1 means the joins are invisible, 3 means every
boundary is a visible flinch, 10 means the arms lurch. It reads `—` when the run
cannot support the comparison — fewer than two boundaries, or a trajectory that
barely moved — because a ratio of noise over noise would read as a result.

Read it beside the other columns, never alone. A strategy can buy a beautiful
ratio by ignoring what the policy just saw: a long `--blend-window`, or
`--new-weight` near zero, smooths the join by declining to act on new
information. `held` and the task outcome are what stop that passing unnoticed.

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
