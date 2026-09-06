# The rig console (`tool/rig_web.py`)

One browser page for the collection drive: review what was recorded, and
manage the datasets themselves. It replaces `tool/dataset_web.py`, which
did the reviewing half only.

```bash
source setup.sh
venv/bin/python tool/rig_web.py --dir /mnt/seagate/so101
venv/bin/python tool/rig_web.py                 # opens on the last drive used
```

Then open <http://127.0.0.1:8000/>. The console binds the loopback
interface only, like every other tool here; from another machine, reach it
through an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 <rig>`), exactly as
`documents/remote_policy_inference.md` describes for the policy server.

The page has four tabs: **Datasets** (review and manage), **Collect**
(readiness, live view, and one collection session), **Signals** (binding
devices to stream names), and **Training**, which is itself split into **Start
Training** (sending a finished dataset to a GPU machine) and **Training Jobs**
(reading what every machine has done).

## The collection directory

The directory in the header is where every dataset the console shows lives,
and clicking it changes that. It can be given with `--dir`, remembered from
the last time (so the flag is optional), or picked in the page — the
console opens on the picker when it has neither.

The dialog browses this machine's directories, marks the ones that hold
datasets, and keeps a short list of the last few used. Changing the
directory is refused while a collection session or a job is running: both
are working inside the current one.

**A drive on another machine** is reached by mounting it. Give
`user@host:/path` and the console mounts it with `sshfs`, then works on the
mount like any local directory. Authentication is your key or agent only —
no password is ever typed into the browser — so the remote machine must
already accept your key, and its host key must already be known:

```bash
sudo apt install sshfs                 # once, on the machine running the console
ssh-copy-id halid@thanos               # once, if the key is not there yet
ssh halid@thanos                       # once, to accept the host key
```

Anything else fails with what to do about it rather than hanging on a
prompt nobody can answer. Mounts the console made are released when it
exits, and can be released from the dialog.

Browsing and playback over the network are comfortable: playing a recording
reads a byte range out of a video file. Merging or compacting a remote
dataset is not — it moves every byte across the link twice — so that work
belongs on the machine holding the drive: run the console there and reach
it through the SSH tunnel above.

## Reviewing recordings

Unchanged from the tool this grew out of. The left column lists the
datasets in the collection directory, the middle one the recordings inside the selected
dataset, and the right pane plays that recording: every camera side by
side on one clock, with the joint traces beside them.

Playing a recording decodes nothing on the rig — each episode is a
timestamp window inside a shared video file, so opening one costs a range
request. The composited view the operator watched while collecting is
still rendered on demand for the two cases direct playback cannot cover: a
dataset that records depth (stored as images, not a playable stream), and
a browser without AV1.

Episode curation is in two steps:

1. **Delete** marks episodes. They vanish from the list at once and the
   decision is written into the dataset, reversibly (**Restore** undoes
   it). They are still on disk, so anything training on the dataset would
   still see them — the real-VLA training scripts refuse to start while
   marks are pending, and the pane says so.
2. **Remove for good** compacts: one rewrite of the whole dataset for the
   whole batch, renumbering the survivors and re-indexing the sidecar
   files with them. This one asks first, because it cannot be undone, and
   then runs as a job — a dataset of any size takes minutes to rewrite —
   so the panel in the corner of the page is where it is watched and where
   a failure is reported. The rewrite builds a sibling copy and swaps it in
   at the end; if it cannot finish, the copy is removed and the dataset is
   untouched.

Marking is deliberately not confirmed — it is the frequent action while
reviewing, and *Restore* takes it back. Everything that cannot be taken
back (compaction, deleting a dataset, deleting the sources of a merge) puts
up a dialog naming what will go and how much of it.

## Collecting

The Collect tab starts and watches one collection session. The session is
the unchanged teleoperation recorder, run as a separate process with
exactly the flags `tool/collect_dataset.py` would have computed — the same
resolver backs both, so a session started here and one started from a
terminal cannot drift apart.

**Before starting.** Open *Rig readiness* for the preflight table:
calibrations, poses, cameras, the depth device, disk headroom, and the
host-tuning items that reduce timing jitter. The device checks are skipped
while a session or a preview holds them; the file checks always answer.
With nothing running, **Start preview** opens the assigned cameras so you
can see where they point — with the same size, pixel format and exposure the
recorder will use, so the preview answers "does this look right", not only
"is this pointing right". A camera that cannot be opened is named under the
tiles with the reason, rather than leaving you to count the tiles: an absent
camera wants replugging, one refused its share of the USB bandwidth wants
fewer cameras (see **The USB budget** below). **Read the arms** opens the two
follower buses
to fill the joint table — calibrated, torque disabled, nothing commanded, so
the arms stay limp and you can push them by hand and watch the numbers move.
Both are released before a session launches, and before the Signals tab
probes a port: one process owns a camera, and one owns a serial port.

**Starting.** Fill in the dataset name, the instruction stored with every
frame, and the streams. Typing the name of an existing dataset switches the
form to *resume*: its recorded streams, depth, EE features and frame rate
are shown ticked and locked, because a resumed dataset must not drift from
how it began, and anything you asked for that disagrees is reported rather
than silently ignored. **Check** resolves the request without running it.

A camera whose assigned device node is gone is offered **unticked and marked
*not connected***, and ticking it anyway is refused rather than started: the
recorder opens every selected camera before it creates the dataset, so a session
started on an absent one exits at once, with its reason buried in the output
tail. Plug it back in and press **Check** again — no reload needed — or clear
its assignment on the Signals tab. The idle preview follows the same rule from
the other side: it opens the assigned cameras that `recording.yaml` still
enables, so a camera the rig no longer carries is neither previewed nor reported
missing on every poll.

The arm buses are checked the same way and for the same reason. They come back
as `root:dialout` after every re-enumeration, so on a machine whose operator is
not in that group a hub reset takes an arm away without anyone touching it —
and the ports that reappear *after* `setup.sh` ran are precisely the ones its
`chmod` did not reach. A port that is missing, or there but unopenable, is
named before Start rather than thrown as a serial traceback thirty seconds
in. The fix it names is `source setup.sh`; joining the `dialout` group fixes it
for good.

**Driving the session.** *Driving the session* lists the steps in order —
what enables the arms, what makes them follow, what closes the grippers,
what records an episode and what ends the session — and, for each, the
surface it lives on. It changes with the input mode, and it is the same
list the terminal prints when teleoperation starts
(`src/common/recording/controls.py`), so the browser cannot describe a rig
it is not driving. A step marked *this page* is also a button here, and
those steps are exactly the keys the session's monitor accepts — which
depends on how the session is driven, for the reason in **Enable arms**
below.

**While it runs.** The tiles are the session's own frames, proxied — no
device is opened for them and the record loop is not touched, so the live
view costs a JPEG encode per viewer. Below them is the proprioception:
both arms' measured joints beside the command last written to their motors,
which is the pair stored as the state and the action of every recorded
frame. A command older than a frame is dimmed and named, because that is
what makes a recorded action fall back to the measured state; beside it are
whether teleoperation is active (the grips) and how stale the joint stream
is. The status line carries the arm state, the recorder state, the episode
count, the frame count of the episode in progress, and any stream that has
gone stale. The recorder's output is tailed below the form.

**Episode** presses the session's own A button: it starts an episode, and
pressing it again stops and saves. It is disabled until the arms are
enabled, and starting an episode drives BOTH ARMS to the ready pose, so
stand clear first.

**The keys work here too.** Y, A and Q can be typed into the page instead of
clicked, when a session is running and this tab is on screen; each does exactly
what its button does, confirmations included, and a disabled button ignores its
key as it ignores a click. Keys typed into a text box stay in the text box. Only
the keys the session itself allows are taken, so a Quest session's page leaves
Y alone exactly as its monitor would refuse it.

The session the console starts is given **no terminal**: it would otherwise
inherit the one the console was launched in, put it into raw mode underneath the
shell sitting there, and then race that shell for every keystroke — which is
what made the control keys look broken. Nothing types into the session any more;
the page presses them.

**Enable arms** appears only for a session driven by the **leader arms**,
and it presses Y. The rule behind that is the same in both modes — a key
that moves an arm belongs where the operator can see the arm — and it lands
differently only because the modes put the operator in different places.
With a headset on, every button is already to hand, so the browser gets
none of them. With leader arms there is no headset: the keys are read from
a terminal, and a session the console started reads them from the terminal
the *console* was launched in, which is not where anyone is standing. So
for those sessions the page is the only surface left, and it gets enable,
episode and quit. Park and home stay physical in both modes.

An episode the console cannot play because it *"is still being written"* is not
a fault: its frames are on disk and the session has not committed its metadata
row yet, which it does as the recording proceeds. The same episode described as
having **no metadata row and no session running** is the other case — the
session ended before committing it — and the count settles when the dataset is
next opened for recording. One that never settles is what **Repair** is for.

**Stop session** asks the session to quit, which is what parks the arms,
finishes an in-flight episode and closes the dataset properly. Only if that
is ignored does the console interrupt, and later terminate; it never kills,
because a killed session abandons an open episode. Closing the console does
NOT end a session — restarting a web page must not cost a recording.

Live view during collection is the reason the recorder gained
`--monitor-port`: it serves this small loopback monitor from inside the
session (frames, status, and an allow-list of two keys). It is off by
default, so nothing changes for an invocation that does not ask for it.

## Managing datasets

The buttons above the dataset list work on whole datasets.

**New…** checks that a name is well formed and free, and hands back the
`tool/collect_dataset.py` command that starts the session. A dataset comes
into being when the first episode is recorded into it, so the console
cannot create an empty one; starting the session from the browser arrives
with the Collect tab.

**Rename** moves the directory. Nothing inside a dataset records its own
name — `meta/info.json` holds no repo id — so a rename changes nothing
about the recordings, and being a rename within one directory it cannot
half-happen.

**Delete** asks first, then moves the dataset to a sibling
`<name>.trash-<stamp>` — atomic, so the dataset is gone from the list at
once — and frees the bytes afterwards as a job. It cannot be undone. A
dataset with no saved episodes at all — a session that quit before
recording — is listed too, flagged as such, precisely so it can be deleted
here rather than by hand.

**Merge…** writes a NEW dataset from two or more sources, in the order
listed. The sources are kept unless *delete the sources* is ticked, in
which case they are removed once the merge has succeeded and not before —
if anything goes wrong the merged dataset stays and every source is still
there. It is refused, with
the reason shown before anything starts, when the sources disagree on
frame rate, robot type or the set of recorded streams; when the output
name is taken or is one of the sources; when a source still has episodes
marked for deletion (their indices would be meaningless afterwards —
compact or restore first); or when the drive has less free space than the
sources add up to.

**Repairing what a merge leaves behind.** Every row of a dataset's episode
metadata names the file it is stored in, and LeRobot's aggregation writes
the source's file numbers into a destination whose files are numbered
differently — so a merged dataset claims its episodes live in files that
were never written, and the next rewrite of it (a compaction, or a second
merge) fails partway through. The console corrects those two columns
against the files they are actually in, before a merge reads its sources,
after a merge writes its output, and before a compaction begins. It is
silent when there is nothing wrong, which is the case for everything that
comes straight off the rig.

A merge re-encodes every episode of every source, so it runs as a job and
the dialog closes: the panel in the bottom corner of the page shows what is
running and how far it has got, from whichever tab you are on, and reloads
the list when it finishes. Freeing a deleted dataset's bytes appears there
too. One job runs at a time. A merge builds into
a sibling `<name>.tmp-<stamp>` and swaps it in at the end, so an
interrupted merge leaves the collection directory as it was. LeRobot's
aggregation rewrites `meta/` from scratch, so the console carries across
what it drops — `meta/action_space.json`, `meta/realsense.json`, and the
per-episode files under `extra/` (sidecar, drift and identity), renumbered
onto the merged episode indices.

### Repairing a dataset that counts an episode nobody wrote

A recording that was counted and never written leaves an episode number with
nothing behind it: the list shows it as `?f`, and **no** rewrite of that dataset
can run — not a deletion, not a merge — because LeRobot judges the whole local
copy incomplete and goes looking for the missing part on the Hub.

The pane says so when it happens, and offers **Repair**. It forgets the empty
slots and renumbers the recordings after them, which is the part worth knowing
before pressing it: what is now recording 5 may become recording 4. Nothing
recorded is lost, marks for deletion follow their recordings, and the repair
runs as a job in the corner like any other whole-dataset work.

A dataset whose recordings are *all* missing is refused rather than repaired:
that is a drive that is not mounted far more often than a dataset that truly
holds nothing, and emptying its metadata would throw away the only record of
what used to be there. So is damage that needs a decision — frames with no
metadata, or metadata with no frames — because repairing either one means
choosing what to discard.

## Assigning signals

The rig is built from identical-looking USB devices: gripper and wrist
cameras that differ only by which socket they are in, and four SO-101 buses
that enumerate in whatever order they were plugged. The Signals tab is
where each one is identified physically and given its name. It does the
same job as `tool/test_sensor_rates.py --assign`, writes the same
per-machine file through the same loader and saver (which is why the file
and the routes still say *sensor*), and is refused while a
collection session is running — that session owns the cameras and the arm
buses.

The left column lists every assignable name beside the device it is bound
to, and says whether that device is **here now**. An assignment that no
longer resolves to a connected device is the failure worth catching early:
a camera moved to another socket simply reads as an absent stream at
collection time, which is a confusing way to find out.

Each device in the working column carries the name it already has, and the
name it carries is marked among the ones on offer, so pressing a different
one visibly moves the mark. The console has to say this: the map stores
each device's stable by-path alias while the buttons show it as the kernel
names it, so only the server can tell which chip is which name.

**Cameras.** *Scan devices* lists the capture nodes that actually deliver
frames. Show one, press a gel or wave in front of it to see which camera it
is, then press the name it should have. Binding a device to a name first
clears it from whatever name it had before, so swapping two names is done
by reassigning, not by hunting for the stale entry.

**The USB budget.** Every camera on the rig is a USB 2.0 device, so each one
lands on a 480 Mbit/s bus whatever socket it is in — and that bandwidth is
allocated by the **host controller**, across the whole bus. This is the single
most important thing to know about it: **a hub does not add capacity.** These
hubs are already USB 3.0 and it makes no difference, because the cameras are
USB 2.0 and enumerate on the controller's 480 Mbit/s side regardless. Buying a
better hub will not fix it.

A stream refused its share does not run slowly — it opens normally and then
delivers nothing at all, and WHICH one loses is random, so the same selection
fails differently on consecutive runs and looks like a flaky camera rather than
a budget.

What a controller carries is **not a stream count**, because the cameras do not
cost the same. One model reproduces every trial run on this rig: a **tactile
camera costs 2 units, a colour camera 1, and a controller carries 4**.

| selection on one controller | units | fits? | measured |
|---|---|---|---|
| `central` + a wrist + 1 tactile | 4 | yes | three ran |
| a wrist + 2 tactile | 5 | no | two ran |
| 2 tactile | 4 | yes | both ran |
| `central` + 2 tactile | 5 | no | one refused |
| `central` + a wrist + 2 tactile | 6 | no | three ran |

**Asking for less does not help, and it cannot.** Each camera reports exactly one
frame rate per format and size — the tactile cameras offer 60 fps at 640×480
MJPG and 30 fps at 320×240, the wrist cameras offer 30 fps, and that is the whole
list. There is no slower mode to select, which is also why the `fps:` key in
`recording.yaml` cannot slow a camera down. Halving the frame size (which does
move a tactile camera to its 30 fps mode) was measured and still bought nothing.
The uvcvideo `FIX_BANDWIDTH` quirk was tried with the module reloaded and every
device re-enumerated, and changed nothing either; uvcvideo appears to skip that
fixup for compressed formats, and everything here is MJPEG.

### Three controllers, five cameras

The remedy was never a better hub — it was **another host controller**, and this
laptop is an AMD Rembrandt with five of them. The rig now spreads its cameras
over three, and all five deliver together:

| controller | cameras | units |
|---|---|---|
| `pci-0000:06:00.3` | `central` | 1 |
| `pci-0000:05:00.4` | `left_arm_left_gripper`, `left_arm_right_gripper` | 4 |
| `pci-0000:06:00.4` | `right_arm_left_gripper`, `right_arm_right_gripper` | 4 |

Both wrist cameras are currently unplugged and disabled in `recording.yaml`;
re-attaching one means a free controller, not a free socket — a fourth port on
either gripper controller has no room in it.

Find the controllers with `lsusb -t`, where each `/: Bus NNN` line is one, and
after any replug re-run `tool/test_sensor_rates.py --assign` (or the Signals
tab) so the by-path aliases follow the cameras to their new sockets. Then check
the readiness table's `usb camera budget` row, which prints the units on each
controller.

**Arms.** Open a port and wiggle ONE arm by hand: the joints that move are
shown live, so the port belonging to that arm is obvious. The bus is opened
uncalibrated and with torque off — this reads raw ticks to tell ports
apart, it never commands an arm. Then press the role: follower or leader,
right or left. A physical port is one arm, so assigning it clears it from
wherever it lived; followers and leaders are separate namespaces, so a
follower-right and a leader-right can coexist.

**Depth camera.** The RealSense has no stable device node, so it is bound
by its serial: pick the connected device.

*Release* lets go of whatever the tab is holding, and so does leaving the
tab — a camera or a bus held open here is one a collection session cannot
have.

## Training

The tab has two subviews, because it does two unrelated jobs.

**Start Training** is the launch form: pick a dataset, some policies and a
machine; the console stages the dataset there, writes a run matrix, and
starts it. Nothing here decides whether a run can work — the rules are
`src/common/training/`, the same ones `tool/train_launch.py` applies from a
terminal, so the page cannot start a run the command line would refuse.

**Training Jobs** is where a run is read: projects, a sortable table of every
run on every machine, the configuration difference between the ones selected,
and a chart per metric. It needs **no collection directory** — only the three
routes that read a dataset off the drive do — which is how it is actually used:
from a laptop, watching a GPU box.

### Two implementations of the same three policies

The policy list is in two groups. **LeRobot** is the installed checkout's own
`act`, `diffusion`, `pi05` and `fastwam`; **This repo** is `src/so101_policies/`
— the three ports of those, plus `so101_flowmatch` and `so101_dreamzero`, which
have no upstream twin. A port is shown against the policy it was ported from.

Both are trained by the same `lerobot-train`; a repo policy simply adds
`--policy.discover_packages_path=so101_policies`, which the driver does for
itself. A port takes its twin's steps, batch and hours, so `act` and `so101_act`
differ *only* in whose code runs — which is what makes the comparison a
comparison. Where a machine has measured a ceiling for a policy, its port is
held to the same one: the number is a fact about the model's activations, and a
port is the model.

MEASURED, on thanos with the real five-camera dataset: `act` and `so101_act`,
and `diffusion` and `so101_diffusion`, logged an identical loss at every point
over 60 steps from the same seed. `tool/compare_port_training.py` is what does
that, and `make test-port-parity DATASET_ROOT=<ds>` is the same check on the
CPU.

**Machines** come from `src/conf/train_destinations.yaml`, and adding one is
an entry there rather than a code change:

| Key | Meaning |
|---|---|
| `ssh` | `[user@]host` as SSH reads it — an `~/.ssh/config` alias keeps the key, port and jump host in one place. Omitted for `local` |
| `kind` | `slurm` (submit an array, ask `squeue`), `ssh` (start the driver under `nohup`, watch a pid), or `local` (this machine) |
| `outputs` | a second place run directories may be, besides `<scratch>/so101_outputs` |
| `repo` | the `so101_garment` checkout on that machine |
| `scratch` | where its `HF_LEROBOT_HOME` and `SO101_OUTPUT_DIR` live |
| `stage` | where staged datasets go; `{scratch}` is substituted |
| `partition` | `slurm` only: the GPU partition (`sinfo -s`) |
| `limits` | per-policy ceilings **measured** on that machine |

`~` and `$USER` in those paths are left alone and expand on the far side, in
the remote login shell — which is the point of writing them. They therefore
reach that shell unquoted, so what may appear in them is checked when the
file is read: a path with a space, a backtick or a `;` is refused there
rather than run.

**Check** resolves the run and lists every refusal without touching the
machine, so the form is usable off the VPN. The refusals are the reason this
exists, and each one has been paid for at least once:

- a camera the dataset does not record (a typo trains on fewer inputs than
  the experiment meant, and the result looks like a finding);
- **more cameras than the policy has slots** — pi0.5 has exactly three, and
  this rig records five;
- a batch size over what that machine has been *measured* to carry
  (pi0.5 at batch 8 raised `OutOfMemoryError` at 39.22 GiB on a 40 GB A100);
- a policy this LeRobot cannot train — `fastwam` today, refused with the two
  files to change rather than hidden from the list.

A blank steps or batch box means *whatever fits here*: it takes the
machine's measured ceiling when there is one, the policy's default when
there is not. A number you type is used as typed, and refused if it is over
— training something other than what was asked for would make the run matrix
a record of the request rather than of the run.

**Test connection** is the one button that reaches out, because SSH is
seconds and the form should not wait for it. An unreachable machine comes
back as an instruction: `ssh-copy-id`, accept the host key in a terminal,
or — for KCL CREATE from outside its network — *are you on the VPN?*

**Start** asks first, then stages and submits on the job dock's worker, so
it outlives the request and cannot race a merge. Staging is minutes for a
few hundred megabytes over a home uplink; the dock names each file as it
goes, because a message that stands still for ten minutes is
indistinguishable from a launch that has hung.

The dataset is copied **as it stands**. A collection still being recorded
therefore trains on a snapshot, so the episode count that went up is shown
before the launch and kept with the run.

### This machine as a destination

`local` is not a lesser destination: the same driver, the same lock (a laptop
GPU is single-tenant too), the same run directories and the same manifest,
with no SSH in front of the command. `repo: .` means *this* checkout, so the
entry is correct on whichever machine the console is running on. Staging is a
**symlink** rather than a copy — the dataset is already on this disk.

What it can carry is *asked of it* rather than written into the destinations
file, which is checked in and would otherwise record whoever committed it. So
the machine line shows the GPU it actually found, and there is one refusal only
this destination can raise: **a run that would fall back to the CPU**. Under
8 GB of VRAM the driver picks the CPU and trains anyway, which for an
80 000-step run is a week of work that looks like it is going fine. Tick *train
on the CPU anyway* (or pass `--allow-cpu`) if that is really what you mean —
the box only appears when the machine has said it would.

## Training Jobs

The table lists every training run **found on every machine**, not just what
this console started: nineteen of the twenty-one run directories on these
machines were started from a terminal, and nine predate the launcher. Sort by
any column; the filter box narrows by run, policy, machine or dataset.

The list is cheap — one call per machine, and the answers are cached — while a
log is only fetched for what is being watched: the rows you opened, the ones
ticked, and anything written to in the last few minutes, which is how a live
run is recognised without reading it. Nothing polls while the tab is off
screen. A run whose log has not been read yet shows no step and no loss, and
sorts to the end of those columns rather than pretending to be zero.

- **Ticking two or more** runs overlays their curves on every panel and shows
  the **configuration difference** between them: the resolved
  `train_config.json` of each, reduced to the fields that actually differ. On
  the three finished thanos runs that is a handful of lines out of 143, which
  is the whole reason it exists.
- **A panel per metric**, grouped into *Training*, *Evaluation* and
  *Prediction* by the metric's own namespace. The grid is built from what the
  runs actually logged, not from a list in the page — see *Adding a curve*
  below.
- **log scale** is on by default, for the metrics that span decades. An ACT run
  here goes 10.1 → 0.075, and on a linear axis everything after the first few
  hundred steps is one flat line along the floor. A near-constant metric — the
  learning rate, the step time — is drawn linear regardless.
- **Smoothing** is an exponential moving average, with the raw curve ghosted
  behind it: a smoothed line alone hides how noisy the run was, which for a
  loss curve is half the reading.
- **by step / by hours / since start** switches the axis. Wall-clock is the one
  that answers "will this finish tonight".
- Hovering reads every selected run at the nearest step, into a row under the
  chart rather than a floating tooltip — a tooltip clips at the right-hand edge
  and can only hold one series.
- The state chip is `running`, `done`, `failed`, `stalled` or `idle`.
  **stalled** is the one worth knowing: a log that has stopped growing relative
  to its own cadence. thanos rebooted under a run in August and nothing said
  so — every count still agreed, the log simply stopped.
- *Stop* is offered only for runs this console launched, since only those have
  a job id or a pid recorded. It asks first.
- **Export** writes what is plotted — not what was fetched, which would
  disagree with the figure it was taken to support.

### Projects

A run directory is named `<dataset>__<cameras>` and lives on one machine, which
says what it trained on and nothing about why. A **project** is a name you give
a set of runs; the dataset stays a column. **All runs** and **Unassigned** are
always there, so the view is useful before anyone creates anything.

Membership is a list of `machine|run|policy` keys in
`$SO101_OUTPUT_DIR/training_projects.yaml`, kept apart from the launch records
because most runs have none. A member no machine answered for is **reported as
missing, not dropped** — a machine off the VPN this morning has not deleted
anything, and a list that silently shrank would be the one way this view could
misreport what was run. Deleting a project removes the label only.

### Adding a curve

The driver's log is the only metric record that exists, so a number has to
survive a grep that runs on the far machine before it can be drawn. One line
format does that, and nothing else has to be edited:

```python
from common.training import metrics
metrics.emit(step, **{"eval/psnr_central": 31.2})
```

`train/`, `eval/` and `pred/` are the namespaces, and they pick the panel block
— a name without one is refused where it is emitted rather than landing in a
group chosen for it by accident. `lerobot`'s own held-out validation loss is
read too, from the `step N: eval_loss=…` line it writes when `--eval-split` and
`--eval-steps` are given to the driver. Both are off by default: a validation
split holds out episodes and therefore changes what is trained, so a run with
one is not comparable to the runs already finished.

**Per-checkpoint rollout success is a simulation-only curve.** `long_vla_sim.sh`
rolls every checkpoint out on the validation seeds and writes
`val/step_<N>.json`, which becomes `eval/success_rate`. The real rig's
equivalent is a person judging trials afterwards, in `outputs/policy_runs/` —
a different measurement at a different time, and never drawn on this axis as
though the training loop had produced it.

Two things about the log decide what the chart can honestly show. Its `step:`
field is **rounded above a thousand** — 10 500 and 10 600 both print as `10K` —
so the axis is reconstructed from the logging interval, which lerobot records in
the configuration it dumps at the head of its own log. And the progress bar that
carries the exact step is **disabled inside Slurm**, so a CREATE log has no
exact step in it at all and a thanos one does. Both are read; neither is
assumed.

### pi0.5's three slots, and the tiled fingertips

Two policies here have a fixed number of views, and this rig has five
cameras, so both need an answer.

pi0.5 was pretrained with three image slots. A camera whose viewpoint it
knows — the overhead one — takes that slot by name; a camera it has never
seen, which is every tactile one, takes the next free slot in order. The
driver prints the map it resolved, and the **Slot map** box pins it outright
(`central=base,left_arm_left_gripper=left_wrist`) when the assignment is the
experiment rather than a detail.

FastWAM concatenates its cameras into a single frame, so it takes exactly
two square views. Hence the **composite**: `tactile_quad` tiles the four
fingertip cameras 2x2 into one 224x224 feature, which sits beside the
overhead view inside FastWAM's 224x448. Name it in the Cameras box like any
camera; `all` never includes it, because spending one view on four cameras
is a choice about what the model sees.

## Options

| Flag | Meaning |
|---|---|
| `--dir` | Collection directory holding the datasets (optional: the last one is remembered, and it can be changed in the page) |
| `--port` | Loopback port (default 8000) |
| `--fps` | Playback frame-rate override (default: the dataset's) |
| `--prerender` | Build composited episode videos in the background while browsing |
| `--monitor-port` | Loopback port a collection session serves its live view on (default 8766) |
| `--mount-dir` | Where remote directories are mounted (default `$SO101_OUTPUT_DIR/rig_web_mounts`) |

Only `--prerender` is worth explaining: it warms the cache for the
composited view, which is the one worth watching only for a depth dataset
or a browser without AV1. Playing the recorded streams needs no rendering
at all.
