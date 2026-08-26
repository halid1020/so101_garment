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

The page has three tabs: **Datasets** (review and manage), **Collect**
(readiness, live view, and one collection session), and **Signals**
(binding devices to stream names).

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
lands on a 480 Mbit/s bus whatever socket it is in, and each bus has one
bandwidth budget shared by everything on it. About **three** streams fit per
bus, and the rig has seven cameras across two buses — so the full set does not
fit, and the four tactile cameras on their own always do.

Past the ceiling a camera does not run slowly; it opens normally and then
delivers nothing at all. **Which** camera loses is random, so the same
selection fails differently on consecutive runs and looks like a flaky camera
rather than a budget. Asking for less does not help: a lower frame rate and a
smaller frame were both measured, and neither admits another stream.

Three things surface this. The readiness check counts the enabled streams
against the bus each sits on and names the over-subscribed one; the Collect
form repeats that warning for the streams you actually selected; and a camera
that opens without delivering is treated as a failed open, so the session
refuses to start rather than looping on it.

The obvious lever is the uvcvideo `FIX_BANDWIDTH` quirk, which makes the driver
compute the real bandwidth need instead of trusting what the camera declares.
**It was tried on this rig and made no difference** — module reloaded, every
device re-enumerated, and the same two streams were still refused. The readiness
check reports whether it is on, so nobody spends an afternoon rediscovering that.

What does work is choosing a set that fits: the four tactile cameras together,
or a mix that keeps each bus at or under three streams. The readiness check's
hub grouping tells you which camera sits on which bus.

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
