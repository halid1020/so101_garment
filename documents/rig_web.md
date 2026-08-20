# The rig console (`tool/rig_web.py`)

One browser page for the collection drive: review what was recorded, and
manage the datasets themselves. It replaces `tool/dataset_web.py`, which
did the reviewing half only.

```bash
source setup.sh
venv/bin/python tool/rig_web.py --dir /mnt/seagate/so101
venv/bin/python tool/rig_web.py --dir /mnt/seagate/so101 --allow-delete
```

Then open <http://127.0.0.1:8000/>. The console binds the loopback
interface only, like every other tool here; from another machine, reach it
through an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 <rig>`), exactly as
`documents/remote_policy_inference.md` describes for the policy server.

The page has three tabs: **Datasets** (review and manage), **Collect**
(readiness, live view, and one collection session), and **Sensors**
(binding devices to stream names).

## Reviewing recordings

Unchanged from the tool this grew out of. The left column lists the
datasets under `--dir`, the middle one the recordings inside the selected
dataset, and the right pane plays that recording: every camera side by
side on one clock, with the joint traces beside them.

Playing a recording decodes nothing on the rig — each episode is a
timestamp window inside a shared video file, so opening one costs a range
request. The composited view the operator watched while collecting is
still rendered on demand for the two cases direct playback cannot cover: a
dataset that records depth (stored as images, not a playable stream), and
a browser without AV1.

Episode curation is in two steps and needs `--allow-delete`:

1. **Delete** marks episodes. They vanish from the list at once and the
   decision is written into the dataset, reversibly (**Restore** undoes
   it). They are still on disk, so anything training on the dataset would
   still see them — the real-VLA training scripts refuse to start while
   marks are pending, and the pane says so.
2. **Remove for good** compacts: one rewrite of the whole dataset for the
   whole batch, renumbering the survivors and re-indexing the sidecar
   files with them.

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
can see where they point. The preview is always released before a session
launches — one process owns a camera.

**Starting.** Fill in the dataset name, the instruction stored with every
frame, and the streams. Typing the name of an existing dataset switches the
form to *resume*: its recorded streams, depth, EE features and frame rate
are shown ticked and locked, because a resumed dataset must not drift from
how it began, and anything you asked for that disagrees is reported rather
than silently ignored. **Check** resolves the request without running it.

**While it runs.** The tiles are the session's own frames, proxied — no
device is opened for them and the record loop is not touched, so the live
view costs a JPEG encode per viewer. The status line carries the arm state,
the recorder state, the episode count against its goal, the frame count of
the episode in progress, and any stream that has gone stale. The
recorder's output is tailed below the form.

**Episode** presses the session's own A button: it starts an episode, and
pressing it again stops and saves. It is disabled until the arms are
enabled, and it is worth reading the warning beside it — starting an
episode drives BOTH ARMS to the ready pose. Enable, park and home are
deliberately absent from the browser: they move the arms with nobody
necessarily looking at the rig, so they stay on the headset and the session
keyboard.

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

**Delete** moves the dataset to a sibling `<name>.trash-<stamp>` and frees
the bytes afterwards on a worker thread, so the click returns at once.
This needs `--allow-delete` and cannot be undone. A dataset with no saved
episodes at all — a session that quit before recording — is listed too,
flagged as such, precisely so it can be deleted here rather than by hand.

**Merge…** writes a NEW dataset from two or more sources, in the order
listed, and leaves the sources exactly as they are. It is refused, with
the reason shown before anything starts, when the sources disagree on
frame rate, robot type or the set of recorded streams; when the output
name is taken or is one of the sources; when a source still has episodes
marked for deletion (their indices would be meaningless afterwards —
compact or restore first); or when the drive has less free space than the
sources add up to.

A merge re-encodes every episode of every source, so it runs as a job: the
dialog polls it and shows where it is, one merge at a time. It builds into
a sibling `<name>.tmp-<stamp>` and swaps it in at the end, so an
interrupted merge leaves the collection directory as it was. LeRobot's
aggregation rewrites `meta/` from scratch, so the console carries across
what it drops — `meta/action_space.json`, `meta/realsense.json`, and the
per-episode files under `extra/` (sidecar, drift and identity), renumbered
onto the merged episode indices.

## Assigning sensors

The rig is built from identical-looking USB devices: gripper and wrist
cameras that differ only by which socket they are in, and four SO-101 buses
that enumerate in whatever order they were plugged. The Sensors tab is
where each one is identified physically and given its name. It does the
same job as `tool/test_sensor_rates.py --assign`, writes the same
per-machine file through the same loader and saver, and is refused while a
collection session is running — that session owns the cameras and the arm
buses.

The left column lists every assignable name beside the device it is bound
to, and says whether that device is **here now**. An assignment that no
longer resolves to a connected device is the failure worth catching early:
a camera moved to another socket simply reads as an absent stream at
collection time, which is a confusing way to find out.

**Cameras.** *Scan devices* lists the capture nodes that actually deliver
frames. Show one, press a gel or wave in front of it to see which camera it
is, then press the name it should have. Binding a device to a name first
clears it from whatever name it had before, so swapping two names is done
by reassigning, not by hunting for the stale entry.

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
| `--dir` | Collection directory holding the datasets (required) |
| `--port` | Loopback port (default 8000) |
| `--allow-delete` | Enable episode and dataset deletion (destructive) |
| `--fps` | Playback frame-rate override (default: the dataset's) |
| `--prerender` | Build composited episode videos in the background while browsing |
| `--monitor-port` | Loopback port a collection session serves its live view on (default 8766) |

Only `--prerender` is worth explaining: it warms the cache for the
composited view, which is the one worth watching only for a depth dataset
or a browser without AV1. Playing the recorded streams needs no rendering
at all.
