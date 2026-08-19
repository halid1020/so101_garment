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

The page has three tabs. **Datasets** is described below. **Collect**
(live view and session control) and **Sensors** (assigning devices to
stream names) are placeholders for now; until they land, collect with
`tool/collect_dataset.py` and assign with
`tool/test_sensor_rates.py --assign`.

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

## Options

| Flag | Meaning |
|---|---|
| `--dir` | Collection directory holding the datasets (required) |
| `--port` | Loopback port (default 8000) |
| `--allow-delete` | Enable episode and dataset deletion (destructive) |
| `--fps` | Playback frame-rate override (default: the dataset's) |
| `--prerender` | Build composited episode videos in the background while browsing |

Only `--prerender` is worth explaining: it warms the cache for the
composited view, which is the one worth watching only for a depth dataset
or a browser without AV1. Playing the recorded streams needs no rendering
at all.
