"""Start a training run on whichever machine is going to do it.

One front door for two destinations. A run on the KCL CREATE cluster and a run
on a plain GPU box differ in how the work is queued and in nothing else -- same
dataset, same camera view, same ``test/system/long_vla_real.sh`` -- so this
resolves the run, refuses it if it cannot work, copies the dataset up and hands
it to whichever executor that machine has. ``src/conf/train_destinations.yaml``
says which machines exist.

The refusals are the point. Every one of them has been paid for: a camera the
dataset does not record, a pi0.5 row with more cameras than the base has slots,
a batch size measured to OOM that particular card, a policy this LeRobot has
never heard of. Each costs a second here and hours on a reserved GPU.

    # what would happen, touching nothing
    venv/bin/python tool/train_launch.py --dir /mnt/seagate/so101 \\
        --dataset fold-short-from-flattend-tactile --dest create \\
        --policies act,diffusion --dry-run

    # for real
    venv/bin/python tool/train_launch.py --dir /mnt/seagate/so101 \\
        --dataset fold-short-from-flattend-tactile --dest thanos --policies act

    # the rows this repo keeps for a dataset, rather than ones named here
    venv/bin/python tool/train_launch.py --dir /mnt/seagate/so101 \\
        --dataset fold-short-from-flattend-tactile --dest create --matrix hpc/runs.tsv

    # what is going, and stopping it
    venv/bin/python tool/train_launch.py --status
    venv/bin/python tool/train_launch.py --stop <id>

The dataset is copied as it stands. A collection that is still growing will
therefore train on a snapshot, and the episode count that was shipped is
printed and recorded so a result can be traced back to it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.recording.dataset_check import DatasetDamaged, ensure_loadable  # noqa: E402
from common.recording.dataset_edit import read_soft_deleted  # noqa: E402
from common.training.destinations import (  # noqa: E402
    dataset_dir,
    destination,
    load_destinations,
    load_runs,
    manifest_dir,
    reachable,
    remember_run,
    rsync_argv,
    save_runs,
    ssh_argv,
    stage_dir,
)
from common.training.matrix import (  # noqa: E402
    POLICY_NAMES,
    MatrixError,
    format_rows,
    make_row,
    parse_rows,
    resolved,
    row_refusals,
)


def runs_file() -> Path:
    out = Path(os.environ.get("SO101_OUTPUT_DIR", _root / "outputs")).expanduser()
    return out / "training_runs.yaml"


# ── running a command, or saying what it would be ────────────────────────────


def show(argv: "list[str]") -> str:
    return " ".join(shlex.quote(a) for a in argv)


def run(argv: "list[str]", dry_run: bool, capture: bool = False) -> str:
    """Run a command, or print it. Raises SystemExit with its output on failure."""
    if dry_run:
        print(f"    $ {show(argv)}")
        return ""
    proc = subprocess.run(argv, capture_output=capture, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise SystemExit(
            f"❌ failed (exit {proc.returncode}): {show(argv)}"
            + (f"\n   {detail}" if detail else "")
        )
    return (proc.stdout or "") if capture else ""


def rsync(argv: "list[str]", dry_run: bool, progress) -> None:
    """Run rsync, reporting each file as it goes.

    A dataset is hundreds of megabytes over a home uplink, so a launch spends
    minutes here. Without this the console's job message stands still for all
    of them, which is indistinguishable from a launch that has hung -- and the
    first thing it hid was a bug that was copying the wrong directory.
    """
    if dry_run:
        print(f"    $ {show(argv)}")
        return
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    total, name, last = 0, "", ""
    # rsync -P rewrites its progress with \r, so one "line" can carry a file
    # name and then many percentages. Both are wanted: the name says what is
    # being copied and the percentage says a big file is still moving.
    for chunk in proc.stdout:
        for piece in chunk.replace("\r", "\n").split("\n"):
            piece = piece.strip()
            if not piece or piece.startswith(("sending", "sent", "total")):
                continue
            if "%" in piece:
                last = next((w for w in piece.split() if w.endswith("%")), last)
            else:
                total += 1
                last = ""
                name = piece[-56:]
        if total:
            progress(f"staging : {total} file(s) — {name} {last}".rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"❌ rsync failed (exit {proc.returncode}): {show(argv)}")


# ── the dataset ──────────────────────────────────────────────────────────────


def read_dataset(root: Path) -> "dict[str, Any]":
    """The dataset's info.json, after the checks that stop a doomed run.

    ``ensure_loadable`` answers a different question -- is this whole -- and
    returns its own report, so the metadata is read separately rather than from
    its return value.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"❌ no dataset at {root}")
    try:
        ensure_loadable(root)
    except DatasetDamaged as exc:
        raise SystemExit(
            f"❌ {root.name} is not whole, so training on it would fail on the "
            f"GPU node rather than here:\n{exc}\n"
            "   Repair it first — the console's Datasets tab has a Repair button."
        )
    marked = read_soft_deleted(root)
    if marked:
        # The same guard hpc/stage_datasets.sh applies. A marked episode is
        # still on disk, so it would be copied up and trained on -- the mark
        # means a reviewer decided it should not be.
        raise SystemExit(
            f"❌ {root.name} has {len(marked)} episode(s) marked for deletion "
            f"but still on disk: {marked[:8]}{'…' if len(marked) > 8 else ''}\n"
            "   Remove them for good in the console's Datasets tab first, or "
            "they will be trained on."
        )
    return json.loads(info_path.read_text())


# ── building the rows ────────────────────────────────────────────────────────


def rows_for(args: argparse.Namespace) -> "list[dict[str, str]]":
    if args.matrix:
        text = Path(args.matrix).expanduser().read_text()
        rows = [r for r in parse_rows(text) if r["dataset"] == args.dataset]
        if not rows:
            raise SystemExit(
                f"❌ {args.matrix} has no rows for {args.dataset!r}. It names: "
                + ", ".join(sorted({r["dataset"] for r in parse_rows(text)}))
            )
        if args.policies:
            want = set(args.policies.split(","))
            rows = [r for r in rows if r["policy"] in want]
        return rows
    return [
        make_row(
            args.dataset,
            policy,
            cameras=args.cameras,
            steps=args.steps or "-",
            batch=args.batch or "-",
            hours=args.hours,
            slots=args.slots or "-",
            extra=args.extra or "-",
        )
        for policy in (args.policies or "act,diffusion").split(",")
        if policy
    ]


def report_rows(rows: "list[dict[str, str]]", info: dict, dest: dict) -> "list[str]":
    """Print each row with what its `-` resolves to; return every refusal."""
    problems: "list[str]" = []
    print(f"  rows for {dest['name']} ({dest['kind']}):")
    for row in rows:
        refusals = row_refusals(row, info, dest)
        mark = "✗" if refusals else "✓"
        print(
            f"   {mark} {row['policy']:<10} cameras={row['cameras']}"
            f"  steps={resolved(row, 'steps', dest)}"
            f"  batch={resolved(row, 'batch', dest)}"
            f"  hours={row['hours']}"
            + (f"  slots={row['slots']}" if row["slots"] != "-" else "")
        )
        for refusal in refusals:
            print(f"       {refusal}")
        problems += refusals
    return problems


# ── the two dispatches ───────────────────────────────────────────────────────


def push_manifest(
    dest: dict, rows: "list[dict[str, str]]", name: str, dry_run: bool
) -> str:
    """Write the rows onto the destination and return the remote path.

    The manifest goes UP rather than being generated there, so what ran is the
    matrix that was checked here -- and a copy of it stays on the machine that
    ran it, which is the only durable record of what a run directory contains.
    """
    remote_dir = manifest_dir(dest)
    remote = f"{remote_dir}/{name}.tsv"
    text = format_rows(rows)
    if dry_run:
        print(
            f"    $ ssh {dest['ssh']} 'mkdir -p {remote_dir}'  # then write {remote}:"
        )
        for line in text.splitlines():
            print(f"        {line}")
        return remote
    # A single quoted heredoc: the rows reach the file byte for byte, with no
    # remote expansion of anything they contain.
    script = f"mkdir -p {remote_dir} && cat > {remote} <<'SO101_MANIFEST_EOF'\n{text}SO101_MANIFEST_EOF"
    run(ssh_argv(dest, script), dry_run=False, capture=True)
    return remote


def dispatch_slurm(
    dest: dict, remote_manifest: str, dry_run: bool = False
) -> "dict[str, Any]":
    submit = [
        f"cd {dest['repo']}",
        " ".join(
            [
                "bash hpc/submit_real.sh",
                # NOT shlex.quote: these come from the destinations file,
                # where '~' and '$USER' are written on purpose and must be
                # expanded by the REMOTE shell. What may appear in them is
                # checked when that file is read.
                f"--manifest {remote_manifest}",
                f"--scratch {dest['scratch']}",
                f"--partition {shlex.quote(str(dest['partition']))}",
            ]
            + (
                [f"--account {shlex.quote(str(dest['account']))}"]
                if dest.get("account")
                else []
            )
            + (["--dry-run"] if dry_run else [])
        ),
    ]
    out = run(ssh_argv(dest, " && ".join(submit)), dry_run, capture=not dry_run)
    if out:
        print(out.rstrip())
    # submit_real.sh prints sbatch's own "Submitted batch job N" lines.
    jobs = [
        word
        for line in out.splitlines()
        if "Submitted batch job" in line
        for word in [line.split()[-1]]
    ]
    return {"jobs": jobs}


def dispatch_ssh(
    dest: dict,
    remote_manifest: str,
    dry_run: bool = False,
    run_tag: "str | None" = None,
) -> "dict[str, Any]":
    cmd = " ".join(
        [
            f"cd {dest['repo']} &&",
            "bash hpc/gpu_box_run.sh",
            # NOT shlex.quote — see dispatch_slurm.
            f"--manifest {remote_manifest}",
            f"--scratch {dest['scratch']}",
            *([f"--run-tag {shlex.quote(run_tag)}"] if run_tag else []),
            "--dry-run" if dry_run else "--detach",
        ]
    )
    out = run(ssh_argv(dest, cmd), dry_run, capture=not dry_run)
    if out:
        print(out.rstrip())
    pid = next((w.split("=", 1)[1] for w in out.split() if w.startswith("pid=")), None)
    return {"pid": pid}


# ── status and stop ──────────────────────────────────────────────────────────


def do_status(args: argparse.Namespace) -> None:
    state = load_runs(runs_file())
    runs = state["runs"]
    if not runs:
        print("no runs recorded on this machine")
        return
    known = load_destinations(args.destinations)
    for record in runs:
        dest = known.get(record.get("dest", ""))
        line = (
            f"{record['id']}  {record.get('dest'):<10} {record.get('dataset')}  "
            f"[{', '.join(record.get('policies') or [])}]  "
            f"{record.get('started', '')}"
        )
        print(line)
        if dest is None or args.no_probe:
            continue
        print(f"    {live_state(dest, record)}")


def live_state(dest: dict, record: "dict[str, Any]") -> str:
    """Ask the machine itself. A machine that cannot be reached says so."""
    problem = reachable(dest)
    if problem:
        return f"(cannot ask {dest['name']}: {problem})"
    if dest["kind"] == "slurm":
        jobs = record.get("jobs") or []
        if not jobs:
            return "(no job id was recorded)"
        cmd = f"squeue --jobs={','.join(jobs)} --noheader --format='%i %T %M' || true"
    else:
        cmd = f"cd {dest['repo']} && bash hpc/gpu_box_run.sh --status"
    proc = subprocess.run(
        ssh_argv(dest, cmd), capture_output=True, text=True, timeout=60
    )
    return (proc.stdout or proc.stderr or "").strip() or "(nothing running)"


def do_stop(args: argparse.Namespace) -> None:
    state = load_runs(runs_file())
    record = next((r for r in state["runs"] if r["id"] == args.stop), None)
    if record is None:
        raise SystemExit(f"❌ no recorded run called {args.stop!r}")
    dest = destination(record["dest"], args.destinations)
    if dest["kind"] == "slurm":
        jobs = record.get("jobs") or []
        if not jobs:
            raise SystemExit("❌ no job id was recorded for this run")
        cmd = f"scancel {' '.join(jobs)}"
    else:
        cmd = f"cd {dest['repo']} && bash hpc/gpu_box_run.sh --stop"
    print(run(ssh_argv(dest, cmd), args.dry_run, capture=True).rstrip())


# ── the launch itself, which both the terminal and the console use ───────────


def run_id_for(dataset: str) -> str:
    """A run's name, which also becomes a remote FILENAME."""
    slug = "".join(c if (c.isalnum() or c in "-_") else "-" for c in dataset)
    return f"{slug[:24]}-{uuid.uuid4().hex[:6]}"


def stage(
    dest: dict,
    collection_dir: Path,
    dataset: str,
    info: dict,
    dry_run: bool = False,
    restage: bool = False,
    no_stage: bool = False,
    progress=lambda message: None,
) -> None:
    """Copy the dataset up, unless it is already there.

    ``collection_dir`` is the DIRECTORY THE DATASETS LIVE IN, not the dataset:
    what is copied is ``collection_dir/dataset``. Taking the parent and joining
    the name here rather than accepting either is deliberate -- passing the
    collection directory to something expecting a dataset root uploads every
    dataset on the drive, under a directory named after the drive, and the only
    symptom is a transfer that takes far too long.

    The count that is shipped is REPORTED, because a collection that is still
    growing means a run trains on a snapshot -- and a result is only traceable
    back to a dataset if somebody wrote down which version of it.
    """
    source = Path(collection_dir) / dataset
    if not (source / "meta" / "info.json").is_file():
        raise SystemExit(f"❌ no dataset at {source}")
    remote = dataset_dir(dest, dataset)
    if no_stage:
        progress(f"staging : skipped; assuming {remote}")
        return
    there = (
        not dry_run
        and subprocess.run(
            ssh_argv(dest, f"test -f {remote}/meta/info.json"), capture_output=True
        ).returncode
        == 0
    )
    if there and not restage:
        progress(f"staging : already at {remote} (restage to copy again)")
        return
    progress(f"staging : {info.get('total_episodes')} episodes -> {remote}")
    run(ssh_argv(dest, f"mkdir -p {stage_dir(dest)}"), dry_run)
    rsync(rsync_argv(source, dest, dry_run=dry_run), dry_run, progress)


def launch(
    dest: dict,
    collection_dir: Path,
    dataset: str,
    rows: "list[dict[str, str]]",
    info: dict,
    dry_run: bool = False,
    restage: bool = False,
    no_stage: bool = False,
    run_tag: "str | None" = None,
    progress=lambda message: None,
) -> "dict[str, Any] | None":
    """Stage, write the manifest and submit. ``None`` for a dry run.

    The one implementation of a launch: the Training tab and the command line
    are two front ends over this, so a run started from the browser is the same
    run, recorded the same way.
    """
    stage(dest, collection_dir, dataset, info, dry_run, restage, no_stage, progress)

    run_id = run_id_for(dataset)
    progress(f"submit  : {dest['name']} ({dest['kind']})")
    remote_manifest = push_manifest(dest, rows, run_id, dry_run)
    launched = (
        dispatch_slurm(dest, remote_manifest, dry_run)
        if dest["kind"] == "slurm"
        else dispatch_ssh(dest, remote_manifest, dry_run, run_tag)
    )
    if dry_run:
        return None

    record = {
        "id": run_id,
        "dest": dest["name"],
        "dataset": dataset,
        "policies": [r["policy"] for r in rows],
        "cameras": sorted({r["cameras"] for r in rows}),
        "episodes": info.get("total_episodes"),
        "manifest": remote_manifest,
        "run_tag": run_tag,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **launched,
    }
    save_runs(runs_file(), remember_run(load_runs(runs_file()), record))
    return record


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", help="Collection directory holding the dataset")
    parser.add_argument("--dataset", help="Dataset directory name")
    parser.add_argument("--dest", help="Destination name (see --list-destinations)")
    parser.add_argument(
        "--policies",
        help=f"Comma list of {'|'.join(POLICY_NAMES)} (default: act,diffusion)",
    )
    parser.add_argument(
        "--cameras", default="all", help="Comma list of camera names, or 'all'"
    )
    parser.add_argument("--slots", help="pi0.5 slot map, camera=slot pairs")
    parser.add_argument("--steps", help="Optimiser steps (default: per policy)")
    parser.add_argument("--batch", help="Batch size (default: per policy/machine)")
    parser.add_argument("--hours", help="Slurm wall time in hours")
    parser.add_argument("--extra", help="Raw lerobot-train flags")
    parser.add_argument(
        "--run-tag",
        help="Name the run directory this instead of the camera view. Use it "
        "for anything throwaway: a short probe under the real name leaves a "
        "checkpoint that makes the long run skip training and report success "
        "at the probe's step count",
    )
    parser.add_argument(
        "--matrix",
        help="Take this dataset's rows from a runs.tsv instead of building them",
    )
    parser.add_argument(
        "--destinations",
        help="Destinations file (default: src/conf/train_destinations.yaml)",
    )
    parser.add_argument(
        "--restage", action="store_true", help="Copy the dataset up again"
    )
    parser.add_argument(
        "--no-stage", action="store_true", help="Assume the dataset is already there"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print every command, transfer nothing"
    )
    parser.add_argument(
        "--list-destinations", action="store_true", help="List the machines and exit"
    )
    parser.add_argument("--status", action="store_true", help="What has been launched")
    parser.add_argument(
        "--no-probe", action="store_true", help="--status without asking the machines"
    )
    parser.add_argument("--stop", metavar="ID", help="Stop a recorded run")
    args = parser.parse_args()

    if args.list_destinations:
        for name, dest in sorted(load_destinations(args.destinations).items()):
            print(
                f"{name:<10} {dest['kind']:<6} {dest['ssh']}  stage={stage_dir(dest)}"
            )
        return
    if args.status:
        return do_status(args)
    if args.stop:
        return do_stop(args)

    for required in ("dataset", "dest"):
        if not getattr(args, required):
            raise SystemExit(f"❌ --{required} is required (or use --status)")

    dest = destination(args.dest, args.destinations)
    collection_dir = Path(args.dir or ".").expanduser()
    info = read_dataset(collection_dir / args.dataset)

    print(f"dataset  : {collection_dir / args.dataset}")
    print(
        f"           {info.get('total_episodes')} episodes, "
        f"{info.get('total_frames')} frames, {info.get('fps')} fps"
    )
    try:
        rows = rows_for(args)
    except MatrixError as exc:
        raise SystemExit(f"❌ {exc}")

    problems = report_rows(rows, info, dest)
    if problems:
        raise SystemExit(
            f"\n❌ {len(problems)} refusal(s); nothing was staged or submitted."
        )

    problem = reachable(dest)
    if problem:
        raise SystemExit(f"\n❌ {problem}")

    record = launch(
        dest,
        collection_dir,
        args.dataset,
        rows,
        info,
        dry_run=args.dry_run,
        restage=args.restage,
        no_stage=args.no_stage,
        run_tag=args.run_tag,
        progress=print,
    )
    if record is None:
        print("\n(dry run — nothing was staged, written or submitted)")
        return
    print(f"\n✅ launched {record['id']} on {dest['name']}")
    print("   watch it: venv/bin/python tool/train_launch.py --status")
    print(f"   stop it : venv/bin/python tool/train_launch.py --stop {record['id']}")


if __name__ == "__main__":
    main()
