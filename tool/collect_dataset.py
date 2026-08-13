"""Friendly front-end for collecting a LeRobot dataset on the dual SO-101 rig.

Instead of the long ``tool/meta_quest_teleopration.py --record ...`` command,
this wraps it with three questions: WHERE (a collection directory), WHAT NAME
(the dataset), and WHICH STREAMS. It then hands off to the real teleop recorder,
so collection behaves exactly as it always has (Quest or leader teleoperation,
the A button records episodes, the same two-rate dataset + sidecar + depth).

If the named dataset already exists under the directory, its recorded stream
settings are read back and FOLLOWED — the same cameras, depth on/off, EE
features and frame rate — and the session resumes it. Any stream flags you pass
in that case are ignored with a warning, so a resumed dataset can never drift
from how it began.

Usage:

    # New dataset (defaults to recording.yaml's enabled cameras):
    venv/bin/python tool/collect_dataset.py \\
        --dir /media/hdd/so101 --name towel_fold --task "fold the towel"

    # Choose streams explicitly:
    venv/bin/python tool/collect_dataset.py --dir /media/hdd/so101 \\
        --name towel_fold --task "fold the towel" \\
        --stream scene --stream wrist_camera_left --central-depth

    # Resume: streams follow the existing dataset, flags below are ignored:
    venv/bin/python tool/collect_dataset.py --dir /media/hdd/so101 \\
        --name towel_fold --task "fold the towel"

    --input leader        # leader-arm teleop instead of the Quest
    --sensor-view         # open the live monitor while collecting
    --dry-run             # print the resolved teleop command and exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_TELEOP = Path(__file__).resolve().parent / "meta_quest_teleopration.py"


def nearest_existing_ancestor(path: Path) -> Path:
    """The closest existing directory at or above ``path``. Pure — unit-tested."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_writable_root(root: Path) -> None:
    """Fail fast (before connecting any hardware) if ``root`` cannot be written.

    Catches the common collection mishaps — an unmounted data drive, a wrong
    ``--dir``, or a quoted path that still contains backslash escapes — so the
    session aborts here instead of after both arms and every camera are open.
    """
    anchor = nearest_existing_ancestor(root)
    if not anchor.exists() or not os.access(anchor, os.W_OK):
        raise SystemExit(
            f"❌ cannot create the dataset at {root}: nothing writable at or above "
            f"it ({anchor}). Check that the data drive is mounted and that --dir is "
            "correct — a quoted path must NOT contain backslash escapes, e.g. use "
            '--dir "/media/you/My Drive/so101", not "/media/you/My\\ Drive/so101".'
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir", required=True, help="Collection directory (e.g. the data drive mount)"
    )
    parser.add_argument("--name", required=True, help="Dataset name (folder + repo id)")
    parser.add_argument(
        "--task", required=True, help="Language task string stored with every frame"
    )
    parser.add_argument(
        "--input",
        default="quest",
        choices=["quest", "leader"],
        help="Teleoperation interface (default quest)",
    )
    parser.add_argument(
        "--stream",
        action="append",
        default=[],
        metavar="NAME",
        help="UVC camera stream to store (repeatable). New datasets only; "
        "default is recording.yaml's enabled cameras.",
    )
    parser.add_argument(
        "--central-depth",
        action="store_true",
        help="Store the central RealSense RGB-D camera (new datasets only)",
    )
    parser.add_argument(
        "--no-ee",
        action="store_true",
        help="Do not store the EE-space features (new datasets only; quest mode)",
    )
    parser.add_argument(
        "--dataset-fps", type=int, default=None, help="Dataset frame rate override"
    )
    parser.add_argument("--ip-address", default=None, help="Quest IP (passthrough)")
    parser.add_argument(
        "--sensor-view", action="store_true", help="Open the live monitor"
    )
    parser.add_argument(
        "--goal",
        type=int,
        default=0,
        metavar="N",
        help="Target episode count; shows an episodes n/N progress bar on the "
        "--sensor-view footer (0 = no goal)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved teleop command and exit without running",
    )
    return parser.parse_args()


def _resolve_new(args, known_cameras, default_enabled):
    """(enabled_uvc, depth, record_ee, fps) for a brand-new dataset."""
    if args.stream:
        unknown = sorted(set(args.stream) - known_cameras)
        if unknown:
            raise SystemExit(
                f"❌ unknown --stream {unknown} (known: {sorted(known_cameras)})"
            )
        enabled_uvc = set(args.stream)
    else:
        enabled_uvc = set(default_enabled)
    record_ee = args.input == "quest" and not args.no_ee
    return enabled_uvc, args.central_depth, record_ee, args.dataset_fps


def _resolve_resume(args, settings, rs_rgb_name):
    """(enabled_uvc, depth, record_ee, fps) recovered from an existing dataset."""
    from common.recording.collection_settings import uvc_cameras

    enabled_uvc = uvc_cameras(
        settings["cameras"], settings["depth_rgb_name"] or rs_rgb_name
    )
    depth, record_ee, fps = settings["depth"], settings["ee"], settings["fps"]

    # A resumed dataset follows its own settings; warn if the operator asked
    # for something different so the ignored flags are not a silent surprise.
    if args.stream and set(args.stream) != enabled_uvc:
        print(
            f"⚠️  --stream ignored on resume; following recorded {sorted(enabled_uvc)}"
        )
    if args.central_depth and not depth:
        print("⚠️  --central-depth ignored on resume; dataset has no depth stream")
    if args.no_ee and record_ee:
        print("⚠️  --no-ee ignored on resume; dataset already has EE features")
    if args.dataset_fps and args.dataset_fps != fps:
        print(f"⚠️  --dataset-fps ignored on resume; dataset fps is {fps}")

    # Leader mode cannot produce EE targets, so an EE dataset cannot be resumed
    # with the leader — the frames would not match its features.
    if record_ee and args.input == "leader":
        raise SystemExit(
            "❌ this dataset has EE features (collected in quest mode); resume it "
            "with --input quest, not --input leader"
        )
    return enabled_uvc, depth, record_ee, fps


def main() -> None:
    args = _parse_args()

    from common.config_parser import load_recording_config
    from common.recording.collection_settings import (
        is_resumable_dataset,
        read_existing_streams,
        selection_to_teleop_flags,
    )

    rec_cfg = load_recording_config()
    known_cameras = set(rec_cfg["cameras"])
    default_enabled = {n for n, c in rec_cfg["cameras"].items() if c["enabled"]}
    rs_rgb_name = (rec_cfg.get("realsense") or {}).get("rgb_name")

    root = Path(args.dir).expanduser() / args.name
    resuming = is_resumable_dataset(root)
    check_writable_root(root)

    # A directory that exists but holds no saved episodes is a stillborn
    # dataset (created, then quit before recording). It can be neither created
    # into (create refuses an existing root) nor resumed (its metadata is
    # incomplete), so stop with an actionable message instead of a deep crash.
    if root.exists() and not resuming:
        raise SystemExit(
            f"❌ {root} exists but has no saved episodes yet (incomplete dataset). "
            f"Record at least one episode, or delete it to start over:\n"
            f"    rm -rf '{root}'"
        )

    if resuming:
        settings = read_existing_streams(root)
        print(f"📂 {root} exists — following its recorded settings (resume)")
        enabled_uvc, depth, record_ee, fps = _resolve_resume(
            args, settings, rs_rgb_name
        )
    else:
        enabled_uvc, depth, record_ee, fps = _resolve_new(
            args, known_cameras, default_enabled
        )

    flags = selection_to_teleop_flags(known_cameras, enabled_uvc, depth, record_ee, fps)
    argv = [
        str(_TELEOP),
        "--input",
        args.input,
        "--record",
        "--repo-id",
        args.name,
        "--dataset-root",
        str(root),
        "--task",
        args.task,
        *flags,
    ]
    if resuming:
        argv.append("--resume")
    if args.ip_address:
        argv += ["--ip-address", args.ip_address]
    if args.sensor_view:
        argv.append("--sensor-view")
    if args.goal:
        argv += ["--episode-goal", str(args.goal)]

    depth_str = "on" if depth else "off"
    ee_str = "on" if record_ee else "off"
    print(
        f"▶ {'resume' if resuming else 'new'} dataset '{args.name}' at {root}\n"
        f"  input={args.input}  cameras={sorted(enabled_uvc)}  depth={depth_str}  "
        f"ee={ee_str}  fps={fps or 'default'}"
    )

    if args.dry_run:
        print("\nteleop command (--dry-run, nothing run):")
        print("  " + " ".join([sys.executable, *argv]))
        return

    # Collection is local-only (the dataset is never pushed), so keep the
    # recorder off the network: a stray Hub lookup during create/resume would
    # otherwise fail with a misleading credentials error instead of staying
    # local. Hand off in THIS process — collection is a full teleop session, so
    # a clean exec (which also inherits the teleop shutdown path) beats
    # duplicating the arm/IK/camera stack here.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, *argv])


if __name__ == "__main__":
    main()
