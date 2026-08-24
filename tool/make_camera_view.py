"""Build a camera-ablation view of a collected dataset.

A view is a dataset directory that LeRobot opens normally but that names only
some of the source's cameras, sharing the source's video files by symlink. It
is what makes a camera ablation a fair comparison: same episodes, same frames,
same actions, only the visible cameras differ. See
``src/common/recording/dataset_view.py`` for what is rewritten and why.

    venv/bin/python tool/make_camera_view.py \\
        --dataset /mnt/seagate/so101/cube-pnp-new --cameras central,wrist_camera_left

    venv/bin/python tool/make_camera_view.py \\
        --dataset .../cube-pnp-new --cameras all --out-dir /scratch/.../local

With no ``--out`` the view is created beside the source, named
``<dataset>__<slug>`` (``cube-pnp-new__central+wrist_left``). ``--list`` prints
the cameras the dataset has and builds nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from common.recording.dataset_view import (  # noqa: E402
    ViewError,
    build_view,
    camera_keys,
    pi05_rename_map,
    resolve_cameras,
    short_name,
    view_slug,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="Source dataset root")
    parser.add_argument(
        "--cameras",
        default="all",
        help="Comma list of camera names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--out", help="View directory (default: <source>__<slug> beside it)"
    )
    parser.add_argument(
        "--out-dir", help="Directory to create the view in, keeping the derived name"
    )
    parser.add_argument(
        "--canonical-task",
        help="Force this task string on every frame (default: whichever spelling "
        "covers the most frames, when a dataset carries more than one)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild an existing view")
    parser.add_argument(
        "--list", action="store_true", help="List the source's cameras and exit"
    )
    parser.add_argument(
        "--print",
        choices=("path", "slug", "pi05-rename-map"),
        help="Print one value and nothing else, for a script to capture. "
        "'pi05-rename-map' maps this dataset's cameras onto pi0.5's pretrained slots.",
    )
    args = parser.parse_args()

    src = Path(args.dataset).expanduser()
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"❌ no dataset at {src}")
    info = json.loads(info_path.read_text())

    if args.list:
        print(
            f"{src.name}: {info['total_episodes']} episodes, {info['total_frames']} frames"
        )
        for key in camera_keys(info):
            shape = info["features"][key]["shape"]
            print(f"  {short_name(key):24s} {shape}")
        return

    try:
        keep = resolve_cameras(info, args.cameras.split(","))
        slug = view_slug(info, keep)

        # Queries answer about the dataset as it stands and build nothing, so a
        # driver can ask what to pass a policy without side effects.
        if args.print == "slug":
            print(slug)
            return
        if args.print == "pi05-rename-map":
            print(json.dumps(pi05_rename_map(keep), separators=(",", ":")))
            return

        if args.out:
            dst = Path(args.out).expanduser()
        else:
            parent = Path(args.out_dir).expanduser() if args.out_dir else src.parent
            dst = parent / f"{src.name}__{slug}"
        existed = dst.exists() and not args.force
        build_view(src, dst, keep, canonical_task=args.canonical_task, force=args.force)
    except ViewError as exc:
        raise SystemExit(f"❌ {exc}")

    if args.print == "path":
        print(dst)
        return
    print(f"{'↷ reusing' if existed else '✓ built'} {dst}")
    print(f"  cameras: {', '.join(short_name(k) for k in keep)}")


if __name__ == "__main__":
    main()
