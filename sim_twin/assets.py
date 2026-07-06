"""Asset pipeline: OpenSCAD -> meshes -> URDF -> twin_params.json.

Run ``python -m sim_twin.assets``. Exports the sim meshes from
``src/platform/export.scad`` (content-hash cached — CGAL renders are
slow), scales them from mm to meters, copies the robot description
meshes alongside, generates ``build/twin/robot.urdf`` and writes the
resolved ``twin_params.json``. ``build/twin/`` ends up fully
self-contained so the Isaac package can zip it up and leave the repo.

Flags:
  --force          ignore the hash cache and re-export everything
  --check          verify CGAL volume counts, report, and exit non-zero
                   on mismatch (no other outputs touched)
  --print-parts    also export every *printable* part to build/print/
                   (mm STLs, ready to slice)
  --package-isaac  zip sim_twin/isaac + build/twin into a portable
                   archive for the Isaac Lab machine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from sim_benchmark.constants import DESCRIPTION_DIR, DUAL_URDF_PATH
from sim_twin import urdf_gen
from sim_twin.params import BUILD_DIR, PLATFORM_DIR, REPO_ROOT, TwinParams

MESH_DIR = BUILD_DIR / "meshes"
CACHE_PATH = BUILD_DIR / ".hashcache.json"
EXPORT_SCAD = PLATFORM_DIR / "export.scad"

# Sim meshes: part name -> (extra .scad deps, expected CGAL volume count).
# Volumes == solids + 1 (the unbounded exterior cell). tower_assembled
# additionally encloses the joint-gap cavities between spigots and
# sockets, hence the larger, empirically confirmed count — treat a
# mismatch there as a warning, not an error.
SIM_PARTS: dict[str, tuple[list[str], int]] = {
    "adapter": (["arm_mount_adapter.scad"], 2),
    "board_assembled": (["board.scad"], 2),
    "tower_assembled": (["camera_tower.scad"], 5),
    "wrist_camera_mount": (["wrist_camera_mount.scad", "cam_tray_lib.scad"], 2),
    "tower_camera_cradle": (["tower_camera_cradle.scad", "cam_tray_lib.scad"], 2),
    "cam_body": (["cam_body.scad"], 2),
}
VOLUME_WARN_ONLY = {"tower_assembled"}

# Printable parts for --print-parts: name -> extra -D defines.
# (drill_template.scad's nut plate is legacy — wooden-board rig only.)
PRINT_PARTS: dict[str, list[str]] = {
    "arm_mount_adapter": ["-D", 'part="adapter"'],
    "board_tile_0": ["-D", 'part="board_tile"', "-D", "seg=0"],
    "board_tile_1": ["-D", 'part="board_tile"', "-D", "seg=1"],
    "board_tile_2": ["-D", 'part="board_tile"', "-D", "seg=2"],
    "splice_bar": ["-D", 'part="splice_bar"'],
    "tower_base_plate": ["-D", 'part="tower_base_plate"'],
    "tower_mast_segment_0": ["-D", 'part="tower_mast_segment"', "-D", "seg=0"],
    "tower_mast_segment_1": ["-D", 'part="tower_mast_segment"', "-D", "seg=1"],
    "tower_mast_segment_2": ["-D", 'part="tower_mast_segment"', "-D", "seg=2"],
    "tower_camera_platform": ["-D", 'part="tower_camera_platform"'],
    "wrist_camera_mount": ["-D", 'part="wrist_camera_mount"'],
    "tower_camera_cradle": ["-D", 'part="tower_camera_cradle"'],
}


def _openscad_version() -> str:
    out = subprocess.run(["openscad", "--version"], capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def _part_hash(part: str, deps: list[str], version: str) -> str:
    h = hashlib.sha256()
    h.update(f"{part}|{version}|".encode())
    for dep in ["config.scad", "export.scad", *deps]:
        h.update((PLATFORM_DIR / dep).read_bytes())
    return h.hexdigest()


def _run_openscad(out_stl: Path, defines: list[str]) -> str:
    """Export one part; returns captured console output (CGAL stats)."""
    result = subprocess.run(
        ["openscad", "-o", str(out_stl), *defines, str(EXPORT_SCAD)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or not out_stl.exists():
        raise RuntimeError(f"openscad failed for {out_stl.name}:\n{output}")
    return output


def _volumes(console: str) -> int | None:
    match = re.search(r"Volumes:\s*(\d+)", console)
    return int(match.group(1)) if match else None


def _check_volumes(part: str, count: int | None, expected: int) -> bool:
    if count == expected:
        return True
    message = f"{part}: CGAL volumes {count}, expected {expected}"
    if part in VOLUME_WARN_ONLY:
        print(f"  WARNING {message} (assembly cavity count is empirical)")
        return True
    print(f"  ERROR   {message} — likely a floating/disconnected solid")
    return False


def export_sim_meshes(force: bool = False, check: bool = False) -> bool:
    """Export + scale the sim meshes. Returns overall volume-check pass."""
    import trimesh

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    version = _openscad_version()
    ok = True

    for part, (deps, expected) in SIM_PARTS.items():
        target = MESH_DIR / f"{part}.stl"
        digest = _part_hash(part, deps, version)
        if not force and not check and target.exists() and cache.get(part) == digest:
            continue
        print(f"  openscad: {part} ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / f"{part}.stl"
            console = _run_openscad(raw, ["-D", f'part="{part}"'])
            ok &= _check_volumes(part, _volumes(console), expected)
            if check:
                continue
            mesh = trimesh.load(raw)
            mesh.apply_scale(0.001)  # SCAD mm -> sim meters
            mesh.export(target)
        cache[part] = digest

    if not check:
        CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n")
    return ok


def copy_robot_meshes() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted((DESCRIPTION_DIR / "meshes").glob("*.stl")):
        dst = MESH_DIR / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)


def export_print_parts() -> None:
    print_dir = REPO_ROOT / "build" / "print"
    print_dir.mkdir(parents=True, exist_ok=True)
    for name, defines in PRINT_PARTS.items():
        target = print_dir / f"{name}.stl"
        print(f"  openscad (print): {name} ...", flush=True)
        _run_openscad(target, defines)
    print(f"printable STLs in {print_dir}")


def package_isaac() -> Path:
    archive = REPO_ROOT / "build" / "so101_twin_isaac.zip"
    isaac_dir = REPO_ROOT / "sim_twin" / "isaac"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(isaac_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, Path("so101_twin") / path.relative_to(isaac_dir))
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file() and path.name != ".hashcache.json":
                zf.write(
                    path,
                    Path("so101_twin") / "twin" / path.relative_to(BUILD_DIR),
                )
    print(f"portable Isaac package: {archive}")
    return archive


def build(force: bool = False) -> TwinParams:
    """Full pipeline; safe to call repeatedly (cache makes it cheap)."""
    params = TwinParams.load()
    if not export_sim_meshes(force=force):
        raise RuntimeError("mesh volume check failed — see errors above")
    copy_robot_meshes()
    urdf_gen.generate(params, BUILD_DIR / "robot.urdf", DUAL_URDF_PATH)
    params.write_json(BUILD_DIR / "twin_params.json")
    return params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-parts", action="store_true")
    parser.add_argument("--package-isaac", action="store_true")
    args = parser.parse_args()

    if args.check:
        return 0 if export_sim_meshes(check=True) else 1
    build(force=args.force)
    print(f"twin assets ready in {BUILD_DIR}")
    if args.print_parts:
        export_print_parts()
    if args.package_isaac:
        package_isaac()
    return 0


if __name__ == "__main__":
    sys.exit(main())
