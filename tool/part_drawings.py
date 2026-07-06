"""Engineering drawing sheets for the printed rig parts.

    python tool/part_drawings.py            # all sheets
    python tool/part_drawings.py adapter    # just one part

For every printable part this renders a third-angle multi-view sheet
(top / front / right feature-edge projections + shaded isometric) with
overall dimensions, key callouts pulled from config.scad, and a title
block. Output: outputs/drawings/<part>.png and one combined
so101_rig_drawings.pdf.

Prerequisite: ``python -m sim_twin.assets --print-parts`` (the sheets
draw the STLs in build/print/).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PolyCollection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim_twin.scad_params import parse_scad_params  # noqa: E402

PRINT_DIR = REPO / "build" / "print"
OUT_DIR = REPO / "outputs" / "drawings"

INK = "#20222c"  # line work
DIM = "#9c4221"  # dimensions (single restrained accent)
MUTED = "#6b7280"  # secondary text
FACE = "#c9cfdd"  # iso shading base
EDGE_ANGLE_DEG = 15  # feature-edge threshold

P = parse_scad_params(REPO / "src/platform/config.scad")

# part -> (stl name, quantity, description, key callouts)
SHEETS: dict[str, tuple[str, str, str, list[str]]] = {
    "arm_mount_adapter": (
        "arm_mount_adapter",
        "2",
        "SO-101 base -> board adapter",
        [
            f"plate {P['adapter_w']:.0f} x {P['adapter_d']:.0f} x "
            f"{P['adapter_thick']:.0f}",
            f"4x M5 board bolts on {P['adapter_hole_spacing']:.0f} sq,"
            " heads counterbored in top",
            f"base trapezoid: front {P['front_hole_spacing']:.3f} /"
            f" back {P['back_hole_spacing']:.3f},"
            f" {P['front_to_back_dist']:.0f} apart (MEASURED)",
            "M3 base screws enter from below",
            "hardware: 4x M5x25 SHCS + nuts, 4x M3x16",
        ],
    ),
    "board_tile_0": (
        "board_tile_0",
        "1",
        "printed board - tile A (clamp wing left)",
        [
            f"grid pitch {P['grid_pitch']:.0f}, holes {P['bolt_dia']:.1f} thru",
            f"M5 hex pockets {P['board_nut_pocket']:.0f} deep in BOTTOM face",
            f"solid clamp wing {P['board_clamp_wing']:.0f} at the outer end",
            "splice recess (8 deep) under the seam edge",
            "print TOP FACE DOWN - pockets need no support",
        ],
    ),
    "board_tile_1": (
        "board_tile_1",
        "1",
        "printed board - tile B (middle)",
        [
            "splice recesses under BOTH seam edges",
            "otherwise identical grid to tile A",
            "print TOP FACE DOWN",
        ],
    ),
    "board_tile_2": (
        "board_tile_2",
        "1",
        "printed board - tile C (clamp wing right)",
        ["mirror of tile A", "print TOP FACE DOWN"],
    ),
    "splice_bar": (
        "splice_bar",
        "2",
        "board seam splice bar",
        [
            f"{P['splice_bar_w']:.0f} x {P['board_depth']:.0f} x "
            f"{P['splice_bar_thick']:.0f}",
            "4 x 5 holes matching the board grid",
            f"M5 hex pockets {P['splice_nut_pocket']:.1f} deep in TOP face",
            "sits in the board's bottom recess, flush",
            ">= 3x M5x12 SHCS per side, staggered",
        ],
    ),
    "tower_base_plate": (
        "tower_base_plate",
        "1",
        "camera tower base",
        [
            f"foot {P['tower_base_plate']:.0f} sq x {P['tower_base_thick']:.0f}",
            f"4x M5x20 on {P['tower_hole_spacing']:.0f} sq",
            "triangular spigot on top (mast drops over it)",
            "spine-rod nut pocket in the spigot top",
        ],
    ),
    "tower_mast_segment_0": (
        "tower_mast_segment_0",
        "1",
        "mast segment 1 (widest, bottom)",
        [
            f"hollow triangle, wall {P['tower_wall']:.1f}",
            "2x M5x10 joint screws per joint (max length 10!)",
            "slide over the base spigot, widest first",
        ],
    ),
    "tower_mast_segment_1": (
        "tower_mast_segment_1",
        "1",
        "mast segment 2 (middle)",
        [],
    ),
    "tower_mast_segment_2": ("tower_mast_segment_2", "1", "mast segment 3 (top)", []),
    "tower_camera_platform": (
        "tower_camera_platform",
        "1",
        "tower top platform",
        [
            f"platform {P['camera_plate']:.0f} sq x "
            f"{P['camera_platform_thick']:.0f}",
            f"1/4\"-20 heat-set insert bore {P['tripod_insert_hole']:.1f}",
            "socket skirt drops over the top mast spigot",
        ],
    ),
    "wrist_camera_mount": (
        "wrist_camera_mount",
        "2",
        "C310 wrist camera mount",
        [
            f"plate {P['wrist_plate_w']:.0f} x {P['wrist_plate_l']:.0f} x "
            f"{P['wrist_plate_thick']:.0f}",
            f"2x M3x8 at {P['wrist_screw_spacing']:.2f} spacing (MEASURED,"
            " official SO-101 wrist interface)",
            f"tray tilted {P['wrist_cam_tilt_deg']:.0f} deg toward fingertips",
            "screws drive through the two vertical shafts",
            "camera zip-ties into the tray (2 ties)",
        ],
    ),
    "tower_camera_cradle": (
        "tower_camera_cradle",
        "1",
        "C310 tower camera cradle",
        [
            f"base {P['tower_cradle_base']:.0f} sq x " f"{P['tower_cradle_thick']:.0f}",
            '1x 1/4"-20 x 1/2" SHCS, 3/16" allen key through the shaft',
            f"tray tilted {P['tower_cam_tilt_deg']:.0f} deg down",
            "aim along the arms' +X before tightening",
        ],
    ),
}

VIEWS = (  # (title, index pair for the projection plane)
    ("TOP  (X-Y)", (0, 1)),
    ("FRONT  (X-Z)", (0, 2)),
    ("RIGHT  (Y-Z)", (1, 2)),
)


def feature_segments(mesh: trimesh.Trimesh) -> np.ndarray:
    angles = mesh.face_adjacency_angles
    sharp = mesh.face_adjacency_edges[angles > np.radians(EDGE_ANGLE_DEG)]
    return mesh.vertices[sharp]  # (n, 2, 3)


def draw_view(ax, segs3d, plane, title):
    i, j = plane
    segs = segs3d[:, :, (i, j)]
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=0.45))
    lo = segs.reshape(-1, 2).min(axis=0)
    hi = segs.reshape(-1, 2).max(axis=0)
    span = hi - lo
    pad = 0.30 * span.max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad * 0.9, hi[1] + pad * 0.9)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.5, color=MUTED, loc="left", pad=3)
    ax.axis("off")
    _dim_h(ax, lo[0], hi[0], lo[1] - 0.12 * span.max(), f"{span[0]:.1f}")
    _dim_v(ax, lo[1], hi[1], hi[0] + 0.12 * span.max(), f"{span[1]:.1f}")


def _dim_h(ax, x0, x1, y, label):
    ax.annotate(
        "", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", lw=0.7, color=DIM)
    )
    for x in (x0, x1):
        ax.plot([x, x], [y - 1, y + 3], lw=0.4, color=DIM, alpha=0.6)
    ax.text((x0 + x1) / 2, y, label, fontsize=7, color=DIM, ha="center", va="bottom")


def _dim_v(ax, y0, y1, x, label):
    ax.annotate(
        "", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<->", lw=0.7, color=DIM)
    )
    for y in (y0, y1):
        ax.plot([x - 1, x + 3], [y, y], lw=0.4, color=DIM, alpha=0.6)
    ax.text(
        x,
        (y0 + y1) / 2,
        label,
        fontsize=7,
        color=DIM,
        ha="left",
        va="center",
        rotation=90,
    )


def draw_iso(ax, mesh: trimesh.Trimesh):
    """Flat-shaded isometric projection (painter's algorithm)."""
    view = np.array([1.0, 1.0, 0.8])
    view /= np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    up2 = np.cross(view, right)
    tris = mesh.vertices[mesh.faces]  # (n, 3, 3)
    depth = tris.mean(axis=1) @ view
    order = np.argsort(depth)
    tris = tris[order]
    normals = mesh.face_normals[order]
    xy = np.stack([tris @ right, tris @ up2], axis=-1)
    light = np.array([0.5, 0.35, 0.8])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(FACE))
    colors = np.clip(shade[:, None] * base[None, :], 0, 1)
    ax.add_collection(PolyCollection(xy, facecolors=colors, edgecolors="none"))
    flat = xy.reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    pad = 0.06 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_title("ISOMETRIC", fontsize=7.5, color=MUTED, loc="left", pad=3)
    ax.axis("off")


def title_block(fig, name, qty, desc):
    y = 0.055
    fig.patches.append(
        plt.Rectangle(
            (0.03, 0.015),
            0.94,
            0.075,
            transform=fig.transFigure,
            fill=False,
            ec=INK,
            lw=0.8,
        )
    )
    fig.text(0.05, y, name.upper(), fontsize=11, weight="bold", color=INK)
    fig.text(0.40, y, desc, fontsize=8.5, color=INK, va="center")
    fig.text(
        0.76, y + 0.012, f"QTY: {qty}    UNITS: mm    PLA", fontsize=7.5, color=MUTED
    )
    fig.text(
        0.76,
        y - 0.014,
        f"SO-101 dual-arm rig  ·  {date.today().isoformat()}",
        fontsize=7.5,
        color=MUTED,
    )


def sheet(name: str, stl: Path, qty: str, desc: str, callouts: list[str]) -> plt.Figure:
    mesh = trimesh.load(stl)
    segs3d = feature_segments(mesh)

    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    grid = {
        "TOP  (X-Y)": fig.add_axes([0.04, 0.52, 0.42, 0.40]),
        "FRONT  (X-Z)": fig.add_axes([0.04, 0.12, 0.42, 0.36]),
        "RIGHT  (Y-Z)": fig.add_axes([0.47, 0.12, 0.26, 0.36]),
    }
    for (title, plane), ax in zip(VIEWS, grid.values()):
        draw_view(ax, segs3d, plane, title)
    iso_ax = fig.add_axes([0.50, 0.50, 0.34, 0.42])
    draw_iso(iso_ax, mesh)

    if callouts:
        fig.text(0.755, 0.46, "NOTES", fontsize=8, weight="bold", color=INK)
        for k, line in enumerate(callouts):
            fig.text(
                0.755,
                0.43 - 0.033 * k,
                f"{k + 1}.  {line}",
                fontsize=7.2,
                color=INK,
                wrap=True,
                ha="left",
                va="top",
            )
    ext = mesh.extents
    fig.text(
        0.755,
        0.46 - 0.033 * (len(callouts) + 1.2),
        f"bounding box: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm",
        fontsize=7.2,
        color=MUTED,
    )
    title_block(fig, name, qty, desc)
    return fig


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "so101_rig_drawings.pdf"
    todo = {k: v for k, v in SHEETS.items() if only is None or only in k}
    with PdfPages(pdf_path) as pdf:
        for name, (stl_name, qty, desc, callouts) in todo.items():
            stl = PRINT_DIR / f"{stl_name}.stl"
            if not stl.exists():
                print(
                    f"skip {name}: {stl} missing "
                    "(run python -m sim_twin.assets --print-parts)"
                )
                continue
            print(f"sheet: {name}")
            fig = sheet(name, stl, qty, desc, callouts)
            fig.savefig(OUT_DIR / f"{name}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"drawings in {OUT_DIR} (+ {pdf_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
