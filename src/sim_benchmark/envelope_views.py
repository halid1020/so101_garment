"""Render the per-arm teleoperation reach envelopes as 3-D figures.

The envelope that gates teleoperation (``common.workspace_envelope``) is an
annulus around each arm's shoulder-lift pivot, ``r_min <= |p - pivot| <=
r_max``, cut by a table-clearance floor ``p.z >= z_floor``. Operators cannot
see this constraint while teleoperating, so when the hand target is silently
clamped (or warned) the arm feels "stuck" — most visibly it refuses to lower
the gripper all the way to the table. These plots make the constraint
concrete: for each arm we draw the outer reach shell (``r_max``) truncated at
the floor, the inner dead-zone sphere (``r_min``), the clearance floor, and
the table surface, from four fixed camera angles plus a vertical cross
section that reads off directly how far above the table the envelope stops.

All coordinates are the flat IK / URDF frame (arm base at z = 0), the same
frame the envelope constants are expressed in. ``table_z`` marks where the
work surface sits in that frame (0 by default: level with the arm base).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.workspace_envelope import ArmEnvelope

# Deferred: matplotlib is only needed for rendering, not for the teleop or
# benchmark run paths that import the rest of sim_benchmark.
_SIDE_COLOUR = {"left": "#1f77b4", "right": "#d62728"}

# Four fixed cameras (elev, azim) in degrees: top-down, front, side, iso.
_VIEWS = {
    "top": (89.0, -90.0),
    "front": (8.0, -90.0),
    "side": (6.0, 0.0),
    "iso": (24.0, -60.0),
}


def _truncated_sphere(
    centre: np.ndarray, radius: float, z_floor: float, n: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sphere mesh with points below ``z_floor`` masked to NaN (a floor cut)."""
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    mask = z < z_floor
    x[mask] = np.nan
    y[mask] = np.nan
    z[mask] = np.nan
    return x, y, z


def _lowest_reachable_z(env: ArmEnvelope) -> float:
    """Deepest EE z the envelope permits (max of floor and radial reach)."""
    return max(env.z_floor, env.pivot_z - env.r_max)


def _draw_scene(ax, envelopes: dict[str, ArmEnvelope], table_z: float) -> None:
    import matplotlib.pyplot as plt  # noqa: F401  (Axes3D already active)

    pans = np.array([e.pan_xy for e in envelopes.values()])
    reach = max(e.r_max for e in envelopes.values())
    xlim = (pans[:, 0].min() - reach, pans[:, 0].max() + reach)
    ylim = (pans[:, 1].min() - reach, pans[:, 1].max() + reach)

    for side, env in envelopes.items():
        colour = _SIDE_COLOUR.get(side, "#555555")
        centre = np.array(
            [env.pan_xy[0] + env.pivot_offset_xy[0], env.pan_xy[1], env.pivot_z]
        )
        xo, yo, zo = _truncated_sphere(centre, env.r_max, env.z_floor)
        ax.plot_surface(xo, yo, zo, color=colour, alpha=0.12, linewidth=0, shade=False)
        xi, yi, zi = _truncated_sphere(centre, env.r_min, env.z_floor, n=24)
        ax.plot_surface(xi, yi, zi, color=colour, alpha=0.35, linewidth=0, shade=False)
        ax.scatter(*centre, color=colour, s=30, marker="o")
        ax.scatter(env.pan_xy[0], env.pan_xy[1], 0.0, color=colour, s=40, marker="^")

    # Table surface and the (higher) clearance floor the envelope enforces.
    gx, gy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
    ax.plot_surface(
        gx, gy, np.full_like(gx, table_z), color="#8c8c8c", alpha=0.25, linewidth=0
    )
    z_floor = next(iter(envelopes.values())).z_floor
    if abs(z_floor - table_z) > 1e-6:
        ax.plot_surface(
            gx, gy, np.full_like(gx, z_floor), color="#2ca02c", alpha=0.18, linewidth=0
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(
        min(table_z, z_floor) - 0.02,
        max(e.pivot_z + e.r_max for e in envelopes.values()),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], 0.5))
    except Exception:
        pass


def render_envelope_views(
    envelopes: dict[str, ArmEnvelope],
    out_dir: str | Path,
    table_z: float = 0.0,
) -> list[Path]:
    """Write the four 3-D views + a vertical cross section as PNGs.

    Returns the list of written paths. ``table_z`` is the work-surface height
    in the IK frame (0 = level with the arm base).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, (elev, azim) in _VIEWS.items():
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        _draw_scene(ax, envelopes, table_z)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Teleop reach envelope — {name} view")
        path = out_dir / f"envelope_{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    written.append(_render_cross_section(envelopes, out_dir, table_z))
    return written


def _render_cross_section(
    envelopes: dict[str, ArmEnvelope], out_dir: Path, table_z: float
) -> Path:
    """Vertical (x–z) slice through each pivot: the clearest 'how low' read."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for idx, (side, env) in enumerate(envelopes.items()):
        colour = _SIDE_COLOUR.get(side, "#555555")
        cx = env.pan_xy[0] + env.pivot_offset_xy[0]
        cz = env.pivot_z
        for r, style in ((env.r_max, "-"), (env.r_min, "--")):
            x = cx + r * np.cos(theta)
            z = cz + r * np.sin(theta)
            keep = z >= env.z_floor
            ax.plot(x[keep], z[keep], style, color=colour, lw=1.5)
        low = _lowest_reachable_z(env)
        # Left/right sections coincide in x-z, so stagger the callouts.
        ax.annotate(
            f"{side}: lowest EE z = {low * 1000:.0f} mm\n"
            f"(= {(low - table_z) * 1000:.0f} mm above table)",
            xy=(cx, low),
            xytext=(cx + 0.06, low + 0.06 + 0.09 * idx),
            fontsize=8,
            color=colour,
            arrowprops=dict(arrowstyle="->", color=colour, lw=1),
        )
    ax.axhline(table_z, color="#8c8c8c", lw=2, label="table surface")
    z_floor = next(iter(envelopes.values())).z_floor
    ax.axhline(z_floor, color="#2ca02c", lw=1.2, ls=":", label="envelope floor")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Reach envelope — vertical section through the lift pivots")
    path = out_dir / "envelope_section.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
