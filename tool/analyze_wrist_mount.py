"""Measure the SO-101 wrist camera-mount interface from reference meshes.

Extracts the two M3 screw positions (and the flat mating face) that the
official SO-ARM100 wrist camera mounts use on the wrist-roll follower
element, and expresses them in the URDF ``gripper_link`` frame. The printed
``src/platform/wrist_camera_mount.scad`` and the camera poses in
``src/sim_twin`` are derived from these numbers; re-run this script if the
upstream meshes ever change and update ``config.scad`` accordingly.

Method: slice the mesh with planes perpendicular to the screw axis and fit
circles to small closed boundary loops (bore = ~3.2 mm circle, hex-nut
pocket = ~5.9 mm equivalent-diameter hexagon).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
WRIST_STL = REPO / "src/so101_dual_description/meshes/wrist_roll_follower_so101_v1.stl"
# Visual/collision origin of that mesh inside {side}_gripper_link (from
# src/so101_dual_description/robot.urdf): xyz + rpy=(-pi, 0, 0).
MESH_XYZ = np.array([0.0, -0.000218214, 0.000949706])
MESH_RPY_X = -np.pi  # rotation about X only


def circles_in_slice(mesh, origin, normal, dia_range, circularity_tol=0.15):
    """Circular-ish closed loops in a planar cross-section of ``mesh``."""
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return []
    planar, to_3d = section.to_2D()
    out = []
    for poly in planar.polygons_closed:
        if poly is None:
            continue
        radius = np.sqrt(poly.area / np.pi)
        if radius <= 0:
            continue
        if not (dia_range[0] < 2 * radius < dia_range[1]):
            continue
        if abs(poly.length / (2 * np.pi * radius) - 1) > circularity_tol:
            continue
        c2 = np.array(poly.centroid.coords[0])
        c3 = trimesh.transform_points([[c2[0], c2[1], 0.0]], to_3d)[0]
        out.append((2 * radius, c3))
    return out


def mesh_to_gripper(p_mm: np.ndarray) -> np.ndarray:
    """Mesh-frame point (mm) -> gripper_link frame (m)."""
    p = p_mm / 1000.0
    rotated = np.array([p[0], -p[1], -p[2]])  # Rx(pi)
    return rotated + MESH_XYZ


def main() -> None:
    mesh = trimesh.load(WRIST_STL)
    mesh.apply_scale(1000.0)  # URDF meshes are in metres; work in mm

    # Mating face: the large planar facet with outward normal -Y.
    face_y = None
    for n, a, o in zip(mesh.facets_normal, mesh.facets_area, mesh.facets_origin):
        if a > 300 and n[1] < -0.98:
            face_y = o[1]
    assert face_y is not None, "mating face (large -Y facet) not found"

    # M3 clearance bores: slice 1.2 mm inside the face, screw axis is +Y.
    bores = circles_in_slice(
        mesh, [0, face_y + 1.2, 0], [0, 1, 0], dia_range=(2.6, 3.8)
    )
    assert len(bores) == 2, f"expected 2 M3 bores, found {len(bores)}"
    bores.sort(key=lambda item: item[1][0])

    # Hex-nut pockets sit behind ~3 mm of wall (nut slides in sideways).
    pockets = circles_in_slice(
        mesh, [0, face_y + 5.0, 0], [0, 1, 0], dia_range=(5.2, 6.8)
    )

    print(f"wrist mesh mating face: y = {face_y:.3f} mm (outward normal -Y)")
    for dia, c in bores:
        g = mesh_to_gripper(np.array([c[0], face_y, c[2]]))
        print(
            f"M3 bore dia {dia:.2f} mm at mesh (x,z)=({c[0]:.3f},{c[2]:.3f})"
            f" -> gripper_link frame ({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}) m"
        )
    spacing = abs(bores[1][1][0] - bores[0][1][0])
    mid_x = (bores[1][1][0] + bores[0][1][0]) / 2
    print(f"screw spacing: {spacing:.3f} mm, midpoint x = {mid_x:.3f} mm")
    print(f"hex pockets found: {len(pockets)} (eq-dia ~5.9 mm expected, M3 nut)")

    mid = mesh_to_gripper(np.array([mid_x, face_y, bores[0][1][2]]))
    print(
        "\nconfig.scad / sim values:\n"
        f"  wrist_cam_screw_spacing = {spacing:.2f}  (mm)\n"
        f"  wrist mount face in gripper_link frame: y = {mid[1]:.4f} m,"
        " outward normal +Y\n"
        f"  screw midpoint in gripper_link frame: ({mid[0]:.4f}, {mid[1]:.4f},"
        f" {mid[2]:.4f}) m"
    )


if __name__ == "__main__":
    main()
