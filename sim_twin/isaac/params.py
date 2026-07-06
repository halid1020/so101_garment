"""Loader for the resolved twin parameters (no repo imports).

Finds the ``twin/`` asset directory (robot.urdf, meshes/,
twin_params.json) in this order:

1. ``$SO101_TWIN_DIR``
2. ``twin/`` next to this package (the unzipped portable layout)
3. ``../../build/twin`` (running from the source repository)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def twin_dir() -> Path:
    candidates = []
    if os.environ.get("SO101_TWIN_DIR"):
        candidates.append(Path(os.environ["SO101_TWIN_DIR"]))
    candidates.append(_HERE / "twin")
    candidates.append(_HERE.parent.parent / "build" / "twin")
    for cand in candidates:
        if (cand / "twin_params.json").exists():
            return cand
    raise FileNotFoundError(
        "twin assets not found — run `python -m sim_twin.assets` in the "
        "source repo (or set $SO101_TWIN_DIR); looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def load_params() -> dict:
    return json.loads((twin_dir() / "twin_params.json").read_text())


# ---------------------------------------------------------------------
# minimal pose algebra (URDF-style fixed-axis rpy), numpy-free
# ---------------------------------------------------------------------


def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def mat_vec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def mat_to_quat(m):
    """Rotation matrix -> (w, x, y, z), Isaac's quaternion order."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k][j] - m[j][k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (m[j][i] + m[i][j]) / s
    q[k + 1] = (m[k][i] + m[i][k]) / s
    return tuple(q)


def compose(pose_a: dict, pose_b: dict) -> tuple[list, list]:
    """a ∘ b -> (xyz, 3x3 matrix); poses are {"xyz": .., "rpy": ..}."""
    ra = rpy_to_mat(pose_a["rpy"])
    rb = rpy_to_mat(pose_b["rpy"])
    xyz = [pose_a["xyz"][i] + mat_vec(ra, pose_b["xyz"])[i] for i in range(3)]
    return xyz, mat_mul(ra, rb)


def scene_camera_world(params: dict) -> tuple[list, tuple]:
    """World pose of the tower C310 optical frame (OpenGL convention:
    -Z look, +Y up — matches CameraCfg(convention="opengl"))."""
    cams = params["cameras"]
    xyz, rot = compose(cams["tower_cradle_world"], cams["tower_cam_in_cradle"])
    # C310 body frame (+Y look, +Z up) -> optical: Rx(+90°)
    opt = rpy_to_mat((math.pi / 2, 0.0, 0.0))
    return xyz, mat_to_quat(mat_mul(rot, opt))


def wrist_optical_offset() -> tuple:
    """Identity — the URDF's *_wrist_cam_optical links are already in
    the OpenGL camera convention."""
    return (1.0, 0.0, 0.0, 0.0)


def focal_length_from_dfov(
    dfov_deg: float, h_aperture: float = 20.955, aspect: float = 4 / 3
) -> float:
    """USD pinhole focal length (same units as aperture) from a
    diagonal FOV, 4:3 sensor."""
    half_diag = math.tan(math.radians(dfov_deg) / 2)
    diag_frac = math.sqrt(1 + 1 / aspect**2)  # horizontal share of diag
    half_h = half_diag * (1 / diag_frac)
    return h_aperture / (2 * half_h)
