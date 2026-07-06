# SO-101 dual-arm rig — Isaac Lab twin (portable package)

Digital twin of the dual SO-101 board rig (two arms on printed adapters,
grid board, camera tower, three Logitech C310s) for **Isaac Lab 2.x**.
Everything geometric comes from the repo's `src/platform/config.scad` via
the resolved `twin/twin_params.json` — no repo imports, so this directory
plus `twin/` runs standalone on the Isaac machine.

Built and locally verified against the MuJoCo twin on 2026-07-06; the
Isaac side is **API-written but not yet executed** (the source machine
has no Isaac-capable GPU) — expect to touch small things, see the
checklist below.

## Requirements (Isaac machine)

- Isaac Sim 4.5+ / Isaac Lab 2.x installed and working
- This package, either as the zip (`so101_twin/` with `twin/` inside)
  or the repo checkout (it finds `../../build/twin` automatically;
  `$SO101_TWIN_DIR` overrides).

## Steps

```bash
# 1. sanity-check the assets came along (no isaaclab needed)
python convert_assets.py --dry-run

# 2. URDF + STL -> USD (writes twin/usd/)
./isaaclab.sh -p convert_assets.py

# 3. spawn, hold neutral pose, dump the three camera views to out/
./isaaclab.sh -p run_demo.py --headless
```

## Verification checklist (mirror of the MuJoCo twin)

- `left_base_link` world position printed by run_demo ≈
  `(0, +arm_spacing/2, arm_base_height)` from `twin/twin_params.json`
  (defaults: `(0, 0.15, 0.0444)`).
- `out/rgb_scene.png`: workspace table filling the frame, both wrist
  assemblies at the top corners, table front edge near the frame top.
- `out/rgb_wrist_left|right.png`: looking down the gripper jaws at the
  grasp point, table beyond.
- Arms hold the neutral pose (no sag/oscillation blowup).

## Known drift points (fix here, not in the geometry)

- **Converter API**: `UrdfConverterCfg` / `JointDriveCfg` field names
  were restructured between Isaac Lab 1.x and 2.x and may move again —
  `convert_assets.py` is the only file touching them.
- **PD gains**: `actuator_kp/kv` (25 / 1.5) are the MuJoCo servo values;
  PhysX implicit actuators interpret gains differently, so treat them as
  starting points and retune until the neutral hold matches the MuJoCo
  drift (≈1.4° sag).
- **Camera conventions**: the URDF's `*_wrist_cam_optical` links and the
  precomputed tower-camera pose are in the OpenGL convention (-Z look,
  +Y up) and every `CameraCfg` uses `convention="opengl"` with identity
  offset. If an image comes out rotated/mirrored, fix the `OffsetCfg`
  in `twin_scene.py`, not the URDF.
- **Dynamic scene attributes**: `TwinSceneCfg.__post_init__` attaches
  the furniture assets with `setattr`; if your Isaac Lab version's
  `InteractiveScene` only discovers declared fields, inline them as
  class attributes instead.

## Updating after a design change

Re-run in the source repo:

```bash
python -m sim_twin.assets --package-isaac
```

and copy the new `build/so101_twin_isaac.zip` over — `twin/` carries all
geometry; nothing in this directory needs editing for parameter changes.
