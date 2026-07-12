# Project: SO-101 dual-arm board rig

Parametric OpenSCAD design for a 3D-printable rig that mounts two SO-101
LeRobot arms plus an overhead camera tower on a shared board, for a
teleoperation / imitation-learning setup. This file is background for you
(Claude Code) — read it before making changes, and keep it in sync if the
design changes.

## What this actually is

- A **fully 3D-printed board** (3 tiles + 2 splice bars, `board.scad`)
  with a uniform grid of through-holes, pegboard-style, so the arms and
  tower can be repositioned later instead of being fixed in one spot.
  Every grid hole has an M5 hex pocket molded into the board's BOTTOM
  face (this replaced the old wooden board + separate under-board nut
  plate). Solid 60mm **clamp wings** at both ends take the table clamps.
- Two **printed adapter plates**, one per arm, that bolt directly into each
  SO-101 base's real (measured) mounting holes on top, and bolt into 4
  adjacent grid holes on the board on the bottom.
- A **printed modular camera tower** (base plate + stacked mast segments +
  camera platform) that bolts to the board at the grid position centered
  between the two arms, with an optional core spine rod for rigidity.

The whole point of the grid + adapter approach: the SO-101 base's own hole
pattern is a small irregular trapezoid that doesn't line up with any
uniform grid, so a small fixed adapter bridges "the base's real holes" to
"any position on a reconfigurable board."

## Files

- `config.scad` — single source of truth. Every dimension is a named,
  commented variable here. **Always edit this file for parameter changes,
  never hardcode a number in another file.**
- `arm_mount_adapter.scad` — print x2. Includes `config.scad`.
- `board.scad` — print 3x board_tile + 2x splice_bar. The perforated
  board itself: tiles butt-join at grid-column midpoints and are tied by
  splice bars bolted (M5x12) into 8mm blind recesses in the underside,
  so the bottom face stays flush for clamping. Tiles print TOP FACE
  DOWN. Asserts every tile fits `printer_bed`. `board_assembled` is the
  sim-twin visual, never printed.
- `camera_tower.scad` — print x1 base_plate, N mast_segments, x1
  camera_platform (all laid out side by side in one render). Includes
  `config.scad`. `tower_yaw_deg = -90` points one triangle corner at
  the front (+X), per explicit user request.
- `drill_template.scad` — LEGACY, don't print. Drill guide / under-board
  nut plate for the original wooden-board rig, superseded by the printed
  board. (History: the user first rejected tiling the nut plate, then on
  2026-07-06 explicitly asked for a printable tiled board with robust
  connectors and clamp room at both ends — that request produced
  `board.scad` and made this file legacy.)
- `README.md` — human-facing summary: what to print how many times,
  assembly order, which numbers are placeholders. **Keep this updated
  whenever you change geometry or hardware sizes** — it's the file the
  user actually reads before printing.
- `wrist_camera_mount.scad` — print x2. Logitech C310 holder on the
  SO-101 wrist. Interface = the official SO-ARM100 wrist mount's two
  M3 holes (8.10mm spacing, MEASURED from the official STL by
  `tool/analyze_wrist_mount.py` — treat like the caliper values below).
  Part frame: origin at the screw midpoint, +Y toward fingertips,
  +Z away from the wrist face.
- `tower_camera_cradle.scad` — print x1. C310 cradle bolting into the
  tower platform's 1/4"-20 insert (single bolt, aim then tighten).
- `cam_tray_lib.scad` — shared tray/pedestal modules (`use <>`d by
  both camera holders). Zip-tie retention instead of a rigid snap fit,
  deliberately: the C310 body dims are CONFIRM-flagged.
- `cam_body.scad` — NOT printable; simplified C310 visual mesh for
  the sim twin. Origin at the optical center, optical axis +Y.
- `export.scad` — headless per-part export dispatcher
  (`-D 'part="..."'`, plus `-D seg=N` for mast segments). `use <>`s
  every part file, so their top-level print layouts don't execute.
  `src/sim_twin/assets.py` drives it; keep part names in sync there.

## Confirmed / measured values (config.scad) — do not casually change these

These came from the user's own caliper measurements on the physical
SO-101 base, not guesses:
- `base_hole_dia = 3.8` mm
- `front_hole_spacing = 55.419` mm (center-to-center)
- `back_hole_spacing = 63.251` mm (center-to-center)
- `front_to_back_dist = 70` mm

If asked to change the adapter's top hole pattern, these are the values
that must stay accurate — everything else in the adapter derives from
them.

Additionally these were measured from the SO-101 reference meshes by
`tool/analyze_wrist_mount.py` (mesh analysis, not calipers — but same
do-not-casually-change status):
- `wrist_screw_spacing = 8.10` mm (official wrist-mount M3 pair)
- `wrist_iface_x/y/z` — screw midpoint in the URDF gripper_link frame
- `base_holes_x = 21.2` mm, `base_bottom_z = -2.4` mm — base-hole
  trapezoid center and base underside in the base_link frame

## The sim digital twin reads this directory

`src/sim_twin/` parses `config.scad` and headless-exports parts
via `export.scad` to build MuJoCo/Isaac Lab scenes. Consequences:
- Adding/renaming a part module: update `export.scad`'s dispatch AND
  the part table in `src/sim_twin/assets.py`.
- The `RIG PLACEMENT` / C310 / tray / anchor sections in `config.scad`
  exist for the twin; keep them parseable as plain `name = number;`
  assignments (simple arithmetic is fine, no conditionals).
- `tower_spigot_h` and `camera_platform_thick` moved INTO config.scad
  because the twin computes tower stack heights from them.

## Placeholder / assumed values — flag these, don't silently trust them

- `base_screw_head_dia` / `base_screw_head_h` — assumed M3 pan/socket
  head, unconfirmed against the user's actual screws.
- `min_arm_spacing` / `max_arm_spacing` (150-400mm) — a guessed
  adjustability range, not a hard requirement.
- `board_thickness` (18mm) — user's stated plan, not yet locked in.
- `board_margin`, `grid_pitch` — design choices (25mm each), reasonable
  defaults, freely adjustable.

## Design decisions already made — don't relitigate without being asked

1. **Bolts from the top, nuts captive in hex pockets molded into the
   board's bottom face.** Every board-to-part M5 bolt drops in from
   above (head counterbored flush into the printed part's top face),
   passes through the board, and threads into a hex nut pushed into an
   8mm blind pocket in the board's underside (`board_nut_pocket`). In
   splice-bar zones the nut sits in the bar's top-face pocket instead —
   the pocket heights coincide, so bolt lengths don't change. The hard
   requirement behind this: the board gets clamped flat to a table, so
   nothing may protrude below it — pockets and recesses are blind, the
   bottom face is flush. Bolt lengths are stack-up-verified against
   standard catalogue sizes in the README (M5x25 adapters, M5x20 tower,
   M5x12 splice bars — **M5x16 on the splice bars bottoms out**);
   re-verify if any part thickness, `board_thickness`, or
   `board_nut_pocket` changes. (This scheme replaced bolts-from-below,
   then recesses-drilled-into-the-board, then the wooden board +
   under-board nut plate, all at the user's request.)
2. **Grid, not fixed holes.** The board is a uniform hole grid so the rig
   is reconfigurable. Don't replace this with fixed-position holes unless
   explicitly asked.
3. **Two-stage adapter, not a direct base-to-board bolt.** The SO-101
   base's hole pattern is measured and irregular; it does not align to
   the grid. The adapter plate is the bridge — top matches the base
   exactly, board bolts are grid-pitch-spaced. The 4 board bolts form a
   100x100mm square (`adapter_hole_spacing = 4*grid_pitch`) placed
   fully *outside* the base's trapezoid, per explicit user request, so
   the top-inserted bolt heads stay reachable with the base attached.
   Assembly order matters and is documented in the README:
   base-to-adapter M3 screws (from the adapter's underside) must be
   installed *before* the adapter is bolted to the board, because the
   underside becomes inaccessible afterward. Preserve this constraint
   in any redesign.
4. **M5 hardware for board/tower, M3 for base-to-adapter.** The user
   initially asked for M5 everywhere including the base's own holes, but
   an M5 shaft (5mm) physically cannot pass through a 3.8mm hole. This was
   corrected and flagged rather than silently complied with — keep that
   posture: if a future instruction is physically inconsistent with a
   measured/confirmed value above, say so explicitly rather than forcing
   it through.
5. **Stability features present for a reason**, added because the rig
   shakes during teleoperation/inference (dynamic torque loads, not just
   static weight):
   - 4-bolt square board mount per adapter (not 2 in a line) — resists
     twisting, not just pull-off.
   - Tapered triangular mast (per explicit user request): hollow
     equilateral-triangle tube, 110mm side at the board tapering to
     45mm at the top, 4mm wall — wide where the bending moment is
     largest, triangular section against twist.
   - Optional M5 core spine rod through the tower, tensioned base-to-top,
     puts joints into compression instead of relying on joint screws
     alone. Terminates below the camera_platform (doesn't pass through
     it) so the tripod insert hole stays clear.
   If asked to further improve rigidity, these are the levers that
   already worked; consider extending them (e.g. spine rod through more
   of the structure, wider adapter footprint) before inventing new
   mechanisms.
6. **Joint style in the tower**: every part has a solid triangular
   spigot on top (a clearance-shrunk continuation of the tube's inner
   cavity — it also locks rotation), and the hollow tube interior is
   the socket, open at the bottom. The part above always sits down over
   the part below's spigot. Each mast segment has a solid floor
   (`cap_h`) under its spigot — without it the spigot is a
   disconnected floating volume (this was a real rendered bug; the
   `Volumes:` count in headless CGAL output is a cheap way to catch
   it: it should be number-of-parts + 1). Keep this convention if
   adding new tower parts.

## Environment notes

- OpenSCAD is installed locally and is on PATH as `openscad`. Verify with
  `openscad --version` before assuming it's available.
- If VS Code was installed via snap, spawning `openscad` from it may hit a
  known snap/glibc symbol-lookup bug
  (`undefined symbol: __libc_pthread_init`). If you hit this from a
  headless render command, it's an environment issue, not a `.scad` bug —
  don't start "fixing" the model in response to it.
- Headless render/export from the command line, useful for verifying a
  change actually renders without opening the GUI:
  ```
  openscad -o output.stl arm_mount_adapter.scad
  ```
  Use `--export-format=asciistl` or check `openscad --help` for current
  flags if that fails — CLI flags have changed across versions.
- No internet access should be assumed necessary for this project — it's
  self-contained OpenSCAD, no external libraries.

## What I'd like help with

Open-ended — this is a living design. Typical asks going forward:
- Parameter tuning (spacing ranges, hole sizes, tower height) — edit
  `config.scad`, keep the README's numbers in sync.
- New parts that follow the existing conventions (hidden bolts from
  below, grid-pitch-aligned mounting where applicable).
- Sanity-checking geometry (e.g. "does this hole collide with that one")
  — this project has a history of exactly that kind of bug, so it's worth
  actually computing coordinates rather than eyeballing them.
- Rendering/exporting STLs once a design is stable.

When a request conflicts with a physical constraint (bolt too big for a
measured hole, part too big for the stated print bed, etc.), say so
explicitly and propose the closest correct alternative — don't silently
"fix" it without flagging, and don't silently comply with something that
won't physically work either.
