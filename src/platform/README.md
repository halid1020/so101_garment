# SO-101 dual-arm board rig

Parametric OpenSCAD files for mounting two SO-101 arms + an overhead camera
tower on a shared, fully 3D-printed perforated board. Free software to open
these: https://openscad.org — open any `.scad` file, it renders in 3D, and
File > Export > STL when you're ready to slice. For batch export use
`python -m sim_twin.assets --print-parts` (STLs land in `build/print/`),
and `python tool/part_drawings.py` renders dimensioned multi-view drawing
sheets for every part into `outputs/drawings/` (PNGs + one combined PDF).

## How the fastening works

Every board-mount bolt drops in from the **top**: head counterbored flush
into the printed part, shaft through the part and the board, threading
into an M5 hex nut held **captive in a hex pocket molded into the board's
bottom face** (8mm deep — push a nut in from below wherever a bolt lands;
it's a snug slip fit). The board's bottom face stays flush and flat —
pockets and splice recesses are blind, nothing protrudes — so the board
clamps directly to a table.

## The board (printed, in tiles)

The board is now 3D-printed: **3 tiles + 2 splice bars**, replacing both
the old wooden board and the separate under-board nut plate
(`drill_template.scad`, kept only as legacy reference).

- **570 x 150 x 18mm** overall — 25mm grid, 17 columns x 5 rows, every
  hole with a bottom-face M5 nut pocket.
- **Clamp wings**: 60mm of solid, hole-free board at **each end**, beyond
  the grid — clamp the board to your table from both sides there.
- **Tiles**: seams fall at grid-column midpoints; each tile is ~197 /
  ~175 / ~197mm wide, all under a 220mm print bed (`printer_bed` in
  `config.scad` — the file asserts if a tile outgrows it).
- **Splice bars** (100 x 150 x 8mm) tie the seams together: each bar sits
  in an 8mm recess in the underside spanning 2 grid columns per side, with
  hex pockets opening toward the board. Drop nuts in the bar, hold it in
  the recess, and bolt M5x12 down through **at least 3 free grid holes per
  side, staggered**. Rig bolts that land inside a bar zone simply use the
  bar's pockets instead of the board's — the pocket heights coincide, so
  bolt lengths don't change.
- **Print orientation**: tiles TOP FACE DOWN (pockets and recesses print
  as upward blind holes, no supports). Bars pocket-side up.

## Files

| File | Prints | What it is |
|---|---|---|
| `config.scad` | — | Shared parameters. Edit this first; everything else reads from it. The **sim digital twin** (`sim_twin/`) also reads this file — change a number here, rebuild the twin, see it in simulation. |
| `board.scad` | 3 tiles + 2 splice bars | The printed perforated board described above. |
| `arm_mount_adapter.scad` | **x2** | Top matches your SO-101 base's real screw holes (M3 from below). Bolts to the board with 4x M5x25 from the top, in a 100x100mm square fully outside the base's footprint. |
| `camera_tower.scad` | x1 base + 3 mast segments + 1 platform | Tapered triangular tower (110mm side at the board → 45mm at the top), one corner facing the front (+X, `tower_yaw_deg`); bolts down with 4x M5x20; optional core spine rod. |
| `wrist_camera_mount.scad` | **x2** | Logitech C310 holder (housing intact, monitor clip removed) on each arm's wrist — see "Camera holders" below for how it holds the camera and installs. |
| `tower_camera_cradle.scad` | x1 | C310 cradle for the tower's top platform (the C310 has no tripod thread) — see "Camera holders" below. |
| `cam_tray_lib.scad` | — | Shared tray/pedestal modules for the two camera holders. |
| `cam_body.scad` | **never** | Simplified C310 visual mesh for the simulation twin only. |
| `export.scad` | — | Headless export dispatcher: `openscad -o out.stl -D 'part="adapter"' export.scad`. Used by `sim_twin/assets.py`. |
| `drill_template.scad` | legacy — don't print | Drill guide / under-board nut plate for the original **wooden**-board rig. Superseded by `board.scad`; kept for reference. |

## Shopping list (standard catalogue hardware)

All lengths assume the 18mm printed board (`board_thickness`) and its 8mm
nut pockets. If you change either, re-verify the stack-ups below.

| Item | Qty | Where | Stack-up check |
|---|---|---|---|
| M5x25 socket head cap screw (DIN 912) | 8 | adapters → board (4 per arm) | under-head grip = 16−5.5 (adapter below head) + 10 (board above the pocket) = 20.5mm → 4.5mm into the 4.7mm nut; tip stays inside the pocket, 3.5mm above the board's bottom face. ✓ |
| M5x20 socket head cap screw (DIN 912) | 4 | tower base → board | grip = 12−5.5 + 10 = 16.5mm → 3.5mm into the nut, tip 4.5mm above the bottom face. ✓ (M5x25 here would still stay inside the pocket, but M5x20 is the intended fit.) |
| M5x12 socket/button head screw | 12 | splice bars → board (6 per bar, ≥3 per side, staggered) | head sits on the board top; grip = 10mm (board above the recess) → **2mm into the nut (~2.5 threads)**. Shallow by design — the joint is loaded in shear across 6 bolts per bar. If you can get M5x14 (common in ISO 7380 button head), use it: 4mm engagement, tip still 0.8mm clear of the pocket floor. **Do not use M5x16 — it bottoms out.** |
| M5 hex nut (ISO 4032, 4.7mm thick) | 26 | 12 board pockets (adapters + tower) + 12 splice-bar pockets + 2 spine rod | pockets are 9.4mm across corners — snug slip fit, nuts stay put |
| M5x10 pan/button head screw | 8 | tower joints, 2 per joint, thread-forming into the 4.5mm pilot | **max length 10mm** — at the narrow top joint the face-to-spine-rod distance is 10.25mm, so a 12mm screw would hit the rod channel |
| M3x16 socket/pan head screw | 8 | SO-101 base → adapter (4 per arm), from below | 12.8mm through the adapter, ~3.2mm into the base. **Confirm how your base receives these** (thread into plastic vs. captive nut) — go M3x20 if it needs more engagement. |
| M5 threaded rod, ~405mm (cut from 500mm stock) | 1 | optional core spine, with 2 of the nuts + 2 washers | runs from the base spigot pocket to the top segment's spigot pocket |
| 1/4"-20 heat-set threaded insert | 1 | camera platform tripod hole (8.0mm bore, 12mm deep plate) | check your insert's OD against `tripod_insert_hole` |
| M3x8 pan/socket head screw | 4 | wrist camera mounts → wrist-roll element (2 per arm) | plate 4mm with 3.2 counterbore → 7.2mm past the plate: 3.2mm wall + full nut engagement. ✓ (matches the official mount's spec) |
| M3 hex nut | 4 | slid into the wrist-roll element's recesses (2 per arm) | remove motor 6, insert nuts, reattach — same as the official SO-ARM100 wrist mount guide |
| 1/4"-20 x 1/2" socket head bolt | 1 | tower cradle → platform insert | grip 3.5mm + 9.2mm into the 12mm insert. ✓ (5/8" risks bottoming out) |
| Zip ties, ~3.5mm wide | 6 | 2 per camera tray (wrists + tower) | thread through the floor slots **before** seating the camera |

## Assembly order (matters!)

1. **Print** everything per the table above (tiles top-face-down!).
2. **Splice the board.** Lay the tiles top-face-down, butted at the
   seams. Drop an M5 nut into each splice-bar pocket you'll use, seat
   each bar in its underside recess (pockets toward the board), and
   drive 6x M5x12 per bar from the board's top side through free grid
   holes — 3 per side of the seam, staggered.
3. **Load the board nuts.** Push an M5 nut into the bottom-face pocket
   of every grid hole a rig bolt will use (8 adapter + 4 tower). Flip
   the board right-side-up onto the table.
4. **Base to adapter.** Bolt each SO-101 base onto an
   `arm_mount_adapter` using 4x M3 screws from the adapter's underside —
   do this before the adapter touches the board, since those screws
   become inaccessible afterward.
5. **Adapter to board.** Set the adapter+base assembly over a 100x100mm
   square of grid holes and drop 4x M5x25 in from the top — they thread
   into the captive nuts below. Move to different grid positions later
   to change arm spacing.
6. **Tower base.** Bolt `base_plate` down the same way with 4x M5x20,
   one triangle corner facing the front (the bolt square is symmetric,
   so index the yaw by eye against `tower_yaw_deg = -90`). Drop a
   washer+nut into its spigot-top pocket and thread in the spine rod.
7. **Mast stack.** Slide each `mast_segment` down over the rod — widest
   first, the taper only assembles one way — 2x M5x10 per joint.
8. **Tension the spine.** At the top segment's spigot pocket, add a
   washer and nut, and tighten — this puts the whole stack into
   compression.
9. **Platform.** Press the 1/4"-20 heat-set insert into
   `camera_platform`'s center hole, then seat it on top (normal
   spigot/socket fit — the rod stops below it).
10. **Clamp down.** Clamp the solid 60mm wings at both ends of the board
    to your table.

## Camera holders (Logitech C310)

Both holders share the same **open tray** (`cam_tray_lib.scad`): the
camera drops in with its long axis across the tray — both ends stay open
(the tray is 48mm long against the 71mm body, so the housing overhangs
each side and the USB cable exits freely). The **front wall is low**
(6mm, below the lens); the **back wall is tall** (12mm) and takes the
camera's weight when tilted. Two zip ties through the floor slots strap
the housing down — deliberately tolerant of small dimension errors.

**Wrist mount (x2)** — how it installs: it replaces the official
SO-ARM100 wrist camera mount and uses the same interface — a 4mm plate
against the wrist-roll element's flat face, 2x M3x8 screws (8.10mm
spacing, measured from the official STL) into hex nuts slid into the
element's recesses (remove motor 6, insert the nuts, reattach). A
pedestal rises 12mm off the plate and pitches the tray **30° toward the
fingertips**, centering the grasp point in frame. Order: drive the two
M3 screws through the vertical shafts **first**, thread the zip ties,
then seat and strap the camera.

**Tower cradle (x1)** — how it installs: a 34mm base block with one
central 1/4"-20 socket-head bolt that drives down into the platform's
heat-set insert — the driver reaches the head **through a shaft in the
tray floor** (3/16" allen key). The tray is fixed at **55° downward**
pitch; yaw the whole cradle on the bolt to aim at the workspace, then
tighten. Zip ties last, same as the wrist.

**Caliper-check the `cam_*` values in `config.scad` before printing**
(defaults are C310 datasheet numbers; the clip-less body depth
especially). The same values drive the simulation digital twin
(`sim_twin/` at the repo root): camera poses, tilt angles, and the
C310's 60° diagonal FOV in MuJoCo/Isaac all derive from this file —
`python -m sim_twin.verify` renders what each camera actually sees, and
the overview render draws the tower camera's view frustum.

## Numbers to double-check before printing

- `front_hole_spacing` (55.419mm), `back_hole_spacing` (63.251mm),
  `front_to_back_dist` (70mm), `base_hole_dia` (3.8mm) — from your caliper
  measurements.
- `cam_body_w/h/d`, `cam_lens_z_offset`, `cam_mass_g` — C310 datasheet
  defaults, CONFIRM with calipers.
- `base_screw_head_dia` / `base_screw_head_h` — assumed M3 pan/socket
  head. Confirm against your actual screws.
- **Check the SO-101 base's outer footprint** against the adapter's bolt
  counterbores: the bolt heads sit at the corners of a 100x100mm square
  (~71mm from the adapter center). If the base's body overhangs those
  spots you can't drop the bolts in — bump `adapter_hole_spacing` to
  `5*grid_pitch` (125mm) and enlarge `adapter_w`/`adapter_d` to ~145.
- `printer_bed` (220mm) — set to your real usable bed; `board.scad`
  asserts if a tile no longer fits.
- `min_arm_spacing` / `max_arm_spacing` — placeholder range (150-400mm).

Everything else (bolt clearances, pocket sizes, grid pitch) is a named
variable in `config.scad` — change a number, re-render, re-export.
