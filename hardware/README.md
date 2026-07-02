# SO-101 dual-arm board rig

Parametric OpenSCAD files for mounting two SO-101 arms + an overhead camera
tower on a shared board. Free software to open these: https://openscad.org
Open any `.scad` file there, it renders in 3D, and File > Export > STL when
you're ready to slice.

## How the fastening works

Every board-mount bolt drops in from the **top**: head counterbored flush
into the printed part, shaft through the part and the board, threading
into an M5 hex nut held **captive in a printed nut plate that lives under
the board**. The nut pockets open toward the board; the plate's bottom
face is solid and flat, so the whole board+plate sandwich clamps to a
table. No recesses are drilled into the board — it only needs plain
5.5mm through-holes, and since you drill them *through the nut plate
itself* (it doubles as the drill guide), board holes and nut pockets
align by construction.

## Files

| File | Prints | What it is |
|---|---|---|
| `config.scad` | — | Shared parameters. Edit this first; everything else reads from it. |
| `arm_mount_adapter.scad` | **x2** | Top matches your SO-101 base's real screw holes (M3 from below). Bolts to the board with 4x M5x35 from the top, in a 100x100mm square fully outside the base's footprint. |
| `camera_tower.scad` | x1 base + 3 mast segments + 1 platform | Tapered triangular tower (110mm side at the board → 45mm at the top); bolts down with 4x M5x30; optional core spine rod. |
| `drill_template.scad` | x1 | Drill guide **and** permanent under-board nut plate: every grid hole has a snug hex pocket facing the board. One full-board piece, **450 x 150mm — needs a large-format printer** (exceeds common 220-256mm beds). |

## Shopping list (standard catalogue hardware)

All lengths assume the 18mm board (`board_thickness`). If your board is
different, bolt lengths change 1:1 with it.

| Item | Qty | Where | Stack-up check |
|---|---|---|---|
| M5x35 socket head cap screw (DIN 912) | 8 | adapters → board (4 per arm) | grip = 16−5.5 (adapter below head) + 18 (board) = 28.5mm; nut adds 4.7 → needs ≥33.2. Tip ends 1.8mm past the nut, 1.5mm above the plate's bottom face. ✓ |
| M5x30 socket head cap screw (DIN 912) | 4 | tower base → board | grip = 12−5.5 + 18 = 24.5mm; needs ≥29.2. Tip ends 0.8mm past the nut, 2.5mm above the plate bottom. ✓ (M5x35 would poke out the bottom — don't substitute.) |
| M5 hex nut (ISO 4032, 4.7mm thick) | 14 | 12 in the nut plate pockets + 2 for the spine rod | pocket is 4.8mm deep x 9.4mm across corners — snug slip fit |
| M5x10 pan/button head screw | 8 | tower joints, 2 per joint, thread-forming into the 4.5mm pilot | **max length 10mm** — at the narrow top joint the face-to-spine-rod distance is 10.25mm, so a 12mm screw would hit the rod channel |
| M3x16 socket/pan head screw | 8 | SO-101 base → adapter (4 per arm), from below | 12.8mm through the adapter, ~3.2mm into the base. **Confirm how your base receives these** (thread into plastic vs. captive nut) — go M3x20 if it needs more engagement. |
| M5 threaded rod, ~405mm (cut from 500mm stock) | 1 | optional core spine, with 2 of the nuts + 2 washers | runs from the base spigot pocket to the top segment's spigot pocket |
| 1/4"-20 heat-set threaded insert | 1 | camera platform tripod hole (8.0mm bore, 12mm deep plate) | check your insert's OD against `tripod_insert_hole` |

## The board itself

Not printed — cut from plywood/MDF/acrylic, 18mm. Current size: **450 x
150mm**, computed from `max_arm_spacing` (400mm) + edge margin. It only
needs the plain 5.5mm grid holes, drilled through the clamped-on nut
plate tiles. No counterbores, no recesses.

## Assembly order (matters!)

1. **Drill the board.** Clamp the nut plate to the board and drill every
   grid hole with a 5.5mm bit through the plate's bores.
2. **Load the nut plate.** Lay it pocket-side-up on the table, drop an
   M5 nut into each pocket you'll use (they press in snugly), and set
   the board on top, holes aligned.
3. **Base to adapter.** Bolt each SO-101 base onto an
   `arm_mount_adapter` using 4x M3 screws from the adapter's underside —
   do this before the adapter touches the board, since those screws
   become inaccessible afterward.
4. **Adapter to board.** Set the adapter+base assembly over a 100x100mm
   square of grid holes and drop 4x M5x35 in from the top — they thread
   into the captive nuts below. Move to different grid positions later
   to change arm spacing.
5. **Tower base.** Bolt `base_plate` down the same way with 4x M5x30.
   Drop a washer+nut into its spigot-top pocket and thread in the spine
   rod.
6. **Mast stack.** Slide each `mast_segment` down over the rod — widest
   first, the taper only assembles one way — 2x M5x10 per joint.
7. **Tension the spine.** At the top segment's spigot pocket, add a
   washer and nut, and tighten — this puts the whole stack into
   compression.
8. **Platform.** Press the 1/4"-20 heat-set insert into
   `camera_platform`'s center hole, then seat it on top (normal
   spigot/socket fit — the rod stops below it).

## Numbers to double-check before printing

- `front_hole_spacing` (55.419mm), `back_hole_spacing` (63.251mm),
  `front_to_back_dist` (70mm), `base_hole_dia` (3.8mm) — from your caliper
  measurements.
- `base_screw_head_dia` / `base_screw_head_h` — assumed M3 pan/socket
  head. Confirm against your actual screws.
- **Check the SO-101 base's outer footprint** against the adapter's bolt
  counterbores: the bolt heads sit at the corners of a 100x100mm square
  (~71mm from the adapter center). If the base's body overhangs those
  spots you can't drop the bolts in — bump `adapter_hole_spacing` to
  `5*grid_pitch` (125mm) and enlarge `adapter_w`/`adapter_d` to ~145.
- `board_thickness` — set to your real board, and re-check the bolt
  lengths in the shopping list if it isn't 18mm.
- `min_arm_spacing` / `max_arm_spacing` — placeholder range (150-400mm).

Everything else (bolt clearances, pocket sizes, grid pitch) is a named
variable in `config.scad` — change a number, re-render, re-export.
