# Project: SO-101 dual-arm board rig

Parametric OpenSCAD design for a 3D-printable rig that mounts two SO-101
LeRobot arms plus an overhead camera tower on a shared board, for a
teleoperation / imitation-learning setup. This file is background for you
(Claude Code) — read it before making changes, and keep it in sync if the
design changes.

## What this actually is

- A **board** (NOT printed — plywood/MDF/acrylic) drilled with a uniform
  grid of plain through-holes, pegboard-style, so the arms and tower can
  be repositioned later instead of being fixed in one spot. A printed
  **nut plate** sits under the whole board holding captive M5 nuts.
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
- `camera_tower.scad` — print x1 base_plate, N mast_segments, x1
  camera_platform (all laid out side by side in one render). Includes
  `config.scad`.
- `drill_template.scad` — print x1, ONE full-board piece (450x150mm; the
  user explicitly rejected splitting it into tiles, accepting that it
  needs a large-format printer — don't reintroduce tiling unasked).
  Dual purpose: drill guide clamped to the board, then the permanent
  under-board nut plate — every hole has a snug hex pocket on its top
  (board-facing) face. Includes `config.scad`.
- `README.md` — human-facing summary: what to print how many times,
  assembly order, which numbers are placeholders. **Keep this updated
  whenever you change geometry or hardware sizes** — it's the file the
  user actually reads before printing.

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

## Placeholder / assumed values — flag these, don't silently trust them

- `base_screw_head_dia` / `base_screw_head_h` — assumed M3 pan/socket
  head, unconfirmed against the user's actual screws.
- `min_arm_spacing` / `max_arm_spacing` (150-400mm) — a guessed
  adjustability range, not a hard requirement.
- `board_thickness` (18mm) — user's stated plan, not yet locked in.
- `board_margin`, `grid_pitch` — design choices (25mm each), reasonable
  defaults, freely adjustable.

## Design decisions already made — don't relitigate without being asked

1. **Bolts from the top, nuts captive in a printed nut plate under the
   board.** Every board-to-part M5 bolt drops in from above (head
   counterbored flush into the printed part's top face), passes through
   the board, and threads into a hex nut sitting in a snug hex pocket
   in the nut plate (`drill_template.scad` — it is the drill guide AND
   the permanent under-board nut plate; pockets open toward the board,
   plate bottom is solid). The hard requirement behind this: the
   board+plate sandwich gets clamped flat to a table, so nothing may
   protrude below it. The board itself needs only plain through-holes —
   no recesses. Bolt lengths are stack-up-verified against standard
   catalogue sizes in the README (M5x35 adapters, M5x30 tower —
   **M5x35 on the tower would poke out the plate bottom**); re-verify
   if any part thickness, `board_thickness`, or `nutplate_thick`
   changes. (This scheme replaced bolts-from-below, then
   recesses-drilled-into-the-board, both at the user's request.)
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
