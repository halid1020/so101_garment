// ============================================================
// SHARED CONFIG — SO-101 dual-arm board rig
// (revision: top-insert bolts + board nut recesses + tapered
//  triangular camera tower)
// Edit values here; all other files include this automatically.
// ============================================================

// ---- Board (NOT printed — plywood/MDF/acrylic, 18mm) ----
board_thickness     = 18;    // mm  <-- SET THIS to your real board thickness
board_margin        = 25;    // mm, edge margin outside the outermost grid holes

// ---- Adjustable mounting grid ----
grid_pitch          = 25;    // mm, spacing between adjacent grid holes
min_arm_spacing     = 150;   // mm  <-- PLACEHOLDER range, adjust freely
max_arm_spacing     = 400;   // mm  <-- PLACEHOLDER range, adjust freely

board_width  = max_arm_spacing + 2*board_margin;
board_depth  = 150;

// ---- Board-mount bolts — M5, inserted from the TOP ----
// Each bolt drops down through the printed part and the board,
// into an M5 hex nut held captive in the printed NUT PLATE that
// lives under the board (see drill_template.scad — it is both the
// drill guide and that nut plate). The nut pockets open toward the
// board; the plate's bottom face is solid and flat, so the whole
// sandwich still clamps to a table. The bolt head is counterbored
// flush into the printed part's top face.
// Hardware: standard DIN 912 / ISO 4762 socket head cap screws
// and ISO 4032 hex nuts — see README for exact lengths.
bolt_dia            = 5.5;   // clearance hole through part + board + plate
bolt_head_dia       = 8.6;   // M5 SHCS head, actual 8.5 (reference)
bolt_head_pocket    = 9.6;   // top-face counterbore dia (head + slop)
bolt_head_h         = 5.5;   // counterbore depth (M5 SHCS head is 5.0 tall)
nut_dia_corners     = 9.4;   // hex pocket size; M5 nut is 9.24 across
                              // corners, so this is a snug slip fit that
                              // keeps nuts from falling out of the plate
nut_h               = 4.7;   // ISO 4032 M5 nut thickness (reference)
nut_pocket_depth    = 4.8;   // pocket depth in the nut plate — a hair
                              // deeper than the nut, so tightening pulls
                              // the nut flush against the board

// ---- Nut plate (printed, sits under the whole board) ----
nutplate_thick      = 8;     // nut pocket (4.8) + solid floor below; the
                              // bolt tip may run up to ~3mm past the nut
                              // before hitting the floor — bolt lengths
                              // in the README respect this
                              // NOTE: the plate prints as ONE full-board
                              // piece (450x150mm) at the user's request —
                              // requires a large-format print bed

// ---- SO-101 base's real mounting holes (measured by you) ----
// NOTE: the requested "M5 screws" for these holes is not physically
// possible — an M5 shaft (5mm) cannot pass through a 3.8mm hole.
// Kept at M3 here to match the measured 3.8mm hole; M5 is used
// everywhere else in the design instead.
base_hole_dia       = 3.8;   // mm, as measured
base_screw_clear    = 4.2;   // mm, adapter clearance hole (M3 shaft + play)
base_screw_head_dia = 6.5;   // mm, assumed M3 pan/socket head — CONFIRM
base_screw_head_h   = 3.2;   // mm, counterbore depth for that head
front_hole_spacing  = 55.419;// mm, measured, center-to-center
back_hole_spacing   = 63.251;// mm, measured, center-to-center
front_to_back_dist  = 70;    // mm, measured

// ---- Arm adapter plate ----
// The 4 board bolts form a 100x100mm square (4x4 grid pitches, so
// still grid-aligned) that sits fully OUTSIDE the SO-101 base's own
// trapezoid hole pattern — nearest board bolt is ~24mm from the
// nearest base hole, and the bolt heads stay reachable from above.
adapter_hole_spacing = 4*grid_pitch;  // 100mm — keep a grid multiple
adapter_w = 120;    // mm, along the arm-spacing axis
adapter_d = 120;    // mm, front-back — trapezoid + bolt square + margin
adapter_thick = 16; // mm, rigidity under motor torque

// ---- Camera tower — tapered TRIANGULAR mast + core spine rod ----
// Equilateral-triangle cross-section, wide at the board and
// narrowing toward the camera. The triangle resists twist, and the
// taper puts material where the bending moment is largest (at the
// bottom), so it reads and behaves like a stable tower.
tower_height_total  = 400;   // mm, overall height above the base plate
tower_segment_max   = 190;   // mm, keep each printed segment under this
tower_base_width    = 110;   // mm, triangle SIDE length at the bottom
tower_top_width     = 45;    // mm, triangle SIDE length at the top
tower_wall          = 4.0;   // mm, wall thickness of the hollow mast
tower_base_plate    = 140;   // mm, square foot bolted to the board
tower_base_thick    = 12;    // mm — sized so a standard M5x30 SHCS lands
                              // correctly in the nut plate (see README)
tower_hole_spacing  = 4*grid_pitch; // 100mm bolt square, grid-aligned,
                              // clears the 110mm triangle footprint
joint_pilot_dia     = 4.5;   // mm, pilot for M5 screws thread-forming
                              // into the printed spigots (2 per joint);
                              // keep joint screws at M5x10 max — longer
                              // ones reach the spine rod channel at the
                              // narrow top joint
camera_plate        = 60;    // mm, square top platform
tripod_insert_hole  = 8.0;   // mm, for a 1/4"-20 heat-set threaded insert

spine_rod_dia        = 5.5;  // mm, clearance for an M5 threaded rod run
                              // through the tower's hollow core, base to
                              // top segment, tensioned with nuts at both
                              // ends. Compresses every joint together —
                              // far stiffer against shake than the joint
                              // screws alone. Strongly recommended; skip
                              // it if you don't want a 400mm+ rod.
spine_nut_dia_corners = 9.4;  // mm, M5 nut pocket for the spine rod ends
spine_nut_h           = 4.4;

$fn = 64;
