// ============================================================
// SHARED CONFIG — SO-101 dual-arm board rig
// (revision: fully 3D-printed tiled board with integrated nut
//  pockets + bolted splice bars, tapered triangular camera
//  tower, Logitech C310 camera holders)
// Edit values here; all other files include this automatically.
// The sim digital twin (sim_twin/) also reads this file.
// ============================================================

// ---- Board (3D-PRINTED, in tiles — see board.scad) ----
// The board prints as 3 tiles joined by splice bars bolted into
// recesses in the underside (bottom face stays flush and flat).
// Every grid hole has an M5 hex pocket molded into the bottom
// face, replacing the old separate nut plate. Solid clamp wings
// at both ends take the table clamps.
board_thickness     = 18;    // mm
board_margin        = 25;    // mm, edge margin outside the outermost grid holes
board_clamp_wing    = 60;    // mm, solid clamp area at EACH end
printer_bed         = 220;   // mm, usable bed — tiles stay under this

// ---- Adjustable mounting grid ----
grid_pitch          = 25;    // mm, spacing between adjacent grid holes
min_arm_spacing     = 150;   // mm  <-- PLACEHOLDER range, adjust freely
max_arm_spacing     = 400;   // mm  <-- PLACEHOLDER range, adjust freely

board_width  = max_arm_spacing + 2*board_margin + 2*board_clamp_wing; // 570
board_depth  = 150;

// ---- Board-mount bolts — M5, inserted from the TOP ----
// Each bolt drops down through the printed part and the board into
// an M5 hex nut held captive in a pocket molded into the board's
// BOTTOM face (or into a splice bar's top pocket in the seam
// zones — the stack heights coincide, so bolt lengths are the
// same either way). Nothing protrudes below the board.
// Hardware: DIN 912 / ISO 4762 SHCS + ISO 4032 hex nuts — see the
// README for the stack-up-verified lengths (M5x25 adapters,
// M5x20 tower, M5x12 splice bars).
bolt_dia            = 5.5;   // clearance hole through parts + board
bolt_head_dia       = 8.6;   // M5 SHCS head, actual 8.5 (reference)
bolt_head_pocket    = 9.6;   // top-face counterbore dia (head + slop)
bolt_head_h         = 5.5;   // counterbore depth (M5 SHCS head is 5.0 tall)
nut_dia_corners     = 9.4;   // hex pocket size; M5 nut is 9.24 across
                              // corners — snug slip fit, nuts stay put
nut_h               = 4.7;   // ISO 4032 M5 nut thickness (reference)

// ---- Board tiles + splice bars (board.scad) ----
board_nut_pocket    = 8;     // mm, hex pocket depth in the board bottom;
                              // sized so a standard M5x25 lands fully in
                              // the nut through a 16mm part + the board
splice_bar_w        = 100;   // mm, splice bar span across each seam
                              // (covers 2 grid columns per side)
splice_bar_thick    = 8;     // mm, bar thickness = recess depth, so the
                              // board bottom stays flush for clamping
splice_nut_pocket   = 4.8;   // mm, hex pockets in the bar TOP face
                              // (nuts drop in before the bar is bolted)
seam_clearance      = 0.3;   // mm, butt-joint gap per seam

// ---- LEGACY: separate under-board nut plate (drill_template.scad)
//      for the original non-printed wooden board. Not part of the
//      printed-board build; kept for reference. ----
nutplate_thick      = 8;
nut_pocket_depth    = 4.8;

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
tower_base_thick    = 12;    // mm — sized so a standard M5x20 SHCS lands
                              // correctly in the board's nut pocket
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

// Tower structural constants (shared with the sim twin's stack-height
// math — that's why they live here and not in camera_tower.scad)
tower_spigot_h        = 20;   // mm, spigot/socket engagement per joint
camera_platform_thick = 12;   // mm, top platform plate thickness

// ============================================================
// RIG PLACEMENT ACTUALLY IN USE  (the sim digital twin reads
// everything below via sim_twin/scad_params.py — edit a value,
// re-run the twin, see the change)
// ============================================================
arm_spacing         = 300;   // mm, center-to-center between the two arm
                              // adapters along the board. MUST be a
                              // multiple of 2*grid_pitch so both adapters'
                              // bolt squares land on grid holes.
tower_y_offset      = 0;     // mm, tower center offset from the midpoint
                              // between the arms (grid-multiple, usually 0)
tower_yaw_deg       = -90;   // deg, tower rotation on its bolt square:
                              // -90 points one triangle corner at the
                              // FRONT (+X, the arms' working direction)

// ---- Table the rig sits on (NOT printed; sim twin only) ----
table_size_x        = 1000;  // mm  <-- CONFIRM against your real table
table_size_y        = 900;   // mm  <-- CONFIRM
table_thick         = 40;    // mm, slab visual thickness
table_height        = 750;   // mm, table top above the floor <-- CONFIRM
table_x_offset      = 150;   // mm, table center ahead of the arms (+X,
                              // more workspace in front) <-- CONFIRM

// ---- Camera: Logitech C310 HD webcam (housing intact, monitor
//      clip removed). Defaults from the datasheet/retail specs —
//      CONFIRM the body numbers with calipers before printing. ----
cam_body_w          = 71.1;  // mm, width (long axis)      <-- CONFIRM
cam_body_h          = 31.2;  // mm, height (front face)    <-- CONFIRM
cam_body_d          = 24.0;  // mm, depth without the clip <-- CONFIRM
cam_lens_x_offset   = 0;     // mm, lens center offset from body center
cam_lens_z_offset   = 0;     // mm, lens vertical offset   <-- CONFIRM
cam_lens_ring_dia   = 16;    // mm, decorative ring on the sim body mesh
cam_dfov_deg        = 60;    // deg, C310 diagonal field of view (datasheet)
cam_mass_g          = 71;    // g, with clip removed + some cable <-- CONFIRM
cam_fit_clear       = 0.6;   // mm, cradle clearance around the housing

// ---- Wrist camera mount, one per arm (interface = the official
//      SO-101 wrist mount: 2x M3 into hex-nut recesses on the
//      wrist-roll element; measured by tool/analyze_wrist_mount.py
//      from the official STL — do not change the measured values) ----
wrist_screw_spacing = 8.10;  // mm, center-to-center — MEASURED
wrist_screw_clear   = 3.4;   // mm, M3 clearance holes in the plate
wrist_screw_head_dia = 6.0;  // mm, M3 pan/socket head
wrist_screw_head_h  = 3.2;   // mm, counterbore depth (head sits flush)
wrist_plate_w       = 24;    // mm, interface plate width (across screws)
wrist_plate_l       = 34;    // mm, interface plate length
wrist_plate_thick   = 4;     // mm (matches the official mount's plate)
wrist_cam_tilt_deg  = 30;    // deg, camera pitch toward the fingertips;
                              // 30 centers the grasp point in frame
wrist_cam_rise      = 12;    // mm, tray floor height above the plate
wrist_cam_fwd       = 4;     // mm, tray center offset toward fingertips

// ---- Shared camera tray (used by wrist mount and tower cradle) ----
tray_len            = 48;    // mm, tray length along the camera width
                              // (shorter than the body — ends stay open)
tray_wall           = 2.5;   // mm, front/back wall thickness
tray_floor          = 3;     // mm, floor under the camera
tray_lip_front      = 6;     // mm, front (lens-side) wall height — must
                              // stay below the lens bottom edge
tray_lip_back       = 12;    // mm, back wall height
zip_slot_w          = 5;     // mm, zip-tie slot width
zip_slot_t          = 3;     // mm, zip-tie slot thickness
zip_slot_x          = 16;    // mm, slot pairs at +/- this from center

// ---- Tower camera cradle (bolts into the platform's 1/4"-20
//      insert; holds the third C310 looking down at the workspace) ----
tower_cam_tilt_deg  = 55;    // deg, downward pitch from horizontal
tower_cam_rise      = 4;     // mm, tray pivot height above the base block
tower_cradle_base   = 34;    // mm, square base block side (platform is 60)
tower_cradle_thick  = 10;    // mm, base block thickness
quarter20_clear_dia = 6.8;   // mm, 1/4"-20 bolt clearance
quarter20_head_dia  = 10.5;  // mm, 1/4" socket head (9.5) + slop
quarter20_head_h    = 6.5;   // mm, counterbore depth

// ---- Measured SO-101 mesh anchors (sim twin placement only;
//      from tool/analyze_wrist_mount.py + base-mesh analysis of
//      src/so101_dual_description — DO NOT CHANGE) ----
wrist_iface_x       = -0.95; // mm, screw midpoint in gripper_link frame
wrist_iface_y       = 24.0;  // mm, mounting face plane (outward normal +Y)
wrist_iface_z       = -23.4; // mm, screw midpoint height
base_holes_x        = 21.2;  // mm, base-hole trapezoid center ahead of
                              // base_link origin (+X, arm-forward)
base_bottom_z       = -2.4;  // mm, base underside below base_link origin

$fn = 64;
