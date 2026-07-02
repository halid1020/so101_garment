include <config.scad>

// ============================================================
// CAMERA TOWER (revision: tapered TRIANGULAR mast + core spine rod)
// Bolts to the board at the grid position centered between the
// two arms.
// Prints as: 1x base_plate, N x mast_segment, 1x camera_platform
//
// SHAPE: the mast is a hollow equilateral-triangle tube that
// tapers linearly from tower_base_width (side length, at the
// board) to tower_top_width (at the camera). Wide bottom + narrow
// top puts material where the bending moment is largest, and the
// triangular section resists twist.
//
// JOINT RULE (same at every level): each part has a solid
// triangular SPIGOT sticking up from its top — a continuation of
// the tube's inner cavity, with clearance — and the hollow tube
// interior of the part above is the SOCKET that drops over it.
// The triangle also locks rotation. Two M5 screws per joint,
// self-tapping through the outer wall into the spigot.
//
// STABILITY: an M5 threaded rod runs up the tower's core from
// base_plate through every mast_segment, tensioned with a
// nut+washer at the bottom (in base_plate's spigot-top pocket,
// before mounting) and another at the top (in the topmost
// segment's spigot-top pocket, before the camera_platform is
// added). This clamps every joint into compression. The
// camera_platform sits on top afterward, untouched by the rod,
// so the tripod hole stays clear.
//
// FASTENING TO BOARD: 4x M5x30 bolts dropped in from the TOP
// (heads flush in the base plate's top face), through the board,
// into hex nuts held captive in the printed nut plate under the
// board (see drill_template.scad).
// ============================================================

k_off        = 2*sqrt(3); // triangle side-length change per unit of inward offset
spigot_clear = 1.4;       // side-length clearance on spigots (~0.4mm per face)
spigot_h     = 20;

seg_count  = ceil(tower_height_total / tower_segment_max);
seg_height = tower_height_total / seg_count;

// outer triangle side length at height z above the base plate's top
function s_at(z) = tower_base_width
                 + (tower_top_width - tower_base_width) * z / tower_height_total;

// equilateral triangle, centroid at origin, one vertex pointing +Y
module tri(s) {
    R = s/sqrt(3); // circumradius
    polygon([[0, R], [-s/2, -R/2], [s/2, -R/2]]);
}

// tapered triangular prism, centroid on the Z axis
module tri_frustum(h, s_bot, s_top) {
    linear_extrude(height=h, scale=s_top/s_bot) tri(s_bot);
}

module spine_hex_pocket(depth) { cylinder(h=depth, d=spine_nut_dia_corners, $fn=6); }

module board_mount_hole(plate_thick) {
    translate([0,0,-1]) cylinder(h=plate_thick+2, d=bolt_dia);
    translate([0,0,plate_thick-bolt_head_h]) cylinder(h=bolt_head_h+1, d=bolt_head_pocket);
}

// two pilot holes per joint, each perpendicular to a flat face of
// the vertex-up triangle (face normals at 30 and 150 degrees)
module joint_screw_holes(z, s) {
    for (a = [30, 150])
        rotate([0,0,a])
            translate([0,0,z])
                rotate([0,90,0])
                    cylinder(h=s*2, d=joint_pilot_dia, center=true);
}

// solid spigot continuing the tube's inner cavity upward from
// absolute tower height z_abs, shrunk by spigot_clear for fit
module spigot(z_abs) {
    tri_frustum(spigot_h,
                s_at(z_abs)            - k_off*tower_wall - spigot_clear,
                s_at(z_abs + spigot_h) - k_off*tower_wall - spigot_clear);
}

module base_plate() {
    plate_thick = tower_base_thick;
    difference() {
        union() {
            translate([-tower_base_plate/2, -tower_base_plate/2, 0])
                cube([tower_base_plate, tower_base_plate, plate_thick]);
            translate([0,0,plate_thick]) spigot(0);
        }
        for (x = [-tower_hole_spacing/2, tower_hole_spacing/2])
            for (y = [-tower_hole_spacing/2, tower_hole_spacing/2])
                translate([x,y,0]) board_mount_hole(plate_thick);
        // spine rod clearance, nut pocket recessed into the spigot top
        translate([0,0,-1]) cylinder(h=plate_thick+spigot_h+2, d=spine_rod_dia);
        translate([0,0,plate_thick+spigot_h-spine_nut_h])
            spine_hex_pocket(spine_nut_h+1);
    }
}

// segment i spans tower heights [i*seg_height, (i+1)*seg_height]
module mast_segment(i) {
    z0 = i*seg_height;
    z1 = z0 + seg_height;
    is_top = (i == seg_count-1);
    cap_h = 4; // solid floor under the spigot, ties it to the walls
    difference() {
        union() {
            tri_frustum(seg_height, s_at(z0), s_at(z1));
            translate([0,0,seg_height]) spigot(z1);
        }
        // hollow interior — its bottom opening is the socket that
        // drops over the spigot of the part below; stops cap_h
        // short of the top so the spigot sits on a solid floor
        translate([0,0,-1])
            tri_frustum(seg_height+1-cap_h,
                        s_at(z0)        - k_off*tower_wall,
                        s_at(z1-cap_h)  - k_off*tower_wall);
        joint_screw_holes(12, s_at(z0));
        translate([0,0,-1]) cylinder(h=seg_height+spigot_h+2, d=spine_rod_dia);
        if (is_top)
            translate([0,0,seg_height+spigot_h-spine_nut_h])
                spine_hex_pocket(spine_nut_h+1);
    }
}

module camera_platform() {
    plate_thick = 12;
    zt = tower_height_total; // sits over the topmost segment's spigot
    difference() {
        union() {
            // triangular socket skirt, continues the mast's taper
            tri_frustum(spigot_h, s_at(zt), s_at(zt+spigot_h));
            translate([-camera_plate/2, -camera_plate/2, spigot_h])
                cube([camera_plate, camera_plate, plate_thick]);
        }
        translate([0,0,-1])
            tri_frustum(spigot_h+1,
                        s_at(zt)          - k_off*tower_wall,
                        s_at(zt+spigot_h) - k_off*tower_wall);
        translate([0,0,spigot_h-1])
            cylinder(h=plate_thick+2, d=tripod_insert_hole);
        joint_screw_holes(spigot_h/2, s_at(zt));
    }
}

// ---- layout for printing (side by side) ----
gap = tower_base_width * 1.3;
base_plate();
for (i = [0:seg_count-1])
    translate([gap*(i+1), 0, 0]) mast_segment(i);
translate([gap*(seg_count+1), 0, 0]) camera_platform();
