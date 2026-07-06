include <config.scad>

// ============================================================
// SHARED C310 TRAY — library only, nothing to print from here.
// Used (via `use <>`) by wrist_camera_mount.scad and
// tower_camera_cradle.scad.
//
// Tray coordinate convention: origin at the center of the tray
// FLOOR TOP surface; the camera sits on it with its long axis
// along X, optical axis along +Y (lens side = +Y wall, which is
// kept low so it stays below the lens), body height along +Z.
// The tray is shorter than the camera body (tray_len < body
// width) so both ends stay open; two pairs of zip-tie slots
// through the floor strap the housing down — tolerant of the
// CONFIRM-flagged body dimensions.
// ============================================================

// inner tray depth (Y) around the camera body
function tray_inner_d() = cam_body_d + 2*cam_fit_clear;
// outer tray depth (Y) including both walls
function tray_outer_d() = tray_inner_d() + 2*tray_wall;

module cam_tray() {
    iw = tray_inner_d();
    difference() {
        union() {
            // floor
            translate([-tray_len/2, -iw/2 - tray_wall, -tray_floor])
                cube([tray_len, tray_outer_d(), tray_floor]);
            // front wall (+Y, lens side) — low, below the lens
            translate([-tray_len/2, iw/2, 0])
                cube([tray_len, tray_wall, tray_lip_front]);
            // back wall (-Y) — tall, takes the tilt load
            translate([-tray_len/2, -iw/2 - tray_wall, 0])
                cube([tray_len, tray_wall, tray_lip_back]);
        }
        // zip-tie slots: two per tie, just inside each wall, at
        // +/- zip_slot_x. Thread the ties before dropping the
        // camera in, then tighten over the housing.
        for (sx = [-1, 1]) for (sy = [-1, 1])
            translate([sx*zip_slot_x - zip_slot_w/2,
                       sy*(iw/2 - 1) - (sy > 0 ? zip_slot_t : 0),
                       -tray_floor - 1])
                cube([zip_slot_w, zip_slot_t, tray_floor + 2]);
    }
}

// Sloped pedestal connecting a flat base (top face at z=base_top,
// footprint pedestal_w wide in X, tray_outer_d deep in Y centered
// on y=fwd) to the underside of the tilted tray. The hull of the
// two slabs gives a solid printable wedge.
//
// tilt > 0 pitches the optical axis (+Y) downward (toward -Z).
module cam_tray_pedestal(base_top, fwd, rise, tilt, pedestal_w) {
    hull() {
        translate([-pedestal_w/2, fwd - tray_outer_d()/2, base_top])
            cube([pedestal_w, tray_outer_d(), 0.1]);
        cam_tray_place(fwd, base_top + rise, tilt)
            translate([-pedestal_w/2, -tray_outer_d()/2, -tray_floor])
                cube([pedestal_w, tray_outer_d(), 0.1]);
    }
}

// Places children from tray coordinates into part coordinates:
// tilt about X (optical +Y dips toward -Z), then move the floor-top
// origin to (0, fwd, z_floor).
module cam_tray_place(fwd, z_floor, tilt) {
    translate([0, fwd, z_floor]) rotate([-tilt, 0, 0]) children();
}
