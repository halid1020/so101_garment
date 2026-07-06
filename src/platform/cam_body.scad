include <config.scad>

// ============================================================
// LOGITECH C310 BODY — NOT FOR PRINTING. Simplified visual mesh
// for the sim digital twin, so the cameras in MuJoCo / Isaac look
// like the real sensor. Pill-shaped housing + lens ring.
//
// PART FRAME: origin at the OPTICAL CENTER (lens), optical axis
// +Y, +Z up — the sim's camera sensors are colocated with this
// origin, so pose math stays trivial.
// ============================================================

module cam_body() {
    r = cam_body_h/2;
    // body center relative to the lens
    cx = -cam_lens_x_offset;
    cz = -cam_lens_z_offset;
    union() {
        // pill housing: hull of two depth-axis cylinders
        translate([cx, -cam_body_d, cz])
            hull()
                for (sx = [-1, 1])
                    translate([sx*(cam_body_w/2 - r), 0, 0])
                        rotate([-90, 0, 0])
                            cylinder(h=cam_body_d, r=r);
        // lens ring
        rotate([-90, 0, 0]) cylinder(h=1.6, d=cam_lens_ring_dia);
    }
}

cam_body();
