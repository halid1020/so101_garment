include <config.scad>
use <cam_tray_lib.scad>

// ============================================================
// WRIST CAMERA MOUNT — Logitech C310 (clip removed) on the SO-101
// wrist. Print this file TWICE — one per arm.
//
// INTERFACE (same as the official SO-ARM100 wrist mounts): two
// M3x8 screws through the plate into hex nuts that slide into the
// recesses on the wrist-roll element (remove motor 6, insert the
// nuts, reattach — see src/platform/README.md). Screw spacing is
// MEASURED from the official mount STL (8.10mm) — do not change.
//
// PART FRAME: origin at the screw-pair midpoint on the wrist face;
// X lateral (along the screw pair), +Y toward the fingertips,
// +Z away from the wrist face. The camera tray sits on a sloped
// pedestal, pitched wrist_cam_tilt_deg toward the fingertips so
// the grasp point is centered in frame.
//
// SCREW ACCESS: the pedestal covers the screw heads, so each
// counterbore continues upward as a driver shaft through the
// pedestal and tray floor — drive the screws through those, then
// drop the camera in and zip-tie it down.
// ============================================================

pedestal_w = 24;  // narrower than the tray so the zip slots stay clear

module wrist_camera_mount() {
    difference() {
        union() {
            // interface plate, centered on the screw midpoint
            translate([-wrist_plate_w/2, -wrist_plate_l/2, 0])
                cube([wrist_plate_w, wrist_plate_l, wrist_plate_thick]);
            cam_tray_pedestal(wrist_plate_thick, wrist_cam_fwd,
                               wrist_cam_rise, wrist_cam_tilt_deg,
                               pedestal_w);
            cam_tray_place(wrist_cam_fwd,
                            wrist_plate_thick + wrist_cam_rise,
                            wrist_cam_tilt_deg)
                cam_tray();
        }
        // M3 holes: clearance through the plate, head counterbored
        // 0.8mm above the wrist face, driver shaft continuing up
        // through pedestal and tray floor
        for (sx = [-1, 1])
            translate([sx*wrist_screw_spacing/2, 0, 0]) {
                translate([0, 0, -1])
                    cylinder(h=wrist_plate_thick + 2, d=wrist_screw_clear);
                translate([0, 0, wrist_plate_thick - wrist_screw_head_h])
                    cylinder(h=100, d=wrist_screw_head_dia + 0.5);
            }
    }
}

wrist_camera_mount();
