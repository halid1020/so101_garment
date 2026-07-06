include <config.scad>
use <cam_tray_lib.scad>

// ============================================================
// TOWER CAMERA CRADLE — Logitech C310 (clip removed) on the camera
// tower's top platform. Print x1. The C310 has no tripod thread,
// so this cradle bolts into the platform's 1/4"-20 heat-set insert
// with a single 1/4"-20 socket-head bolt and pitches the camera
// tower_cam_tilt_deg down at the workspace.
//
// PART FRAME: origin at the bolt axis on the platform surface;
// +Y is the camera's look direction (aim it at the workspace —
// along the arms' forward axis — before tightening), +Z up.
//
// BOLT ACCESS: the head is counterbored inside the base block and
// reached with a 3/16" allen key through the shaft that continues
// up through the pedestal and tray floor. One bolt only — tighten
// firmly; the sim assumes the tray points along the arms' +X.
// ============================================================

pedestal_w = 24;

module tower_camera_cradle() {
    difference() {
        union() {
            translate([-tower_cradle_base/2, -tower_cradle_base/2, 0])
                cube([tower_cradle_base, tower_cradle_base,
                      tower_cradle_thick]);
            cam_tray_pedestal(tower_cradle_thick, 0, tower_cam_rise,
                               tower_cam_tilt_deg, pedestal_w);
            cam_tray_place(0, tower_cradle_thick + tower_cam_rise,
                            tower_cam_tilt_deg)
                cam_tray();
        }
        // 1/4"-20: clearance through the base, head counterbored,
        // allen-key shaft continuing up through pedestal and tray
        translate([0, 0, -1])
            cylinder(h=tower_cradle_thick + 2, d=quarter20_clear_dia);
        translate([0, 0, tower_cradle_thick - quarter20_head_h])
            cylinder(h=100, d=quarter20_head_dia + 0.5);
    }
}

tower_camera_cradle();
