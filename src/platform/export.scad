include <config.scad>
use <arm_mount_adapter.scad>
use <board.scad>
use <drill_template.scad>
use <camera_tower.scad>
use <wrist_camera_mount.scad>
use <tower_camera_cradle.scad>
use <cam_body.scad>

// ============================================================
// EXPORT DISPATCHER — single entry point for headless STL/PNG
// export (used by sim_twin/assets.py and tool/part_drawings.py):
//
//   openscad -o out.stl -D 'part="adapter"' export.scad
//   openscad -o out.stl -D 'part="tower_mast_segment"' -D seg=1 export.scad
//   openscad -o out.stl -D 'part="board_tile"' -D seg=0 export.scad
//
// `use <>` imports the part modules without executing each file's
// top-level print layout, so exactly one part renders per run.
// "*_assembled" and "*_assembly" parts are for the sim twin and
// the assembly diagrams — never print those.
// ============================================================

part = "all";  // which part to render
seg  = 0;      // index for tower_mast_segment / board_tile

// same derivations as camera_tower.scad / board.scad
seg_count = ceil(tower_height_total / tower_segment_max);
seg_h     = tower_height_total / seg_count;

module tower_assembled() {
    base_plate();
    for (i = [0:seg_count-1])
        translate([0, 0, tower_base_thick + i*seg_h])
            mast_segment(i);
    translate([0, 0, tower_base_thick + tower_height_total])
        camera_platform();
}

// camera body seated in the wrist mount, plus the wrist-roll
// element it screws onto — assembly diagram only
module wrist_assembly(with_wrist=true) {
    color([0.25, 0.35, 0.55]) wrist_camera_mount();
    color([0.12, 0.12, 0.12])
        cam_tray_place(wrist_cam_fwd, wrist_plate_thick + wrist_cam_rise,
                       wrist_cam_tilt_deg)
            translate([cam_lens_x_offset,
                       cam_body_d/2,
                       cam_body_h/2 + cam_lens_z_offset])
                cam_body();
    if (with_wrist)
        color([0.85, 0.75, 0.2])
            rotate([90, 0, 0])
                translate([0.95, -24.218, 24.35])
                    rotate([180, 0, 0])
                        scale(1000)
                            import("../so101_dual_description/meshes/wrist_roll_follower_so101_v1.stl");
}

// cradle + camera on the tower's top platform — assembly diagram only
module cradle_assembly() {
    color([0.85, 0.75, 0.2])
        translate([0, 0, -tower_spigot_h - camera_platform_thick])
            camera_platform();
    color([0.25, 0.35, 0.55]) tower_camera_cradle();
    color([0.12, 0.12, 0.12])
        cam_tray_place(0, tower_cradle_thick + tower_cam_rise,
                       tower_cam_tilt_deg)
            translate([cam_lens_x_offset,
                       cam_body_d/2,
                       cam_body_h/2 + cam_lens_z_offset])
                cam_body();
}

use <cam_tray_lib.scad>  // cam_tray_place for the assembly views

if (part == "adapter")                   adapter_plate();
else if (part == "board_tile")           board_tile(seg);
else if (part == "splice_bar")           splice_bar();
else if (part == "board_assembled")      board_assembled();
else if (part == "nut_plate")            nut_plate();  // legacy (wood board)
else if (part == "tower_base_plate")     base_plate();
else if (part == "tower_mast_segment")   mast_segment(seg);
else if (part == "tower_camera_platform") camera_platform();
else if (part == "tower_assembled")      tower_assembled();
else if (part == "wrist_camera_mount")   wrist_camera_mount();
else if (part == "tower_camera_cradle")  tower_camera_cradle();
else if (part == "cam_body")             cam_body();
else if (part == "wrist_assembly")       wrist_assembly();
else if (part == "wrist_assembly_bare")  wrist_assembly(with_wrist=false);
else if (part == "cradle_assembly")      cradle_assembly();
else if (part == "all") {
    // side-by-side sanity layout for the GUI; not for export
    board_assembled();
    translate([0, 250, 0]) adapter_plate();
    translate([200, 250, 0]) tower_assembled();
    translate([350, 250, 0]) wrist_camera_mount();
    translate([450, 250, 0]) tower_camera_cradle();
    translate([550, 250, 0]) cam_body();
} else {
    assert(false, str("unknown part: ", part));
}
