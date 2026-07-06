include <config.scad>

// ============================================================
// ARM MOUNT ADAPTER (revision: top-insert bolts, bolt square
// OUTSIDE the base's own hole pattern)
// Print this file TWICE — one per arm.
//
// FASTENING:
//  - Base-to-adapter: 4x M3 screws from BELOW the adapter (heads
//    recessed flush in the bottom counterbores, shafts pass up
//    into the base's own holes).
//  - Adapter-to-board: 4x M5x35 bolts dropped in from the TOP
//    (heads recessed flush in the adapter's top face), through the
//    board, into hex nuts held captive in the printed nut plate
//    under the board (see drill_template.scad). Nothing protrudes
//    below the plate.
//  The M5 square (100x100mm = 4x4 grid pitches) sits fully outside
//  the base's trapezoid, so the bolt heads stay reachable from
//  above even with the base attached.
//
// ASSEMBLY ORDER STILL MATTERS:
//  1. First bolt the SO-101 base onto this adapter (M3, from
//     below) — the adapter's underside becomes unreachable once
//     it's on the board.
//  2. Then set the assembly on the board and drop the 4 M5 bolts
//     in from the top; fit nuts from under the board.
// ============================================================

module base_mount_hole() {
    translate([0,0,-1]) cylinder(h=base_screw_head_h+1, d=base_screw_head_dia);
    translate([0,0,base_screw_head_h-0.5]) cylinder(h=adapter_thick-base_screw_head_h+1.5, d=base_screw_clear);
}

module board_mount_hole() {
    translate([0,0,-1]) cylinder(h=adapter_thick+2, d=bolt_dia);
    translate([0,0,adapter_thick-bolt_head_h]) cylinder(h=bolt_head_h+1, d=bolt_head_pocket);
}

front_y = (adapter_d - front_to_back_dist)/2;
back_y  = front_y + front_to_back_dist;
cx      = adapter_w/2;
cy      = adapter_d/2;

// wrapped in a module so export.scad can render it via `use <>`;
// the call below keeps this file working when opened directly
module adapter_plate() {
    difference() {
        cube([adapter_w, adapter_d, adapter_thick]);

        // trapezoid pattern matching the SO-101 base exactly
        translate([cx - front_hole_spacing/2, front_y, 0]) base_mount_hole();
        translate([cx + front_hole_spacing/2, front_y, 0]) base_mount_hole();
        translate([cx - back_hole_spacing/2,  back_y,  0]) base_mount_hole();
        translate([cx + back_hole_spacing/2,  back_y,  0]) base_mount_hole();

        // 4 board-mount holes in an adapter_hole_spacing square, fully
        // outside the trapezoid above — torsionally stiff, top-loading
        for (dx = [-adapter_hole_spacing/2, adapter_hole_spacing/2])
            for (dy = [-adapter_hole_spacing/2, adapter_hole_spacing/2])
                translate([cx+dx, cy+dy, 0]) board_mount_hole();
    }
}

adapter_plate();
