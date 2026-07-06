include <config.scad>

// ============================================================
// 3D-PRINTED PERFORATED BOARD — tiles + splice bars.
// Prints as: 3x board_tile (each under printer_bed) + 2x
// splice_bar. Replaces the old wooden board AND the separate
// under-board nut plate (drill_template.scad, now legacy).
//
// GRID: same 25mm hole grid as before (17 columns x 5 rows for
// the default sizes), each hole with an M5 hex pocket molded
// into the board's BOTTOM face (board_nut_pocket deep) — push a
// nut in from below wherever a bolt lands. The bottom face stays
// flush and flat: pockets and splice recesses are blind, nothing
// protrudes, so the board clamps to a table.
//
// CLAMP WINGS: board_clamp_wing of solid, hole-free board at
// both ends, beyond the grid — that's where the table clamps go.
//
// SEAMS: the tiles butt-join at grid-column midpoints and are
// tied by SPLICE BARS bolted into 8mm recesses in the underside.
// Each bar spans 2 grid columns on each side of the seam (4 x 5
// holes) with M5 hex pockets opening TOWARD the board, exactly
// like the old nut plate: drop nuts in the bar, hold it in the
// recess, and bolt M5x12 down through free grid holes (>= 3 per
// side, staggered). Rig bolts that land inside a bar zone simply
// use the bar's pockets instead of the board's — the pocket
// heights coincide, so bolt lengths don't change (M5x25
// adapters, M5x20 tower base — see README).
//
// PRINT ORIENTATION: tiles TOP FACE DOWN (pockets and recesses
// then print as upward blind holes, no supports). Bars pocket
// side up.
// ============================================================

// ---- derived grid ----
usable_w = board_width - 2*(board_margin + board_clamp_wing);
usable_d = board_depth - 2*board_margin;
cols     = floor(usable_w / grid_pitch) + 1;
rows     = floor(usable_d / grid_pitch) + 1;
grid_x0  = board_clamp_wing + board_margin
           + (usable_w - (cols-1)*grid_pitch) / 2;
grid_y0  = board_margin + (usable_d - (rows-1)*grid_pitch) / 2;

// ---- derived seams (grid-column midpoints nearest the equal-split
//      positions), 3 tiles ----
n_tiles = 3;
function seam_pos(k) = grid_x0
    + (round((k*board_width/n_tiles - grid_x0)/grid_pitch - 0.5) + 0.5)
      * grid_pitch;
seams = [ for (k = [1:n_tiles-1]) seam_pos(k) ];

function tile_x0(i) = (i == 0) ? 0 : seams[i-1] + seam_clearance/2;
function tile_x1(i) = (i == n_tiles-1) ? board_width
                                       : seams[i] - seam_clearance/2;

// splice bar footprint around a seam (bar itself has no clearance;
// the recess gets seam_clearance of play per side)
function bar_x0(s) = s - splice_bar_w/2;

function in_a_bar(x) = len([ for (s = seams)
                             if (abs(x - s) < splice_bar_w/2 + 1) s ]) > 0;

echo(str("board: ", board_width, " x ", board_depth, " x ", board_thickness,
         "mm, ", cols, " cols x ", rows, " rows"));
echo(str("seams at ", seams));
for (i = [0:n_tiles-1]) {
    tw = tile_x1(i) - tile_x0(i);
    echo(str("tile ", i, ": ", tw, " x ", board_depth, "mm",
             tw > printer_bed ? "  ** EXCEEDS printer_bed! **" : ""));
    assert(tw <= printer_bed, "tile exceeds printer_bed — adjust n_tiles");
}
assert(splice_bar_w <= printer_bed, "splice bar exceeds printer_bed");

// ---- pieces ----

// grid hole: M5 clearance all the way through; hex nut pocket in
// the bottom face except where a splice-bar recess replaces it
module board_hole(pocketed=true) {
    translate([0,0,-1]) cylinder(h=board_thickness+2, d=bolt_dia);
    if (pocketed)
        translate([0,0,-0.01])
            cylinder(h=board_nut_pocket+0.01, d=nut_dia_corners, $fn=6);
}

module board_tile(i) {
    x0 = tile_x0(i);
    x1 = tile_x1(i);
    difference() {
        translate([x0, 0, 0]) cube([x1-x0, board_depth, board_thickness]);
        // grid holes falling inside this tile
        for (c = [0:cols-1]) {
            x = grid_x0 + c*grid_pitch;
            if (x > x0 + bolt_dia/2 && x < x1 - bolt_dia/2)
                for (r = [0:rows-1])
                    translate([x, grid_y0 + r*grid_pitch, 0])
                        board_hole(pocketed = !in_a_bar(x));
        }
        // splice-bar recesses in the underside at adjacent seams
        for (s = seams)
            if (s > x0 - 1 && s < x1 + 1)
                translate([bar_x0(s) - seam_clearance, -1, -0.01])
                    cube([splice_bar_w + 2*seam_clearance,
                          board_depth + 2,
                          splice_bar_thick + 0.01]);
    }
}

// splice bar: 4 cols x 5 rows of holes, hex pockets opening on the
// TOP (board-facing) side; bottom face solid and flat
module splice_bar() {
    difference() {
        cube([splice_bar_w, board_depth, splice_bar_thick]);
        for (c = [0:3]) for (r = [0:rows-1])
            translate([splice_bar_w/2 + (c-1.5)*grid_pitch,
                       grid_y0 + r*grid_pitch, 0]) {
                translate([0,0,-1])
                    cylinder(h=splice_bar_thick+2, d=bolt_dia);
                translate([0,0,splice_bar_thick-splice_nut_pocket])
                    cylinder(h=splice_nut_pocket+1, d=nut_dia_corners,
                             $fn=6);
            }
    }
}

// full board as assembled (sim twin visual mesh — never print this)
module board_assembled() {
    for (i = [0:n_tiles-1]) board_tile(i);
    for (s = seams) translate([bar_x0(s), 0, 0]) splice_bar();
}

// ---- layout for printing (side by side) ----
for (i = [0:n_tiles-1])
    translate([tile_x0(i)*0 + i*(printer_bed+20), 0, 0])
        translate([-tile_x0(i), 0, 0]) board_tile(i);
translate([0, board_depth + 30, 0]) splice_bar();
translate([printer_bed + 20, board_depth + 30, 0]) splice_bar();
