include <config.scad>

// ============================================================
// DRILL TEMPLATE + NUT PLATE — one print, two jobs.
//
// 1) DRILL GUIDE: clamp the assembled tiles to the board and
//    drill every grid hole (5.5mm bit) through the round bores.
// 2) NUT PLATE: the plate then lives UNDER the board permanently.
//    Every hole has a hex pocket opening on the TOP face (the
//    face against the board): drop an M5 nut into each position
//    you use, set the board on top, and the bolts inserted from
//    above thread into those captive nuts. The pocket is snug
//    (nut_dia_corners) so nuts don't fall out, and the plate's
//    bottom face is solid and flat — the whole board+plate
//    sandwich clamps to a table. No recess drilling in the board.
//
// Because the board is drilled through this exact plate, the
// board holes and the nut pockets align by construction.
//
// NOTE: prints as ONE piece, full board size (450 x 150mm) — this
// needs a large-format printer; it exceeds common 220-256mm beds.
// ============================================================

usable_w = board_width - 2*board_margin;
usable_d = board_depth - 2*board_margin;
cols = floor(usable_w / grid_pitch) + 1;
rows = floor(usable_d / grid_pitch) + 1;
x_off = board_margin + (usable_w - (cols-1)*grid_pitch) / 2;
y_off = board_margin + (usable_d - (rows-1)*grid_pitch) / 2;

module plate_hole() {
    // drill-guide / bolt-clearance bore, all the way through
    translate([0,0,-1]) cylinder(h=nutplate_thick+2, d=bolt_dia);
    // hex nut pocket, opening at the top (board-facing) surface
    translate([0,0,nutplate_thick-nut_pocket_depth])
        cylinder(h=nut_pocket_depth+1, d=nut_dia_corners, $fn=6);
}

difference() {
    cube([board_width, board_depth, nutplate_thick]);
    for (c = [0:cols-1])
        for (r = [0:rows-1])
            translate([x_off + c*grid_pitch, y_off + r*grid_pitch, 0])
                plate_hole();
}

echo(str("board_width = ", board_width, "mm, board_depth = ", board_depth, "mm"));
echo(str("grid: ", cols, " columns x ", rows, " rows at ", grid_pitch, "mm pitch"));
