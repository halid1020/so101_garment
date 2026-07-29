// Simple 3D-printed validation cube — the task object for pick-and-place
// sanity checks during data collection and policy evaluation. Sized to be
// graspable by the SO-101's capped jaw (default 25 mm side; the sim payload
// cross-section is ~22 mm). All dimensions come from config.scad; edit them
// there, not here.
//
// Rounded edges make it easier to grasp and cleaner to print; an optional
// shallow square recess on the top face seats a fiducial/marker sticker so the
// object pose is visible to the overhead camera.
//
// Export a printable STL with:
//   openscad -o task_cube.stl -D 'part="task_cube"' export.scad

include <config.scad>

module task_cube(
    size = task_cube_size,
    chamfer = task_cube_chamfer,
    marker = task_cube_marker_recess,
    marker_depth = task_cube_marker_depth
) {
    difference() {
        // Rounded-edge cube: Minkowski of a shrunken cube with a sphere gives
        // an outer dimension of exactly `size` with `chamfer`-radius edges.
        if (chamfer > 0)
            minkowski() {
                cube(size - 2 * chamfer, center = true);
                sphere(r = chamfer);
            }
        else
            cube(size, center = true);

        // Shallow marker recess on the +Z face (skip when marker == 0).
        if (marker > 0)
            translate([0, 0, size / 2 - marker_depth])
                cube([marker, marker, 2 * marker_depth + 1], center = true);
    }
}

task_cube();
