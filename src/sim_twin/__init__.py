"""Simulation digital twin of the dual SO-101 board rig.

``src/platform/config.scad`` is the single source of truth for the
physical design; this package parses it, exports the printed parts to
meshes, generates a dual-arm URDF with the real mounting stack and the
C310 cameras, and builds matching MuJoCo (local) and Isaac Lab
(portable, ``src/sim_twin/isaac``) scenes.

Typical use::

    python -m sim_twin.assets        # (re)build build/twin from the SCAD
    python tool/view_twin.py --watch # live viewer, follows SCAD edits
    python -m sim_twin.verify        # end-to-end checks + camera renders
"""
