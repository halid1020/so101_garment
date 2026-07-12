"""Portable Isaac Lab twin package.

Self-contained on purpose: modules here import only the standard
library, numpy, and (on the Isaac machine) ``isaaclab`` — never the
rest of this repository. All rig geometry arrives via the resolved
``twin_params.json`` produced by ``python -m sim_twin.assets``.
"""
