"""Scripted-oracle demonstration collection for the sim-VLA pipeline.

This package drives the digital-twin payload scene (``sim_twin.scene`` with
``payload=True``) with a scripted expert that grasps a bar by REAL contact
and places it on a marked target, producing LeRobot-format datasets for
training and evaluating VLA policies. The oracle reuses the teleop IK stack
(``sim_benchmark.methods``) so the collected actions live in the same space
as the real-robot recorder.

Modules:
    oracle -- scripted single-arm and bimanual-handover pick-and-place experts.
    env    -- ``PickPlaceTwinEnv``, the shared collection/eval environment.
"""
