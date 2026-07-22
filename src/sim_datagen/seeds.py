"""Disjoint scenario-seed pools shared by collection and evaluation.

One demonstration is generated per seed: the scenario for a seed ``s`` is
``generate_scenarios(1, seed=s)[0]`` (or ``generate_single_scenarios``). Keeping
the train, validation and evaluation seeds disjoint guarantees the policy is
never evaluated on a scenario it was trained on.
"""

from __future__ import annotations

TRAIN_SEEDS = range(0, 1000)
VAL_SEEDS = range(10000, 10010)
EVAL_SEEDS = range(20000, 20030)
