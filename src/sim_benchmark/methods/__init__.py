"""Registry of benchmarked teleoperation methods."""

from typing import Callable

import mujoco

from sim_benchmark.methods.base import Targets, TeleopMethod
from sim_benchmark.methods.dls import DampedLeastSquares
from sim_benchmark.methods.mink_ik import MinkQP
from sim_benchmark.methods.pink_qp import PinkFull, PinkRelaxed
from sim_benchmark.methods.scipy_ik import ScipyLeastSquares
from sim_benchmark.methods.telegrip_split import TelegripSplit

MethodFactory = Callable[[mujoco.MjModel], TeleopMethod]

METHODS: dict[str, MethodFactory] = {
    cls.name: cls
    for cls in (
        PinkFull,
        PinkRelaxed,
        DampedLeastSquares,
        MinkQP,
        ScipyLeastSquares,
        TelegripSplit,
    )
}

__all__ = ["METHODS", "MethodFactory", "Targets", "TeleopMethod"]
