"""Shared teleop CLI + IK-stack wiring for the real and sim Quest tools.

Both entry points — ``tool/meta_quest_teleopration.py`` (real arms) and
``tool/quest_sim_teleop.py`` (MuJoCo rehearsal) — must build the SAME IK layer
and pass the SAME kwargs to ``dual_ik_solver_thread`` or they silently drift
(a method that behaves one way in the sim and another on the robot defeats the
whole point of the rehearsal). This module owns that wiring once:

* ``add_teleop_cli_args`` registers the method/wrist-mode/envelope arguments;
* ``create_teleop_stack`` turns the parsed args into an ``(ik_solver,
  thread_kwargs)`` pair, resolving the ``mymethod`` variants and the
  orientation-cost policy.

Each tool keeps only its own side (hardware buses / button callbacks, or the
MuJoCo stepping loop). British English throughout.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from common.config_parser import load_method_params
from common.configs import (
    DUAL_URDF_PATH,
    END_EFFECTOR_FRAME_NAMES,
    IK_SOLVER_RATE,
    NEUTRAL_JOINT_ANGLES_DUAL,
    WORKSPACE_OOB_MODE,
)
from common.envelope_feedback import (
    CompositeFeedback,
    NullFeedback,
    SpeakerBeepFeedback,
    TerminalBellFeedback,
)
from common.pink_ik_solver import PinkIKSolver
from common.workspace_envelope import OOE_POLICIES
from sim_benchmark.methods import METHODS  # type: ignore[attr-defined]

# --wrist-mode -> the dual_ik_solver_thread orientation_mode it selects (only
# meaningful with --method mymethod).
_WRIST_MODE_TO_ORIENTATION = {
    "hold": "hold",  # attitude held from grip; sticks are the only change
    "wrist": "incremental",  # existing incremental wrist mapping + stick trims
    "soft": "armplane",  # weak absolute follow (0.05) + stick trims
}


def add_teleop_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_joint_vel: float,
    default_method: str = "armplane",
) -> None:
    """Register the teleop method / wrist-mode / envelope CLI arguments.

    ``default_max_joint_vel`` is the rate-limit default (sim vs hardware);
    ``default_method`` differs per tool (armplane on the real robot,
    pink_relaxed in the sim), so each caller passes its own.
    """
    parser.add_argument(
        "--method",
        type=str,
        default=default_method,
        choices=["armplane", "production", "mymethod", *sorted(METHODS)],
        help=(
            "IK layer: 'armplane' (alias: production, deprecated) is the tuned "
            "Pink solver with the armplane orientation mapping; 'mymethod' is "
            "the pink_relaxed solver plus the thumbstick wrist trims (see "
            "--wrist-mode); the others are the sim-benchmark methods behind a "
            "joint-space rate limiter. Rehearse in the sim first."
        ),
    )
    parser.add_argument(
        "--wrist-mode",
        type=str,
        default="hold",
        choices=["hold", "wrist", "soft"],
        help=(
            "Gripper attitude behaviour for --method mymethod (ignored "
            "otherwise): 'hold' keeps the attitude captured at grip and only "
            "the thumbsticks change it; 'wrist' keeps the incremental wrist "
            "mapping and lets the sticks trim on top; 'soft' uses pink_relaxed's "
            "weak absolute follow with stick trims that decay toward the hand."
        ),
    )
    parser.add_argument(
        "--max-joint-vel",
        type=float,
        default=default_max_joint_vel,
        help="Joint-space rate limit (rad/s) applied to benchmark methods",
    )
    parser.add_argument(
        "--oob-mode",
        type=str,
        default=WORKSPACE_OOB_MODE,
        choices=sorted(OOE_POLICIES),
        help="Out-of-envelope target policy (see common/workspace_envelope.py)",
    )
    parser.add_argument(
        "--orientation-cost",
        type=float,
        default=None,
        help="Override the IK orientation-task cost (Pink methods only). Raises "
        "gripper pitch/roll tracking on a relaxed method like pink_relaxed "
        "(benchmark default 0.05); try ~0.2-0.4. None keeps the method's YAML "
        "value (except mymethod+wrist, which defaults to 0.3).",
    )
    parser.add_argument(
        "--envelope-feedback",
        type=str,
        default="audio",
        choices=["audio", "bell", "none"],
        help="Operator out-of-envelope cueing: 'audio' (default) rings the "
        "terminal bell AND plays an audible speaker beep (lower tone = LEFT "
        "arm, higher = RIGHT); 'bell' is the terminal bell only; 'none' "
        "disables it (the throttled diagnostic print stays in every case).",
    )


def _build_envelope_feedback(choice: str) -> Any:
    """Map the ``--envelope-feedback`` choice to a feedback backend.

    'audio' runs the terminal bell and the speaker beep together; 'bell' is the
    terminal bell alone; 'none' disables operator cueing.
    """
    if choice == "audio":
        return CompositeFeedback([TerminalBellFeedback(), SpeakerBeepFeedback()])
    if choice == "bell":
        return TerminalBellFeedback()
    return NullFeedback()


def _resolve_method(method: str, wrist_mode: str) -> tuple[str, str, bool]:
    """Return (solver_method, orientation_mode, joystick_wrist) for --method.

    'mymethod' resolves to the pink_relaxed solver, the orientation mode its
    --wrist-mode selects, and the thumbstick trims enabled; everything else is
    the armplane orientation mapping with no thumbstick trims.
    """
    if method == "mymethod":
        return "pink_relaxed", _WRIST_MODE_TO_ORIENTATION[wrist_mode], True
    solver_method = "armplane" if method in ("armplane", "production") else method
    return solver_method, "armplane", False


def create_teleop_stack(
    args: argparse.Namespace, *, dt: float
) -> tuple[Any, dict[str, Any]]:
    """Build the IK solver and the dual_ik_solver_thread kwargs from args.

    Returns ``(ik_solver, thread_kwargs)``. The armplane/production methods
    build a ``PinkIKSolver`` from src/ik_conf/methods/armplane.yaml; every
    other method (including mymethod's pink_relaxed) builds a
    ``MethodIKAdapter``. Both tools consume the identical result.
    """
    solver_method, orientation_mode, joystick_wrist = _resolve_method(
        args.method, args.wrist_mode
    )
    is_armplane = args.method in ("armplane", "production")

    if is_armplane:
        if args.method == "production":
            print("⚠️  --method production is deprecated; use --method armplane")
        print("\n🔧 Creating dual-arm Pink IK solver (armplane)...")
        # armplane solver weights come from src/ik_conf/methods/armplane.yaml
        # (strict load). Neutral/posture are expanded to the dual (10-DOF)
        # configuration here.
        ap = load_method_params("armplane")
        posture_dual = np.array(ap["posture_cost_vector"], dtype=float)
        neutral_dual = np.radians(
            [*ap["neutral_joint_angles_deg"], *ap["neutral_joint_angles_deg"]]
        )
        ik_solver: Any = PinkIKSolver(
            urdf_path=DUAL_URDF_PATH,
            end_effector_frames=END_EFFECTOR_FRAME_NAMES,
            solver_name=ap["solver"],
            position_cost=ap["position_cost"],
            # Anisotropic: zero cost on the EE-local yaw axis (no wrist-yaw joint)
            orientation_cost=ap["orientation_cost"]
            * np.asarray(ap["ee_orientation_cost_mask"]),
            frame_task_gain=ap["frame_task_gain"],
            lm_damping=ap["lm_damping"],
            damping_cost=ap["damping_cost"],
            solver_damping_value=ap["solver_damping_value"],
            integration_time_step=1.0 / IK_SOLVER_RATE,
            initial_configuration=neutral_dual,
            posture_cost_vector=posture_dual,
        )
    else:
        from sim_benchmark.method_adapter import MethodIKAdapter

        print(f"\n🔧 Creating benchmark IK method '{solver_method}'...")
        ik_solver = MethodIKAdapter(
            solver_method,
            dt=dt,
            max_joint_vel=args.max_joint_vel,
            initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
        )

    # Orientation-task cost. --orientation-cost wins if given; otherwise
    # mymethod+wrist (the incremental mapping) needs the attitude actually
    # tracked, so it defaults to 0.3 (pink_relaxed's 0.05 is near position-
    # only); mymethod hold/soft and every other method keep their YAML value.
    effective_ocost = args.orientation_cost
    if (
        effective_ocost is None
        and args.method == "mymethod"
        and args.wrist_mode == "wrist"
    ):
        effective_ocost = 0.3
    if effective_ocost is not None:
        if is_armplane:  # PinkIKSolver
            ik_solver.update_task_parameters(orientation_cost=effective_ocost)
            applied = True
        else:  # MethodIKAdapter (Pink QP methods only expose the task cost)
            applied = ik_solver.set_orientation_cost(effective_ocost)
        if applied:
            print(f"🎚️  Orientation-cost → {effective_ocost}")
        else:
            print(
                f"⚠️  --orientation-cost has no effect for method "
                f"'{args.method}' (no Pink orientation task); ignored"
            )

    thread_kwargs: dict[str, Any] = {
        "oob_mode": args.oob_mode,
        "envelope_feedback": _build_envelope_feedback(args.envelope_feedback),
        "orientation_mode": orientation_mode,
        "joystick_wrist": joystick_wrist,
    }
    return ik_solver, thread_kwargs
