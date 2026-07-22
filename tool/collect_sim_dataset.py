#!/usr/bin/env python3
"""Collect scripted-oracle pick-and-place demonstrations in the digital twin.

Two collection modes share the task scripts, the twin environment, the
demo-gating logic and the dataset schema; only the layer that turns EE targets
into joint commands differs:

* ``--oracle teleop`` (default) -- a scripted operator device drives the FULL
  production teleop pipeline (One-Euro filter, grip clutch, operator-frame
  retargeting, ``dual_ik_solver_thread``), exactly as ``quest_sim_teleop``. The
  recorded action is the joint command the IK thread published. Real-time and
  asynchronous; episodes are wall-clock.
* ``--oracle direct`` -- the insurance path: per 30 Hz tick the script's EE
  targets go straight into ``METHODS[method].solve`` and then the env. The
  recorded action is that solved command. Synchronous and deterministic.

Both write a LeRobot dataset (``observation.state`` + the three twin cameras +
``action``) and gate every episode on the bar landing on the target, settled,
with the grippers released; failures are discarded and retried on a fresh
scenario. ``--dry-run`` skips the dataset entirely (for the contact-grasp
tuning loop), and ``--gif DIR`` writes a debugging animation per episode.

Examples
--------
    # tune contacts (no dataset), single-arm, default teleop oracle
    python tool/collect_sim_dataset.py --task single --episodes 30 --dry-run \
        --gif outputs/oracle_gifs

    # collect a handover dataset with the direct oracle
    python tool/collect_sim_dataset.py --task handover --episodes 150 \
        --oracle direct --repo-id local/so101_sim_handover
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.configs import (  # noqa: E402
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    GRIPPER_OPEN_MAX_FRAC,
    IK_SOLVER_RATE,
    MAX_JOINT_VEL_SIM_RAD_S,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.recording.features import (  # noqa: E402
    assemble_frame,
    build_dataset_features,
    build_observation_state,
)
from common.teleop_setup import add_teleop_cli_args, create_teleop_stack  # noqa: E402
from sim_benchmark.constants import SIDES  # noqa: E402
from sim_datagen.env import CAMERAS, TASKS, PickPlaceTwinEnv  # noqa: E402
from sim_datagen.oracle import (  # noqa: E402
    NOMINAL_TABLE_Z_IK,
    HandoverContactScript,
    SinglePickPlaceScript,
    generate_relay_scenarios,
    generate_single_scenarios,
)
from sim_datagen.seeds import TRAIN_SEEDS  # noqa: E402

FPS_SUBSTEPS_NOTE = "30 Hz control, 1/600 s physics -> 20 substeps per tick"

# Orientation-task cost used while collecting (unless --orientation-cost is
# given). pink_relaxed's benchmark value (0.05) is near position-only, which
# lets the wrist pitch drift with position and rotates the rigidly-gripped bar
# past its topple margin during the carry; 0.4 makes the IK hold the oracle's
# fixed grasp attitude (see sim_datagen.oracle.GRASP_PITCH_DEG) end to end.
ORACLE_ORIENTATION_COST = 0.4


@dataclass
class EpisodeResult:
    """Outcome of one attempted demonstration."""

    success: bool
    phase: str  # failure phase, or "success"
    length: int  # recorded frames


@dataclass
class Stats:
    """Aggregate collection statistics for the JSON sidecar."""

    attempts: int = 0
    successes: int = 0
    lengths: list[int] = field(default_factory=list)
    failure_phases: dict[str, int] = field(default_factory=dict)

    def record(self, result: EpisodeResult) -> None:
        self.attempts += 1
        if result.success:
            self.successes += 1
            self.lengths.append(result.length)
        else:
            self.failure_phases[result.phase] = (
                self.failure_phases.get(result.phase, 0) + 1
            )

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
def _classify(
    env: PickPlaceTwinEnv,
    lifted: bool,
    dropped: bool,
    grips_open: bool,
) -> tuple[bool, str]:
    """Return (success, phase) for a finished episode."""
    if not lifted:
        return False, "grasp"
    if dropped:
        return False, "carry"
    if env.place_error() >= 0.02:
        return False, "place"
    if not env.payload_settled():
        return False, "settle"
    if not grips_open:
        return False, "release"
    return True, "success"


class _FrameSink:
    """Adds frames to a dataset (or discards them) plus optional GIF capture."""

    def __init__(
        self,
        dataset: Any,
        task_str: str,
        gif: Any,
    ) -> None:
        self.dataset = dataset
        self.task_str = task_str
        self.gif = gif
        self.count = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
        sim_data: Any,
        sim_time: float,
    ) -> None:
        if self.dataset is not None:
            self.dataset.add_frame(assemble_frame(state, action, images, self.task_str))
        if self.gif is not None:
            self.gif.maybe_capture(sim_data, sim_time)
        self.count += 1


# ---------------------------------------------------------------------------
# Closed-loop grasp + place correction (the teleoperator's eyes), shared by
# both oracle paths.
# ---------------------------------------------------------------------------
class _AlignmentServo:
    """Per-side closed-loop XY correction added to the script's EE targets.

    Two coupled corrections, both driven by the sim's measured cube/EE poses —
    exactly the visual feedback a human teleoperator uses:

    * grasp servo — while an arm's gripper is still OPEN and poised over the
      cube on the table, centre the pinch on the cube, then freeze at jaw
      close. The differential IK leaves an azimuth-dependent ~8 mm steady-state
      offset that the 2.2 cm cube will not tolerate (the jaws close beside it).
    * hold servo — while the arm squeezes the lifted cube, drive the cube onto
      the scripted-intended position (scripted EE + baked grasp offset),
      correcting the larger place-pose IK error so the OBJECT lands on the mark.

    Low gains + magnitude clips: the IK lags the target by ~10 ticks with a
    persistent steady-state error, so a high gain winds up and throws the pinch
    off. ``offset(side)`` is the net XY shift to ADD to that arm's EE target.
    """

    GRASP_GAIN = 0.12
    GRASP_CLIP = 0.03  # m
    HOLD_GAIN = 0.18
    HOLD_CLIP = 0.06  # m — place IK error can reach ~40 mm

    def __init__(self, holders: list[str], script: Any) -> None:
        self.holders = holders
        self.script = script
        self.grasp: dict[str, np.ndarray] = {s: np.zeros(2) for s in holders}
        self.frozen: dict[str, bool] = {s: False for s in holders}
        self.hold: dict[str, np.ndarray] = {s: np.zeros(2) for s in holders}
        self._lifted: dict[str, bool] = {s: False for s in holders}
        self._placed: dict[str, bool] = {s: False for s in holders}

    def update(
        self, env: PickPlaceTwinEnv, grips: dict[str, float], raw_targets: dict
    ) -> None:
        bar = env.sim.payload_pos()
        for side in self.holders:
            holding = grips[side] < 0.05
            lifted = bar[2] >= env._payload_rest_z + 0.03
            if holding and lifted:
                self._lifted[side] = True
            # Freeze the hold correction once the cube is set back down on the
            # table (was lifted, now resting) while still gripped: through the
            # teleop pipeline lag the servo would otherwise keep nudging and
            # DRAG the placed cube off the target before the jaws open.
            if self._lifted[side] and holding and bar[2] < env._payload_rest_z + 0.015:
                self._placed[side] = True
            if not (holding and lifted) or self._placed[side]:
                continue
            raw_xy = raw_targets[side][0][:2]
            intended = raw_xy + self.script.baked_offset(side, raw_xy)
            self.hold[side] = np.clip(
                self.hold[side] + self.HOLD_GAIN * (bar[:2] - intended),
                -self.HOLD_CLIP,
                self.HOLD_CLIP,
            )
        for side in self.holders:
            # grasp servo: only while the jaws are open over the resting cube
            if self.frozen[side]:
                continue
            if grips[side] <= 0.8:  # jaws closing -> lock this arm's alignment
                self.frozen[side] = True
                continue
            if bar[2] > env._payload_rest_z + 0.02:
                continue
            ee_xy = env.sim.eef_pose(side)[0][:2]
            pinch_xy = ee_xy + self.script.baked_offset(side, ee_xy)
            if np.linalg.norm(pinch_xy - bar[:2]) > 0.06:
                continue  # this arm is not approaching the cube yet
            self.grasp[side] = np.clip(
                self.grasp[side] + self.GRASP_GAIN * (bar[:2] - pinch_xy),
                -self.GRASP_CLIP,
                self.GRASP_CLIP,
            )

    def offset(self, side: str) -> np.ndarray:
        """Net XY correction to ADD to ``side``'s EE target."""
        if side not in self.holders:
            return np.zeros(2)
        return self.grasp[side] - self.hold[side]


class _CorrectedScript:
    """Wraps a task script, adding the servo's per-side XY offset to targets.

    The teleop device reads ``targets`` asynchronously from the IK thread; the
    collector loop updates the shared servo at the control rate, so the same
    closed-loop correction flows through the full teleoperation pipeline.
    """

    def __init__(self, base: Any, servo: _AlignmentServo) -> None:
        self._base = base
        self._servo = servo
        self.duration = base.duration

    def targets(self, t: float) -> dict:
        out = self._base.targets(t)
        for side, (pos, rot) in out.items():
            off = self._servo.offset(side)
            if off.any():
                out[side] = (pos + np.array([off[0], off[1], 0.0]), rot)
        return out

    def grip_fractions(self, t: float) -> dict[str, float]:
        return self._base.grip_fractions(t)

    def baked_offset(self, side: str, ee_xy: np.ndarray) -> np.ndarray:
        return self._base.baked_offset(side, ee_xy)


def _holders_for(task: str, scenario: Any) -> list[str]:
    if task == "single":
        return [scenario.side]
    return [scenario.pick_side, scenario.place_side]


# ---------------------------------------------------------------------------
# Direct oracle: script EE targets -> METHODS[method].solve -> env
# ---------------------------------------------------------------------------
def run_episode_direct(
    env: PickPlaceTwinEnv,
    scenario: Any,
    script: Any,
    method: Any,
    task: str,
    fps: int,
    camera_wh: tuple[int, int],
    sink: _FrameSink,
) -> EpisodeResult:
    env.reset(scenario)
    neutral_q = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
    method.reset(neutral_q)
    dt = 1.0 / fps
    n_ticks = int(np.ceil(script.duration / dt))

    lifted = dropped = False
    holders = _holders_for(task, scenario)
    servo = _AlignmentServo(holders, script)
    grips = script.grip_fractions(0.0)
    for k in range(n_ticks):
        t = k * dt
        targets = script.targets(t)
        grips = script.grip_fractions(t)
        servo.update(env, grips, targets)
        for side in holders:
            off = servo.offset(side)
            if off.any():
                pos, rot = targets[side]
                targets[side] = (pos + np.array([off[0], off[1], 0.0]), rot)
        q_cmd = method.solve(targets, dt)
        env.tick(q_cmd, grips)

        state, images = env.observe(camera_wh)
        # Action gripper channels store the capped full-range command, exactly
        # like the real recorder ((1 - trigger) x GRIPPER_OPEN_MAX_FRAC).
        cmd_grips = {s: g * GRIPPER_OPEN_MAX_FRAC for s, g in grips.items()}
        action = build_observation_state(np.rad2deg(q_cmd), cmd_grips)
        sink.add(state, action, images, env.sim.data, t)

        bar_z = env.sim.payload_pos()[2]
        if bar_z > env._payload_rest_z + 0.03:
            lifted = True
        if lifted and env.payload_dropped():
            dropped = True
            break

    grips_open = all(grips[s] > 0.8 for s in SIDES)
    success, phase = _classify(env, lifted, dropped, grips_open)
    return EpisodeResult(success, phase, sink.count)


# ---------------------------------------------------------------------------
# Teleop oracle: scripted device -> full pipeline -> env
# ---------------------------------------------------------------------------
def run_episode_teleop(
    env: PickPlaceTwinEnv,
    scenario: Any,
    script: Any,
    device: Any,
    data_manager: Any,
    task: str,
    fps: int,
    camera_wh: tuple[int, int],
    sink: _FrameSink,
) -> EpisodeResult:
    from common.data_manager_dual import RobotActivityState

    env.reset(scenario)
    neutral_deg = np.array(NEUTRAL_JOINT_ANGLES_DUAL, dtype=float)
    held_target = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
    data_manager.set_current_joint_angles(neutral_deg)
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    # Release any previous grip, let the clutch reset, then arm this episode.
    device.disengage()
    time.sleep(0.15)
    # Same closed-loop grasp/place correction as the direct oracle, injected
    # through the device: the collector updates the servo from the measured
    # cube each tick and the device reads the corrected hand targets, so the
    # alignment flows through the full teleoperation pipeline.
    holders = _holders_for(task, scenario)
    servo = _AlignmentServo(holders, script)
    device.load_episode(_CorrectedScript(script, servo))

    dt = 1.0 / fps
    lifted = dropped = False
    grip_frac = {s: 1.0 for s in SIDES}
    while not device.episode_finished():
        tick_start = time.time()
        if data_manager.get_teleop_active():
            target_deg = data_manager.get_target_joint_angles()
            if target_deg is not None:
                held_target = np.radians(target_deg)
        # Gripper command from the One-Euro-filtered trigger (as the sim tool).
        for side in SIDES:
            _, _, trigger = data_manager.get_controller_state(side)
            grip_frac[side] = float(np.clip(1.0 - trigger, 0.0, 1.0))
        servo.update(env, grip_frac, script.targets(device.script_time()))
        env.tick(held_target, grip_frac)
        data_manager.set_current_joint_angles(np.rad2deg(env.sim.arm_q()))

        state, images = env.observe(camera_wh)
        target_deg = data_manager.get_target_joint_angles()
        if data_manager.get_teleop_active() and target_deg is not None:
            cmd_grips = {s: g * GRIPPER_OPEN_MAX_FRAC for s, g in grip_frac.items()}
            action = build_observation_state(np.asarray(target_deg), cmd_grips)
        else:
            action = state.copy()
        sink.add(state, action, images, env.sim.data, sink.count * dt)

        bar_z = env.sim.payload_pos()[2]
        if bar_z > env._payload_rest_z + 0.03:
            lifted = True
        if lifted and env.payload_dropped():
            dropped = True
            break
        sleep = dt - (time.time() - tick_start)
        if sleep > 0:
            time.sleep(sleep)

    device.disengage()
    grips_open = all(grip_frac[s] > 0.8 for s in SIDES)
    success, phase = _classify(env, lifted, dropped, grips_open)
    return EpisodeResult(success, phase, sink.count)


# ---------------------------------------------------------------------------
def _scenario_for_seed(task: str, seed: int) -> Any:
    """One deterministic scenario for a single seed (per-seed protocol)."""
    if task == "single":
        return generate_single_scenarios(1, seed=seed)[0]
    return generate_relay_scenarios(1, seed=seed)[0]


def _make_script(task: str, scenario: Any, env: PickPlaceTwinEnv) -> Any:
    if task == "single":
        return SinglePickPlaceScript(scenario, env.neutral_ik_poses, env.table_z)
    return HandoverContactScript(scenario, env.neutral_ik_poses, env.table_z)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument(
        "--oracle",
        choices=["teleop", "direct"],
        default="teleop",
        help="teleop = full pipeline via a scripted operator device (default); "
        "direct = synchronous METHODS[method].solve insurance path",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--seeds",
        choices=["simple", "full"],
        default="full",
        help="full = one demo per TRAIN seed, walking 0,1,2,…; simple = every "
        "demo uses only seed 0's scenario (overfit-one-scenario sanity mode)",
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--max-attempts-factor", type=int, default=3)
    parser.add_argument("--view", action="store_true", help="live MuJoCo viewer")
    parser.add_argument("--gif", type=str, default=None, metavar="DIR")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write a dataset (contact-tuning iterations)",
    )
    parser.add_argument("--stats-out", type=str, default=None, metavar="PATH")
    add_teleop_cli_args(
        parser,
        default_max_joint_vel=MAX_JOINT_VEL_SIM_RAD_S,
        default_method="pink_relaxed",
    )
    args = parser.parse_args()
    # A scripted oracle never needs operator out-of-envelope cueing, and the
    # bar-carry needs the grasp attitude actually held (see the constant).
    args.envelope_feedback = "none"
    if args.orientation_cost is None:
        args.orientation_cost = ORACLE_ORIENTATION_COST

    task_str = TASKS[args.task]
    camera_wh = (args.camera_width, args.camera_height)
    n = args.episodes
    # Deterministic direct mode gains nothing from retrying the same seed;
    # the asynchronous teleop mode can differ run to run, so it retries.
    per_seed_attempts = args.max_attempts_factor if args.oracle == "teleop" else 1

    env = PickPlaceTwinEnv(args.task)
    print(
        f"📐 IK→twin offset {np.round(env.scene_offset, 4)} m; table top at "
        f"IK z={env.table_z:.4f} (nominal {NOMINAL_TABLE_Z_IK:.4f})"
    )
    if abs(env.table_z - NOMINAL_TABLE_Z_IK) > 0.005:
        print("⚠️  measured table_z differs from nominal by >5 mm — check the twin")

    # Dataset (unless dry-run).
    dataset = None
    if not args.dry_run:
        if not args.repo_id:
            parser.error("--repo-id is required unless --dry-run")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME

        root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
        if root.exists():
            raise SystemExit(
                f"❌ {root} already exists — choose another --repo-id/--root"
            )
        features = build_dataset_features(
            [(name, args.camera_height, args.camera_width) for name in CAMERAS]
        )
        print(f"📂 Creating dataset {args.repo_id} at {root} ({args.fps} fps)")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=root,
            robot_type="so101_dual",
            use_videos=True,
            video_backend="pyav",
        )

    # Mode-specific target->command layer.
    device = data_manager = ik_thread = method = None
    if args.oracle == "teleop":
        from common.data_manager_dual import DualDataManager
        from common.threads.dual_ik_solver import dual_ik_solver_thread
        from sim_datagen.oracle_device import OracleQuestDevice

        data_manager = DualDataManager()
        data_manager.set_controller_filter_params(
            CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
        )
        data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)
        ik_solver, thread_kwargs = create_teleop_stack(args, dt=1.0 / IK_SOLVER_RATE)
        # The twin's table top sits ~34 mm BELOW the IK frame's base plane
        # (measured at runtime as env.table_z), so table-level grasp targets
        # would violate the shared-YAML z_floor tuned for the real rig. Scope
        # the fix to this collection scene: lower the envelope floor to 5 mm
        # above the measured table top (still guards the jaws against the
        # table; the lowest scripted EE target — the handover placer's
        # low-grasp descend, ~table_z + 30 mm — clears it comfortably). The
        # shared YAML and the real teleop stack are untouched. NOTE: the
        # direct oracle mode bypasses the envelope entirely (METHODS adapters
        # never see it), so no override is needed there.
        thread_kwargs["envelope_z_floor"] = env.table_z + 0.005
        device = OracleQuestDevice(env.neutral_ik_poses, TRANSLATION_SCALE)
        ik_thread = threading.Thread(
            target=dual_ik_solver_thread,
            args=(data_manager, ik_solver, device),
            kwargs=thread_kwargs,
            daemon=True,
        )
        ik_thread.start()
    else:
        from sim_benchmark.methods import METHODS  # type: ignore[attr-defined]
        from sim_benchmark.scene import DualArmSim

        if args.method not in METHODS:
            parser.error(
                f"--oracle direct needs --method in {sorted(METHODS)}; got {args.method}"
            )
        method = METHODS[args.method](DualArmSim().model)
        solver = getattr(method, "solver", None)
        if solver is not None and hasattr(solver, "update_task_parameters"):
            solver.update_task_parameters(orientation_cost=args.orientation_cost)
            print(f"🎚️  Orientation-cost → {args.orientation_cost}")

    gif_dir = Path(args.gif) if args.gif else None

    def run_one(scenario: Any, gif: Any) -> EpisodeResult:
        script = _make_script(args.task, scenario, env)
        sink = _FrameSink(dataset, task_str, gif)
        if args.oracle == "teleop":
            return run_episode_teleop(
                env,
                scenario,
                script,
                device,
                data_manager,
                args.task,
                args.fps,
                camera_wh,
                sink,
            )
        return run_episode_direct(
            env,
            scenario,
            script,
            method,
            args.task,
            args.fps,
            camera_wh,
            sink,
        )

    def seed_walk() -> Any:
        if args.seeds == "simple":
            for _ in range(n * args.max_attempts_factor):
                yield 0
        else:
            yield from TRAIN_SEEDS

    stats = Stats()
    per_seed_outcomes: list[dict[str, Any]] = []
    print(
        f"🎬 Collecting {n} '{args.task}' demos via the {args.oracle} oracle, "
        f"{args.seeds} seeds ({FPS_SUBSTEPS_NOTE})"
    )
    try:
        for seed in seed_walk():
            if stats.successes >= n:
                break
            scenario = _scenario_for_seed(args.task, seed)
            seed_ok = False
            phase = "grasp"
            attempts_here = 0
            # Simple mode counts each yield as one attempt; full mode retries
            # the same seed up to per_seed_attempts before skipping it.
            retries = 1 if args.seeds == "simple" else per_seed_attempts
            for _ in range(retries):
                attempts_here += 1
                gif = None
                if gif_dir is not None:
                    from sim_benchmark.gif import GifRecorder

                    gif = GifRecorder(env.sim.model, label=f"{args.task} seed{seed}")
                result = run_one(scenario, gif)
                stats.record(result)
                phase = result.phase
                # HARD INVARIANT: a failed demo is never saved.
                if dataset is not None:
                    if result.success:
                        dataset.save_episode()
                    else:
                        dataset.clear_episode_buffer()
                if gif is not None and gif_dir is not None:
                    status = "ok" if result.success else f"fail-{result.phase}"
                    gif.save(gif_dir / f"{args.task}_seed{seed:04d}_{status}.gif")
                    gif.close()
                if result.success:
                    seed_ok = True
                    break
            per_seed_outcomes.append(
                {
                    "seed": int(seed),
                    "attempts": attempts_here,
                    "success": seed_ok,
                    "phase": "success" if seed_ok else phase,
                }
            )
            tag = "✅" if seed_ok else f"❌ {phase}"
            print(
                f"  seed {seed:>4} {tag}  place_err {env.place_error()*1e3:5.1f} mm  "
                f"({stats.successes}/{n} done, {stats.attempts} eps tried)",
                flush=True,
            )
    finally:
        if data_manager is not None:
            data_manager.request_shutdown()
        if device is not None:
            device.stop()
        if ik_thread is not None:
            ik_thread.join(timeout=2.0)
        if dataset is not None:
            dataset.finalize()

    n_seeds = len(per_seed_outcomes)
    per_seed_rate = (
        sum(o["success"] for o in per_seed_outcomes) / n_seeds if n_seeds else 0.0
    )
    mean_len = float(np.mean(stats.lengths)) if stats.lengths else 0.0
    summary: dict[str, Any] = {
        "task": args.task,
        "oracle_mode": args.oracle,
        "method": args.method,
        "seeds_mode": args.seeds,
        "episodes_requested": n,
        "episodes_collected": stats.successes,
        "scenarios_tried": n_seeds,
        "episode_attempts": stats.attempts,
        "per_seed_success_rate": per_seed_rate,
        "oracle_success_rate": stats.success_rate,
        "failure_phase_histogram": stats.failure_phases,
        "mean_episode_length": mean_len,
        "camera_wh": list(camera_wh),
        "per_seed_outcomes": per_seed_outcomes,
    }
    if dataset is not None:
        summary["view"] = _view_block(dataset, args.repo_id, stats.successes)
    print("\n=== collection summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "view"}, indent=2))
    if dataset is not None:
        for line in summary["view"]:
            print(line)
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(summary, indent=2))
        print(f"📝 stats → {args.stats_out}")

    if stats.successes < n and per_seed_rate < 0.5:
        print("❌ oracle per-seed success below 50% — contact/retargeting needs tuning")
        return 1
    return 0


def _view_block(dataset: Any, repo_id: str, n_episodes: int) -> list[str]:
    """Copy-pasteable commands to inspect the collected dataset (rerun/lerobot)."""
    root = str(dataset.root)
    cams = ", ".join(CAMERAS)
    base = f"venv/bin/lerobot-dataset-viz --repo-id {repo_id} --root {root}"
    return [
        "",
        "=== view your data ===",
        f"data: {root}  ({n_episodes} episodes, cameras: {cams})",
        "local viewer (rerun):",
        f"  {base} --episode-index 0",
        "headless/remote — stream:",
        f"  {base} --episode-index 0 --mode distant --grpc-port 9876",
        "  then on your machine:  rerun rerun+http://<HOST>:9876/proxy",
        "headless/remote — export:",
        f"  {base} --episode-index 0 --save 1 --output-dir <dir>",
        "  then scp the .rrd and:  rerun <file>.rrd",
        "note: the HF hub web viewer only serves hub-pushed datasets; these are "
        "local-only by design (never auto-pushed).",
    ]


if __name__ == "__main__":
    sys.exit(main())
