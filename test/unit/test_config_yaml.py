#!/usr/bin/env python3
"""Unit tests for the teleop parameter-YAML loading layer.

Guards the migration of teleop tuning constants into
``src/ik_conf/teleop_shared.yaml`` and ``src/ik_conf/methods/<name>.yaml``:

* frozen-value regression — the YAML-loaded constants still equal today's
  literals, so an accidental edit to a YAML is caught;
* strict validation — a missing/unknown/extra key raises a clear error;
* every registered benchmark method (plus the production ``armplane`` solver)
  has a loadable YAML with exactly the expected keys.

Run via: python -m unittest test.unit.test_config_yaml
(requires PYTHONPATH=.:src, as set by `source setup.sh`).
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from common import configs
from common.config_parser import load_method_params, load_teleop_shared
from sim_benchmark.methods import METHODS  # type: ignore[attr-defined]

# Every method that must have a companion YAML: the benchmark registry plus the
# production armplane solver.
_ALL_METHODS = sorted(set(METHODS) | {"armplane"})


class TestFrozenSharedValues(unittest.TestCase):
    """The bound configs constants must equal today's literal values."""

    def test_filtering(self):
        self.assertEqual(configs.CONTROLLER_MIN_CUTOFF, 0.8)
        self.assertEqual(configs.CONTROLLER_BETA, 5.0)
        self.assertEqual(configs.CONTROLLER_D_CUTOFF, 0.9)

    def test_clutch(self):
        self.assertEqual(configs.GRIP_THRESHOLD, 0.9)
        self.assertEqual(configs.ORIENTATION_BLEND_TIME_S, 1.0)

    def test_gripper(self):
        self.assertEqual(configs.GRIPPER_OPEN_MAX_FRAC, 0.3)

    def test_handle(self):
        self.assertEqual(configs.HANDLE_PITCH_OFFSET_DEG, 65.0)
        self.assertEqual(list(configs.HANDLE_AXIS), [0.8242, 0.2110, -0.5255])

    def test_operator_frame(self):
        self.assertEqual(configs.OPERATOR_FRAME_BACK_M, 0.20)
        self.assertEqual(configs.OPERATOR_FRAME_UP_M, 0.20)

    def test_envelope(self):
        self.assertEqual(configs.WORKSPACE_R_MIN, 0.0837)
        self.assertEqual(configs.WORKSPACE_R_MAX, 0.4110)
        self.assertEqual(configs.WORKSPACE_Z_FLOOR, -0.01)
        self.assertEqual(configs.WORKSPACE_SAFETY_MARGIN, 0.01)
        self.assertEqual(configs.WORKSPACE_SOFT_MARGIN, 0.04)
        self.assertEqual(configs.WORKSPACE_OOB_MODE, "warn")

    def test_rates(self):
        self.assertEqual(configs.CONTROLLER_DATA_RATE, 50.0)
        self.assertEqual(configs.IK_SOLVER_RATE, 100)
        self.assertEqual(configs.VISUALIZATION_RATE, 60.0)
        self.assertEqual(configs.ROBOT_RATE, 100.0)
        self.assertEqual(configs.JOINT_STATE_STREAMING_RATE, 100.0)
        self.assertEqual(configs.CAMERA_FRAME_STREAMING_RATE, 30.0)

    def test_scaling(self):
        self.assertEqual(configs.TRANSLATION_SCALE, 0.8)
        self.assertEqual(configs.ROTATION_SCALE, 1.0)

    def test_rate_limit(self):
        self.assertEqual(configs.MAX_JOINT_VEL_SIM_RAD_S, 3.0)
        self.assertEqual(configs.MAX_JOINT_VEL_HW_RAD_S, 2.0)

    def test_feedback(self):
        self.assertEqual(configs.FEEDBACK_REPEAT_PERIOD_S, 0.25)

    def test_ratchet(self):
        self.assertEqual(configs.RATCHET_LIMIT_GUARD_DEG, 8.0)

    def test_joystick(self):
        self.assertEqual(configs.JOYSTICK_DEADZONE, 0.15)
        self.assertEqual(configs.JOYSTICK_ROLL_RATE_DEG_S, 60.0)
        self.assertEqual(configs.JOYSTICK_FLEX_RATE_DEG_S, 45.0)
        self.assertEqual(configs.JOYSTICK_EXPO, 2.0)
        self.assertEqual(configs.JOYSTICK_ROLL_SIGN, 1.0)
        self.assertEqual(configs.JOYSTICK_FLEX_SIGN, 1.0)

    def test_operator_height(self):
        self.assertEqual(configs.OPERATOR_HEIGHT_M, 1.71)

    def test_derived_dual_constants(self):
        # Composed in Python from the per-arm base lists.
        self.assertEqual(
            configs.NEUTRAL_JOINT_ANGLES_DUAL,
            [0.0, -10.0, 20.0, 25.0, 0.0] * 2,
        )
        self.assertEqual(configs.POSTURE_COST_VECTOR_DUAL, [0.0] * 10)


class TestSharedStrictValidation(unittest.TestCase):
    """load_teleop_shared must reject missing files and bad keys."""

    def _write(self, data: dict) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(data, tmp)
        tmp.close()
        return tmp.name

    def _valid_dict(self) -> dict:
        return {
            "filtering": {"min_cutoff": 0.8, "beta": 5.0, "d_cutoff": 0.9},
            "clutch": {"grip_threshold": 0.9, "orientation_blend_time_s": 1.0},
            "gripper": {"open_max_frac": 0.5},
            "handle": {"pitch_offset_deg": 65.0, "axis": [0.1, 0.2, 0.3]},
            "operator_frame": {"back_m": 0.2, "up_m": 0.2},
            "envelope": {
                "r_min": 0.08,
                "r_max": 0.41,
                "z_floor": 0.01,
                "safety_margin": 0.01,
                "soft_margin": 0.04,
                "oob_mode": "warn",
            },
            "rates": {
                "controller_data": 50.0,
                "ik_solver": 100,
                "visualization": 60.0,
                "robot": 100.0,
                "joint_state_streaming": 100.0,
                "camera_frame_streaming": 30.0,
            },
            "scaling": {"translation_scale": 0.8, "rotation_scale": 1.0},
            "rate_limit": {
                "max_joint_vel_sim_rad_s": 3.0,
                "max_joint_vel_hw_rad_s": 2.0,
            },
            "feedback": {"repeat_period_s": 0.25},
            "ratchet": {"limit_guard_deg": 8.0},
            "joystick": {
                "deadzone": 0.15,
                "roll_rate_deg_s": 60.0,
                "flex_rate_deg_s": 45.0,
                "expo": 2.0,
                "roll_sign": 1.0,
                "flex_sign": 1.0,
            },
            "operator": {"height_m": 1.71},
        }

    def test_valid_dict_loads(self):
        path = self._write(self._valid_dict())
        self.addCleanup(lambda: Path(path).unlink())
        cfg = load_teleop_shared(path)
        self.assertEqual(cfg["scaling"]["translation_scale"], 0.8)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_teleop_shared("/no/such/teleop_shared.yaml")

    def test_unknown_key_raises(self):
        data = self._valid_dict()
        data["scaling"]["bogus"] = 1.0
        path = self._write(data)
        self.addCleanup(lambda: Path(path).unlink())
        with self.assertRaises(ValueError):
            load_teleop_shared(path)

    def test_missing_key_raises(self):
        data = self._valid_dict()
        del data["filtering"]["beta"]
        path = self._write(data)
        self.addCleanup(lambda: Path(path).unlink())
        with self.assertRaises(ValueError):
            load_teleop_shared(path)

    def test_missing_section_raises(self):
        data = self._valid_dict()
        del data["operator"]
        path = self._write(data)
        self.addCleanup(lambda: Path(path).unlink())
        with self.assertRaises(ValueError):
            load_teleop_shared(path)


class TestMethodYamls(unittest.TestCase):
    """Every method has a loadable YAML with exactly the expected keys."""

    def test_every_method_loads(self):
        for name in _ALL_METHODS:
            with self.subTest(method=name):
                params = load_method_params(name)
                self.assertIsInstance(params, dict)
                self.assertTrue(params)  # non-empty

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            load_method_params("does_not_exist")

    def test_armplane_keys(self):
        p = load_method_params("armplane")
        self.assertEqual(p["solver"], "quadprog")
        self.assertEqual(p["position_cost"], 1.0)
        self.assertEqual(p["orientation_cost"], 0.75)
        self.assertEqual(p["ee_orientation_cost_mask"], [1.0, 1.0, 1.0])
        self.assertEqual(p["frame_task_gain"], 0.4)
        self.assertEqual(p["lm_damping"], 0.0)
        self.assertEqual(p["damping_cost"], 0.25)
        self.assertEqual(p["solver_damping_value"], 1e-12)
        self.assertEqual(p["posture_cost_vector"], [0.0] * 10)
        self.assertEqual(p["neutral_joint_angles_deg"], [0.0, -10.0, 20.0, 25.0, 0.0])


class TestMethodValueRegression(unittest.TestCase):
    """Frozen per-method values — must match the previously hardcoded ones so
    the benchmark tables reproduce exactly."""

    def test_pink(self):
        self.assertEqual(load_method_params("pink_full")["orientation_cost"], 0.75)
        self.assertEqual(load_method_params("pink_relaxed")["orientation_cost"], 0.05)
        for name in ("pink_full", "pink_relaxed"):
            p = load_method_params(name)
            self.assertEqual(p["position_cost"], 1.0)
            self.assertEqual(p["frame_task_gain"], 0.4)
            self.assertEqual(p["lm_damping"], 0.0)
            self.assertEqual(p["damping_cost"], 0.25)
            self.assertEqual(p["solver_damping_value"], 1e-12)

    def test_dls(self):
        p = load_method_params("dls")
        self.assertEqual(p["ori_weight"], 0.5)
        self.assertEqual(p["damping"], 0.05)
        self.assertEqual(p["gain"], 0.4)
        self.assertEqual(p["max_joint_vel"], 3.0)

    def test_mink(self):
        p = load_method_params("mink")
        self.assertEqual(p["position_cost"], 1.0)
        self.assertEqual(p["orientation_cost"], 0.75)
        self.assertEqual(p["gain"], 0.4)
        self.assertEqual(p["lm_damping"], 1e-3)
        self.assertEqual(p["posture_cost"], 1e-3)
        self.assertEqual(p["velocity_limit"], 3.0)

    def test_scipy_ls(self):
        p = load_method_params("scipy_ls")
        self.assertEqual(p["ori_weight"], 0.1)
        self.assertEqual(p["max_nfev"], 25)
        self.assertEqual(p["ftol"], 1e-6)
        self.assertEqual(p["xtol"], 1e-6)

    def test_telegrip(self):
        p = load_method_params("telegrip")
        self.assertEqual(p["damping"], 0.05)
        self.assertEqual(p["gain"], 0.6)
        self.assertEqual(p["max_joint_vel"], 3.0)


if __name__ == "__main__":
    unittest.main()
