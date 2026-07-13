"""Utilities for loading teleoperation and IK configuration from YAML files.

Two strictness regimes live here on purpose:

* ``load_ik_config`` is the historical, permissive loader used by the live
  tuning tool (tool/tune_teleop.py): a missing file silently yields ``{}`` and
  the caller falls back to its own per-key defaults.
* ``load_teleop_shared`` / ``load_method_params`` are STRICT: they underpin the
  parameter-YAML migration, where a missing file, a missing key, or an unknown
  key is a configuration bug that must fail loudly (naming the file and key)
  rather than silently reverting to a stale default.
"""

from pathlib import Path

import yaml  # type: ignore[import]

# Directory holding the teleop parameter YAMLs (this file lives in src/common).
_IK_CONF_DIR = Path(__file__).resolve().parent.parent / "ik_conf"
_DEFAULT_SHARED_PATH = _IK_CONF_DIR / "teleop_shared.yaml"
_METHODS_DIR = _IK_CONF_DIR / "methods"

# Frozen schema for teleop_shared.yaml: section -> exact set of allowed keys.
# Kept in lock-step with the YAML and with the constant bindings in
# common/configs.py; test/unit/test_config_yaml.py guards it.
_SHARED_SCHEMA: dict[str, frozenset[str]] = {
    "filtering": frozenset({"min_cutoff", "beta", "d_cutoff"}),
    "clutch": frozenset({"grip_threshold", "orientation_blend_time_s"}),
    "handle": frozenset({"pitch_offset_deg", "axis"}),
    "operator_frame": frozenset({"back_m", "up_m"}),
    "envelope": frozenset(
        {"r_min", "r_max", "z_floor", "safety_margin", "soft_margin", "oob_mode"}
    ),
    "rates": frozenset(
        {
            "controller_data",
            "ik_solver",
            "visualization",
            "robot",
            "joint_state_streaming",
            "camera_frame_streaming",
        }
    ),
    "scaling": frozenset({"translation_scale", "rotation_scale"}),
    "rate_limit": frozenset({"max_joint_vel_sim_rad_s", "max_joint_vel_hw_rad_s"}),
    "operator": frozenset({"height_m"}),
}

# Frozen schema for each methods/<name>.yaml: method -> exact set of allowed
# keys. Every registered benchmark method plus the production "armplane" solver
# must appear here.
_METHOD_SCHEMA: dict[str, frozenset[str]] = {
    "armplane": frozenset(
        {
            "solver",
            "position_cost",
            "orientation_cost",
            "ee_orientation_cost_mask",
            "frame_task_gain",
            "lm_damping",
            "damping_cost",
            "solver_damping_value",
            "posture_cost_vector",
            "neutral_joint_angles_deg",
        }
    ),
    "pink_full": frozenset(
        {
            "position_cost",
            "orientation_cost",
            "frame_task_gain",
            "lm_damping",
            "damping_cost",
            "solver_damping_value",
        }
    ),
    "pink_relaxed": frozenset(
        {
            "position_cost",
            "orientation_cost",
            "frame_task_gain",
            "lm_damping",
            "damping_cost",
            "solver_damping_value",
        }
    ),
    "dls": frozenset({"ori_weight", "damping", "gain", "max_joint_vel"}),
    "mink": frozenset(
        {
            "position_cost",
            "orientation_cost",
            "gain",
            "lm_damping",
            "posture_cost",
            "velocity_limit",
        }
    ),
    "scipy_ls": frozenset({"ori_weight", "max_nfev", "ftol", "xtol"}),
    "telegrip": frozenset({"damping", "gain", "max_joint_vel"}),
}


def load_ik_config(config_path: str) -> dict:
    """Loads the teleoperation and IK parameters from a YAML file.

    Permissive: a missing file yields an empty dict so callers can fall back to
    their own defaults. Used by tool/tune_teleop.py.
    """
    path = Path(config_path)
    if not path.exists():
        print(
            f"⚠️ Warning: Config file not found at {path}. Using safe fallback defaults."
        )
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_yaml_strict(path: Path) -> dict:
    """Load a YAML file, raising a clear error if it is missing or not a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Required teleop config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a top-level mapping, got {type(data).__name__}"
        )
    return data


def _validate_keys(
    path: Path, where: str, present: object, expected: frozenset[str]
) -> None:
    """Raise if ``present`` (a mapping) is not exactly keyed by ``expected``."""
    if not isinstance(present, dict):
        raise ValueError(f"{path}: section '{where}' must be a mapping")
    keys = set(present)
    missing = expected - keys
    unknown = keys - expected
    if missing:
        raise ValueError(
            f"{path}: {where} is missing required key(s): {sorted(missing)}"
        )
    if unknown:
        raise ValueError(f"{path}: {where} has unknown key(s): {sorted(unknown)}")


def load_teleop_shared(path: str | None = None) -> dict:
    """Load and strictly validate the shared teleop parameters.

    Validates that the top-level sections and every key within each section
    exactly match the frozen schema; any missing file, missing key, or unknown
    key raises a clear error naming the file and the offending key. Unlike
    ``load_ik_config`` there is NO silent fallback.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_SHARED_PATH
    data = _load_yaml_strict(cfg_path)
    _validate_keys(cfg_path, "top-level", data, frozenset(_SHARED_SCHEMA))
    for section, expected in _SHARED_SCHEMA.items():
        _validate_keys(cfg_path, section, data[section], expected)
    return data


def load_method_params(method_name: str) -> dict:
    """Load and strictly validate the parameters for a single IK method.

    Reads src/ik_conf/methods/<method_name>.yaml and validates its keys against
    the frozen per-method schema. Same strictness as ``load_teleop_shared``.
    """
    if method_name not in _METHOD_SCHEMA:
        raise ValueError(
            f"Unknown IK method {method_name!r} " f"(known: {sorted(_METHOD_SCHEMA)})"
        )
    cfg_path = _METHODS_DIR / f"{method_name}.yaml"
    data = _load_yaml_strict(cfg_path)
    _validate_keys(cfg_path, method_name, data, _METHOD_SCHEMA[method_name])
    return data
