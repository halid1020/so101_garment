import sys
from pathlib import Path

import yaml

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from src.so101_dual_arm import SO101DualArm


def load_yaml(filepath):
    """Helper function to load a YAML file as a dictionary."""
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main():
    # 1. Build the comprehensive config dictionary
    # This reads all three of your config files into one central state
    config = {
        "robot": load_yaml(_root / "src/conf/robot.yaml"),
        "rest_pos": load_yaml(_root / "src/conf/rest_pos.yaml"),
        "mid_pos": load_yaml(_root / "src/conf/mid_pos.yaml"),
    }

    # 2. Initialize the dual arm class with the unified config
    print("Initializing Dual Arms...")
    dual_arm = SO101DualArm(config)

    # Configuration for movement
    move_time = 2.0
    hold_time = 2.0

    # 3. Execute movement sequence using the new class methods!
    dual_arm.send_to_rest(move_time)
    dual_arm.hold_position(hold_time)

    dual_arm.send_to_middle(move_time)
    dual_arm.hold_position(hold_time)

    dual_arm.send_to_rest(move_time)

    # 4. Finish
    print("Sequence complete.")
    dual_arm.disable_torque()


if __name__ == "__main__":
    main()
