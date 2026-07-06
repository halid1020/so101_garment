import sys
import time
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

    # Disable torque so you can move the motors freely by hand
    dual_arm.disable_torque()

    print("\nTorque disabled! You can now physically move both robot arms.")
    print("Reading joint positions... (Press Ctrl+C to stop)\n")

    try:
        while True:
            # Read the current position of all joints from our class
            pos_0, pos_1 = dual_arm.read_positions()

            # Format the dictionary to round the numbers to 2 decimal places
            fmt_0 = {k: round(v, 2) for k, v in pos_0.items()}
            fmt_1 = {k: round(v, 2) for k, v in pos_1.items()}

            # Print side-by-side cleanly in the terminal
            print(f"ARM 0: {fmt_0} | ARM 1: {fmt_1}", end="\r", flush=True)

            # Sleep to prevent overloading the serial bus (10Hz loop)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped reading joint angles. Exiting...")


if __name__ == "__main__":
    main()
