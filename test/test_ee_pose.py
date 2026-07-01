import yaml

from so101_garment.src.so101_dual_arm import SO101DualArm


# 1. Create a dummy IK solver for testing purposes
class DummyIKSolver:
    """
    A placeholder class. In production, replace the 'compute_ik' method
    with a library like Pinocchio, PyBullet, or a custom Jacobian solver.
    """

    def compute_ik(self, ee_pose):
        # Simply returns a hardcoded joint state for testing communication
        return {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 45.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 20.0,
        }


def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load configuration
    config = {
        "robot": load_yaml("conf/robot.yaml"),
        "rest_pos": load_yaml("conf/rest_pos.yaml"),
        "mid_pos": load_yaml("conf/mid_pos.yaml"),
    }

    # Initialize
    dual_arm = SO101DualArm(config)
    ik_solver = DummyIKSolver()

    # Define dummy Cartesian poses
    # Note: These values are arbitrary; your real IK solver will dictate the units.
    test_pose = {"x": 0.2, "y": 0.1, "z": 0.3}

    print("Testing Cartesian End-Effector movement...")
    try:
        # Move both arms to the calculated IK position
        dual_arm.move_to_ee_pose(
            test_pose, test_pose, duration=2.0, ik_solver=ik_solver
        )
        print("Movement command sent successfully.")

    except Exception as e:
        print(f"Error during EE pose test: {e}")
    finally:
        dual_arm.disable_torque()


if __name__ == "__main__":
    main()
