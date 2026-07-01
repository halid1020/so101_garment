import time
from pathlib import Path
from typing import Any, Dict, Tuple

import draccus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


class SO101DualArm:
    """
    Hardware interface for dual SO-101 robotic arms.
    Provides standard joint-space control, normalized teleoperation control,
    and end-effector Cartesian control wrappers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes both arms based on a single configuration dictionary.

        Args:
            config (Dict): Comprehensive YAML-loaded configuration dictionary.
        """
        # 1. Extract robot connection settings
        robot_conf = config.get("robot", {})
        port_0 = robot_conf.get("PORT_ID_0")
        port_1 = robot_conf.get("PORT_ID_1")
        name_0 = robot_conf.get("ROBOT_NAME_0")
        name_1 = robot_conf.get("ROBOT_NAME_1")

        # Extracted limits for normalized mapping (assumes a 'joint_limits_deg' section in robot.yaml)
        self.limits = robot_conf.get("joint_limits_deg", {})

        # 2. Extract and store predefined positions
        self.rest_pos = config.get("rest_pos", {})
        self.mid_pos = config.get("mid_pos", {})

        print("Loading calibrations...")
        self.calib_0 = self._load_calibration(name_0)
        self.calib_1 = self._load_calibration(name_1)

        print("Setting up motor buses...")
        self.bus_0 = self._setup_motors(self.calib_0, port_0)
        self.bus_1 = self._setup_motors(self.calib_1, port_1)

    def _load_calibration(self, robot_name: str) -> MotorCalibration:
        fpath = Path(f"calibration_files/{robot_name}.json")
        with open(fpath) as f, draccus.config_type("json"):
            return draccus.load(Dict[str, MotorCalibration], f)

    def _setup_motors(
        self, calibration: MotorCalibration, port_id: str
    ) -> FeetechMotorsBus:
        bus = FeetechMotorsBus(
            port=port_id,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        bus.connect(True)

        with bus.torque_disabled():
            bus.configure_motors()
            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                bus.write("P_Coefficient", motor, 16)
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 32)
        return bus

    def move_to_joint_pose(
        self,
        desired_pos_0: Dict[str, float],
        desired_pos_1: Dict[str, float],
        duration: float,
    ) -> None:
        """
        Interpolates and moves both arms synchronously to target joint angles over a set duration.

        Args:
            desired_pos_0: Target degrees/ranges for Arm 0.
            desired_pos_1: Target degrees/ranges for Arm 1.
            duration: Time in seconds to complete the motion.
        """
        start_time = time.time()
        starting_pose_0 = self.bus_0.sync_read("Present_Position")
        starting_pose_1 = self.bus_1.sync_read("Present_Position")

        while True:
            t = time.time() - start_time
            if t > duration:
                break

            alpha = min(t / duration, 1)

            position_dict_0 = {
                j: (1 - alpha) * starting_pose_0[j] + alpha * desired_pos_0[j]
                for j in desired_pos_0
            }
            position_dict_1 = {
                j: (1 - alpha) * starting_pose_1[j] + alpha * desired_pos_1[j]
                for j in desired_pos_1
            }

            self.bus_0.sync_write("Goal_Position", position_dict_0, normalize=True)
            self.bus_1.sync_write("Goal_Position", position_dict_1, normalize=True)

            time.sleep(0.02)

    def move_to_norm_pose(
        self,
        norm_pos_0: Dict[str, float],
        norm_pos_1: Dict[str, float],
        duration: float,
    ) -> None:
        """
        Moves the arms based on normalized inputs [-1.0, 1.0].
        Designed for raw analog inputs from devices like Meta-Quest controllers.
        Requires 'joint_limits_deg' to be defined in robot.yaml.
        """
        if not self.limits:
            raise ValueError(
                "Cannot move to normalized pose: 'joint_limits_deg' missing from robot config."
            )

        def map_norm_to_deg(norm_dict: Dict[str, float]) -> Dict[str, float]:
            deg_dict = {}
            for joint, norm_val in norm_dict.items():
                min_val, max_val = self.limits[joint]
                # Map [-1, 1] to [min_val, max_val]
                deg_dict[joint] = min_val + ((norm_val + 1.0) / 2.0) * (
                    max_val - min_val
                )
            return deg_dict

        target_deg_0 = map_norm_to_deg(norm_pos_0)
        target_deg_1 = map_norm_to_deg(norm_pos_1)

        self.move_to_joint_pose(target_deg_0, target_deg_1, duration)

    def move_to_ee_pose(
        self,
        ee_pose_0: Dict[str, float],
        ee_pose_1: Dict[str, float],
        duration: float,
        ik_solver: Any,
    ) -> None:
        """
        Moves both arms to a Cartesian End-Effector (EE) pose relative to the robot base.

        Args:
            ee_pose_0: Target x, y, z, rx, ry, rz for Arm 0.
            ee_pose_1: Target x, y, z, rx, ry, rz for Arm 1.
            duration: Time in seconds for the motion.
            ik_solver: An external inverse kinematics solver instance (e.g., Pinocchio/PyBullet wrapper)
                       that contains a `compute_ik(ee_pose)` method.
        """
        if ik_solver is None:
            raise NotImplementedError(
                "An external IK solver is required to map Cartesian EE poses to joint angles."
            )

        # Resolve Cartesian poses to Joint Angles via the external IK solver
        target_joints_0 = ik_solver.compute_ik(ee_pose_0)
        target_joints_1 = ik_solver.compute_ik(ee_pose_1)

        self.move_to_joint_pose(target_joints_0, target_joints_1, duration)

    # --- ROUTINE HELPER FUNCTIONS ---
    def send_to_rest(self, duration: float = 2.0) -> None:
        """Moves both arms to the rest position loaded from config."""
        print("Moving to rest position...")
        if not self.rest_pos:
            print("Warning: No rest_pos found in config!")
            return
        self.move_to_joint_pose(self.rest_pos, self.rest_pos, duration)

    def send_to_middle(self, duration: float = 2.0) -> None:
        """Moves both arms to the middle position loaded from config."""
        print("Moving to middle position...")
        if not self.mid_pos:
            print("Warning: No mid_pos found in config!")
            return
        self.move_to_joint_pose(self.mid_pos, self.mid_pos, duration)

    # --- STATE & HARDWARE LIFECYCLE ---
    def read_positions(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns the current joint positions of both arms."""
        return (
            self.bus_0.sync_read("Present_Position"),
            self.bus_1.sync_read("Present_Position"),
        )

    def hold_position(self, duration: float) -> None:
        """Pauses execution while actively holding the current pose."""
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.02)

    def disable_torque(self) -> None:
        """Relaxes both arms, dropping hardware torque."""
        print("Disabling torque...")
        self.bus_0.disable_torque()
        self.bus_1.disable_torque()
