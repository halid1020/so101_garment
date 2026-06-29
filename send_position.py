from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from pathlib import Path
import draccus
import time

# --- CONFIGURATION VARIABLES ---
PORT_ID_0 = "/dev/ttyACM0"
PORT_ID_1 = "/dev/ttyACM1"  
ROBOT_NAME_0 = "follower_0" 
ROBOT_NAME_1 = "follower_1" 

def load_calibration(ROBOT_NAME) -> None:
    """
    Helper to load calibration data from the specified file.
    """
    fpath = Path(f'calibration_files/{ROBOT_NAME}.json')
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
        return calibration
    
def move_to_pose_dual(bus_0, bus_1, desired_pos_0, desired_pos_1, duration):
    """
    Moves two robot arms simultaneously to their desired positions.
    """
    start_time = time.time()
    starting_pose_0 = bus_0.sync_read("Present_Position")
    starting_pose_1 = bus_1.sync_read("Present_Position")
    
    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Interpolation factor [0,1]
        alpha = min(t / duration, 1)

        # Interpolate each joint for Arm 0
        position_dict_0 = {}
        for joint in desired_pos_0:
            p0 = starting_pose_0[joint]
            pf = desired_pos_0[joint]
            position_dict_0[joint] = (1 - alpha) * p0 + alpha * pf

        # Interpolate each joint for Arm 1
        position_dict_1 = {}
        for joint in desired_pos_1:
            p0 = starting_pose_1[joint]
            pf = desired_pos_1[joint]
            position_dict_1[joint] = (1 - alpha) * p0 + alpha * pf

        # Send commands to both buses
        bus_0.sync_write("Goal_Position", position_dict_0, normalize=True)
        bus_1.sync_write("Goal_Position", position_dict_1, normalize=True)

        time.sleep(0.02)  # 50 Hz loop
    
def hold_position(duration):
    """
    Pauses the script while the arms hold their current position.
    """
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        time.sleep(0.02)  # 50 Hz loop

def setup_motors(calibration, PORT_ID):
    norm_mode_body = MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
                port=PORT_ID,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                    "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                    "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                    "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                    "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=calibration,
            )
    bus.connect(True)

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32) 
    return bus


# --- SPECIFIED PARAMETERS ---
# You can define separate dictionaries if you want the arms to do different things,
# but for now, we will use the same target variables for both.
rest_position = {
    'shoulder_pan': -6.02,  
    'shoulder_lift': -110.68,
    'elbow_flex': 96.53,
    'wrist_flex': 75.03,
    'wrist_roll': -0.48,
    'gripper': 1.39
}

desired_position = {
    'shoulder_pan': 0.0,  
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 10.0,
    'gripper': 0.0
}

move_time = 2.0  # seconds to reach desired position
hold_time = 2.0  # total time to hold


# --- MAIN EXECUTION ---
def main():
    print("Loading calibrations...")
    calib_0 = load_calibration(ROBOT_NAME_0)
    calib_1 = load_calibration(ROBOT_NAME_1)

    print("Setting up motor buses...")
    bus_0 = setup_motors(calib_0, PORT_ID_0)
    bus_1 = setup_motors(calib_1, PORT_ID_1)

    # Note: Torque is automatically enabled after setup_motors finishes.
    # Do NOT disable torque here, otherwise the arms cannot move to the rest position!

    print("Moving to rest position...")
    move_to_pose_dual(bus_0, bus_1, rest_position, rest_position, move_time)
    hold_position(hold_time)

    print("Moving to desired position...")
    move_to_pose_dual(bus_0, bus_1, desired_position, desired_position, move_time)
    hold_position(hold_time)

    print("Returning to rest position...")
    move_to_pose_dual(bus_0, bus_1, rest_position, rest_position, move_time)
    
    print("Sequence complete. Disabling torque...")
    bus_0.disable_torque()
    bus_1.disable_torque()

if __name__ == "__main__":
    main()