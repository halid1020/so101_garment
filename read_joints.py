from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from pathlib import Path
import draccus
import time

# CONFIGURATION VARIABLES
PORT_ID = "/dev/ttyACM0" # REPLACE WITH YOUR PORT! 
ROBOT_NAME = "my_awesome_follower_arm" # REPLACE WITH YOUR ROBOT NAME! 

def load_calibration(ROBOT_NAME) -> None:
    """
    Helper to load calibration data from the specified file.
    """
    fpath = Path(f'calibration_files/{ROBOT_NAME}.json')
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
        return calibration
    
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

def main():
    print(f"Loading calibration for {ROBOT_NAME}...")
    calibration = load_calibration(ROBOT_NAME)
    
    print(f"Connecting to motors on {PORT_ID}...")
    bus = setup_motors(calibration, PORT_ID)

    # Disable torque so that you can move the motors freely by hand
    bus.disable_torque()
    
    print("\nTorque disabled! You can now physically move the robot arm.")
    print("Reading joint positions... (Press Ctrl+C to stop)\n")

    try:
        while True:
            # Read the current position of all joints
            present_pos = bus.sync_read("Present_Position")
            
            # Format the dictionary to round the numbers to 2 decimal places for cleaner terminal output
            formatted_pos = {joint: round(angle, 2) for joint, angle in present_pos.items()}
            
            # Use end='\r' to overwrite the same line in the terminal instead of printing a massive list
            print(f"Current Angles: {formatted_pos}      ", end="\r", flush=True)
            
            # Sleep to prevent overloading the serial bus and the CPU (10Hz loop)
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        # Gracefully handle the user pressing Ctrl+C
        print("\n\nStopped reading joint angles. Exiting...")
        
if __name__ == "__main__":
    main()