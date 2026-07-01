import time

from send_position import SO101DualArm

# Assuming you save the class from the file above as 'so101_dual_arm.py'
# you could just do: `from so101_dual_arm import SO101DualArm`
# But for completeness, I will assume the class is accessible here.


# --- CONFIGURATION VARIABLES ---
PORT_ID_0 = "/dev/ttyACM0"
PORT_ID_1 = "/dev/ttyACM1"
ROBOT_NAME_0 = "follower_0"
ROBOT_NAME_1 = "follower_1"


def main():
    print("Initializing Dual Arm Reader...")
    dual_arm = SO101DualArm(PORT_ID_0, PORT_ID_1, ROBOT_NAME_0, ROBOT_NAME_1)

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
