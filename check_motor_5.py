from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor
import sys

# Dynamically grab the enum from the same module where Motor lives
# This avoids needing to guess the exact import path
MotorNormMode = sys.modules[Motor.__module__].MotorNormMode

# Use the port you confirmed
port = "/dev/ttyACM0"

# 1. Define the motor configuration with the newly required 'norm_mode'
motors_config = {
    "wrist_roll": Motor(id=5, model="sts3215", norm_mode=MotorNormMode.DEGREES) 
}

# 2. Initialize the bus
bus = FeetechMotorsBus(port=port, motors=motors_config)

try:
    # Open the serial port
    bus.connect(handshake=False)
    
    # Explicitly set baudrate to 1,000,000 for SO-101 motors
    bus.set_baudrate(1_000_000)

    # 3. Read the raw position. 
    # normalize=False is crucial here because we haven't loaded calibration files!
    pos = bus.read("Present_Position", "wrist_roll", normalize=False)
    
    print(f"\n✅ Success! Wrist Roll (ID 5) is communicating.")
    print(f"Raw position value: {pos}\n")
    
except Exception as e:
    print(f"\n❌ Failed to communicate with ID 5: {e}\n")
finally:
    # Use the proper disconnect method for the new class
    if bus.is_connected:
        bus.disconnect()