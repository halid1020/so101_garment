Testing & Architectural Tracking

## Untested Functionality (Requires Physical Hardware)
1. **Serial Bus Latency (`sts3215_bus.py`)**: Cannot guarantee 1,000,000 baud rate stability over extended dual-arm teleoperation sessions without physical SO-101 units.
2. **Meta Quest Input (`quest_reader.py`)**: Cannot verify UDP/TCP data packet drops from the Meta Quest headset.
3. **IK Divergence (`ik_solver.py`)**: Pink IK solving times and singularity handling near joint limits cannot be safely validated without physical execution to ensure the robot does not self-collide.

## Architectural Debt & Redundancy (Targeted for Refactor)
1. **Duplicate Controllers:** `so101_dual_arm.py` and `common/so101_dual_controller.py` implement overlapping logic for dual-arm orchestration. `so101_dual_arm.py` relies on `lerobot.motors` while the `common/` modules rely on a custom `sts3215_bus.py`. We must standardize on one driver.
2. **Configuration Sprawl:** `meta_quest_teleopration.py` hardcodes configurations at the top of the file, while `so101_dual_arm.py` loads from `conf/robot.yaml`.
