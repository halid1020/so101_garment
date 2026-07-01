# AI Agent Interaction Guidelines

This document provides strict operational constraints and architectural context for any AI agent modifying this codebase.

## 🏗️ Directory Structure
```text
.
├── src/
│   ├── common/                 # Shared teleoperation state, threading, and utilities
│   ├── conf/                   # Hardware mappings and default poses
│   ├── ik_conf/                # Inverse Kinematics tuning parameters
│   ├── calibration_files/      # Feetech servo JSON calibrations
│   └── so101_dual_arm.py       # Primary LeRobot hardware wrapper
├── tool/                       # Execution scripts (teleop, data reading, VLA training)
├── test/                       # Isolated component testing
├── markdowns/                  # Project tracking and agent instructions
├── install.sh                  # Automated environment setup
└── setup.sh                    # PYTHONPATH configuration for tools
```

## General Instruction

* Keep industrial-level documentation.

* Reduce redundancy between files.

* Keep good level of modularisation.

* Be surgical of your changes.

* Do not make changes if it is unnecessary

* Keep an industrial-level README file for installation and running.

* Keep an industrial-level testing for all functionality and report the ones that you cannot deal with in the Problem.md file.

* Keep the project files organised.
