# Vision-Tactile World Model with Dual SO-101 Garment Manipulation

This repository provides an independent, industrial-grade pipeline for dexterous garment manipulation using dual SO-101 robotic arms equipped with tactile sensors. The framework facilitates Meta-Quest teleoperation, autonomous data collection, and direct integration with Vision-Language-Action (VLA) models via the LeRobot ecosystem.

**Author:** Abudureyimu Halite
**Framework Status:** Active Development
**Dependencies:** LeRobot 0.5.1, Pink IK, pyrealsense2

## 🚀 System Architecture

1. **Hardware Interface:** Direct `sts3215` serial bus control utilizing `lerobot.motors` for SO-101 leader/follower synchronization.
2. **Teleoperation:** Meta-Quest VR tracking mapped to dual-arm Cartesian poses via Pink IK, filtered through adaptive 1€ smoothing.
3. **Data Pipeline:** Synchronized, offline collection of RGB feeds, joint states, and tactile data formatted natively for Hugging Face LeRobot dataset standards.

## ⚙️ Installation

1. Clone the repository and navigate to the root directory.
2. Create and activate a Python 3.10+ virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install 'lerobot[dataset,pi]==0.5.1'
   pip install pre-commit
   pre-commit install
