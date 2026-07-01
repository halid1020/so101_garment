#!/usr/bin/env python3
"""
Test script for locally training a VLA model using LeRobot.
Designed for high-frequency tactile and visual datasets (e.g., cloth manipulation).
"""

import os

# SURGICAL CHANGE: Updated to target the pi05 policy configuration per LeRobot specs
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.train import train


def main():
    print("Initiating LeRobot VLA Local Training Sequence...")

    # Define the PI05 configuration
    # Optimized for precise, high-dimensional dexterous tasks
    config = PI05Config(
        dataset_repo_id="lerobot/aloha_sim_insertion_human",  # Replace with local cloth/tactile dataset
        training_steps=50000,
        batch_size=8,
        learning_rate=1e-5,
        vision_backbone="resnet18",
        chunk_size=100,  # Receding horizon chunk size
        n_action_steps=100,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )

    try:
        # Execute the LeRobot training pipeline
        train(config)
        print("✓ Training completed successfully.")
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
