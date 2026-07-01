#!/usr/bin/env python3
"""
End-to-End Simulation Pipeline for PI05.
Orchestrates training and evaluation entirely within native Python, without CLI wrappers.
"""

from pathlib import Path

# Evaluation Imports
import torch

# LeRobot Dataset & Feature Imports
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.factory import make_env

# PI05 Specific Imports
from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader

# --- Configuration ---
SIM_ENV = "gym_aloha"
DATASET_NAME = "lerobot/aloha_sim_insertion_scripted"
OUTPUT_DIR = Path("outputs/train/pi05_sim_test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pi05():
    """Trains the PI05 policy natively using a manual PyTorch training loop."""
    print("==========================================")
    print(f"1. Training PI05 Policy on {DEVICE}")
    print("==========================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_steps = 10000
    log_freq = 100

    # 1. Load Dataset Metadata & Configure Features
    dataset_metadata = LeRobotDatasetMetadata(DATASET_NAME)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    # 2. Configure PI05 Policy
    cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        n_action_steps=100,
    )

    policy = PI05Policy(cfg)
    policy.train()
    policy.to(DEVICE)

    # Use the dedicated PI05 processor pipeline
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    # 3. Define Temporal Delta Indices (Frame Chunking)
    # PI05 returns None for observation_delta_indices, so we only map actions
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # 4. Initialize Dataset and Dataloader
    dataset = LeRobotDataset(DATASET_NAME, delta_timestamps=delta_timestamps)

    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True,
        pin_memory=DEVICE.type != "cpu",
        drop_last=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

    # 5. Native Training Loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # Shift tensors to target device
            batch = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch = preprocessor(batch)

            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} | Loss: {loss.item():.4f}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # Save artifacts
    policy.save_pretrained(OUTPUT_DIR)
    preprocessor.save_pretrained(OUTPUT_DIR)
    postprocessor.save_pretrained(OUTPUT_DIR)
    print("✓ PI05 Training completed.")


def evaluate_in_sim():
    """Evaluates the trained PI05 policy directly in the simulation environment."""
    print("\n==========================================")
    print("2. Evaluating Policy in Simulation")
    print("==========================================")

    # 1. Load the trained policy
    policy = PI05Policy.from_pretrained(OUTPUT_DIR)
    policy.eval()
    policy.to(DEVICE)

    # 2. Instantiate the environment natively
    env = make_env(SIM_ENV)

    n_episodes = 10
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        # Reset policy internal state for the new episode (receding horizon)
        policy.reset()

        while not done and step < 1000:  # 1000-step safeguard
            # Format observation for the model (add batch dimension)
            policy_obs = {
                k: torch.tensor(v).unsqueeze(0).to(DEVICE) for k, v in obs.items()
            }

            with torch.no_grad():
                action = policy.select_action(policy_obs)

            # Squeeze batch dimension to feed into the environment
            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step += 1

        print(
            f"Episode {ep + 1}/{n_episodes} completed in {step} steps. Reward: {reward}"
        )

    env.close()
    print("✓ Simulation Evaluation completed.")


if __name__ == "__main__":
    try:
        # Step 1: Train
        train_pi05()

        # Step 2: Evaluate
        evaluate_in_sim()

        print("\n✓ Full native simulation pipeline executed successfully.")
    except Exception as e:
        print(f"\n❌ Pipeline halted due to error: {e}")
