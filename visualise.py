"""
visualise.py
============
Load a trained PPO model and watch it track trajectories.

Usage:
    python visualise.py                         # best model, circle
    python visualise.py --traj figure8          # different trajectory
    python visualise.py --model models/ppo_circle_final  # specific model
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from franka_env import FrankaTrackingEnv, TRAJECTORIES


def visualise(
    model_path: str = None,
    traj_name:  str = "circle",
    episodes:   int = 5,
):
    # Find model
    if model_path is None:
        candidates = [
            f"models/best_model.zip",
            f"models/ppo_{traj_name}_final.zip",
            f"models/best_model",
            f"models/ppo_{traj_name}_final",
        ]
        for c in candidates:
            if os.path.exists(c) or os.path.exists(c + ".zip"):
                model_path = c
                break

    if model_path is None:
        print("No trained model found. Train first:\n  python train.py")
        return

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = FrankaTrackingEnv(trajectory=traj_name, render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        done   = False
        total_r = 0
        dists   = []
        step    = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            dists.append(info["distance"])
            done = terminated or truncated
            step += 1

        print(f"  Episode {ep+1}: "
              f"reward={total_r:.1f}  "
              f"mean_dist={np.mean(dists)*100:.1f}cm  "
              f"min_dist={np.min(dists)*100:.1f}cm  "
              f"steps={step}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,    help="Path to model zip")
    parser.add_argument("--traj",  default="circle",choices=list(TRAJECTORIES.keys()))
    parser.add_argument("--eps",   type=int, default=5, help="Episodes to run")
    args = parser.parse_args()

    visualise(model_path=args.model, traj_name=args.traj, episodes=args.eps)