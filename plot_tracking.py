"""
plot_tracking.py
================
Load a trained model and plot:
  1. EE position vs target trajectory (3D and XYZ over time)
  2. Tracking error over time

Usage:
    python plot_tracking.py --traj circle
    python plot_tracking.py --traj figure8
    python plot_tracking.py --traj random
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from franka_env import FrankaTrackingEnv, TRAJECTORIES


def run_episode(model, traj_name, deterministic=True):
    env = FrankaTrackingEnv(trajectory=traj_name, render_mode=None, max_steps=500)
    obs, _ = env.reset(seed=42)

    ee_positions     = []
    target_positions = []
    errors           = []
    times            = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract from obs
        ee_pos     = obs[:3].copy()
        target_pos = obs[6:9].copy()

        ee_positions.append(ee_pos)
        target_positions.append(target_pos)
        errors.append(info["distance"])
        times.append(info["time"])

        done = terminated or truncated

    env.close()

    return (
        np.array(ee_positions),
        np.array(target_positions),
        np.array(errors),
        np.array(times),
    )


def plot(traj_name, model_path=None):
    # ── Find model ──
    if model_path is None:
        for path in [
            f"models/best_model",
            f"models/ppo_{traj_name}_final",
            f"models/best_model.zip",
            f"models/ppo_{traj_name}_final.zip",
        ]:
            if os.path.exists(path) or os.path.exists(path + ".zip"):
                model_path = path
                break

    if model_path is None:
        print(f"No model found for {traj_name}. Train first.")
        return

    print(f"Loading: {model_path}")
    model = PPO.load(model_path)

    ee, target, errors, times = run_episode(model, traj_name)

    print(f"  Steps:       {len(errors)}")
    print(f"  Mean error:  {np.mean(errors)*100:.1f}cm")
    print(f"  Min error:   {np.min(errors)*100:.1f}cm")
    print(f"  Max error:   {np.max(errors)*100:.1f}cm")
    print(f"  Final error: {errors[-1]*100:.1f}cm")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Franka Trajectory Tracking — {traj_name.capitalize()}",
                 fontsize=14, fontweight="bold")

    # ── Plot 1: 3D trajectory ──
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(target[:, 0], target[:, 1], target[:, 2],
             "r--", linewidth=2, label="Target", alpha=0.8)
    ax1.plot(ee[:, 0], ee[:, 1], ee[:, 2],
             "b-", linewidth=1.5, label="EE", alpha=0.9)
    ax1.scatter(*ee[0], color="green", s=60, zorder=5, label="Start")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8)

    # ── Plot 2: X over time ──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(times, target[:, 0], "r--", label="Target X", linewidth=1.5)
    ax2.plot(times, ee[:, 0],     "b-",  label="EE X",     linewidth=1.2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X (m)")
    ax2.set_title("X Axis Tracking")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Y over time ──
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, target[:, 1], "r--", label="Target Y", linewidth=1.5)
    ax3.plot(times, ee[:, 1],     "b-",  label="EE Y",     linewidth=1.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Y Axis Tracking")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Z over time ──
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(times, target[:, 2], "r--", label="Target Z", linewidth=1.5)
    ax4.plot(times, ee[:, 2],     "b-",  label="EE Z",     linewidth=1.2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Z (m)")
    ax4.set_title("Z Axis Tracking")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Tracking error over time ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, errors * 100, "purple", linewidth=1.2)
    ax5.axhline(y=np.mean(errors) * 100, color="orange", linestyle="--",
                label=f"Mean: {np.mean(errors)*100:.1f}cm")
    ax5.axhline(y=2.0, color="green", linestyle=":", alpha=0.7,
                label="2cm threshold")
    ax5.fill_between(times, errors * 100, alpha=0.2, color="purple")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Error (cm)")
    ax5.set_title("Tracking Error over Time")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Error histogram ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(errors * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax6.axvline(x=np.mean(errors) * 100, color="orange", linestyle="--",
                label=f"Mean: {np.mean(errors)*100:.1f}cm")
    ax6.axvline(x=2.0, color="green", linestyle=":", label="2cm target")
    ax6.set_xlabel("Error (cm)")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Error Distribution")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    out = f"results/tracking_{traj_name}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",  default="circle", choices=list(TRAJECTORIES.keys()))
    parser.add_argument("--model", default=None)
    parser.add_argument("--all",   action="store_true", help="Plot all trajectories")
    args = parser.parse_args()

    if args.all:
        for traj in TRAJECTORIES:
            try:
                plot(traj)
            except Exception as e:
                print(f"Skipping {traj}: {e}")
    else:
        plot(traj_name=args.traj, model_path=args.model)