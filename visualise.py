# =============================================================
#   VISUALISER
#   Watch trained agent track trajectories
#   Also generates evaluation plots
# =============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from trajectories import make_trajectory
from tracking_env import TrackingEnv


def load_model(traj_name, model_dir="models"):
    """Load trained model with normalisation"""
    model_path = f"{model_dir}/{traj_name}/best/best_model"
    norm_path  = f"{model_dir}/{traj_name}/vec_normalize.pkl"

    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}.zip")
        print(f"Train first: python train.py --traj {traj_name}")
        return None, None

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    return model, norm_path


def run_episode(model, norm_path, traj_name,
                render=True, max_steps=1000):
    """
    Run one episode and collect tracking data.
    Returns arrays of positions and errors for plotting.
    """
    traj = make_trajectory(traj_name)

    env = DummyVecEnv([lambda: TrackingEnv(
        trajectory  = traj,
        render_mode = "human" if render else None
    )])

    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training    = False
        env.norm_reward = False

    obs = env.reset()

    # Storage for plotting
    ee_positions  = []  # where arm tip was
    tgt_positions = []  # where target was
    errors        = []  # distance between them
    times         = []  # time at each step

    t = 0.0
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Extract positions from info
        if len(info) > 0 and "tracking_error" in info[0]:
            errors.append(info[0]["tracking_error"])
            t += 0.05
            times.append(t)

            # Get actual positions
            base_env = env.envs[0].unwrapped
            ee_pos   = base_env.data.xpos[3].copy()
            tgt_pos  = traj.get_position(t)

            ee_positions.append(ee_pos[:2])   # x, y only
            tgt_positions.append(tgt_pos[:2])

        if done[0]:
            break

    env.close()

    return (np.array(times),
            np.array(ee_positions),
            np.array(tgt_positions),
            np.array(errors))


def plot_results(times, ee_pos, tgt_pos, errors,
                 traj_name, save_path="results"):
    """
    Generate evaluation plots:
    1. Tracking error over time
    2. XY trajectory comparison (where arm went vs target)
    3. Error distribution histogram
    """
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Tracking Performance: {traj_name.replace('_', ' ').title()}",
        fontsize=16
    )

    # ----------------------------------------------------------
    # Plot 1: Tracking error over time
    # Shows how well agent tracks throughout episode
    # ----------------------------------------------------------
    ax = axes[0]
    ax.plot(times, errors * 100, 'b-', linewidth=1.5,
            label='Tracking error')
    ax.axhline(y=1.0, color='g', linestyle='--',
               label='1cm threshold')
    ax.axhline(y=5.0, color='orange', linestyle='--',
               label='5cm threshold')
    ax.fill_between(times, errors * 100,
                    alpha=0.2, color='blue')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Tracking Error (cm)', fontsize=12)
    ax.set_title('Error Over Time', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics box
    mean_e = np.mean(errors) * 100
    max_e  = np.max(errors) * 100
    p1cm   = np.mean(errors < 0.01) * 100
    p5cm   = np.mean(errors < 0.05) * 100

    stats_text = (f"Mean: {mean_e:.1f}cm\n"
                  f"Max:  {max_e:.1f}cm\n"
                  f"<1cm: {p1cm:.0f}%\n"
                  f"<5cm: {p5cm:.0f}%")
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat',
                      alpha=0.5),
            fontsize=10)

    # ----------------------------------------------------------
    # Plot 2: XY trajectory comparison
    # Shows actual path vs target path
    # ----------------------------------------------------------
    ax = axes[1]
    if len(tgt_pos) > 0 and len(ee_pos) > 0:
        ax.plot(tgt_pos[:, 0], tgt_pos[:, 1],
                'g--', linewidth=2, label='Target', alpha=0.8)
        ax.plot(ee_pos[:, 0], ee_pos[:, 1],
                'b-', linewidth=1.5, label='Arm tip', alpha=0.8)
        ax.plot(tgt_pos[0, 0], tgt_pos[0, 1],
                'go', markersize=10, label='Start')

    ax.set_xlabel('X (metres)', fontsize=12)
    ax.set_ylabel('Y (metres)', fontsize=12)
    ax.set_title('Trajectory Comparison', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # ----------------------------------------------------------
    # Plot 3: Error distribution
    # Shows spread of errors - tight = consistent tracking
    # ----------------------------------------------------------
    ax = axes[2]
    ax.hist(errors * 100, bins=50, color='blue',
            alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='g', linestyle='--',
               label='1cm', linewidth=2)
    ax.axvline(x=5.0, color='orange', linestyle='--',
               label='5cm', linewidth=2)
    ax.axvline(x=np.mean(errors)*100, color='red',
               linestyle='-', label=f'Mean={mean_e:.1f}cm',
               linewidth=2)
    ax.set_xlabel('Error (cm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_file = f"{save_path}/{traj_name}_results.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved: {save_file}")

    return {
        "mean_error_cm": mean_e,
        "max_error_cm":  max_e,
        "pct_under_1cm": p1cm,
        "pct_under_5cm": p5cm
    }


def visualise(traj_name="circle", render=True, plot=True):
    """Main function - load model, run episode, show results"""

    print(f"\n{'='*60}")
    print(f"  TRAJECTORY TRACKING VISUALISER")
    print(f"  Trajectory: {traj_name}")
    print(f"{'='*60}\n")

    model, norm_path = load_model(traj_name)
    if model is None:
        return

    print(f"Running episode...")
    times, ee_pos, tgt_pos, errors = run_episode(
        model, norm_path, traj_name,
        render   = render,
        max_steps = 1000
    )

    print(f"\nResults:")
    print(f"  Steps:        {len(errors)}")
    print(f"  Mean error:   {np.mean(errors)*100:.1f}cm")
    print(f"  Max error:    {np.max(errors)*100:.1f}cm")
    print(f"  Under 1cm:    {np.mean(errors<0.01)*100:.0f}%")
    print(f"  Under 5cm:    {np.mean(errors<0.05)*100:.0f}%")

    if plot:
        plot_results(times, ee_pos, tgt_pos, errors, traj_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",     type=str,
                        default="circle",
                        choices=["circle","figure_eight","moving"])
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    visualise(
        traj_name = args.traj,
        render    = not args.no_render,
        plot      = not args.no_plot
    )
