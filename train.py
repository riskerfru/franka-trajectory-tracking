# =============================================================
#   TRAINING SCRIPT
#   Trains PPO agent to track trajectories
#
#   WHAT THIS FILE DOES:
#   1. Creates TrackingEnv with your trajectory
#   2. Creates PPO agent with your state/action/reward design
#   3. Trains for N steps
#   4. Saves model checkpoints
#   5. Prints progress so you can watch learning happen
#
#   HOW TO RUN:
#   python train.py --traj circle    (start here)
#   python train.py --traj figure_eight
#   python train.py --traj moving
#   python train.py --traj all      (train all three)
# =============================================================

import os
import argparse
import numpy as np
import torch
from franka_env import FrankaTrackingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback
)

from trajectories import (
    CircleTrajectory,
    FigureEightTrajectory,
    MovingTargetTrajectory,
    make_trajectory
)
from tracking_env import TrackingEnv


# =============================================================
#   CUSTOM CALLBACK - prints tracking error during training
#   So you can watch the agent improve in real time
# =============================================================

class TrackingMonitor(BaseCallback):
    """
    Prints tracking error every N steps.
    Shows whether agent is actually learning to track
    not just getting high reward.
    """

    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq  = print_freq
        self.best_error  = float("inf")

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            # Get recent episode info
            if len(self.model.ep_info_buffer) > 0:
                rewards = [
                    ep["r"] for ep in self.model.ep_info_buffer
                ]
                mean_r  = np.mean(rewards)

                marker = ""
                if mean_r > getattr(self, "_best_reward", -np.inf):
                    self._best_reward = mean_r
                    marker = " ← best"

                print(f"  Step {self.n_calls:>8,} | "
                      f"Reward: {mean_r:6.2f}{marker}")
        return True


# =============================================================
#   MAKE ENVIRONMENT FUNCTION
#   Creates environment for given trajectory name
# =============================================================

def make_env_fn(traj_name, render=False, use_franka=False):
    def _make():
        if use_franka:
            centre = [0.4, 0.0, 0.4]
            traj   = make_trajectory(traj_name, centre=centre)
            return FrankaTrackingEnv(
                trajectory  = traj,
                render_mode = "human" if render else None
            )
        else:
            traj = make_trajectory(traj_name)
            return TrackingEnv(
                trajectory  = traj,
                render_mode = "human" if render else None
            )
    return _make


# =============================================================
#   MAIN TRAINING FUNCTION
# =============================================================

def train(traj_name      = "circle",
          total_steps    = 500_000,
          n_envs         = 4,
          save_dir       = "models",
          device         = "auto",
          use_franka     = False):

    print("\n" + "="*60)
    print(f"  TRAJECTORY TRACKING TRAINING")
    print(f"  Trajectory: {traj_name}")
    print(f"  Steps:      {total_steps:,}")
    print(f"  Envs:       {n_envs}")
    print("="*60)

    if use_franka:
        save_dir = save_dir.replace("models", "models_franka")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # ----------------------------------------------------------
    # Create parallel training environments
    # n_envs copies run simultaneously
    # Gives more diverse experience faster
    # ----------------------------------------------------------
    print("\nCreating environments...")
    env = make_vec_env(
        make_env_fn(traj_name, use_franka=use_franka),
        n_envs = n_envs,
        seed   = 42
    )

    # Normalise observations and rewards
    # Makes training more stable
    # Without this numbers can be too big or small for network
    env = VecNormalize(
        env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        gamma       = 0.99
    )

    # Evaluation environment - single env, no normalisation of reward
    eval_env = make_vec_env(
        make_env_fn(traj_name, use_franka=use_franka),
        n_envs = 1,
        seed   = 100
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs    = True,
        norm_reward = False,
        clip_obs    = 10.0
    )

    # ----------------------------------------------------------
    # PPO Agent Configuration
    # Using your state/action/reward design
    # ----------------------------------------------------------
    print("Creating PPO agent...")

    # Network architecture
    # Two hidden layers of 256 neurons each
    # Smaller than humanoid (simpler task)
    policy_kwargs = dict(
        net_arch      = [256, 256],
        activation_fn = torch.nn.Tanh
    )

    model = PPO(
        policy              = "MlpPolicy",
        env                 = env,
        device              = device,

        # Core PPO parameters
        learning_rate       = 3e-4,
        n_steps             = 1024,    # steps per env before update
        batch_size          = 256,
        n_epochs            = 10,
        gamma               = 0.99,    # discount factor
        gae_lambda          = 0.95,
        clip_range          = 0.2,
        ent_coef            = 0.01,    # small exploration bonus
        vf_coef             = 0.5,
        max_grad_norm       = 0.5,
        normalize_advantage = True,

        policy_kwargs       = policy_kwargs,
        tensorboard_log     = "logs",
        verbose             = 0
    )

    print(f"  Network: {policy_kwargs['net_arch']}")
    print(f"  Device:  {model.device}")

    # ----------------------------------------------------------
    # Callbacks - things that happen during training
    # ----------------------------------------------------------

    # Save checkpoint every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq         = 50_000 // n_envs,
        save_path         = f"{save_dir}/{traj_name}",
        name_prefix       = f"tracking_{traj_name}",
        save_vecnormalize = True
    )

    # Evaluate every 25k steps
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = f"{save_dir}/{traj_name}/best",
        log_path             = "logs",
        eval_freq            = 25_000 // n_envs,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1
    )

    # Custom monitor - prints reward progress
    monitor_cb = TrackingMonitor(print_freq=10_000)

    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------
    print(f"\nTraining for {total_steps:,} steps...")
    print("Watch reward increase as agent learns to track\n")

    model.learn(
        total_timesteps      = total_steps,
        callback             = [checkpoint_cb, eval_cb, monitor_cb],
        progress_bar         = True,
        reset_num_timesteps  = True
    )

    # Save final model
    final_path = f"{save_dir}/{traj_name}/final"
    model.save(final_path)
    env.save(f"{save_dir}/{traj_name}/vec_normalize.pkl")

    print(f"\nTraining complete!")
    print(f"Model saved: {final_path}.zip")

    env.close()
    eval_env.close()
    return model


# =============================================================
#   ENTRY POINT
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train trajectory tracking agent"
    )
    parser.add_argument(
        "--traj",
        type    = str,
        default = "circle",
        choices = ["circle", "figure_eight", "moving", "all"],
        help    = "Which trajectory to train on"
    )
    parser.add_argument(
        "--franka",
        action  = "store_true",
        help    = "Use Franka Panda instead of Reacher"
    )
    parser.add_argument(
        "--steps",
        type    = int,
        default = 500_000,
        help    = "Total training steps"
    )
    parser.add_argument(
        "--envs",
        type    = int,
        default = 4,
        help    = "Number of parallel environments"
    )
    args = parser.parse_args()

    if args.traj == "all":
        # Train all three in sequence
        # Each builds on previous learning
        for traj in ["circle", "figure_eight", "moving"]:
            print(f"\n{'='*60}")
            print(f"Training: {traj}")
            train(
                traj_name   = traj,
                total_steps = args.steps,
                n_envs      = args.envs
            )
    else:
        train(
            traj_name   = args.traj,
            total_steps = args.steps,
            n_envs      = args.envs
        )