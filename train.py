"""
train.py
========
Train a PPO agent to track trajectories with the Franka Panda.

Usage:
    python train.py                          # circle, 1M steps
    python train.py --traj figure8           # figure8 trajectory
    python train.py --steps 2000000          # longer training
    python train.py --traj circle --render   # watch while training

Requirements:
    pip install stable-baselines3 gymnasium mujoco
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from franka_env import FrankaTrackingEnv, TRAJECTORIES


def make_env(traj_name: str, seed: int = 0):
    """Factory function for vectorised environments."""
    def _make():
        env = FrankaTrackingEnv(trajectory=traj_name, render_mode=None)
        env = Monitor(env)
        return env
    return _make


def train(
    traj_name:   str   = "circle",
    total_steps: int   = 1_000_000,
    n_envs:      int   = 4,
    save_dir:    str   = "models",
):
    print(f"\n{'='*50}")
    print(f"  FRANKA TRAJECTORY TRACKING — PPO TRAINING")
    print(f"  Trajectory: {traj_name}")
    print(f"  Steps:      {total_steps:,}")
    print(f"  Envs:       {n_envs}")
    print(f"{'='*50}\n")

    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ── Vectorised training environments ──
    if n_envs > 1:
        env = SubprocVecEnv([make_env(traj_name, seed=i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(traj_name)])

    # ── Evaluation environment (single, no noise during eval) ──
    eval_env = DummyVecEnv([make_env(traj_name, seed=99)])

    # ── PPO model ──
    model = PPO(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = 3e-4,
        n_steps            = 2048,
        batch_size         = 512,
        n_epochs           = 10,
        gamma              = 0.99,
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.005,       # small entropy bonus
        vf_coef            = 0.5,
        max_grad_norm      = 0.5,
        policy_kwargs      = dict(
            net_arch=[256, 256, 128],     # 3 hidden layers
        ),
        verbose            = 1,
        tensorboard_log    = log_dir,
    )

    # ── Callbacks ──
    checkpoint = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = save_dir,
        name_prefix = f"ppo_{traj_name}",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = save_dir,
        log_path             = log_dir,
        eval_freq            = 25_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    # ── Train ──
    model.learn(
        total_timesteps  = total_steps,
        callback         = [checkpoint, eval_cb],
        progress_bar     = True,
    )

    # ── Save final model ──
    final_path = os.path.join(save_dir, f"ppo_{traj_name}_final")
    model.save(final_path)
    print(f"\nSaved final model: {final_path}.zip")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Franka trajectory tracker")
    parser.add_argument("--traj",  default="circle",
                        choices=list(TRAJECTORIES.keys()),
                        help="Trajectory type")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total training steps")
    parser.add_argument("--envs",  type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--all",   action="store_true",
                        help="Train all trajectories sequentially")
    args = parser.parse_args()

    if args.all:
        for traj in TRAJECTORIES:
            train(traj_name=traj, total_steps=args.steps, n_envs=args.envs)
    else:
        train(traj_name=args.traj, total_steps=args.steps, n_envs=args.envs)