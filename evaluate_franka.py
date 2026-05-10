# Quick evaluation script for Franka tracking
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from trajectories import CircleTrajectory
from franka_env import FrankaTrackingEnv

traj  = CircleTrajectory(centre=[0.4, 0.0, 0.4], radius=0.08, speed=1.0)
model = PPO.load('models/circle/final', device='cpu')
env   = DummyVecEnv([lambda: FrankaTrackingEnv(traj, render_mode=None)])

obs    = env.reset()
errors = []

for _ in range(1000):
    print("Model obs space:", model.observation_space.shape)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if info[0].get('tracking_error'):
        errors.append(info[0]['tracking_error'])
    if done[0]:
        break

errors = np.array(errors)
print(f'Steps:      {len(errors)}')
print(f'Mean error: {errors.mean()*100:.1f}cm')
print(f'Max error:  {errors.max()*100:.1f}cm')
print(f'Under 5cm:  {(errors<0.05).mean()*100:.0f}%')
print(f'Under 1cm:  {(errors<0.01).mean()*100:.0f}%')
env.close()