# =============================================================
#   TRACKING ENVIRONMENT
#   MuJoCo Reacher-v5 with custom observation and reward
#
#   State (12 numbers):
#     [0:3]  end effector position    [x, y, z]
#     [3:6]  end effector velocity    [vx, vy, vz]
#     [6:9]  target position          [x, y, z]
#     [9:12] target velocity          [vx, vy, vz]
#
#   Reward (your design):
#     exp(-10*d) + (-d²) + (-d) + smoothness + unreachable
#
#   Uncertainty:
#     1. Observation noise (5mm)
#     2. Control delay (2 steps)
#     3. Unreachable positions
# =============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrackingEnv(gym.Wrapper):

    # Reward weights
    W_EXPONENTIAL = 1.0
    W_QUADRATIC   = 0.5
    W_LINEAR      = 0.3
    W_SMOOTH      = 0.1
    W_UNREACHABLE = 0.5

    # Uncertainty parameters
    NOISE_STD     = 0.005   # 5mm observation noise
    DELAY_STEPS   = 2       # 2 step control delay
    MAX_REACH     = 0.19    # reachable radius for Reacher-v5

    def __init__(self, trajectory, render_mode=None):
        env = gym.make(
            "Reacher-v5",
            render_mode       = render_mode,
            max_episode_steps = 1000
        )
        super().__init__(env)

        self.trajectory = trajectory

        # Override observation space to our 12-number design
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (12,),
            dtype = np.float32
        )

        # Time tracking
        self.t  = 0.0
        self.dt = 0.05  # 50ms per step

        # Control delay buffer
        n_actions          = self.action_space.shape[0]
        self.action_buffer = [np.zeros(n_actions)] * self.DELAY_STEPS
        self.prev_action   = np.zeros(n_actions)

        # Episode tracking
        self.step_count = 0
        self.max_steps  = 1000
        self.errors     = []

        print(f"[ENV] TrackingEnv initialised")
        print(f"  Trajectory: {trajectory.name}")
        print(f"  Obs space:  {self.observation_space.shape}")
        print(f"  Act space:  {self.action_space.shape}")
        print(f"  Noise std:  {self.NOISE_STD}m")
        print(f"  Delay:      {self.DELAY_STEPS} steps")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.t          = 0.0
        self.step_count = 0
        self.errors     = []

        n_actions          = self.action_space.shape[0]
        self.action_buffer = [np.zeros(n_actions)] * self.DELAY_STEPS
        self.prev_action   = np.zeros(n_actions)

        return self._get_observation(), info

    def step(self, action):
        # Control delay - execute oldest action
        self.action_buffer.append(action.copy())
        delayed_action = self.action_buffer.pop(0)

        obs, _, terminated, truncated, info = self.env.step(
            delayed_action
        )

        self.t          += self.dt
        self.step_count += 1

        if self.step_count >= self.max_steps:
            truncated = True

        # Get positions
        ee_pos  = self._get_ee_position()
        tgt_pos = self.trajectory.get_position(self.t)

        # Tracking error
        error = np.linalg.norm(ee_pos - tgt_pos)
        self.errors.append(error)

        # Reachability check
        tgt_dist  = np.linalg.norm(tgt_pos[:2])
        reachable = tgt_dist <= self.MAX_REACH

        # Compute reward
        reward = self._compute_reward(
            ee_pos, tgt_pos, action, reachable
        )

        observation = self._get_observation()

        info.update({
            "tracking_error": error,
            "reachable":      reachable,
            "t":              self.t,
            "trajectory":     self.trajectory.name
        })

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """12-number observation with noise"""
        ee_pos  = self._get_ee_position()
        ee_vel  = self._get_ee_velocity()
        tgt_pos = self.trajectory.get_position(self.t)
        tgt_vel = self.trajectory.get_velocity(self.t)

        # Observation noise - simulates real sensor error
        noise  = np.random.normal(0, self.NOISE_STD, 6)
        ee_pos = ee_pos + noise[:3]
        ee_vel = ee_vel + noise[3:]

        obs = np.concatenate([
            ee_pos,   # [0:3]  where arm tip is
            ee_vel,   # [3:6]  how fast arm tip moving
            tgt_pos,  # [6:9]  where target is
            tgt_vel   # [9:12] how fast target moving
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, ee_pos, tgt_pos,
                        action, reachable):
        """Your three-term reward function"""
        distance = np.linalg.norm(ee_pos - tgt_pos)

        # Term 1: Exponential - strong pull to target
        exp_reward  = np.exp(-10.0 * distance) * self.W_EXPONENTIAL

        # Term 2: Quadratic - penalise being far
        quad_reward = -(distance ** 2) * self.W_QUADRATIC

        # Term 3: Linear - consistent gradient
        lin_reward  = -distance * self.W_LINEAR

        # Smoothness - penalise sudden changes
        jerk        = np.linalg.norm(action - self.prev_action)
        smooth      = -(jerk ** 2) * self.W_SMOOTH
        self.prev_action = action.copy()

        # Unreachable penalty
        unreachable = -self.W_UNREACHABLE if not reachable else 0.0

        return float(
            exp_reward  +
            quad_reward +
            lin_reward  +
            smooth      +
            unreachable
        )

    def _get_ee_position(self):
        """Get fingertip position from MuJoCo"""
        try:
            return self.unwrapped.data.xpos[3].copy()
        except Exception:
            return np.zeros(3)

    def _get_ee_velocity(self):
        """Get fingertip velocity from MuJoCo"""
        try:
            return self.unwrapped.data.cvel[3, 3:6].copy()
        except Exception:
            return np.zeros(3)

    def get_episode_stats(self):
        """Statistics for evaluation and plotting"""
        if not self.errors:
            return {}
        errors = np.array(self.errors)
        return {
            "mean_error": float(np.mean(errors)),
            "max_error":  float(np.max(errors)),
            "min_error":  float(np.min(errors)),
            "std_error":  float(np.std(errors)),
            "under_1cm":  float(np.mean(errors < 0.01)),
            "under_5cm":  float(np.mean(errors < 0.05)),
        }


# =============================================================
#   TEST
# =============================================================

if __name__ == "__main__":
    from trajectories import CircleTrajectory

    print("Testing TrackingEnv...")

    traj   = CircleTrajectory()
    env    = TrackingEnv(traj, render_mode=None)
    obs, _ = env.reset()

    print(f"\nInitial observation:")
    print(f"  EE pos:     {obs[0:3].round(3)}")
    print(f"  EE vel:     {obs[3:6].round(3)}")
    print(f"  Target pos: {obs[6:9].round(3)}")
    print(f"  Target vel: {obs[9:12].round(3)}")

    # Check target is in workspace
    tgt = obs[6:9]
    print(f"\nTarget check:")
    print(f"  X: {tgt[0]:.3f} (workspace: -0.182 to 0.187)")
    print(f"  Y: {tgt[1]:.3f} (workspace: -0.151 to 0.197)")
    print(f"  Z: {tgt[2]:.3f} (should be 0.01)")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"\nTest episode (random actions):")
    print(f"  Steps:        {env.step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Mean error:   {stats['mean_error']*100:.1f}cm")
    print(f"  Max error:    {stats['max_error']*100:.1f}cm")

    env.close()
    print("\nEnvironment test passed!")