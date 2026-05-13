"""
franka_env.py
=============
Franka Panda end-effector trajectory tracking environment.
Uses MuJoCo via mujoco-py / dm_control.

Observation (18,):
  [0:3]  ee_pos          end effector position (xyz)
  [3:6]  ee_vel          end effector velocity (xyz)
  [7:9]  target_pos      target position (xyz)
  [9:12] target_vel      target velocity (xyz)
  [12:19] joint_angles   all 7 joint angles

Action (7,):
  joint velocity targets for all 7 joints, clipped to [-1, 1]

Reward:
  exp(-10 * dist)   dense exponential — peaks at 1.0 when perfect
  - 0.01 * ||u||²  small control penalty for smooth motion
  + 1.0 if dist < 0.02  bonus for being within 2cm
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Try MuJoCo imports ──────────────────────────────────────
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

import os

# ── Trajectory generators ────────────────────────────────────

def circle_trajectory(t, centre=(0.5, 0.0, 0.4), r=0.12, speed=0.5):
    angle = speed * t
    x = centre[0] + r * np.cos(angle)
    y = centre[1] + r * np.sin(angle)
    z = centre[2]
    vx = -r * speed * np.sin(angle)
    vy =  r * speed * np.cos(angle)
    vz = 0.0
    return np.array([x, y, z]), np.array([vx, vy, vz])


def figure8_trajectory(t, centre=(0.5, 0.0, 0.4), r=0.10, speed=0.4):
    angle = speed * t
    x = centre[0] + r * np.sin(2 * angle)
    y = centre[1] + r * np.sin(angle)
    z = centre[2]
    vx = 2 * r * speed * np.cos(2 * angle)
    vy = r * speed * np.cos(angle)
    vz = 0.0
    return np.array([x, y, z]), np.array([vx, vy, vz])


def linear_trajectory(t, start=(0.4, -0.2, 0.4), end=(0.4, 0.2, 0.4), period=4.0):
    frac = (np.sin(2 * np.pi * t / period) + 1) / 2
    pos = np.array(start) + frac * (np.array(end) - np.array(start))
    speed = (np.pi / period) * np.cos(2 * np.pi * t / period)
    vel = speed * (np.array(end) - np.array(start))
    return pos, vel


TRAJECTORIES = {
    "circle":   circle_trajectory,
    "figure8":  figure8_trajectory,
    "linear":   linear_trajectory,
}


# ── Environment ──────────────────────────────────────────────

class FrankaTrackingEnv(gym.Env):
    """
    Franka Panda end-effector trajectory tracking.

    The agent controls joint velocities to keep the end
    effector on a moving target trajectory.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        trajectory: str = "circle",
        render_mode: str = None,
        max_steps: int   = 500,
        dt: float        = 0.02,
        noise_std: float = 0.003,
    ):
        super().__init__()

        self.traj_fn     = TRAJECTORIES[trajectory]
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.dt          = dt
        self.noise_std   = noise_std
        self.traj_name   = trajectory

        # ── Action space: 7 joint velocity targets [-1, 1] ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # ── Observation space ──
        # ee_pos(3) + ee_vel(3) + target_pos(3) + target_vel(3) + joints(7) = 19
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )

        # ── MuJoCo model ──
        self._load_model()

        self._step       = 0
        self._time       = 0.0
        self._data       = None

    def _load_model(self):
        """Load Franka Panda MuJoCo XML."""
        # Try to find model file
        search_paths = [
            "mjx_panda_nohand.xml",
            "panda_nohand.xml",
            "mjx_panda.xml",
            "panda.xml",
            os.path.join("assets", "panda_nohand.xml"),
            os.path.join("mujoco_menagerie", "franka_emika_panda", "panda_nohand.xml"),
        ]

        self._model_path = None
        for path in search_paths:
            if os.path.exists(path):
                self._model_path = path
                break

        if self._model_path is None:
            # Generate minimal inline model
            self._model_xml = self._minimal_xml()
            self.model = mujoco.MjModel.from_xml_string(self._model_xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(self._model_path)

        self.data  = mujoco.MjData(self.model)
        self._viewer = None

        # Find end effector body
        try:
            self._ee_id = self.model.body("hand").id
        except Exception:
            try:
                self._ee_id = self.model.body("panda_hand").id
            except Exception:
                self._ee_id = self.model.nbody - 1

        # Joint limits
        self._qpos_min = self.model.jnt_range[:7, 0]
        self._qpos_max = self.model.jnt_range[:7, 1]
        self._vel_limit = 2.0  # rad/s max joint velocity

        print(f"  [ENV] Loaded: {self._model_path or 'inline XML'}")
        print(f"  [ENV] EE body id: {self._ee_id}")
        print(f"  [ENV] Trajectory: {self.traj_name}")

    def _minimal_xml(self):
        """Fallback minimal Franka Panda model string."""
        return """
<mujoco model="franka_panda_minimal">
  <compiler angle="radian" />
  <option timestep="0.002" gravity="0 0 -9.81" />
  <default>
    <joint limited="true" damping="1.0" armature="0.1"/>
    <geom contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="cylinder" size="0.06 0.15" rgba="0.8 0.8 0.8 1"/>
        <body name="link2" pos="0 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
          <geom type="cylinder" size="0.06 0.15" rgba="0.8 0.8 0.8 1"/>
          <body name="link3" pos="0 -0.316 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="cylinder" size="0.06 0.12" rgba="0.8 0.8 0.8 1"/>
            <body name="link4" pos="0.0825 0 0">
              <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0718 -0.0698"/>
              <geom type="cylinder" size="0.06 0.12" rgba="0.8 0.8 0.8 1"/>
              <body name="link5" pos="-0.0825 0.384 0">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="cylinder" size="0.05 0.1" rgba="0.8 0.8 0.8 1"/>
                <body name="link6" pos="0 0 0">
                  <joint name="joint6" type="hinge" axis="0 1 0" range="-0.0175 3.7525"/>
                  <geom type="cylinder" size="0.05 0.08" rgba="0.8 0.8 0.8 1"/>
                  <body name="link7" pos="0.088 0 0">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="cylinder" size="0.04 0.06" rgba="0.8 0.8 0.8 1"/>
                    <body name="hand" pos="0 0 0.107">
                      <geom type="sphere" size="0.03" rgba="0.2 0.6 0.8 1"/>
                      <site name="ee_site" pos="0 0 0.05" size="0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos="0.5 0 0.4">
      <geom type="sphere" size="0.025" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
      <site name="target_site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="act1" joint="joint1" kv="10"/>
    <velocity name="act2" joint="joint2" kv="10"/>
    <velocity name="act3" joint="joint3" kv="10"/>
    <velocity name="act4" joint="joint4" kv="10"/>
    <velocity name="act5" joint="joint5" kv="10"/>
    <velocity name="act6" joint="joint6" kv="10"/>
    <velocity name="act7" joint="joint7" kv="10"/>
  </actuator>
</mujoco>"""

    def _get_ee_pos(self):
        """Get end effector position and velocity."""
        pos = self.data.xpos[self._ee_id].copy()
        vel = self.data.cvel[self._ee_id][:3].copy()
        return pos, vel

    def _get_obs(self):
        target_pos, target_vel = self.traj_fn(self._time)
        ee_pos, ee_vel         = self._get_ee_pos()

        # Add sensor noise (realistic depth camera noise)
        noisy_ee_pos = ee_pos + np.random.normal(0, self.noise_std, 3)

        joints = self.data.qpos[:7].copy()

        obs = np.concatenate([
            noisy_ee_pos,    # 3
            ee_vel,          # 3
            target_pos,      # 3
            target_vel,      # 3
            joints,          # 7
        ]).astype(np.float32)

        return obs, ee_pos, target_pos

    def _compute_reward(self, ee_pos, target_pos, action):
        dist = np.linalg.norm(ee_pos - target_pos)

        exp_reward     = np.exp(-10.0 * dist)
        control_cost   = 0.01 * np.sum(action ** 2)
        success_bonus  = 1.0 if dist < 0.02 else 0.0

        return float(exp_reward - control_cost + success_bonus)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start at a reasonable home position
        home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = home
        mujoco.mj_forward(self.model, self.data)

        # Randomise start time so agent sees different trajectory phases
        self._time    = np.random.uniform(0, 10.0)
        self._step    = 0

        obs, ee_pos, target_pos = self._get_obs()

        # Move target sphere to initial position
        self._update_target_visual(target_pos)

        return obs, {}

    def _update_target_visual(self, target_pos):
        """Move the red target sphere to current trajectory position."""
        try:
            tid = self.model.body("target").id
            self.data.xpos[tid] = target_pos
        except Exception:
            pass

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Scale action to joint velocity limits
        self.data.ctrl[:7] = action * self._vel_limit

        # Step physics (multiple substeps for stability)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._time += self.dt
        self._step += 1

        obs, ee_pos, target_pos = self._get_obs()
        self._update_target_visual(target_pos)

        reward      = self._compute_reward(ee_pos, target_pos, action)
        terminated  = False
        truncated   = self._step >= self.max_steps
        dist        = float(np.linalg.norm(ee_pos - target_pos))

        info = {"distance": dist, "time": self._time}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ── Standalone test ──────────────────────────────────────────

if __name__ == "__main__":
    print("Testing FrankaTrackingEnv...")
    env = FrankaTrackingEnv(trajectory="circle", render_mode=None)
    obs, _ = env.reset()

    print(f"  Obs space:  {env.observation_space.shape}")
    print(f"  Act space:  {env.action_space.shape}")
    print(f"  Trajectory: circle")
    print(f"  Initial obs shape: {obs.shape}")
    print(f"  EE pos: {obs[:3]}")
    print(f"  Target: {obs[6:9]}")

    rewards, dists = [], []
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        rewards.append(r)
        dists.append(info["distance"])
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"\nRandom action stats (200 steps):")
    print(f"  Mean reward:   {np.mean(rewards):.3f}")
    print(f"  Mean distance: {np.mean(dists)*100:.1f}cm")
    print(f"\nEnvironment test passed!")
    env.close()