# =============================================================
#   FRANKA PANDA TRAJECTORY TRACKING ENVIRONMENT
#   PyBullet + Franka Panda 7-DOF arm
#   Replaces Reacher with proper 7-DOF robot
#
#   WHY FRANKA INSTEAD OF REACHER:
#   Reacher has 2 joints - limited workspace control
#   Franka has 7 joints - full 3D workspace control
#   Expected accuracy improvement: 8-9cm → 1-4cm
#
#   State (14 numbers):
#     [0:3]   end effector position    [x, y, z]
#     [3:6]   end effector velocity    [vx, vy, vz]
#     [6:9]   target position          [x, y, z]
#     [9:12]  target velocity          [vx, vy, vz]
#     [12:14] joint limit proximity    [left, right]
#
#   Action (7 numbers):
#     Joint velocity commands for all 7 joints [-1, +1]
# =============================================================

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time


class FrankaTrackingEnv(gym.Env):
    """
    Franka Panda 7-DOF arm for trajectory tracking.
    Built from scratch with PyBullet.
    Much better workspace than Reacher (2-DOF).
    """

    # Reward weights - same design as before
    W_EXPONENTIAL = 1.0
    W_QUADRATIC   = 0.5
    W_LINEAR      = 0.3
    W_SMOOTH      = 0.1
    W_UNREACHABLE = 0.5

    # Uncertainty parameters
    NOISE_STD   = 0.005   # 5mm observation noise
    DELAY_STEPS = 2       # 2 step control delay

    # Franka Panda joint limits (radians)
    JOINT_LIMITS_LOW  = np.array([-2.9, -1.8, -2.9,
                                   -3.1, -2.9, -0.1, -2.9])
    JOINT_LIMITS_HIGH = np.array([ 2.9,  1.8,  2.9,
                                    0.0,  2.9,  3.8,  2.9])

    # Joint velocity limits (rad/s)
    VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175,
                            2.610, 2.610, 2.610])

    # Arm joint indices in PyBullet
    ARM_JOINTS   = [0, 1, 2, 3, 4, 5, 6]
    EE_LINK      = 11   # end effector link index

    # Workspace
    MAX_REACH    = 0.85  # Franka max reach
    MIN_REACH    = 0.25  # too close to base

    def __init__(self, trajectory,
                 render_mode = None,
                 max_steps   = 1000):

        super().__init__()

        self.trajectory  = trajectory
        self.render_mode = render_mode
        self.max_steps   = max_steps

        # Observation: 14 numbers
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (12,),
            dtype = np.float32
        )

        # Action: 7 joint velocities normalised to [-1, +1]
        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (7,),
            dtype = np.float32
        )

        # Connect to PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance      = 1.5,
                cameraYaw           = 45,
                cameraPitch         = -30,
                cameraTargetPosition = [0.4, 0, 0.3]
            )
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load robot and environment
        self.plane  = p.loadURDF("plane.urdf")
        self.robot  = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition    = [0, 0, 0],
            useFixedBase    = True
        )

        # Target visual marker
        self.target_visual = self._create_target_marker()

        # Time and tracking
        self.t          = 0.0
        self.dt         = 0.05
        self.step_count = 0
        self.errors     = []

        # Control delay buffer
        self.action_buffer = [np.zeros(7)] * self.DELAY_STEPS
        self.prev_action   = np.zeros(7)

        # Home position
        self.home_joints = np.array([0, -0.3, 0, -2.0,
                                      0,  1.8, 0.8])

        print(f"[FRANKA ENV] Initialised")
        print(f"  Trajectory: {trajectory.name}")
        print(f"  Obs space:  {self.observation_space.shape}")
        print(f"  Act space:  {self.action_space.shape}")
        print(f"  Joints:     7 DOF")

    def _create_target_marker(self):
        """Create red sphere to show target position"""
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius    = 0.02,
            rgbaColor = [1, 0, 0, 0.8]
        )
        body = p.createMultiBody(
            baseVisualShapeIndex = visual,
            basePosition         = [0.4, 0, 0.4]
        )
        return body

    def _move_to_home(self):
        """Move arm to home position"""
        for i, joint_idx in enumerate(self.ARM_JOINTS):
            p.resetJointState(
                self.robot,
                joint_idx,
                self.home_joints[i]
            )
        # Let settle
        for _ in range(50):
            p.stepSimulation()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset to home position
        self._move_to_home()

        # Reset tracking
        self.t          = 0.0
        self.step_count = 0
        self.errors     = []
        self.action_buffer = [np.zeros(7)] * self.DELAY_STEPS
        self.prev_action   = np.zeros(7)

        # Move target marker to start position
        tgt = self.trajectory.get_position(0.0)
        p.resetBasePositionAndOrientation(
            self.target_visual, tgt, [0, 0, 0, 1]
        )

        obs  = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # Control delay
        self.action_buffer.append(action.copy())
        delayed = self.action_buffer.pop(0)

        # Scale action to joint velocity limits
        joint_vels = delayed * self.VEL_LIMITS

        # Apply joint velocity control
        p.setJointMotorControlArray(
            self.robot,
            self.ARM_JOINTS,
            p.VELOCITY_CONTROL,
            targetVelocities = joint_vels.tolist(),
            forces           = [87, 87, 87, 87, 12, 12, 12]
        )

        # Step simulation
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(self.dt)

        self.t          += self.dt
        self.step_count += 1

        # Get positions
        ee_pos  = self._get_ee_position()
        tgt_pos = self.trajectory.get_position(self.t)

        # Move target marker
        p.resetBasePositionAndOrientation(
            self.target_visual, tgt_pos.tolist(), [0, 0, 0, 1]
        )

        # Tracking error
        error = np.linalg.norm(ee_pos - tgt_pos)
        self.errors.append(error)

        # Reachability
        dist      = np.linalg.norm(tgt_pos)
        reachable = self.MIN_REACH <= dist <= self.MAX_REACH

        # Reward
        reward = self._compute_reward(
            ee_pos, tgt_pos, action, reachable
        )

        terminated = False
        truncated  = self.step_count >= self.max_steps

        obs  = self._get_observation()
        info = {
            "tracking_error": error,
            "reachable":      reachable,
            "t":              self.t
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """14-number observation with noise"""
        ee_pos  = self._get_ee_position()
        ee_vel  = self._get_ee_velocity()
        tgt_pos = self.trajectory.get_position(self.t)
        tgt_vel = self.trajectory.get_velocity(self.t)

        # Joint limit proximity (how close to limits)
        joint_pos = np.array([
            p.getJointState(self.robot, j)[0]
            for j in self.ARM_JOINTS
        ])
        range_    = self.JOINT_LIMITS_HIGH - self.JOINT_LIMITS_LOW
        normalised = (joint_pos - self.JOINT_LIMITS_LOW) / range_
        left_prox  = np.min(normalised)        # closest to lower limit
        right_prox = np.min(1 - normalised)    # closest to upper limit

        # Add noise
        noise   = np.random.normal(0, self.NOISE_STD, 6)
        ee_pos  = ee_pos + noise[:3]
        ee_vel  = ee_vel + noise[3:]

        obs = np.concatenate([
            ee_pos,              # [0:3]
            ee_vel,              # [3:6]
            tgt_pos,             # [6:9]
            tgt_vel,             # [9:12]
            [left_prox,          # [12]
             right_prox]         # [13]
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, ee_pos, tgt_pos,
                        action, reachable):
        distance = np.linalg.norm(ee_pos - tgt_pos)

        exp_reward  = np.exp(-10.0 * distance) * self.W_EXPONENTIAL
        quad_reward = -(distance ** 2) * self.W_QUADRATIC
        lin_reward  = -distance * self.W_LINEAR

        jerk        = np.linalg.norm(action - self.prev_action)
        smooth      = -(jerk ** 2) * self.W_SMOOTH
        self.prev_action = action.copy()

        unreachable = -self.W_UNREACHABLE if not reachable else 0.0

        # Joint limit penalty
        joint_pos = np.array([
            p.getJointState(self.robot, j)[0]
            for j in self.ARM_JOINTS
        ])
        near_limit = np.sum(
            np.maximum(joint_pos - self.JOINT_LIMITS_HIGH + 0.1, 0) +
            np.maximum(self.JOINT_LIMITS_LOW + 0.1 - joint_pos, 0)
        )
        limit_penalty = -near_limit * 0.5

        return float(
            exp_reward  +
            quad_reward +
            lin_reward  +
            smooth      +
            unreachable +
            limit_penalty
        )

    def _get_ee_position(self):
        """Get end effector position"""
        state = p.getLinkState(self.robot, self.EE_LINK)
        return np.array(state[4])  # world position

    def _get_ee_velocity(self):
        """Get end effector velocity"""
        state = p.getLinkState(
            self.robot, self.EE_LINK,
            computeLinkVelocity = 1
        )
        return np.array(state[6])  # linear velocity

    def get_episode_stats(self):
        if not self.errors:
            return {}
        errors = np.array(self.errors)
        return {
            "mean_error": float(np.mean(errors)),
            "max_error":  float(np.max(errors)),
            "under_1cm":  float(np.mean(errors < 0.01)),
            "under_5cm":  float(np.mean(errors < 0.05)),
        }

    def close(self):
        p.disconnect(self.client)

    def render(self):
        pass


# =============================================================
#   TEST
# =============================================================

if __name__ == "__main__":
    from trajectories import CircleTrajectory

    print("Testing FrankaTrackingEnv...")

    # Circle centred in front of robot
    traj = CircleTrajectory(
        centre = [0.4, 0.0, 0.4],
        radius = 0.08,
        speed  = 1.0
    )

    env    = FrankaTrackingEnv(traj, render_mode=None)
    obs, _ = env.reset()

    print(f"\nInitial observation:")
    print(f"  EE pos:     {obs[0:3].round(3)}")
    print(f"  Target pos: {obs[6:9].round(3)}")
    print(f"  Obs shape:  {obs.shape}")

    # Check initial distance to target
    ee  = obs[0:3]
    tgt = obs[6:9]
    dist = np.linalg.norm(ee - tgt)
    print(f"  Initial distance to target: {dist*100:.1f}cm")

    total = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"\nTest (random actions):")
    print(f"  Steps:      {env.step_count}")
    print(f"  Reward:     {total:.2f}")
    print(f"  Mean error: {stats['mean_error']*100:.1f}cm")

    env.close()
    print("\nFranka environment test passed!")
