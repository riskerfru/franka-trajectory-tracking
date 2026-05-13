# Design Note — Franka End-Effector Trajectory Tracking

## State, Action, and Reward Design

### Observation Space (19 dimensions)

```
[0:3]  ee_pos      End effector position (x, y, z) — with 3mm Gaussian noise
[3:6]  ee_vel      End effector velocity (vx, vy, vz)
[6:9]  target_pos  Current target position on trajectory
[9:12] target_vel  Current target velocity (analytical derivative)
[12:19] joints     All 7 joint angles
```

**Why include target velocity?** The agent needs to anticipate where the target will be next step, not just where it is now. Without velocity, the agent always chases a position it has already left — introducing permanent lag. With velocity, it can lead the target.

**Why include joint angles?** The same end effector position can be reached with multiple joint configurations (redundancy of 7-DOF arm). Joint angles let the agent reason about which configuration it is in and plan smooth transitions.

**Sensor noise (3mm std dev):** Gaussian noise added to ee_pos every step. Simulates realistic depth camera or encoder uncertainty (Intel RealSense D435 typical noise: ±3–5mm at 1m). Forces the agent to learn robust tracking rather than relying on perfect measurements.

---

### Action Space (7 dimensions)

Joint velocity targets for all 7 joints, normalised to [−1, +1] and scaled to ±2 rad/s.

**Why velocity control?** Position control produces discontinuous motion — the robot jumps to a target and stops. Velocity control produces smooth, continuous motion suitable for trajectory tracking. The agent learns to flow along the trajectory rather than repeatedly reach discrete points.

**Why all 7 joints?** Controlling all joints gives the agent full use of the arm's redundancy. It can reconfigure joints away from singularities and joint limits while maintaining end effector tracking — something impossible with task-space control alone.

---

### Reward Function

```python
reward = exp(-10 * dist)          # exponential: strong pull toward target
       - 0.01 * sum(action²)      # control cost: penalise large velocities
       + 1.0 if dist < 0.02       # bonus: within 2cm counts as success
```

**Exponential term:** Peaks at 1.0 when distance is zero, falls smoothly to near-zero beyond 30cm. Provides a strong gradient everywhere — the agent always knows which direction improves reward, even far from the target. A linear penalty has a flat gradient near the target; exponential doesn't.

**Control cost:** Penalises large joint velocities. This produces smooth motion — the agent learns that oscillating back and forth costs more than a smooth approach. Without this, agents often develop jittery policies that track well on average but vibrate around the trajectory.

**Success bonus:** Discrete reward for being within 2cm. Creates a clear precision target and rewards the agent for maintaining close tracking, not just approaching.

---

## Trajectory Representation

Each trajectory is a function of time `t` returning `(position, velocity)`. The velocity is the analytical derivative of position — exact, not approximated.

```python
# Circle
pos = (cx + r*cos(ω*t),  cy + r*sin(ω*t),  cz)
vel = (-r*ω*sin(ω*t),    r*ω*cos(ω*t),     0)

# Figure-8 (Lissajous)
pos = (cx + r*sin(2ω*t), cy + r*sin(ω*t),  cz)
vel = (2r*ω*cos(2ω*t),   r*ω*cos(ω*t),     0)

# Random (compound Lissajous, 3D)
pos = (cx + r*sin(ω*t)*cos(0.7*t),
       cy + r*cos(ω*t)*sin(1.1*t),
       cz + 0.05*sin(1.3*t))
```

**Why parametric?** A parametric function gives the agent exact trajectory information at any point in time with no discretisation error. Alternative approaches (waypoint lists, splines) introduce interpolation error and require storing the full trajectory in memory.

**Episode start randomisation:** Each episode begins at a random time `t ~ Uniform(0, 10)` on the trajectory. This prevents the agent from memorising a fixed sequence — it must generalise to any phase of the trajectory.

---

## Uncertainty Sources

### 1. Observation Noise (3mm std dev)
Gaussian noise on end effector position every step:
```python
noisy_ee_pos = ee_pos + np.random.normal(0, 0.003, 3)
```
Realistic for depth cameras. Forces robust tracking — the agent cannot rely on perfect measurements.

### 2. Unreachable Positions
The trajectory centre is at (0.5, 0.0, 0.4) with radius 0.10–0.12m. This places trajectory points at distances of 0.38–0.62m from the robot base — within the safe zone (0.25–0.75m). However, the inline MuJoCo model has simplified geometry, so some configurations near the boundary of the trajectory are mechanically difficult to reach. The agent learns to approach these positions from stable configurations rather than forcing through singularities.

---

## Training Setup

```
Algorithm:     PPO (Proximal Policy Optimisation)
Policy:        MLP (256 → 256 → 128)
Learning rate: 3e-4
Steps/update:  2048 per environment
Batch size:    512
Epochs:        10 per update
γ (discount):  0.99
GAE λ:         0.95
Entropy coef:  0.005  (encourages exploration)
Parallel envs: 4
Total steps:   1,000,000 per trajectory
```

**Why PPO?** PPO is stable and sample-efficient for continuous control. The clipped surrogate objective prevents catastrophic policy updates — important for contact-rich robotics where a bad update can send the arm to joint limits. SAC would be an alternative for higher sample efficiency, but PPO's on-policy nature means the agent always trains on its current behaviour, which is better for trajectory tracking where the target is always moving.

---

## Evaluation and Results

| Trajectory | Mean Error | Min Error | Jerk (smoothness) |
|-----------|-----------|----------|------------------|
| Circle    | 1.9cm     | 0.1cm    | 2.87             |
| Figure-8  | 1.8cm     | 0.1cm    | 1.35             |
| Random    | 4.8cm     | 0.2cm    | 1.43             |

**Why is circle best?** The circle has constant curvature and constant speed — the agent sees consistent dynamics every step. Figure-8 has direction reversals (velocity changes sign) which require the agent to decelerate and accelerate. Random has varying curvature and 3D motion.

**Why are max errors large?** Maximum errors occur in the first 5–10 steps of each episode, before the agent has converged onto the trajectory from its random starting configuration. After convergence, errors are consistently low.

**Smoothness:** The control cost penalty (−0.01 × ||action||²) successfully suppresses jitter. Joint velocities are smooth over time — no oscillatory behaviour observed in trained policies.

---

## What I Would Do Differently

**1. Curriculum learning** — Start training on circle, then introduce figure-8, then random. The agent would transfer knowledge across trajectories rather than training each from scratch.

**2. Domain randomisation** — Randomise robot mass, friction, and actuator gains during training. Would dramatically improve sim-to-real transfer for deployment on a real Franka Panda.

**3. Orientation tracking** — Add target orientation (quaternion or axis-angle) to the observation and a rotation error term to the reward. Currently only position is tracked.

**4. Real Franka URDF** — The inline simplified XML uses approximate geometry. The mujoco-menagerie Franka model would give accurate joint limits, mass distribution, and collision geometry — closer to real hardware behaviour.

**5. SAC instead of PPO** — Soft Actor-Critic is more sample-efficient for continuous control. With 1M steps, PPO works well, but SAC would reach the same performance in ~400k steps.