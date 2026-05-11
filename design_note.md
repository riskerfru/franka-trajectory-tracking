# Design Note — 3D End-Effector Tracking

## Why These Design Choices

### State Space: 12 Numbers Including Velocity

Early versions used only 6 numbers — position of end effector
and position of target. The agent consistently lagged behind
the target, always reacting after the target had already moved.

Adding target velocity (4 more numbers) changed the behaviour
fundamentally. The agent can now predict where the target will
be next step and move to meet it rather than chase it. This
reduced mean error on the circle trajectory from 15cm to 8.7cm.

End effector velocity is included for the same reason — the agent
needs to know its own momentum to plan deceleration before
overshoot happens.

### Action Space: Joint Velocities Not Positions

Position control commands a joint to be at a specific angle.
The controller tries to jump there instantly, producing jerky
discontinuous motion that makes smooth tracking impossible.

Velocity control commands how fast each joint should move.
This produces natural smooth acceleration and deceleration.
The smoothness penalty in the reward function then reinforces
this — the agent learns to prefer gradual velocity changes.

### Reward: Three Terms Combined

A single reward term cannot capture everything needed for
good tracking:

**Exponential** `exp(-10*d)` creates a strong reward signal
near the target. At 1cm error the reward is 0.90. At 5cm it
is 0.61. At 10cm it is 0.37. The agent always has a clear
gradient to follow toward the target.

**Quadratic** `-d²` penalises large errors heavily but tolerates
small errors. A 10cm error costs 100x more than a 1cm error.
This pushes the agent to eliminate large deviations first.

**Linear** `-d` provides a consistent signal everywhere including
when the agent is far from the target. Without this term the
agent sometimes gets stuck in local minima where the exponential
gradient is too small to pull it back.

The three terms are complementary — none alone would produce
the same result as all three together.

### Smoothness Penalty

Without the smoothness penalty `-jerk²`, the agent produces
rapid oscillating joint commands that cancel each other out.
The arm vibrates at high frequency around the target without
actually following it smoothly.

The penalty is applied to the change in action between steps.
Large changes in action mean sudden velocity reversals — mechanical
stress in real hardware and poor tracking in simulation.

### Three Uncertainty Sources

**Observation noise** (5mm Gaussian) was chosen because it is
the most realistic source of uncertainty for real deployment.
An Intel RealSense D435 at 1m range has approximately 3-5mm
depth accuracy. Training with this noise level means the policy
is robust to the actual sensor the system would use.

**Control delay** (2 steps = 100ms) simulates the communication
latency between a control computer and robot hardware. Without
delay training, a policy that works in simulation often fails
on hardware where commands take time to execute.

**Unreachable positions** arise naturally from the moving target
trajectory. The agent learns to move to the nearest reachable
point rather than thrashing against joint limits — a behaviour
that would damage real hardware.

### What Did Not Work

**Larger circle radius (0.15m):** The Reacher arm workspace is
approximately 0.19m radius. A 0.15m circle puts the target
consistently near the workspace boundary where joint limits
cause sudden direction reversals. Reduced to 0.10m.

**Position action space:** Initial version used joint position
commands. The arm would jump between positions producing
tracking errors of 25-40cm. Switched to velocity control
which reduced this to 8-15cm for the same training budget.

**Separate models per trajectory:** Training separate models
for circle and figure-eight worked well. A single model trained
on all three performed worse on each individual trajectory
because the observation space did not include trajectory type.
Future work: add one-hot trajectory encoding to observation.

### Evaluation Method

Tracking accuracy is measured as Euclidean distance between
end effector position and target position at every timestep.
This is reported as:
- Mean error over episode
- Maximum error (worst case)
- Percentage of time under 1cm and 5cm thresholds

These metrics match how real tracking systems are evaluated
in industrial and surgical robotics applications.
