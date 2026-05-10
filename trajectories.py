# =============================================================
#   TRAJECTORIES
#   Defines the three target trajectories the arm must track
#   Circle, Figure-8, and Moving Target
#
#   WHAT THIS FILE DOES:
#   At any time t, returns where the target should be [x, y, z]
#   The arm must follow this moving point continuously
#
#   WORKSPACE (from Reacher-v5 measurements):
#   X: -0.182 to 0.187  centre = 0.0
#   Y: -0.151 to 0.197  centre = 0.02
#   Z: 0.01 (fixed - Reacher is a 2D arm)
#   Safe radius: 0.10m
# =============================================================

import numpy as np


class CircleTrajectory:
    """
    Target moves in a perfect circle.
    Easiest trajectory - constant speed, predictable.

    How it works:
      x = centre_x + radius * cos(speed * t)
      y = centre_y + radius * sin(speed * t)
      z = 0.01 (fixed - Reacher only moves in XY plane)
    """

    def __init__(self,
                 centre = [0.0, 0.02, 0.01],
                 radius = 0.10,
                 speed  = 1.0):
        self.centre = np.array(centre)
        self.radius = radius
        self.speed  = speed
        self.name   = "circle"

    def get_position(self, t):
        """Get target position at time t"""
        x = self.centre[0] + self.radius * np.cos(self.speed * t)
        y = self.centre[1] + self.radius * np.sin(self.speed * t)
        z = self.centre[2]
        return np.array([x, y, z])

    def get_velocity(self, t):
        """
        Get target velocity at time t.
        Derivative of position - agent uses this to predict
        where target will be next step.
        """
        vx = -self.radius * self.speed * np.sin(self.speed * t)
        vy =  self.radius * self.speed * np.cos(self.speed * t)
        vz = 0.0
        return np.array([vx, vy, vz])

    def get_period(self):
        """How long one full circle takes"""
        return 2 * np.pi / self.speed


class FigureEightTrajectory:
    """
    Target traces a figure-8 shape.
    Harder - direction reverses at crossing point.

    How it works:
      x = centre_x + scale * sin(speed * t)
      y = centre_y + scale * sin(speed * t) * cos(speed * t)
    """

    def __init__(self,
                 centre = [0.0, 0.02, 0.01],
                 scale  = 0.08,
                 speed  = 1.0):
        self.centre = np.array(centre)
        self.scale  = scale
        self.speed  = speed
        self.name   = "figure_eight"

    def get_position(self, t):
        s = self.speed * t
        x = self.centre[0] + self.scale * np.sin(s)
        y = self.centre[1] + self.scale * np.sin(s) * np.cos(s)
        z = self.centre[2]
        return np.array([x, y, z])

    def get_velocity(self, t):
        s  = self.speed * t
        vx = self.scale * self.speed * np.cos(s)
        vy = self.scale * self.speed * (
            np.cos(s) ** 2 - np.sin(s) ** 2
        )
        vz = 0.0
        return np.array([vx, vy, vz])

    def get_period(self):
        return 2 * np.pi / self.speed


class MovingTargetTrajectory:
    """
    Target moves randomly but smoothly.
    Sometimes goes to edge of workspace - tests robustness.
    Uses Ornstein-Uhlenbeck process for smooth random motion.
    """

    def __init__(self,
                 centre     = [0.0, 0.02, 0.01],
                 max_radius = 0.12,
                 speed      = 0.5,
                 seed       = 42):
        self.centre     = np.array(centre)
        self.max_radius = max_radius
        self.speed      = speed
        self.name       = "moving_target"

        # Pre-generate smooth random trajectory
        rng     = np.random.default_rng(seed)
        n_steps = 10000
        dt      = 0.01
        theta   = 0.3   # pull back to centre
        sigma   = 0.15  # randomness amount

        pos = np.zeros((n_steps, 2))
        vel = np.zeros((n_steps, 2))

        for i in range(1, n_steps):
            vel[i] = (vel[i-1]
                      - theta * vel[i-1] * dt
                      + sigma * rng.normal(0, 1, 2) * np.sqrt(dt))
            pos[i] = pos[i-1] + vel[i] * dt

            # Keep within max_radius
            r = np.linalg.norm(pos[i])
            if r > max_radius:
                pos[i] = pos[i] / r * max_radius

        self._pos_table = pos
        self._vel_table = vel
        self._dt        = dt

    def get_position(self, t):
        idx    = int((t * self.speed) / self._dt) % len(self._pos_table)
        offset = self._pos_table[idx]
        return np.array([
            self.centre[0] + offset[0],
            self.centre[1] + offset[1],
            self.centre[2]
        ])

    def get_velocity(self, t):
        idx = int((t * self.speed) / self._dt) % len(self._vel_table)
        vel = self._vel_table[idx] * self.speed
        return np.array([vel[0], vel[1], 0.0])

    def get_period(self):
        return len(self._pos_table) * self._dt / self.speed


def make_trajectory(name, **kwargs):
    """
    Factory function - creates trajectory by name.

    Usage:
      traj = make_trajectory("circle")
      pos  = traj.get_position(t=0.5)
    """
    trajectories = {
        "circle":       CircleTrajectory,
        "figure_eight": FigureEightTrajectory,
        "moving":       MovingTargetTrajectory,
    }
    if name not in trajectories:
        raise ValueError(
            f"Unknown trajectory: {name}. "
            f"Choose from {list(trajectories.keys())}"
        )
    return trajectories[name](**kwargs)


# =============================================================
#   TEST - plot all three trajectories
# =============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t_vals    = np.linspace(0, 6.28, 500)

    trajs = [
        CircleTrajectory(),
        FigureEightTrajectory(),
        MovingTargetTrajectory(),
    ]

    # Workspace boundary for reference
    workspace_x = [-0.182, 0.187, 0.187, -0.182, -0.182]
    workspace_y = [-0.151, -0.151, 0.197, 0.197, -0.151]

    for ax, traj in zip(axes, trajs):
        positions = np.array([traj.get_position(t) for t in t_vals])

        # Show workspace boundary
        ax.plot(workspace_x, workspace_y, 'k--',
                alpha=0.3, label='Workspace')

        ax.plot(positions[:, 0], positions[:, 1],
                'b-', linewidth=2, label='Trajectory')
        ax.plot(positions[0, 0], positions[0, 1],
                'go', markersize=10, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1],
                'rs', markersize=10, label='End')
        ax.set_title(traj.name, fontsize=14)
        ax.set_xlabel('X (metres)')
        ax.set_ylabel('Y (metres)')
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)

    plt.suptitle('Target Trajectories (Reacher Workspace)',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('trajectories.png', dpi=150)
    plt.show()
    print("Saved: trajectories.png")