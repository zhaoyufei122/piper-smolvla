"""Time sampling of a joint trajectory (no ROS dependency, easy to unit test)."""
import numpy as np


class TrajectorySampler:
    """Cubic Hermite interpolation when velocities are given, linear otherwise.

    If the first point is not at t=0, start_positions is prepended as the t=0 point
    (the FollowJointTrajectory convention).
    """

    def __init__(self, times, positions, velocities=None, start_positions=None):
        times = np.asarray(times, dtype=float)
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        velocities = None if velocities is None else np.atleast_2d(np.asarray(velocities, dtype=float))
        if times.size != positions.shape[0]:
            raise ValueError("times and positions have different lengths")
        if np.any(np.diff(times) < 0):
            raise ValueError("trajectory times must be non-decreasing")
        if times[0] > 0.0 and start_positions is not None:
            times = np.concatenate(([0.0], times))
            positions = np.vstack((start_positions, positions))
            if velocities is not None:
                velocities = np.vstack((np.zeros(positions.shape[1]), velocities))
        self.times = times
        self.positions = positions
        self.velocities = velocities

    @property
    def duration(self):
        return float(self.times[-1])

    @property
    def final_positions(self):
        return self.positions[-1]

    def sample(self, t):
        """Returns (positions, velocities) at time t, clamped to the trajectory ends."""
        n_joints = self.positions.shape[1]
        if t <= self.times[0]:
            return self.positions[0].copy(), np.zeros(n_joints)
        if t >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(n_joints)
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        t0, t1 = self.times[k], self.times[k + 1]
        h = t1 - t0
        p0, p1 = self.positions[k], self.positions[k + 1]
        if h <= 0.0:
            return p1.copy(), np.zeros(n_joints)
        s = (t - t0) / h
        if self.velocities is None:
            return p0 + s * (p1 - p0), (p1 - p0) / h
        v0, v1 = self.velocities[k], self.velocities[k + 1]
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
        d00, d10 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1
        d01, d11 = -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        vel = (d00 * p0 + d01 * p1) / h + d10 * v0 + d11 * v1
        return pos, vel
