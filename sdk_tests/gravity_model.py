"""Gravity torque model of the Piper arm, built from its URDF with numpy only.

tau_g(q) is the joint torque needed to hold the arm static at q (same convention as
pinocchio.computeGeneralizedGravity). Links hanging off the chain (gripper_base and both
fingers) are lumped into link6 when include_tool=True.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_URDF = (Path(__file__).resolve().parents[1]
                / "ros2_ws/src/piper_description/urdf/piper_description.urdf")
# Same arm kinematics, different end-effector inertia. The pika file also carries AgileX's
# newer link2/link3 centres of mass, so pick the one that matches the hardware.
TOOL_URDF = {
    "piper": DEFAULT_URDF,
    "pika": DEFAULT_URDF.parent / "piper_pika_gripper.urdf",
}
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def _vec(text, default="0 0 0"):
    return np.array([float(v) for v in (text or default).split()])


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(element):
    T = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        T[:3, :3] = _rpy_matrix(*_vec(origin.get("rpy")))
        T[:3, 3] = _vec(origin.get("xyz"))
    return T


def _inertial_com(inertial):
    origin = inertial.find("origin")
    return _vec(origin.get("xyz")) if origin is not None else np.zeros(3)


def _rotation(axis, angle):
    """Homogeneous rotation about a unit axis (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return T


def rotation_error(r_desired, r_current):
    """Rotation vector taking r_current onto r_desired, in base coordinates."""
    e = r_desired @ r_current.T
    axis = np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]]) / 2.0
    sin_angle = np.linalg.norm(axis)
    if sin_angle < 1e-9:
        return np.zeros(3)
    return axis / sin_angle * np.arctan2(sin_angle, (np.trace(e) - 1.0) / 2.0)


def rotate(rotation, angular_step):
    """Apply a rotation vector (axis * angle) to a rotation matrix."""
    angle = np.linalg.norm(angular_step)
    if angle < 1e-12:
        return rotation
    return _rotation(angular_step / angle, angle)[:3, :3] @ rotation


def ik_step(model, q, target_position, target_rotation, tip_offset=(0.0, 0.0, 0.0),
            damping=0.08, max_step=0.15, orientation_weight=1.0, limits=None, centering=0.0):
    """One damped-least-squares step of inverse kinematics.

    Damped least squares keeps the step finite near singularities, where a plain inverse
    would ask for enormous joint speeds.

    orientation_weight below 1 tells the solver that position matters more than
    orientation. A full 6-DoF pose uses up every joint of this arm, so insisting on the
    exact orientation while translating is what drives it into limits; giving the
    orientation less weight lets it drift a few degrees and keeps the position accurate.

    centering pulls the joints toward mid-range through the nullspace, which moves the arm
    away from its limits without changing where the tool is.
    """
    position, rotation = model.tip_pose(q, tip_offset)
    error = np.concatenate([target_position - position,
                            orientation_weight * rotation_error(target_rotation, rotation)])
    jacobian = model.jacobian(q, tip_offset).copy()
    jacobian[3:] *= orientation_weight

    # Limit-aware: a joint that is being pushed past its limit gets locked and the others
    # re-solve to compensate. Clamping the result afterwards instead would silently drop
    # that joint's contribution and wreck the whole solution.
    free = np.ones(jacobian.shape[1], dtype=bool)
    for _ in range(3):
        restricted = jacobian * free
        dq = restricted.T @ np.linalg.solve(
            restricted @ restricted.T + damping**2 * np.eye(6), error)
        dq = np.where(free, dq, 0.0)
        if limits is None:
            break
        # Only a joint that is already against a limit and still being pushed outwards:
        # locking every joint whose full step would overshoot starves the solution instead.
        margin = 1e-4
        blocked = free & (((q <= limits[:, 0] + margin) & (dq < 0))
                          | ((q >= limits[:, 1] - margin) & (dq > 0)))
        if not blocked.any():
            break
        free &= ~blocked

    if centering > 0.0 and limits is not None:
        middle = limits.mean(axis=1)
        half_span = np.maximum((limits[:, 1] - limits[:, 0]) / 2.0, 1e-6)
        # Cubed, so the pull is negligible through the middle of the range and only bites
        # near the ends. A linear pull has to be kept tiny to avoid fighting the operator
        # everywhere, and then it is far too weak to stop a joint walking into its limit.
        pull = -centering * ((q - middle) / half_span) ** 3
        nullspace = pull - jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), jacobian @ pull)
        dq = dq + nullspace
    return q + np.clip(dq, -max_step, max_step)


class GravityModel:
    def __init__(self, urdf_path=DEFAULT_URDF, joints=ARM_JOINTS, include_tool=True,
                 gravity=(0.0, 0.0, -9.81)):
        root = ET.parse(urdf_path).getroot()
        joint_elems = {j.get("name"): j for j in root.findall("joint")}
        self._children = {}
        for j in root.findall("joint"):
            self._children.setdefault(j.find("parent").get("link"), []).append(j)
        self._inertial = {}
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                self._inertial[link.get("name")] = (float(inertial.find("mass").get("value")),
                                                    _inertial_com(inertial))

        chain = [joint_elems[name] for name in joints]
        for parent, child in zip(chain, chain[1:]):
            if parent.find("child").get("link") != child.find("parent").get("link"):
                raise ValueError(f"{parent.get('name')} -> {child.get('name')} is not a serial chain")

        self.joint_names = list(joints)
        self.origins = [_origin(j) for j in chain]
        self.axes = [_vec(j.find("axis").get("xyz")) / np.linalg.norm(_vec(j.find("axis").get("xyz")))
                     for j in chain]
        self.masses = np.zeros(len(chain))
        self.coms = np.zeros((len(chain), 3))  # COM of each lumped body, in its child-link frame
        for k, joint in enumerate(chain):
            is_last = k == len(chain) - 1
            skip = None if is_last else chain[k + 1].get("name")
            parts = []
            self._collect(joint.find("child").get("link"), np.eye(4), skip,
                          include_tool or not is_last, parts)
            mass = sum(m for m, _ in parts)
            self.masses[k] = mass
            if mass > 0:
                self.coms[k] = sum(m * c for m, c in parts) / mass
        self.gravity = np.asarray(gravity, dtype=float)
        # Fixed mount from the last chain link to the tool body. On the Pika gripper this
        # is a -90 deg pitch, so the tool points along link6 +Z, not +X: anything that
        # treats the tool as "forward" has to go through this.
        self.tool_mount = np.eye(4)
        for joint in self._children.get(chain[-1].find("child").get("link"), []):
            if joint.get("type") == "fixed":
                self.tool_mount = _origin(joint)
                break

    def tool_to_link(self, point_in_tool):
        """A point given in the tool frame, expressed in the last chain link's frame."""
        return (self.tool_mount @ np.append(np.asarray(point_in_tool, dtype=float), 1.0))[:3]

    def _collect(self, link, T, skip_joint, descend, parts):
        mass, com = self._inertial.get(link, (0.0, np.zeros(3)))
        if mass > 0:
            parts.append((mass, T[:3, :3] @ com + T[:3, 3]))
        if not descend:
            return
        for j in self._children.get(link, []):
            if j.get("name") != skip_joint:
                # Off-chain joints (fixed tool mount, gripper fingers) are taken at zero position.
                self._collect(j.find("child").get("link"), T @ _origin(j), None, True, parts)

    def add_payload(self, mass, com_in_last_link):
        """Point mass rigidly attached to the last link (e.g. a grasped object)."""
        k = len(self.masses) - 1
        total = self.masses[k] + mass
        if total > 0:
            self.coms[k] = (self.masses[k] * self.coms[k] + mass * np.asarray(com_in_last_link)) / total
        self.masses[k] = total

    def link_frames(self, q):
        """World (= base_link) transforms of the child link of every chain joint."""
        T = np.eye(4)
        frames = []
        for origin, axis, angle in zip(self.origins, self.axes, q):
            T = T @ origin @ _rotation(axis, angle)
            frames.append(T)
        return frames

    def tip_pose(self, q, tip_offset=(0.0, 0.0, 0.0)):
        """Tool pose in base coordinates: (position, rotation matrix).

        tip_offset is the TCP expressed in the last link's frame.
        """
        last = self.link_frames(q)[-1]
        return last[:3, :3] @ np.asarray(tip_offset, dtype=float) + last[:3, 3], last[:3, :3]

    def jacobian(self, q, tip_offset=(0.0, 0.0, 0.0)):
        """Geometric Jacobian of the tool, in base coordinates: [v; omega] = J @ qdot."""
        frames = self.link_frames(q)
        tip = frames[-1][:3, :3] @ np.asarray(tip_offset, dtype=float) + frames[-1][:3, 3]
        J = np.zeros((6, len(frames)))
        for i, F in enumerate(frames):
            axis = F[:3, :3] @ self.axes[i]
            J[:3, i] = np.cross(axis, tip - F[:3, 3])  # revolute joints only
            J[3:, i] = axis
        return J

    def gravity_torques(self, q):
        frames = self.link_frames(q)
        joint_origins = [F[:3, 3] for F in frames]
        joint_axes = [F[:3, :3] @ a for F, a in zip(frames, self.axes)]
        tau = np.zeros(len(frames))
        for i, F in enumerate(frames):
            if self.masses[i] == 0:
                continue
            com = F[:3, :3] @ self.coms[i] + F[:3, 3]
            force = self.masses[i] * self.gravity
            for j in range(i + 1):
                tau[j] -= joint_axes[j] @ np.cross(com - joint_origins[j], force)
        return tau

