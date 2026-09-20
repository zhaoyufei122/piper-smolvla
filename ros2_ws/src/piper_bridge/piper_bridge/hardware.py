"""Arm back-ends for the bridge: the real Piper through piper_sdk, or a kinematic simulator.

Both expose the same interface in SI units (rad, rad/s, N*m, m).
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

# Joint limits from the SDK JointCtrl docstring [rad].
JOINT_LIMITS = np.array([
    [-2.6179, 2.6179],
    [0.0, 3.14],
    [-2.967, 0.0],
    [-1.745, 1.745],
    [-1.22, 1.22],
    [-2.09439, 2.09439],
])
GRIPPER_MAX_WIDTH = 0.07  # m
MDEG_PER_RAD = 180000.0 / math.pi  # SDK joint unit is 0.001 deg


@dataclass
class ArmState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_width: float = 0.0
    gripper_effort: float = 0.0


class SdkArm:
    def __init__(self, can_port, speed_percent=100):
        from piper_sdk import C_PiperInterface_V2  # imported here so sim mode works without the SDK

        self._piper = C_PiperInterface_V2(can_port)
        self._piper.ConnectPort()
        self._speed = int(speed_percent)

    def wait_for_feedback(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._piper.GetArmJointMsgs().time_stamp > 0:
                return True
            time.sleep(0.05)
        return False

    def read(self):
        js = self._piper.GetArmJointMsgs().joint_state
        hs = self._piper.GetArmHighSpdInfoMsgs()
        gs = self._piper.GetArmGripperMsgs().gripper_state
        motors = [hs.motor_1, hs.motor_2, hs.motor_3, hs.motor_4, hs.motor_5, hs.motor_6]
        return ArmState(
            position=np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
                              dtype=float) / MDEG_PER_RAD,
            velocity=np.array([m.motor_speed for m in motors], dtype=float) / 1000.0,
            effort=np.array([m.effort for m in motors], dtype=float) / 1000.0,
            gripper_width=gs.grippers_angle / 1e6,
            gripper_effort=gs.grippers_effort / 1000.0,
        )

    def feedback_age(self):
        return time.time() - self._piper.GetArmJointMsgs().time_stamp

    def enable(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._piper.EnablePiper():
                return True
            time.sleep(0.05)
        return False

    def disable(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self._piper.DisablePiper():
                return True
            time.sleep(0.05)
        return False

    def command_joints(self, q):
        mdeg = [int(round(v * MDEG_PER_RAD)) for v in np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])]
        self._piper.MotionCtrl_2(0x01, 0x01, self._speed, 0x00)
        self._piper.JointCtrl(*mdeg)

    def command_gripper(self, width, effort):
        width_um = int(round(np.clip(width, 0.0, GRIPPER_MAX_WIDTH) * 1e6))
        self._piper.GripperCtrl(width_um, int(round(effort * 1000)), 0x01, 0)

    def status_text(self):
        s = self._piper.GetArmStatus().arm_status
        return f"ctrl_mode={s.ctrl_mode} arm_status={s.arm_status} move_mode={s.mode_feed}"


class SimArm:
    """Tracks the last command with a joint speed limit. Good enough to test MoveIt plumbing."""

    def __init__(self, initial_positions=None, max_velocity=1.5, gripper_velocity=0.05):
        self._q = np.zeros(6) if initial_positions is None else np.asarray(initial_positions, dtype=float)
        self._dq = np.zeros(6)
        self._q_cmd = self._q.copy()
        self._width = 0.0
        self._width_cmd = 0.0
        self._max_vel = max_velocity
        self._gripper_vel = gripper_velocity
        self._last = time.monotonic()

    def wait_for_feedback(self, timeout):
        return True

    def read(self):
        now = time.monotonic()
        dt = min(now - self._last, 0.1)
        self._last = now
        step = np.clip(self._q_cmd - self._q, -self._max_vel * dt, self._max_vel * dt)
        self._q = self._q + step
        self._dq = step / dt if dt > 0 else np.zeros(6)
        self._width += float(np.clip(self._width_cmd - self._width, -self._gripper_vel * dt,
                                     self._gripper_vel * dt))
        return ArmState(position=self._q.copy(), velocity=self._dq.copy(), gripper_width=self._width)

    def feedback_age(self):
        return 0.0

    def enable(self, timeout=5.0):
        return True

    def disable(self, timeout=5.0):
        return True

    def command_joints(self, q):
        self._q_cmd = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def command_gripper(self, width, effort):
        self._width_cmd = float(np.clip(width, 0.0, GRIPPER_MAX_WIDTH))

    def status_text(self):
        return "simulated arm"
