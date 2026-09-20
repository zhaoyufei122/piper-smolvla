"""ROS 2 bridge between MoveIt and the AgileX Piper arm.

- publishes /joint_states from CAN feedback (joint1..6, plus joint7/joint8 for the fingers)
- serves arm_controller/follow_joint_trajectory and gripper_controller/follow_joint_trajectory:
  the trajectory is sampled at `rate` Hz and streamed to the arm as position setpoints
- ~/enable (std_srvs/SetBool) enables or disables the motors

With use_sim:=true a kinematic simulator replaces the arm, so the MoveIt side can be
tested without hardware.
"""
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from piper_bridge.hardware import GRIPPER_MAX_WIDTH, JOINT_LIMITS, SdkArm, SimArm
from piper_bridge.trajectory import TrajectorySampler

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]  # joint7 = width / 2, joint8 = -width / 2
GRIPPER_CMD_EVERY = 10  # re-send an unchanged gripper command every N cycles
Result = FollowJointTrajectory.Result


class Controller:
    """Command slot and active goal of one FollowJointTrajectory interface."""

    def __init__(self, name, joints, lower, upper, strict):
        self.name = name
        self.joints = joints
        self.lower = lower
        self.upper = upper
        # strict: enforce path/goal tolerances. The gripper is not strict because a
        # grasped object legitimately stops the fingers short of the target.
        self.strict = strict
        self.command = None  # setpoint streamed by the timer; None = send nothing
        self.goal = None     # the goal handle currently allowed to write `command`


class PiperBridge(Node):
    def __init__(self):
        super().__init__("piper_bridge")
        use_sim = self.declare_parameter("use_sim", False).value
        can_port = self.declare_parameter("can_port", "can0").value
        self.rate = float(self.declare_parameter("rate", 200.0).value)
        speed_percent = self.declare_parameter("speed_percent", 100).value
        self.gripper_exist = self.declare_parameter("gripper_exist", True).value
        self.gripper_effort = float(self.declare_parameter("gripper_effort", 1.0).value)
        # joint7 opens positive on the standard gripper and negative on the Pika one.
        self.finger_sign = float(self.declare_parameter("finger_sign", 1.0).value)
        auto_enable = self.declare_parameter("auto_enable", True).value
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.02).value)
        self.path_tolerance = float(self.declare_parameter("path_tolerance", 0.5).value)
        self.goal_time_margin = float(self.declare_parameter("goal_time_margin", 2.0).value)

        if use_sim:
            self.hw = SimArm()
        else:
            self.hw = SdkArm(can_port, speed_percent)
            if not self.hw.wait_for_feedback(3.0):
                raise RuntimeError(f"no feedback from the arm on {can_port} "
                                   "(powered on? CAN up at 1000000 bit/s?)")
        self.get_logger().info(f"connected: {self.hw.status_text()}")

        self._lock = threading.Lock()
        self._enabled = False
        self._state = self.hw.read()
        self._cycle = 0
        self._last_gripper_width = None
        self.arm = Controller("arm_controller", ARM_JOINTS, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], True)
        finger_limit = self.finger_sign * GRIPPER_MAX_WIDTH / 2.0
        self.gripper = Controller("gripper_controller", ["joint7"],
                                  np.full(1, min(0.0, finger_limit)),
                                  np.full(1, max(0.0, finger_limit)), False)

        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(JointState, "~/joint_command", self._on_joint_command, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)
        action_group = ReentrantCallbackGroup()
        self._action_servers = [self._make_action_server(self.arm, action_group)]
        if self.gripper_exist:
            self._action_servers.append(self._make_action_server(self.gripper, action_group))
        self.create_timer(1.0 / self.rate, self._on_timer, callback_group=MutuallyExclusiveCallbackGroup())

        if auto_enable:
            self._set_enabled(True)
        else:
            self.get_logger().info("motors not enabled (auto_enable:=false); call ~/enable to enable")

    # ------------------------------------------------------------------ state / command loop
    def _on_timer(self):
        state = self.hw.read()
        with self._lock:
            self._state = state
            enabled = self._enabled
            arm_cmd = self.arm.command
            gripper_cmd = self.gripper.command
        if enabled and arm_cmd is not None:
            self.hw.command_joints(arm_cmd)
        if enabled and self.gripper_exist and gripper_cmd is not None:
            width = 2.0 * self.finger_sign * float(gripper_cmd[0])
            if width != self._last_gripper_width or self._cycle % GRIPPER_CMD_EVERY == 0:
                self.hw.command_gripper(width, self.gripper_effort)
                self._last_gripper_width = width
        self._cycle += 1

        if self.hw.feedback_age() > 0.5:
            self.get_logger().warn("arm feedback is stale (CAN cable / power?)", throttle_duration_sec=2.0)

        names = list(ARM_JOINTS)
        position = state.position.tolist()
        velocity = state.velocity.tolist()
        effort = state.effort.tolist()
        if self.gripper_exist:
            half = self.finger_sign * state.gripper_width / 2.0
            names += FINGER_JOINTS
            position += [half, -half]
            velocity += [0.0, 0.0]
            effort += [state.gripper_effort, 0.0]
        msg = JointState(name=names, position=position, velocity=velocity, effort=effort)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._joint_pub.publish(msg)

    def _measured(self, ctrl):
        """Current positions of a controller's joints. Call with the lock held."""
        if ctrl is self.arm:
            return self._state.position.copy()
        return np.array([self.finger_sign * self._state.gripper_width / 2.0])

    def _set_enabled(self, enable):
        if enable:
            if not self.hw.enable():
                self.get_logger().error("enable failed (e-stop? after MIT/teach mode the arm needs a reset)")
                return False
            with self._lock:
                self.arm.command = self._state.position.copy()  # hold where it is
                self._enabled = True
            self.get_logger().info("motors enabled, holding the current pose")
            return True
        with self._lock:
            self._enabled = False
            for ctrl in (self.arm, self.gripper):
                ctrl.command = None
                ctrl.goal = None
        self.get_logger().warn("disabling motors: the arm drops unless it rests at the zero pose")
        return self.hw.disable()

    def _on_joint_command(self, msg):
        """Streaming setpoints for teleoperation and policy rollout.

        Names may be joint1..joint6 (rad) plus either 'joint7' (one finger, signed) or
        'gripper' (total opening in m). Unnamed messages are taken as joint1..joint6 in
        order. A running trajectory goal keeps ownership, so the two cannot fight.
        """
        if not self._enabled:
            self.get_logger().warn("joint_command ignored: motors are not enabled",
                                   throttle_duration_sec=5.0)
            return
        names = list(msg.name)
        values = dict(zip(names, msg.position)) if names else \
            dict(zip(ARM_JOINTS, msg.position[:6]))
        arm = [values.get(j) for j in ARM_JOINTS]
        with self._lock:
            if self.arm.goal is not None:
                self.get_logger().warn("joint_command ignored: a trajectory goal is running",
                                       throttle_duration_sec=5.0)
                return
            if all(v is not None for v in arm):
                self.arm.command = np.clip(np.array(arm, dtype=float),
                                           JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            if self.gripper_exist and self.gripper.goal is None:
                if "gripper" in values:  # total opening in metres
                    finger = self.finger_sign * float(values["gripper"]) / 2.0
                elif "joint7" in values:
                    finger = float(values["joint7"])
                else:
                    return
                self.gripper.command = np.clip([finger], self.gripper.lower, self.gripper.upper)

    def _on_enable(self, request, response):
        response.success = self._set_enabled(request.data)
        response.message = ("enabled" if request.data else "disabled") if response.success else "failed"
        return response

    # ------------------------------------------------------------------ trajectory actions
    def _make_action_server(self, ctrl, callback_group):
        return ActionServer(
            self, FollowJointTrajectory, f"{ctrl.name}/follow_joint_trajectory",
            execute_callback=lambda goal_handle: self._execute(ctrl, goal_handle),
            goal_callback=lambda goal: self._on_goal(ctrl, goal),
            handle_accepted_callback=lambda goal_handle: self._on_accepted(ctrl, goal_handle),
            cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )

    def _on_goal(self, ctrl, goal):
        traj = goal.trajectory
        if not self._enabled:
            self.get_logger().error(f"{ctrl.name}: rejecting goal, motors are not enabled")
            return GoalResponse.REJECT
        if sorted(traj.joint_names) != sorted(ctrl.joints) or not traj.points:
            self.get_logger().error(f"{ctrl.name}: rejecting goal for joints {list(traj.joint_names)}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_accepted(self, ctrl, goal_handle):
        with self._lock:
            ctrl.goal = goal_handle  # an older goal still executing notices this and aborts
        goal_handle.execute()

    def _execute(self, ctrl, goal_handle):
        traj = goal_handle.request.trajectory
        order = [list(traj.joint_names).index(j) for j in ctrl.joints]
        try:
            times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points]
            positions = [[p.positions[k] for k in order] for p in traj.points]
            has_velocities = all(len(p.velocities) == len(traj.joint_names) for p in traj.points)
            velocities = [[p.velocities[k] for k in order] for p in traj.points] if has_velocities else None
            with self._lock:
                start = self._measured(ctrl)
            sampler = TrajectorySampler(times, positions, velocities, start)
        except (IndexError, ValueError) as exc:
            return self._finish(ctrl, goal_handle, Result.INVALID_GOAL, f"bad trajectory: {exc}")

        self.get_logger().info(f"{ctrl.name}: executing {len(traj.points)} points "
                               f"over {sampler.duration:.2f} s")
        t0 = time.monotonic()
        next_feedback = 0.0
        while rclpy.ok():
            t = time.monotonic() - t0
            desired = np.clip(sampler.sample(t)[0], ctrl.lower, ctrl.upper)
            with self._lock:
                active = ctrl.goal is goal_handle and self._enabled
                measured = self._measured(ctrl)
                if active:
                    ctrl.command = measured if goal_handle.is_cancel_requested else desired
            if not active:
                return self._finish(ctrl, goal_handle, Result.INVALID_GOAL,
                                    "preempted by a newer goal or motors disabled")
            if goal_handle.is_cancel_requested:
                return self._finish(ctrl, goal_handle, None, "canceled, holding the current pose")

            error = desired - measured
            if ctrl.strict and self.path_tolerance > 0 and np.max(np.abs(error)) > self.path_tolerance:
                return self._finish(ctrl, goal_handle, Result.PATH_TOLERANCE_VIOLATED,
                                    f"tracking error {np.max(np.abs(error)):.3f} rad > path_tolerance",
                                    hold=measured)
            if t >= sampler.duration:
                final_error = float(np.max(np.abs(sampler.final_positions - measured)))
                if final_error <= self.goal_tolerance or (not ctrl.strict and t >= sampler.duration + 0.5):
                    return self._finish(ctrl, goal_handle, Result.SUCCESSFUL, "")
                if t > sampler.duration + self.goal_time_margin:
                    return self._finish(ctrl, goal_handle, Result.GOAL_TOLERANCE_VIOLATED,
                                        f"final error {final_error:.3f} > goal_tolerance")
            if t >= next_feedback:
                feedback = FollowJointTrajectory.Feedback()
                feedback.header.stamp = self.get_clock().now().to_msg()
                feedback.joint_names = ctrl.joints
                feedback.desired.positions = desired.tolist()
                feedback.actual.positions = measured.tolist()
                feedback.error.positions = error.tolist()
                goal_handle.publish_feedback(feedback)
                next_feedback = t + 0.05
            time.sleep(1.0 / self.rate)
        return self._finish(ctrl, goal_handle, Result.INVALID_GOAL, "shutting down")

    def _finish(self, ctrl, goal_handle, error_code, message, hold=None):
        """Release the controller and report the outcome. error_code None means canceled."""
        with self._lock:
            if ctrl.goal is goal_handle:
                if hold is not None:
                    ctrl.command = hold
                ctrl.goal = None
        result = Result()
        result.error_string = message
        if error_code is None:
            goal_handle.canceled()
            self.get_logger().info(f"{ctrl.name}: {message}")
        elif error_code == Result.SUCCESSFUL:
            result.error_code = error_code
            goal_handle.succeed()
            self.get_logger().info(f"{ctrl.name}: goal reached")
        else:
            result.error_code = error_code
            goal_handle.abort()
            self.get_logger().error(f"{ctrl.name}: aborted: {message}")
        return result


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PiperBridge()
    except (RuntimeError, ConnectionError) as exc:
        print(f"[piper_bridge] {exc}\n"
              "[piper_bridge] check arm power and CAN: bash scripts/can_activate.sh can0 1000000")
        rclpy.try_shutdown()
        return 1
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Motors are intentionally left enabled: the firmware keeps holding the last setpoint.
        try:
            node.destroy_node()
        except KeyboardInterrupt:  # a second Ctrl+C from ros2 launch during cleanup
            pass
        rclpy.try_shutdown()
    return 0
