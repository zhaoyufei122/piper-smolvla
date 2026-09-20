#!/usr/bin/env python3
"""Jog the Piper with a SpaceMouse: Cartesian input, position control on the joints.

    SpaceMouse -> axis filter -> tool twist -> target pose -> damped least squares IK
               -> joint limits + speed clamp -> arm

Two backends, same control code:
    --backend sdk   straight to the arm over CAN (default)
    --backend ros   publish to piper_bridge, which drives the real arm or the simulated
                    one, so you can try the mapping in RViz before touching hardware:
                        ros2 launch piper_bridge display.launch.py use_sim:=true auto_enable:=true
                        python3 teleop_spacemouse.py --backend ros

Buttons (hold them, they are continuous, not toggles):
    left  = open the gripper         right = close the gripper (one press, not held)

Keys, in this terminal:
    a     re-anchor the target on where the arm actually is (use after a stall)
    1     strict mode: only the single largest axis is allowed through
    [ / ] shift down / up a gear: PRECISE - SLOW - NORMAL - FAST - TURBO
    w     save the current pose as a waypoint;  g  flies back to it
    h     fly back to the pose this session started in
    m     switch control frame: tool (first-person) <-> base (fixed table directions)
    f     rotate the input frame by 90 deg, to line it up with the camera or with you
    t / r toggle translation / rotation independently (rotation starts off)
    space stop jogging and hold
    R     start / stop recording an episode (needs --record)
    T     switch task, when several --task were given (between episodes only)
    X     stop and discard the current episode
    q     quit

When it will not go any further: the arm has 6 joints and a full position+orientation
target uses all of them, so holding the tool orientation exactly is what runs joint5
(+-70 deg, the narrowest) out of range. Instead of stalling there, the orientation hold
fades out while the arm cannot follow, the tool tilts, and the position keeps going; the
hold fades back in and the tool turns back once there is room. The status line shows
"ori 8%" while that is happening.

Why the axis filter: pushing the puck forward also produces some pitch, which a deadzone
alone will not remove because the bleed is genuinely large. An axis is therefore zeroed
when it is smaller than --axis-ratio times the biggest axis, so pure pushes stay pure
while deliberate combined motions still work.
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import piper_common as pc
from gravity_compensation import Rate, announce
import hud
from gravity_model import TOOL_URDF, GravityModel, ik_step, rotate, rotation_error
from recorder import EpisodeWriter, RealSenseCamera, UvcCamera, next_episode_dir
from spacemouse import SpaceMouse

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

# One damped IK step per cycle leaves a tracking error proportional to how fast the target
# is moving: measured at the working pose, about 0.15 s worth of travel (3 mm at 12 cm/s,
# 25 mm at 22 cm/s). The leash has to allow for that, or every fast move looks like a stall.
LEASH_TIME = 0.15


# SpaceMouse axes in its own frame -> robot base frame (base X forward, Y left, Z up).
# Push forward = +X, push right = -Y, lift = +Z.
#
# Rotation has to go through the SAME mapping, or tilting the puck turns the tool the wrong
# way. Careful with the driver's names: it unpacks the HID rotation report as
# (pitch, roll, yaw) but the report order is (rx, ry, rz), so its "pitch" is really rotation
# about the device X axis and its "roll" about device Y -- swapped, and negated on the way
# in. Working back to the raw device axes and applying the translation mapping:
#     raw_rx = -state.pitch,  raw_ry = -state.roll,  raw_rz = state.yaw
#     angular = [-raw_ry, -raw_rx, -raw_rz] = [state.roll, state.pitch, -state.yaw]
def to_base_twist(state):
    linear = np.array([state.y, -state.x, state.z])
    angular = np.array([state.roll, state.pitch, -state.yaw])
    return linear, angular


def shape(value, deadzone, exponent):
    """Deadzone, rescale to a full 0..1 range, then bend the curve.

    Linear response makes a SpaceMouse almost impossible to aim: a light touch already
    commands a third of full speed. With an exponent the first half of the travel is slow
    and precise, and the last part is fast.
    """
    magnitude = np.clip((np.abs(value) - deadzone) / (1.0 - deadzone), 0.0, 1.0)
    return np.sign(value) * magnitude ** exponent


def yaw_matrix(degrees):
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def filter_axes(linear, angular, deadzone, rot_deadzone, ratio, strict, mode, exponent=2.0):
    """Deadzone plus response curve, then suppress axes far smaller than the dominant one."""
    linear = shape(linear, deadzone, exponent)
    angular = shape(angular, rot_deadzone, exponent)
    if mode in ("translation", "none"):
        angular = np.zeros(3)
    if mode in ("rotation", "none"):
        linear = np.zeros(3)
    axes = np.concatenate([linear, angular])
    peak = np.max(np.abs(axes))
    if peak > 0:
        keep = np.abs(axes) >= (peak if strict else ratio * peak)
        axes = np.where(keep, axes, 0.0)
    return axes[:3], axes[3:]


class SdkArm:
    """Straight to the arm over CAN."""

    def __init__(self, args):
        self._piper = pc.connect(args.can)
        self._speed = args.speed
        self._effort = args.gripper_effort

    def wait_ready(self):
        pass  # pc.connect already waited for feedback

    def enable(self):
        if not pc.enable(self._piper):
            raise SystemExit("enable failed (after MIT mode run: python3 arm_tool.py reset)")

    def read(self):
        """(joint positions [rad], joint torques [Nm] or None, gripper width [m])"""
        return (pc.joint_positions(self._piper),
                pc.joint_efforts(self._piper, x4_j123=True),
                pc.gripper_state(self._piper)[0])

    def stale(self):
        return pc.feedback_age(self._piper) > 0.2

    def command(self, q, width):
        pc.send_joint_positions(self._piper, q, self._speed)
        pc.send_gripper(self._piper, width, self._effort)

    def stop(self):
        pass


class RosArm:
    """Through piper_bridge, which drives either the real arm or the simulated one."""

    def __init__(self, args):
        import sys
        import threading

        import rclpy

        # Three threads share this interpreter: the control loop (CPU bound), the
        # SpaceMouse reader and the ROS spinner. With the default 5 ms switch interval the
        # CPU-bound one starves the others for hundreds of ms at a time.
        sys.setswitchinterval(0.001)
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        self._rclpy, self._msg_type = rclpy, JointState
        rclpy.init()
        self._node = Node("spacemouse_teleop")
        self._pub = self._node.create_publisher(JointState, args.command_topic, 10)
        self._node.create_subscription(JointState, args.state_topic, self._on_state, 10)
        self._state = None
        self._last = 0.0
        # Callbacks run on their own thread. Spinning from inside the control loop instead
        # left messages sitting in the queue for over 100 ms at a time.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinner = threading.Thread(target=self._executor.spin, daemon=True)
        self._spinner.start()

    def _on_state(self, msg):
        self._state = msg
        self._last = time.time()

    def wait_ready(self):
        deadline = time.time() + 5.0
        while self._state is None and time.time() < deadline:
            time.sleep(0.05)
        if self._state is None:
            raise SystemExit("no /joint_states. Start the bridge first, e.g.\n"
                             "  ros2 launch piper_bridge display.launch.py use_sim:=true auto_enable:=true")

    def enable(self):
        pass  # the bridge owns enabling (auto_enable, or its ~/enable service)

    def read(self):
        msg = self._state  # kept current by the spinner thread
        index = {name: i for i, name in enumerate(msg.name)}
        q = np.array([msg.position[index[j]] for j in ARM_JOINTS])
        effort = None
        if len(msg.effort) >= 6:
            effort = np.array([msg.effort[index[j]] for j in ARM_JOINTS])
            if not np.any(effort):
                effort = None  # the simulated arm reports zeros: no collision detection there
        width = abs(msg.position[index["joint7"]]) * 2.0 if "joint7" in index else 0.0
        return q, effort, width

    def stale(self):
        # A liveness check (has the bridge died?), not a latency one: the spinner competes
        # for the GIL with the control loop, so single samples are routinely 100 ms old
        # even when the bridge is publishing a clean 200 Hz.
        return time.time() - self._last > 2.0

    def command(self, q, width):
        msg = self._msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = ARM_JOINTS + ["gripper"]
        msg.position = list(map(float, q)) + [float(width)]
        self._pub.publish(msg)

    def stop(self):
        # Order matters: destroying the node while the spinner is still inside spin() crashes.
        self._executor.shutdown()
        self._spinner.join(timeout=2.0)
        self._node.destroy_node()
        self._rclpy.try_shutdown()


def build_model(args):
    urdf = args.urdf or TOOL_URDF[args.tool]
    model = GravityModel(urdf)
    if args.payload > 0:
        model.add_payload(args.payload, (args.payload_x, 0.0, args.payload_z))
    print(f"model: {Path(urdf).name}, tool mass {model.masses[-1]:.3f} kg")
    return model


def layout_suggestion(rng):
    """A random table arrangement to set up before the next episode.

    Without this every demonstration ends up looking the same, and a policy can reach the
    right cube by memorising where it was rather than by reading its colour or the
    instruction -- which is exactly the thing the two-cube setup is meant to test.
    """
    left, right = rng.sample(["RED", "WHITE"], 2)
    return (f"{left} on the LEFT, {right} on the RIGHT, "
            f"{rng.choice(['near', 'middle', 'far'])} from the base, "
            f"box on the {rng.choice(['left', 'right'])}")


def run(arm, model, mouse, args):
    tasks = list(args.task) or ["pick up the cube and put it in the box"]
    task_index = 0
    rng = random.Random()
    layout = layout_suggestion(rng) if args.randomize else None
    tcp = model.tool_to_link(args.tcp)  # --tcp is given in the tool frame
    tool_mount = model.tool_mount[:3, :3]
    box_low = np.array([args.box_x[0], args.box_y[0], args.box_z[0]])
    box_high = np.array([args.box_x[1], args.box_y[1], args.box_z[1]])
    lin_speed, rot_speed = args.lin_speed, np.radians(args.rot_speed)
    max_joint_step = args.max_joint_vel / args.rate
    # Stay a margin inside the SDK limits: commanding a joint onto its limit is what makes
    # the firmware raise a joint error. Joint5 gets a much wider margin than the rest --
    # measured on this arm, its driver trips its own protection at about 65 deg, well inside
    # the 69.9 deg the URDF and the SDK both claim, and a trip disables the joint and freezes
    # the whole arm. Keeping it under 60 costs almost nothing: a vertical gripper needs 52 deg
    # at 0.32 m reach and 61 deg at 0.40 m, and past that the tool simply tilts instead.
    margin = np.radians(np.broadcast_to(np.asarray(args.limit_margin, dtype=float), (6,)))
    limits = pc.JOINT_LIMITS + np.stack([margin, -margin], axis=1)

    arm.wait_ready()
    q_cmd, _, gripper_width = arm.read()
    # Second opinion on the start pose. Everything downstream is anchored on it, and an
    # anchor on a pose the arm is not in sends the arm there at full speed.
    time.sleep(0.1)
    again = arm.read()[0]
    if np.max(np.abs(again - q_cmd)) > np.radians(2.0):
        raise SystemExit(
            f"the arm's reported pose changed by "
            f"{np.degrees(np.max(np.abs(again - q_cmd))):.1f} deg between two reads "
            f"100 ms apart:\n  {np.degrees(q_cmd).round(1)}\n  {np.degrees(again).round(1)}\n"
            "It is either still moving or the feedback is incomplete. Wait for it to settle "
            "and start again.")
    target_pos, target_rot = model.tip_pose(q_cmd, tcp)
    print(f"start pose: {np.round(target_pos, 3)} m,  workspace box "
          f"x{args.box_x} y{args.box_y} z{args.box_z} m")
    pc.confirm("Keep the workspace clear: the arm will follow the SpaceMouse.", args.yes)
    arm.enable()

    cameras, show_every = {}, max(int(args.rate / 15), 1)
    if args.record or args.view:
        if args.camera_serial:
            cameras["wrist"] = RealSenseCamera(args.camera_serial, fps=args.record_fps,
                                               depth=args.depth != "off")
        for spec in args.camera_uvc:
            name, _, index = spec.rpartition(":") if ":" in spec else ("", "", spec)
            cameras[name or f"cam{index}"] = UvcCamera(int(index), fps=args.record_fps)
        for name, cam in cameras.items():
            if not cam.wait_ready():
                raise SystemExit(f"no frames from camera '{name}' (is another program holding it?)")
            print(f"camera {name}: {cam.size[0]}x{cam.size[1]}")
        if args.record:
            print(f"ready to record into {args.record} at {args.record_fps} Hz.")
            print("NOTHING is recorded until you press R. R again saves, X discards.")
            for i, task in enumerate(tasks):
                print(f"  task {i + 1}{' (active)' if i == task_index else ''}: {task}")
            if len(tasks) > 1:
                print("  press T between episodes to switch task")
            if args.randomize:
                print(f"  set up now: {layout}")
    writer, next_sample = None, 0.0

    t_start = time.time()
    keys = pc.KeyReader()
    keys.start()
    rate = Rate(args.rate)
    dt = 1.0 / args.rate
    strict, jogging, stalled = args.strict, True, ""
    allow_lin = args.mode in ("translation", "both")
    allow_rot = args.mode in ("rotation", "both")
    feedback_paused = False
    straining = 0
    ori_weight = args.ori_weight
    blocked_for = 0
    buttons_before = (False, False)
    # One yaw per frame, so m switches modes without having to re-aim every time.
    # In base mode "forward" should be the arm's own working direction, which is joint1:
    # this arm works at j1 = -90 deg, so forward is base -Y, not base +X. Taken from the
    # pose the arm is already in, rounded to a quarter turn, unless --frame-yaw says otherwise.
    auto_yaw = round(float(np.degrees(q_cmd[0])) / 45.0) * 45.0
    yaw_by_mode = {"tool": 0.0, "base": auto_yaw}
    if args.frame_yaw is not None:
        yaw_by_mode = {"tool": args.frame_yaw, "base": args.frame_yaw}
    collision_armed = 0.0
    refused = np.zeros(3, dtype=bool)
    radius_block, blocked_radius = False, 0
    last_residual = None
    lag, leash, actual_pos = 0.0, args.leash, target_pos.copy()
    trace = None
    trace_path = args.trace
    if trace_path is None and args.record:
        # Recording without a trace loses exactly the information needed to explain a
        # session afterwards, and remembering a flag at the bench is not realistic.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(args.record).mkdir(parents=True, exist_ok=True)
        trace_path = str(Path(args.record) / f"session-{stamp}.trace.jsonl")
    if trace_path:
        trace = Path(trace_path).open("w")
        # First line is the setup, so the analysis does not have to be told the flags.
        trace.write(json.dumps({"meta": {
            "box": [list(args.box_x), list(args.box_y), list(args.box_z)],
            "leash_m": args.leash, "lin_speed": args.lin_speed, "rot_speed": args.rot_speed,
            "rate": args.rate, "tool": args.tool, "tcp_m": list(args.tcp),
            "ori_weight": args.ori_weight, "ori_min": args.ori_min,
            "limit_margin_deg": list(np.broadcast_to(
                np.asarray(args.limit_margin, dtype=float), (6,))),
            "joint_command_limits_deg": np.degrees(limits).round(1).tolist(),
            "frame": args.frame, "tasks": tasks,
            "min_radius": args.min_radius, "collision_torque": args.collision_torque,
            "joint_limits_deg": np.degrees(pc.JOINT_LIMITS).round(1).tolist(),
        }}) + "\n")
        print(f"tracing the control loop into {trace_path} "
              f"({args.rate / max(args.trace_every, 1):.0f} Hz)")
    # Gears. A SpaceMouse is asked to do two different jobs -- crossing the table and
    # lining up on a 30 mm cube -- and no single speed is right for both.
    # Measured against what the arm can actually track at the working pose: NORMAL runs
    # 12 cm/s with 3 mm of lag, and above about 20 cm/s the joint speed clamp saturates and
    # more command buys nothing, so the ladder stops there.
    gears = [("PRECISE", 0.25), ("SLOW", 0.5), ("NORMAL", 1.0), ("FAST", 1.4), ("TURBO", 1.8)]
    gear = 2
    home_q = q_cmd.copy()          # where this session started: always somewhere sane
    waypoint = None
    auto = None                    # a min-jerk flight back to a saved pose, cancel with space
    frame_mode = args.frame
    frame_yaw = yaw_by_mode[frame_mode]
    frame = yaw_matrix(frame_yaw)
    print(f"control frame: {frame_mode}"
          + (f" (forward = where the gripper points)" if frame_mode == "tool"
             else f" (forward = base yaw {frame_yaw:+.0f} deg, the arm's own direction)")
          + "   press m to switch, f to turn the input 90 deg")
    cycle = 0
    try:
        while True:
            key = keys.get()
            if key == "q":
                break
            elif key in ("R", "X") and args.record:
                if writer is None and key == "R":
                    writer = EpisodeWriter(next_episode_dir(args.record), tasks[task_index],
                                           {n: c.size for n, c in cameras.items()}, args.record_fps,
                                           depth_sizes={n: c.size for n, c in cameras.items()
                                                        if getattr(c, "wants_depth", False)},
                                           depth_mode=args.depth,
                                           depth_range_m=tuple(args.depth_range),
                                           depth_scale=next((c.depth_scale for c in cameras.values()
                                                             if getattr(c, "depth_scale", None)), 0.001),
                                           meta={"layout": layout,
                                                 "gripper_open_mm": args.gripper_open,
                                                 "gripper_close_mm": args.gripper_close,
                                                 "gripper_effort_nm": args.gripper_effort,
                                                 "tool": args.tool, "tcp_m": list(args.tcp)})
                    next_sample = time.perf_counter()
                    announce(f"recording {writer.path.name} -- R to stop, X to discard")
                elif writer is not None:
                    info = writer.close("kept" if key == "R" else "discarded")
                    announce(f"saved {info}" if info else "episode discarded")
                    writer = None
                    if args.randomize:
                        layout = layout_suggestion(rng)
                        announce(f"set up for the next one: {layout}")
            elif key == "T" and len(tasks) > 1:
                # Not while recording: the episode's task is fixed when it starts, so
                # switching mid-episode would label it with something it never did.
                if writer is not None:
                    announce("stop the episode first (R), then switch task")
                else:
                    task_index = (task_index + 1) % len(tasks)
                    announce(f"task {task_index + 1}/{len(tasks)}: {tasks[task_index]}")
            elif key == "a":
                q_cmd = arm.read()[0]
                target_pos, target_rot = model.tip_pose(q_cmd, tcp)
                stalled = ""
                collision_armed = time.time() + args.collision_grace
                announce(f"re-anchored at {np.round(target_pos, 3)}")
            elif key in ("[", "]"):
                gear = int(np.clip(gear + (1 if key == "]" else -1), 0, len(gears) - 1))
                announce(f"gear: {gears[gear][0]}  ({gears[gear][1]:.2f}x)")
            elif key in ("h", "g"):
                goal = home_q if key == "h" else waypoint
                if goal is None:
                    announce("no waypoint saved yet -- press w to save this pose")
                else:
                    travel = float(np.max(np.abs(goal - arm.read()[0])))
                    auto = {"q0": arm.read()[0], "q1": goal.copy(), "t0": time.time(),
                            "duration": max(1.0, travel / 0.4)}
                    # A flight has to un-pause: otherwise the command flies to the waypoint
                    # while the arm stays put, which looks like it worked and is not.
                    was_paused, jogging = not jogging, True
                    collision_armed = time.time() + args.collision_grace
                    announce(f"flying to {'home' if key == 'h' else 'waypoint'} "
                             f"in {auto['duration']:.1f} s -- space aborts"
                             + ("  (resumed from HOLD)" if was_paused else ""))
            elif key == "w":
                waypoint = arm.read()[0].copy()
                announce(f"waypoint saved: {np.round(np.degrees(waypoint), 1)} deg  (g to return)")
            elif key == "1":
                strict = not strict
                announce(f"strict single-axis {'on' if strict else 'off'}")
            elif key in ("t", "r"):
                if key == "t":
                    allow_lin = not allow_lin
                else:
                    allow_rot = not allow_rot
                announce(f"translation {'on' if allow_lin else 'OFF'}, "
                         f"rotation {'on' if allow_rot else 'OFF'}")
            elif key == "m":
                yaw_by_mode[frame_mode] = frame_yaw
                frame_mode = "base" if frame_mode == "tool" else "tool"
                frame_yaw = yaw_by_mode[frame_mode]
                frame = yaw_matrix(frame_yaw)
                announce(f"control frame: {frame_mode}"
                         + ("  (forward = where the gripper points)" if frame_mode == "tool"
                            else f"  (forward = base yaw {frame_yaw:+.0f} deg, fixed on the table)"))
            elif key == "f":
                frame_yaw = (frame_yaw + 90) % 360
                frame = yaw_matrix(frame_yaw)
                announce(f"input frame rotated to {frame_yaw} deg "
                         "(turn it until pushing the puck matches where you stand)")
            elif key == " ":
                if auto is not None:
                    auto = None
                    announce("flight aborted")
                jogging = not jogging
                if jogging:
                    # Always re-anchor on resume: the target is stale by definition after a
                    # pause, and needing a separate key for it just strands people.
                    q_cmd = arm.read()[0]
                    target_pos, target_rot = model.tip_pose(q_cmd, tcp)
                    stalled = ""
                    # Whatever tripped the stop is still there, so the check has to stay off
                    # long enough to drive out of it; otherwise resuming trips it again
                    # within one cycle and there is no way back.
                    collision_armed = time.time() + args.collision_grace
                announce(("jogging (re-anchored, collision check off for "
                          f"{args.collision_grace:.0f} s)") if jogging
                         else "holding: press space to resume")

            q_now, effort, measured_width = arm.read()
            # A hiccup should pause jogging, not end the session: quitting mid-motion is
            # worse than holding still, and it recovers on its own.
            if arm.stale():
                if not feedback_paused:
                    feedback_paused, jogging = True, False
                    announce("arm feedback stalled: jogging paused, waiting for it to come back")
            elif feedback_paused:
                feedback_paused, jogging = False, True
                q_cmd = q_now.copy()
                target_pos, target_rot = model.tip_pose(q_cmd, tcp)
                announce("feedback is back: re-anchored and jogging again")

            # Throttled: the model call is the most expensive thing in this loop, and a
            # collision is still caught within 50 ms.
            if (args.collision_torque > 0 and effort is not None and not stalled
                    and time.time() >= collision_armed and cycle % 5 == 0):
                residual = effort - model.gravity_torques(q_now)
                last_residual = residual
                # A joint parked on its limit always shows a big residual: the hard stop is
                # taking the load, not an obstacle. Counting that as a collision locks the
                # arm up exactly when you need to drive back out of it.
                on_stop = ((q_now <= pc.JOINT_LIMITS[:, 0] + margin)
                           | (q_now >= pc.JOINT_LIMITS[:, 1] - margin))
                checked = np.where(on_stop, 0.0, residual)
                if np.max(np.abs(checked)) > args.collision_torque:
                    jogging = False
                    stalled = f"collision? torque residual {np.round(residual, 1)} Nm"
                    announce(stalled + f"  -- stopped. Press space: the check stays off for "
                             f"{args.collision_grace:.0f} s so you can jog back out.")

            state = mouse.get_state()
            mode = ("both" if allow_lin and allow_rot else
                    "translation" if allow_lin else "rotation" if allow_rot else "none")
            linear, angular = filter_axes(*to_base_twist(state), args.deadzone,
                                          args.rot_deadzone, args.axis_ratio, strict, mode,
                                          args.curve)
            linear_raw, angular_raw = linear.copy(), angular.copy()
            # Tool frame = first-person: "forward" is wherever the gripper (and its camera)
            # currently points, so the mapping follows the view instead of the table.
            control = (target_rot @ tool_mount @ frame) if frame_mode == "tool" else frame
            linear, angular = control @ linear, control @ angular
            gear_name, gear_scale = gears[gear]

            if auto is not None:
                # Flying back to a saved pose. Any real stick input takes over immediately:
                # being carried somewhere you no longer want to go is the worst kind of
                # surprise on a robot.
                if np.any(np.abs(np.concatenate([linear, angular])) > 0.05):
                    auto = None
                    announce("flight cancelled: you moved the stick")
                else:
                    elapsed = time.time() - auto["t0"]
                    q_cmd = pc.min_jerk(auto["q0"], auto["q1"], auto["duration"], elapsed)
                    # Drag the jog target along, so the flight hands back to the stick with
                    # no offset stored up and the displayed lag stays honest throughout.
                    target_pos, target_rot = model.tip_pose(q_cmd, tcp)
                    if elapsed >= auto["duration"]:
                        auto = None
                        collision_armed = time.time() + args.collision_grace
                        announce("arrived")
            if jogging and auto is None:
                # The box limits motion, it never drags the arm: a step is allowed unless it
                # pushes further outside. Starting outside is fine, you can still come back.
                actual_pos = model.tip_pose(q_now, tcp)[0]
                proposed = target_pos + linear * lin_speed * gear_scale * dt
                # Keep out of the column above the base. Inside about 25 cm the IK has many
                # equally good solutions and wanders into a folded-back one. Measured on a
                # real session: at radius 0.25-0.45 m joint1 sat at -90 deg and |joint5|
                # stayed under 40; at 0.21-0.22 m joint1 swung to -148 (2 deg from its limit)
                # and joint5 to 67, and the joint5 driver tripped its protection twice.
                radius_next = float(np.hypot(proposed[0], proposed[1]))
                radius_block = (radius_next < args.min_radius
                                and radius_next < float(np.hypot(target_pos[0], target_pos[1])))
                if radius_block:
                    proposed = np.array([target_pos[0], target_pos[1], proposed[2]])
                    blocked_radius += 1
                    if blocked_radius % int(args.rate * 2) == 1:
                        announce(f"too close to the base ({radius_next * 100:.0f} cm, limit "
                                 f"{args.min_radius * 100:.0f} cm): the wrist folds back and "
                                 "joint1/joint5 run into their limits there. Move outwards.")
                else:
                    blocked_radius = 0
                outside_now = np.maximum(box_low - target_pos, 0) + np.maximum(target_pos - box_high, 0)
                outside_next = np.maximum(box_low - proposed, 0) + np.maximum(proposed - box_high, 0)
                refused = outside_next > outside_now + 1e-9
                target_pos = np.where(refused, target_pos, proposed)
                # Say so: a box that silently eats the input feels exactly like a dead robot.
                blocked_for = blocked_for + 1 if (refused & (np.abs(linear) > 0)).any() else 0
                if blocked_for and blocked_for % int(args.rate * 2) == 1:
                    axes = ",".join("xyz"[i] for i in range(3) if refused[i])
                    announce(f"workspace box is blocking {axes} (tcp {np.round(actual_pos, 3)}, "
                             f"box x{args.box_x} y{args.box_y} z{args.box_z}) -- "
                             "push the other way, or restart with a bigger --box-*")
                target_rot = rotate(target_rot, angular * rot_speed * gear_scale * dt)
                # Leash: never let the target run further than this from where the arm is,
                # otherwise a stall winds up an offset that snaps back later.
                # The leash allows for the speed-proportional tracking error above, so a
                # healthy fast move is not mistaken for a stall and does not start tilting
                # the tool. What is left over is genuine: the arm really cannot get there.
                leash = args.leash + lin_speed * gear_scale * LEASH_TIME
                offset = target_pos - actual_pos
                lag = np.linalg.norm(offset)
                if lag > leash:
                    target_pos = actual_pos + offset / lag * leash
                # The same leash in rotation, but on what the IK is asked for, never on
                # target_rot itself: the stored target is where the operator wants the tool
                # pointed, and overwriting it would lose that intent for good. Clamping only
                # the request means a tool that had to tilt turns back gradually instead of
                # lurching once it has room again.
                actual_rot = model.tip_pose(q_now, tcp)[1]
                twist = rotation_error(target_rot, actual_rot)
                angle = float(np.linalg.norm(twist))
                rot_leash = np.radians(args.rot_leash)
                rot_request = (rotate(actual_rot, twist / angle * rot_leash)
                               if angle > rot_leash else target_rot)

                # Sitting at the leash means the arm cannot follow: usually joint5 (+-70 deg,
                # the narrowest joint) has run out while holding this tool orientation.
                # Rather than stall, give up orientation: let the tool tilt so the position
                # keeps going, and take the orientation back when the arm catches up. A
                # 6-DoF pose uses every joint of this arm, so holding the exact orientation
                # is what drives the wrist into its limit in the first place.
                straining = straining + 1 if lag > 0.8 * leash else 0
                # Hysteresis: relax while clearly stuck, take the orientation back only once
                # the arm is comfortably following again, so it does not flap between the two.
                if lag > 0.5 * leash:
                    wanted = args.ori_min
                elif lag < 0.25 * leash:
                    wanted = args.ori_weight
                else:
                    wanted = ori_weight
                ori_weight += (wanted - ori_weight) * min(dt / args.ori_relax_time, 1.0)
                if straining == int(args.rate * 0.5):
                    announce(f"out of reach here: letting the tool tilt to keep going "
                             f"(orientation hold {ori_weight / max(args.ori_weight, 1e-6):.0%}). "
                             "Back off, or press r and turn the wrist "
                             "(joint5, the +-70 deg one, is usually what ran out)")

                q_next = np.clip(ik_step(model, q_cmd, target_pos, rot_request, tcp,
                                         args.ik_damping, orientation_weight=max(ori_weight, 1e-3),
                                         limits=limits, centering=args.ik_centering),
                                 limits[:, 0], limits[:, 1])
                if not np.all(np.isfinite(q_next)):
                    jogging = False
                    announce("IK produced a non-finite solution: jogging stopped, press a then space")
                else:
                    q_cmd = q_cmd + np.clip(q_next - q_cmd, -max_joint_step, max_joint_step)

                # The command must not run away from where the arm actually is: if it does,
                # the arm is blocked or stalled and pushing further only stores up a lunge.
                drift = float(np.max(np.abs(q_cmd - q_now)))
                if drift > args.max_drift:
                    jogging = False
                    announce(f"command is {np.degrees(drift):.0f} deg ahead of the arm: "
                             "jogging stopped, press a to re-anchor then space to resume")

            # Binary gripper: a press sets one of two widths and the gripper's own controller
            # does the ramp. Holding a button to ramp the width instead would bake the
            # operator's finger timing into the recorded action.
            left, right = bool(state.buttons.get(0)), bool(state.buttons.get(1))
            if args.gripper_continuous:
                if left or right:
                    step = args.gripper_speed / 1000.0 * dt
                    gripper_width = float(np.clip(gripper_width + (step if left else -step),
                                                  0.0, pc.GRIPPER_MAX_WIDTH))
            else:
                if left and not buttons_before[0]:
                    gripper_width = args.gripper_open / 1000.0
                    announce(f"gripper -> open {args.gripper_open:.0f} mm")
                elif right and not buttons_before[1]:
                    gripper_width = args.gripper_close / 1000.0
                    announce(f"gripper -> close {args.gripper_close:.0f} mm")
            buttons_before = (left, right)
            arm.command(q_cmd, gripper_width)

            if writer is not None:
                now = time.perf_counter()
                if now >= next_sample:
                    next_sample = max(next_sample + 1.0 / args.record_fps, now)
                    writer.add(now - writer.clock0,
                               list(q_now) + [measured_width],
                               list(q_cmd) + [gripper_width],
                               {n: c.frame for n, c in cameras.items()},
                               {n: c.depth for n, c in cameras.items()
                                if getattr(c, "wants_depth", False)})

            # The recorder owns the camera, so it has to show the view as well: otherwise
            # you would be teleoperating blind while recording. The overlay goes on the
            # displayed copy only -- what gets written to the video stays clean.
            if cameras and not args.no_show and cycle % show_every == 0:
                telemetry = {
                    "joints": q_now, "limits": pc.JOINT_LIMITS,
                    "tcp": model.tip_pose(q_now, tcp)[0],
                    "error_m": float(np.linalg.norm(target_pos - model.tip_pose(q_now, tcp)[0])),
                    "gripper": (gripper_width, measured_width, pc.GRIPPER_MAX_WIDTH),
                    "jogging": jogging or auto is not None, "frame_mode": frame_mode,
                    "allow_lin": allow_lin, "allow_rot": allow_rot, "strict": strict,
                    "speed_name": "FLYING" if auto is not None else gear_name,
                    "speed_scale": gear_scale,
                    "ori_frac": ori_weight / max(args.ori_weight, 1e-6),
                    "linear": linear_raw, "angular": angular_raw,
                    "recording": ({"samples": writer.samples,
                                   "seconds": time.time() - writer.started}
                                  if writer is not None else None),
                    "can_record": bool(args.record), "task": tasks[task_index],
                    "message": announce.last if time.time() - announce.last_at < 3.0 else None,
                }
                for name, cam in cameras.items():
                    if cam.frame is not None:
                        cv2.imshow(name, hud.draw(cam.frame, telemetry))
                cv2.waitKey(1)

            # Full internal state, for working out after the fact why a push did nothing.
            # The episode files record what the arm did; this records what the controller
            # was thinking, which is what you need when the answer is "it refused".
            if trace is not None and cycle % max(args.trace_every, 1) == 0:
                actual_now, rot_now = model.tip_pose(q_now, tcp)
                trace.write(json.dumps({
                    "t": round(time.time() - t_start, 3),
                    "q": [round(float(v), 5) for v in q_now],
                    "q_cmd": [round(float(v), 5) for v in q_cmd],
                    "tcp": [round(float(v), 4) for v in actual_now],
                    "target": [round(float(v), 4) for v in target_pos],
                    "lag_mm": round(float(lag) * 1000, 1),
                    "leash_mm": round(float(leash) * 1000, 1),
                    "stick": [round(float(v), 3) for v in linear_raw],
                    "spin": [round(float(v), 3) for v in angular_raw],
                    "cmd_twist": [round(float(v), 3) for v in linear],
                    "refused": [bool(v) for v in refused],
                    "radius_block": bool(radius_block),
                    "residual": ([round(float(v), 2) for v in last_residual]
                                 if last_residual is not None else None),
                    "radius": round(float(np.hypot(actual_now[0], actual_now[1])), 4),
                    "blocked_for": int(blocked_for),
                    "straining": int(straining),
                    "ori": round(float(ori_weight), 3),
                    "rot_err_deg": round(float(np.degrees(np.linalg.norm(
                        rotation_error(target_rot, rot_now)))), 1),
                    "gear": gear_name, "frame": frame_mode, "task": task_index,
                    "jogging": bool(jogging), "auto": auto is not None,
                    "stalled": stalled or None,
                    "grip": [round(float(gripper_width), 4), round(float(measured_width), 4)],
                }) + "\n")

            if cycle % max(int(args.rate / 5), 1) == 0:
                pos = model.tip_pose(q_now, tcp)[0]
                moving = np.concatenate([linear, angular])
                axis = "xyzRPY"[int(np.argmax(np.abs(moving)))] if np.any(moving) else "-"
                rec = f"REC {writer.samples:5d}" if writer else ("not rec " if args.record else "")
                print(f"\r{rec}[{'JOG' if jogging else 'HOLD'}:{frame_mode}:{mode[:4]}"
                      f"{'!' if strict else ''}] "
                      f"tcp {np.round(pos, 3)} m  err {np.linalg.norm(target_pos - pos) * 1000:4.0f} mm "
                      f"grip {gripper_width * 1000:4.1f}/{measured_width * 1000:4.1f} mm "
                      f"axis {axis}"
                      + (f"  ori {ori_weight / max(args.ori_weight, 1e-6):3.0%}"
                         if ori_weight < 0.95 * args.ori_weight else "")
                      + "      ", end="", flush=True)
            if not jogging and cycle % int(args.rate * 3) == 0:
                announce("still paused -- press space to resume (space always re-anchors)")
            cycle += 1
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if trace is not None:
            trace.close()
            print(f"\ntrace written to {trace_path}  "
                  f"(python3 analyse_trace.py {trace_path})")
        if writer is not None:
            print(f"\nepisode closed on exit: {writer.close('kept')}")
        for cam in cameras.values():
            cam.stop()
        if cameras and not args.no_show:
            cv2.destroyAllWindows()
        keys.stop()
        mouse.stop()
        arm.stop()
    print("\nstopped; the arm holds this pose "
          "(joint_position_ctrl.py --zero, then arm_tool.py disable to park)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sdk", "ros"], default="sdk")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--command-topic", default="/piper_bridge/joint_command")
    ap.add_argument("--state-topic", default="/joint_states")
    ap.add_argument("--tool", choices=sorted(TOOL_URDF), default="pika")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--payload", type=float, default=0.42)
    ap.add_argument("--payload-x", type=float, default=0.03)
    ap.add_argument("--payload-z", type=float, default=0.07)
    ap.add_argument("--tcp", type=float, nargs=3, default=[0.13, 0.0, 0.0],
                    help="tool centre point in the TOOL frame [m]: +X out of the gripper")
    ap.add_argument("--curve", type=float, default=1.7,
                    help="response exponent: 1 is linear, higher is finer near the centre but "
                         "makes half-pushes feel dead. 1.7 gives 30%% speed at a half push, "
                         "2.0 gave 22%%; use the PRECISE gear for fine work instead")
    ap.add_argument("--frame", choices=["tool", "base"], default="base",
                    help="tool: first-person, forward is where the gripper camera points; "
                         "base: forward is always the same direction on the table")
    ap.add_argument("--frame-yaw", type=float, default=None,
                    help="rotate the input about the control frame's Z [deg], or press f. "
                         "Default: 0 for the tool frame, and for the base frame the arm's "
                         "own joint1 direction, so pushing forward reaches away from you")
    ap.add_argument("--lin-speed", type=float, default=0.10,
                    help="full deflection speed at the NORMAL gear [m/s]")
    ap.add_argument("--rot-speed", type=float, default=45.0,
                    help="full deflection speed at the NORMAL gear [deg/s]")
    ap.add_argument("--deadzone", type=float, default=0.05, help="translation deadzone")
    ap.add_argument("--rot-deadzone", type=float, default=0.12, help="rotation deadzone (bleeds more)")
    ap.add_argument("--axis-ratio", type=float, default=0.4,
                    help="zero any axis below this fraction of the dominant axis")
    ap.add_argument("--mode", choices=["translation", "rotation", "both"], default="translation",
                    help="rotation is the hard part of a SpaceMouse, so it is off by default")
    ap.add_argument("--strict", action="store_true",
                    help="start in strict mode: only the single largest axis gets through")
    ap.add_argument("--gripper-open", type=float, default=70.0, help="open width [mm], left button")
    ap.add_argument("--gripper-close", type=float, default=20.0,
                    help="close width [mm], right button. Must be SMALLER than the object, "
                         "or the fingers stop before touching it; 0 works for any size")
    ap.add_argument("--gripper-continuous", action="store_true",
                    help="old behaviour: hold a button to ramp the width")
    ap.add_argument("--gripper-speed", type=float, default=25.0,
                    help="ramp speed [mm/s], only with --gripper-continuous")
    ap.add_argument("--gripper-effort", type=float, default=1.0, help="gripper force [Nm]")
    ap.add_argument("--box-x", type=float, nargs=2, default=[-0.65, 0.65],
                    help="workspace limit [m]; generous by default because the box silently\n                         eats input, and joint limits are the real guard")
    ap.add_argument("--box-y", type=float, nargs=2, default=[-0.65, 0.65])
    # The floor is BELOW the mounting plane on purpose: base z = 0 is the table surface,
    # so a floor at +0.02 blocks the last 2 cm of every reach down to an object lying on
    # the table -- which looks exactly like "it refuses to go down" and is not a collision
    # guard anyway (that is the torque check and the operator).
    ap.add_argument("--box-z", type=float, nargs=2, default=[-0.05, 0.75])
    ap.add_argument("--min-radius", type=float, default=0.25,
                    help="keep the tool at least this far from the base's vertical axis [m]. "
                         "Closer in, the IK folds the arm back on itself and joint1/joint5 "
                         "end up on their limits, which trips the joint5 driver")
    ap.add_argument("--leash", type=float, default=0.03,
                    help="max target-vs-actual offset [m], on top of the tracking error "
                         "that the commanded speed itself produces")
    ap.add_argument("--ik-damping", type=float, default=0.08)
    ap.add_argument("--ori-weight", type=float, default=0.4,
                    help="how hard to hold the orientation, 1 = as hard as position, 0 = ignore it")
    ap.add_argument("--limit-margin", type=float, nargs="+", default=[2, 2, 2, 2, 10, 2],
                    metavar="DEG",
                    help="keep this far inside each joint limit [deg], one value or six. "
                         "Joint5 defaults to 10 because its driver trips its protection near "
                         "65 deg and that disables the joint and freezes the arm")
    ap.add_argument("--ori-min", type=float, default=0.03,
                    help="orientation weight to fall back to when the arm cannot follow: "
                         "the tool tilts instead of stalling at a wrist limit")
    ap.add_argument("--ori-relax-time", type=float, default=0.4,
                    help="seconds to fade the orientation hold out and back in [s]")
    ap.add_argument("--rot-leash", type=float, default=20.0,
                    help="max target-vs-actual orientation offset [deg]")
    ap.add_argument("--ik-centering", type=float, default=0.08,
                    help="nullspace pull toward mid-range joints, 0 disables")
    ap.add_argument("--max-joint-vel", type=float, default=1.0,
                    help="joint speed clamp [rad/s]. At 0.5 this, not the Cartesian speed, "
                         "was what limited the arm: it lagged 10-15 mm even at 12 cm/s")
    ap.add_argument("--max-drift", type=float, default=0.26,
                    help="stop if the command gets this far ahead of the measured joints [rad]")
    ap.add_argument("--collision-grace", type=float, default=2.0,
                    help="after a resume, hold the collision check off this long [s] so you "
                         "can drive back out of whatever tripped it")
    ap.add_argument("--collision-torque", type=float, default=5.0,
                    help="stop when |measured - model| exceeds this [Nm], 0 disables")
    ap.add_argument("--speed", type=int, default=100, help="firmware MOVE J speed percent")
    ap.add_argument("--rate", type=float, default=100.0, help="control rate [Hz]")
    ap.add_argument("--record", help="record episodes into this directory")
    ap.add_argument("--task", action="append", default=[], metavar="TEXT",
                    help="language instruction stored with every episode. Repeat the flag to "
                         "set up several tasks and cycle them with T, so swapping the object "
                         "on the table does not mean restarting. Keep ONE fixed wording per "
                         "task: paraphrases are worth adding at training time, not here")
    ap.add_argument("--randomize", action="store_true",
                    help="after each episode, print a random table layout to set up next. "
                         "Stops the policy from reaching the right cube by memorising where "
                         "it was instead of reading its colour")
    ap.add_argument("--record-fps", type=int, default=30)
    ap.add_argument("--depth", choices=["off", "scaled", "lossless"], default="off",
                    help="record the D405 depth too: scaled = 8-bit over --depth-range, "
                         "about the size of the colour video; lossless = exact 16-bit, ~300x bigger")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.0, 0.6], metavar="M",
                    help="depth range mapped to 0..255 in scaled mode [m]")
    ap.add_argument("--trace", metavar="FILE",
                    help="log the whole control loop to a jsonl file: stick input, target, "
                         "lag, which axis the box refused, orientation hold, joint angles. "
                         "Feed it to analyse_trace.py afterwards")
    ap.add_argument("--trace-every", type=int, default=2,
                    help="write one trace sample every N control cycles (2 = 50 Hz)")
    ap.add_argument("--view", action="store_true",
                    help="open the wrist camera with the HUD even when not recording")
    ap.add_argument("--no-show", action="store_true", help="do not open the camera window")
    ap.add_argument("--camera-uvc", action="append", default=[], metavar="NAME:INDEX",
                    help="extra plain USB camera, e.g. top:0 for the laptop webcam; repeatable")
    ap.add_argument("--camera-serial", default="315122272699",
                    help="D405 on the gripper (the Sense one is 315122271085)")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    model = build_model(args)
    mouse = SpaceMouse(scale=350.0, deadzone=0.0)  # filtering happens here, not in the driver
    try:
        mouse.start()  # raises on failure, returns None on success
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nThe device is exclusive: if it is plugged in, some other "
                         "program still holds it (check: fuser -v /dev/bus/usb/*/*)")
    print(f"SpaceMouse: {mouse.device_name}")
    run(RosArm(args) if args.backend == "ros" else SdkArm(args), model, mouse, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
