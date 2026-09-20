#!/usr/bin/env python3
"""Plan and execute a joint-space goal through MoveIt's move_group action.

  ros2 run piper_moveit_config moveit_goal.py --named ready
  ros2 run piper_moveit_config moveit_goal.py --joints 0 60 -45 0 30 0   # degrees
  ros2 run piper_moveit_config moveit_goal.py --gripper 40               # opening [mm]
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def named_state(name):
    srdf = Path(get_package_share_directory("piper_moveit_config")) / "config" / "piper.srdf"
    for state in ET.parse(srdf).getroot().findall("group_state"):
        if state.get("name") == name:
            return state.get("group"), {j.get("name"): float(j.get("value")) for j in state.findall("joint")}
    raise SystemExit(f"no group_state '{name}' in {srdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    goal = ap.add_mutually_exclusive_group(required=True)
    goal.add_argument("--named", help="group_state from piper.srdf (zero, ready, open, close)")
    goal.add_argument("--joints", type=float, nargs=6, metavar="DEG", help="arm joint targets [deg]")
    goal.add_argument("--gripper", type=float, metavar="MM", help="gripper opening [mm], 0..70")
    ap.add_argument("--vel", type=float, default=0.2, help="max velocity scaling (0, 1]")
    ap.add_argument("--acc", type=float, default=0.2, help="max acceleration scaling (0, 1]")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    if args.named:
        group, targets = named_state(args.named)
    elif args.joints:
        group, targets = "arm", {n: math.radians(v) for n, v in zip(ARM_JOINTS, args.joints)}
    else:
        group, targets = "gripper", {"joint7": args.gripper / 2000.0}  # joint7 is one finger [m]

    rclpy.init()
    node = Node("moveit_goal")
    client = ActionClient(node, MoveGroup, "move_action")
    if not client.wait_for_server(timeout_sec=10.0):
        raise SystemExit("move_group action server not available (is piper_moveit.launch.py running?)")

    goal_msg = MoveGroup.Goal()
    request = goal_msg.request
    request.group_name = group
    request.num_planning_attempts = 5
    request.allowed_planning_time = 5.0
    request.max_velocity_scaling_factor = args.vel
    request.max_acceleration_scaling_factor = args.acc
    request.start_state.is_diff = True
    request.goal_constraints = [Constraints(joint_constraints=[
        JointConstraint(joint_name=name, position=value, tolerance_above=0.001, tolerance_below=0.001, weight=1.0)
        for name, value in targets.items()
    ])]
    goal_msg.planning_options.plan_only = args.plan_only

    print(f"{'planning' if args.plan_only else 'moving'} group '{group}' to {targets}")
    send_future = client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if not handle.accepted:
        raise SystemExit("goal rejected by move_group")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    code = result_future.result().result.error_code.val
    print("success" if code == MoveItErrorCodes.SUCCESS else f"failed, MoveItErrorCodes = {code}")

    node.destroy_node()
    rclpy.shutdown()
    return 0 if code == MoveItErrorCodes.SUCCESS else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\naborted")
        sys.exit(130)
