#!/usr/bin/env python3
"""Step 2: move Piper joints with direct SDK position commands (MOVE J over CAN).

A min-jerk trajectory from the current pose to the target is streamed at 100 Hz, so the
arm never jumps even if the target is far away. The arm stays enabled and holds the
final pose when the script exits.

Examples:
  python3 joint_position_ctrl.py --demo                     # small joint1/joint6 moves around the current pose
  python3 joint_position_ctrl.py --target 0 60 -45 0 30 0   # degrees
  python3 joint_position_ctrl.py --target 0 60 -45 0 30 0 --gripper 40
  python3 joint_position_ctrl.py --zero                     # back to the folded rest pose
"""
import argparse
import time

import numpy as np

import piper_common as pc


def move_to(piper, q_target, max_vel, speed_percent, rate=100.0, settle_tol=np.radians(0.5),
            settle_timeout=3.0):
    """Stream a min-jerk trajectory to q_target. Returns the final joint error [rad]."""
    q0 = pc.joint_positions(piper)
    q_target = pc.clamp_joints(q_target)
    duration = max(1.0, float(np.max(np.abs(q_target - q0))) / max_vel)
    t0 = time.time()
    warned = False
    while True:
        t = time.time() - t0
        pc.send_joint_positions(piper, pc.min_jerk(q0, q_target, duration, t), speed_percent)
        problem = pc.fault(piper)
        if problem:
            raise SystemExit(f"arm fault while moving: {problem}\nThe arm stopped where it is. "
                             "Clear the cause, then at the rest pose run 'python3 arm_tool.py reset'.")
        if not all(piper.GetArmEnableStatus()):
            raise SystemExit("the motors disabled themselves mid-motion. This usually means they "
                             "were enabled too soon after a reset; wait a few seconds and retry.")
        status = pc.arm_status(piper).arm_status
        if status != 0 and not warned:
            print(f"  warning: arm_status = {status}")
            warned = True
        if t >= duration:
            err = q_target - pc.joint_positions(piper)
            if np.max(np.abs(err)) < settle_tol or t > duration + settle_timeout:
                return err
        time.sleep(1.0 / rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0")
    goal = ap.add_mutually_exclusive_group(required=True)
    goal.add_argument("--target", type=float, nargs=6, metavar="DEG", help="joint1..joint6 target [deg]")
    goal.add_argument("--zero", action="store_true", help="go to the rest pose (all joints 0)")
    goal.add_argument("--demo", action="store_true", help="joint1 +-15 deg and joint6 +30 deg, then back")
    ap.add_argument("--gripper", type=float, metavar="MM", help="gripper opening [mm], 0..70")
    ap.add_argument("--max-vel", type=float, default=30.0, help="trajectory peak-ish joint speed [deg/s]")
    ap.add_argument("--speed", type=int, default=50, help="firmware MOVE J speed percent (1..100)")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    piper = pc.connect(args.can)
    q_start = pc.joint_positions(piper)
    print(pc.status_line(piper))
    print(f"current [deg]: {pc.fmt(np.degrees(q_start))}")

    if args.target is not None:
        waypoints = [np.radians(args.target)]
    elif args.zero:
        waypoints = [np.zeros(6)]
    else:
        offsets = [(0, 15.0), (0, -15.0), (5, 30.0), None]
        waypoints = []
        for item in offsets:
            q = q_start.copy()
            if item is not None:
                q[item[0]] += np.radians(item[1])
            waypoints.append(q)

    for i, q in enumerate(waypoints):
        clamped = pc.clamp_joints(q)
        note = "  (clamped to joint limits)" if not np.allclose(clamped, q) else ""
        print(f"waypoint {i + 1} [deg]: {pc.fmt(np.degrees(clamped))}{note}")
    pc.confirm("Keep the workspace clear: the arm is about to enable and move.", args.yes)

    if not pc.enable(piper):
        raise SystemExit("Enable failed. Check the e-stop and arm status (python3 arm_tool.py status).\n"
                         "If the arm was in MIT/teach mode, run: python3 arm_tool.py reset")
    print("enabled")
    if args.gripper is not None:
        pc.send_gripper(piper, args.gripper / 1000.0)

    for i, q in enumerate(waypoints):
        err = move_to(piper, q, np.radians(args.max_vel), args.speed)
        reached = float(np.max(np.abs(err))) < np.radians(1.0)
        print(f"waypoint {i + 1} {'reached' if reached else 'NOT reached'}, "
              f"error [deg]: {pc.fmt(np.degrees(err), 2)}")
        if not reached:
            raise SystemExit("the arm did not reach the waypoint, stopping here")

    print("Done. The arm stays enabled and holds this pose.\n"
          "To power down safely: --zero first, then python3 arm_tool.py disable")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
