#!/usr/bin/env python3
"""Step 1: read Piper joint feedback over CAN. Read-only: never enables or moves the arm."""
import argparse
import time

import numpy as np

import piper_common as pc


def render(piper, effort_x4):
    q = pc.joint_positions(piper)
    dq = pc.joint_velocities(piper)
    tau = pc.joint_efforts(piper, effort_x4)
    enabled = piper.GetArmEnableStatus()
    width, grip_effort = pc.gripper_state(piper)
    joint_hz = piper.GetArmJointMsgs().Hz

    lines = [
        pc.status_line(piper),
        f"CAN {piper.GetCanFps():.0f} frames/s   joint feedback {joint_hz:.0f} Hz, "
        f"age {pc.feedback_age(piper) * 1000:.0f} ms",
        "",
        "joint    pos[deg]  pos[rad]  vel[rad/s]  effort[Nm]  enabled",
    ]
    for i, name in enumerate(pc.JOINT_NAMES):
        lines.append(f"{name:<7} {np.degrees(q[i]):9.3f} {q[i]:9.4f} {dq[i]:11.3f} "
                     f"{tau[i]:11.3f}  {enabled[i]}")
    lines.append(f"gripper  width {width * 1000:.1f} mm   effort {grip_effort:.3f} Nm")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--rate", type=float, default=10.0, help="refresh rate [Hz]")
    ap.add_argument("--once", action="store_true", help="print one sample and exit")
    ap.add_argument("--effort-x4", action="store_true",
                    help="scale J1-J3 effort by 4 (firmware <= V1.8-2)")
    args = ap.parse_args()

    piper = pc.connect(args.can)
    header = f"Piper on {args.can}, firmware {pc.firmware_version(piper)}"
    if args.once:
        print(header)
        print(render(piper, args.effort_x4))
        return

    try:
        while True:
            # Clear screen and redraw in place.
            print("\033[2J\033[H" + header + "   (Ctrl+C to quit)\n" + render(piper, args.effort_x4),
                  flush=True)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
