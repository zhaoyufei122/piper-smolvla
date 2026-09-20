#!/usr/bin/env python3
"""Small Piper utilities: status / enable / disable / stop / reset.

  status   firmware, mode, joint angles, and the per-joint driver table
  protect  show or set the collision protection level of each joint (0 = off, 8 = loosest)
  clear    clear a joint driver's latched error code, without cutting power to the arm.
           Use this when one joint shows OFF in the status table: 'enable' alone will not
           bring it back, and 'reset' would drop the whole arm.
  enable   enable all joint motors (holds the current pose)
  disable  disable all joint motors -- the arm DROPS unless it rests at the zero pose
  stop     firmware emergency stop
  reset    clear errors / leave MIT or teach mode so position control works again.
           The motors lose power briefly, so do it at the rest pose.
"""
import argparse
import time

import numpy as np

import piper_common as pc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["status", "protect", "clear", "enable", "disable", "stop", "reset"])
    ap.add_argument("--joint", type=int, default=7, choices=range(1, 8), metavar="N",
                    help="clear: which joint, 1-6, or 7 for all (default)")
    ap.add_argument("--set", nargs=6, type=int, metavar="L",
                    help="protect: six levels, 0-8, one per joint")
    ap.add_argument("--can", default="can0")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    piper = pc.connect(args.can)

    if args.command == "status":
        print(f"firmware: {pc.firmware_version(piper)}")
        print(pc.status_line(piper))
        print(f"joints [deg]: {pc.fmt(np.degrees(pc.joint_positions(piper)), 2)}")
        problem = pc.fault(piper)
        print(f"arm fault: {problem}" if problem else "arm fault: none")
        # The arm-level status stays NORMAL when one joint driver disables itself, so the
        # per-joint table is the only way to see "green light but no torque".
        print(f"\n{'joint':6s} {'driver':7s} {'volt':>6s} {'drv':>5s} {'mot':>5s}  faults")
        for row in pc.driver_info(piper):
            print(f"joint{row['joint']} {'ON ' if row['enabled'] else 'OFF':7s} "
                  f"{row['voltage']:5.1f}V {row['driver_temp']:4d}C {row['motor_temp']:4d}C  "
                  + (", ".join(row["faults"]) if row["faults"] else "-"))
        levels = pc.protection_levels(piper)
        if levels:
            print(f"\ncollision protection levels: {levels}  (0 = off, 1 = trips easiest)")
    elif args.command == "protect":
        levels = pc.protection_levels(piper)
        print(f"current levels: {levels}")
        if args.set:
            if not all(0 <= v <= 8 for v in args.set):
                raise SystemExit("levels must be 0-8")
            pc.confirm(f"Set collision protection to {args.set}. Level 0 disables protection "
                       "on that joint entirely.", args.yes)
            piper.CrashProtectionConfig(*args.set)
            time.sleep(0.5)
            print(f"now: {pc.protection_levels(piper)}")
    elif args.command == "clear":
        before = pc.driver_info(piper)
        target = "all joints" if args.joint == 7 else f"joint{args.joint}"
        print(f"clearing error codes on {target}...")
        # 0xAE is the SDK's "this field is meaningful" magic value; everything else stays 0
        # so nothing but the error code is touched.
        piper.JointConfig(joint_num=args.joint, clear_err=0xAE)
        time.sleep(0.5)
        for old, new in zip(before, pc.driver_info(piper)):
            if old["faults"] or new["faults"]:
                print(f"  joint{new['joint']}: {', '.join(old['faults']) or '-'}"
                      f"  ->  {', '.join(new['faults']) or 'clear'}")
        print("\nnow run: python3 arm_tool.py enable")
    elif args.command == "enable":
        print("enabled" if pc.enable(piper) else "enable FAILED")
    elif args.command == "disable":
        pc.confirm("The arm will go limp and fall unless it is at the rest pose or supported.", args.yes)
        print("disabled" if pc.disable(piper) else "disable FAILED")
    elif args.command == "stop":
        piper.EmergencyStop(0x01)
        print("emergency stop sent (use 'reset' to recover)")
    elif args.command == "reset":
        pc.confirm("Reset cuts motor power briefly: the arm falls unless it is at the rest pose "
                   "or supported.", args.yes)
        piper.MotionCtrl_1(0x02, 0x00, 0x00)
        # The drivers power-cycle here: enabling too soon gives JOINT_COMMUNICATION_ERR.
        print("reset sent, waiting for the drivers to come back...")
        time.sleep(3.0)
        print(pc.status_line(piper))
        problem = pc.fault(piper)
        print(f"fault: {problem}" if problem else "no fault; enable again before moving")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
