#!/usr/bin/env python3
"""Step 4: zero-torque drag mode (gravity compensation) with MIT joint control.

Every cycle the URDF gravity torque tau_g(q) is sent as MIT feed-forward torque with
kp = 0 and a little damping, so the arm floats and can be pushed around by hand.

  --check        arm holds its pose in position mode; prints URDF model torque next to
                 the motor effort feedback, to validate the model before going to MIT.
  (default)      MIT free drive. Start from the rest pose (all zeros) with a hand ready.
  --joints 2 3   only these joints are free; the others hold their start angle.

Safety: gains ramp in over --ramp seconds, joints get a soft spring near their limits,
and a joint speed above --max-vel drops into position hold. Ctrl+C holds the pose and
offers to return to the rest pose before disabling.

MIT command torque is multiplied by 4 inside the joint driver (piper_sdk Q&A), hence the
default --tau-scale 0.25. Calibrate it per joint if the arm sinks or rises.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import piper_common as pc
from gravity_model import TOOL_URDF, GravityModel
from joint_position_ctrl import move_to

MIT_T_LIMIT = 8.0            # the SDK encodes t_ref in [-8, 8]
KP_HOLD, KD_HOLD = 10.0, 0.8  # SDK reference gains for MIT position hold
MODE_CMD_EVERY = 5           # re-send the MIT mode command every N cycles
LIMIT_MARGIN = np.radians(5.0)


def per_joint(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(6, arr.item())
    if arr.size == 6:
        return arr
    raise SystemExit(f"{name} needs 1 or 6 values, got {arr.size}")


def announce(text):
    """Key feedback on its own line, so the refreshing status line cannot swallow it.

    The last message is kept on the function so an on-screen overlay can show it too:
    while teleoperating you are looking at the camera, not at the terminal.
    """
    announce.last, announce.last_at = text, time.time()
    print(f"\n  {text}", flush=True)


announce.last, announce.last_at = "", 0.0


def setup_gripper(piper, mode):
    """Deal with the gripper motor, which EnableArm(7) switches on together with the arm.

    Left alone it keeps servoing to whatever target the firmware still holds -- often a
    mechanical stop -- and hums. Returns the width to keep re-sending, or None.
    """
    if mode == "keep":
        return None
    if mode == "off":
        piper.GripperCtrl(0, 0, 0x00, 0)  # gripper_code 0x00 = disable the gripper motor
        return None
    width = pc.gripper_state(piper)[0]
    pc.send_gripper(piper, width, 0.5)
    return width


def enter_mit(piper):
    piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)


def send_mit(piper, pos_ref, kp, kd, t_ref):
    for i in range(6):
        piper.JointMitCtrl(i + 1, float(pos_ref[i]), 0.0, float(kp[i]), float(kd[i]), float(t_ref[i]))


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.next = time.perf_counter()

    def sleep(self):
        self.next += self.period
        delay = self.next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next = time.perf_counter()


def build_model(args):
    urdf = args.urdf or TOOL_URDF[args.tool]
    model = GravityModel(urdf, include_tool=not args.no_gripper)
    if args.payload > 0:
        model.add_payload(args.payload, (args.payload_x, 0.0, args.payload_z))
    print(f"model: {Path(urdf).name}, tool mass {model.masses[-1]:.3f} kg"
          + (f" (includes {args.payload:.2f} kg payload)" if args.payload > 0 else ""))
    return model


def run_check(piper, model, args):
    print(pc.status_line(piper))
    print(f"current [deg]: {pc.fmt(np.degrees(pc.joint_positions(piper)))}")
    if args.target is not None:
        print(f"target  [deg]: {pc.fmt(np.asarray(args.target, dtype=float))}")
        pc.confirm("Keep the workspace clear: the arm is about to enable and move.", args.yes)
    else:
        pc.confirm("The motors will be enabled and hold the current pose.", args.yes)
    if not pc.enable(piper):
        raise SystemExit("enable failed (MIT/teach mode? run: python3 arm_tool.py reset)")
    setup_gripper(piper, args.gripper)
    if args.target is not None:
        move_to(piper, np.radians(args.target), np.radians(30.0), 50)
    q_hold = pc.joint_positions(piper)
    try:
        while True:
            pc.send_joint_positions(piper, q_hold, 20)
            q = pc.joint_positions(piper)
            model_tau = model.gravity_torques(q)
            measured = pc.joint_efforts(piper, args.effort_x4)
            lines = ["\033[2J\033[H--check: model vs measured holding torque (Ctrl+C to quit)",
                     pc.status_line(piper), "",
                     "joint    q[deg]   model[Nm]  measured[Nm]  measured/model"]
            for i, name in enumerate(pc.JOINT_NAMES):
                ratio = f"{measured[i] / model_tau[i]:8.2f}" if abs(model_tau[i]) > 0.3 else "      --"
                lines.append(f"{name:<7} {np.degrees(q[i]):8.2f} {model_tau[i]:10.3f} "
                             f"{measured[i]:12.3f}  {ratio}")
            print("\n".join(lines), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nArm left enabled, holding its pose.")


def run_free_drive(piper, model, args):
    free = np.zeros(6, dtype=bool)
    free[np.asarray(args.joints) - 1] = True
    tau_scale = per_joint(args.tau_scale, "--tau-scale")
    kd_free = per_joint(args.kd, "--kd")
    kp_hold = per_joint(args.kp_hold, "--kp-hold")
    kd_hold = per_joint(args.kd_hold, "--kd-hold")
    lower = pc.JOINT_LIMITS[:, 0] + LIMIT_MARGIN
    upper = pc.JOINT_LIMITS[:, 1] - LIMIT_MARGIN

    print(f"firmware {pc.firmware_version(piper)} (MIT needs >= V1.5-2)")
    print(f"free joints: {[i + 1 for i in range(6) if free[i]]}  gain={args.gain}  "
          f"tau_scale={tau_scale}  kd={kd_free}")
    print("while running:  [ / ] damping,  , / . feed-forward,  1-6 free/hold that joint,  "
          "h hold everything,  q stop   (gain keys act on the free joints)")
    pc.confirm("Keep one hand near the arm: MIT free drive is about to start.", args.yes)
    if not pc.enable(piper):
        raise SystemExit("enable failed")

    gripper_width = setup_gripper(piper, args.gripper)
    q_start = pc.joint_positions(piper)
    hold_ref = q_start.copy()  # where held joints are locked; updated when a joint is re-held
    rate = Rate(args.rate)
    half_ramp = max(args.ramp, 1e-3) / 2.0
    t0 = time.perf_counter()
    cycle = 0
    fast_cycles = 0
    reason = "user"
    enter_mit(piper)
    keys = pc.KeyReader()
    keys.start()
    try:
        while True:
            key = keys.get()
            if key in ("]", "="):
                kd_free = np.minimum(kd_free + 0.05, 5.0)
                announce(f"free-joint damping kd -> {kd_free[0]:.2f}")
            elif key == "[":
                kd_free = np.maximum(kd_free - 0.05, 0.0)
                announce(f"free-joint damping kd -> {kd_free[0]:.2f}")
            elif key in (".", ","):
                step = 0.01 if key == "." else -0.01
                tau_scale = np.where(free, np.clip(tau_scale + step, 0.0, 2.0), tau_scale)
                announce(f"feed-forward scale -> {pc.fmt(tau_scale[free], 2, 5)}")
            elif key in ("1", "2", "3", "4", "5", "6"):
                i = int(key) - 1
                free[i] = not free[i]
                if not free[i]:
                    hold_ref[i] = pc.joint_positions(piper)[i]  # lock it where it is now
                announce(f"joint{i + 1} is now {'FREE' if free[i] else 'held'}")
            elif key == "h":
                hold_ref = pc.joint_positions(piper)
                free[:] = False
                announce("all joints held at the current pose")
            elif key == "q":
                break
            if pc.feedback_age(piper) > 0.2:
                reason = "feedback lost"
                break
            q = pc.joint_positions(piper)
            dq = pc.joint_velocities(piper)
            tau = model.gravity_torques(q) * args.gain

            # Ramp: first fade in the feed-forward while holding q_start, then fade out kp.
            t = time.perf_counter() - t0
            ff = min(t / half_ramp, 1.0)
            hold = 1.0 - min(max(t - half_ramp, 0.0) / half_ramp, 1.0)
            kp = np.where(free, kp_hold * hold, kp_hold)
            kd = np.where(free, np.maximum(kd_free, kd_hold * hold), kd_hold)
            pos_ref = hold_ref.copy()

            if hold == 0.0:
                pos_ref[free] = q[free]
                wall = free & ((q < lower) | (q > upper))
                pos_ref[wall] = np.clip(q, lower, upper)[wall]
                kp[wall] = args.k_wall

            t_ref = np.clip(tau * tau_scale * ff, -MIT_T_LIMIT, MIT_T_LIMIT)
            if cycle % MODE_CMD_EVERY == 0:
                enter_mit(piper)
            if gripper_width is not None and cycle % 20 == 0:
                pc.send_gripper(piper, gripper_width, 0.5)
            send_mit(piper, pos_ref, kp, kd, t_ref)

            fast_cycles = fast_cycles + 1 if np.any(np.abs(dq) > args.max_vel) else 0
            if fast_cycles >= 3:
                reason = f"joint speed above {args.max_vel} rad/s: {pc.fmt(dq, 2)}"
                break

            if cycle % max(int(args.rate / 5), 1) == 0:
                phase = "ramping" if hold > 0 else "FREE"
                active = ",".join(str(i + 1) for i in range(6) if free[i]) or "none"
                print(f"\r[{phase}] free {active:<11} kd {pc.fmt(kd_free[free], 2, 5)} "
                      f"scale {pc.fmt(tau_scale[free], 2, 5)} | q[deg] {pc.fmt(np.degrees(q))} "
                      f"| tau[Nm] {pc.fmt(tau, 2, 6)}   ", end="", flush=True)
            cycle += 1
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        print(f"\ntuning now: --kd {' '.join(f'{v:.2f}' for v in kd_free)}"
              f" --tau-scale {' '.join(f'{v:.2f}' for v in tau_scale)}")
    print()
    finish(piper, model, args, tau_scale, reason)


def finish(piper, model, args, tau_scale, reason):
    """Hold the pose, optionally return to rest, then disable."""
    use_ff = reason == "user"
    if not use_ff:
        print(f"STOPPED: {reason}. Holding pose without model feed-forward.")
    q_hold = pc.joint_positions(piper)
    kp = per_joint(args.kp_hold, "--kp-hold")
    kd = per_joint(args.kd_hold, "--kd-hold")
    rate = Rate(args.rate)

    def feed_forward(q):
        if not use_ff:
            return np.zeros(6)
        return np.clip(model.gravity_torques(q) * args.gain * tau_scale, -MIT_T_LIMIT, MIT_T_LIMIT)

    print("Holding.  [Enter] = park at the rest pose and disable,  d = disable now (SUPPORT THE ARM),"
          "  Ctrl+C = disable now")
    print(f"          [ / ] = hold damping (now {pc.fmt(kd, 2, 5)}),  {{ / }} = hold stiffness "
          f"(now {pc.fmt(kp, 1, 5)})  <- use these if it buzzes while holding")
    keys = pc.KeyReader()
    keys.start()
    choice = None
    try:
        while choice is None:
            key = keys.get()
            if key in ("\n", "\r"):
                choice = "park"
            elif key == "d":
                choice = "disable"
            elif key in ("[", "]"):
                kd = np.clip(kd + (0.05 if key == "]" else -0.05), 0.0, 5.0)
                announce(f"hold damping -> {pc.fmt(kd, 2, 5)}")
            elif key in ("{", "}"):
                kp = np.clip(kp + (1.0 if key == "}" else -1.0), 0.0, 500.0)
                announce(f"hold stiffness -> {pc.fmt(kp, 1, 5)}")
            enter_mit(piper)
            send_mit(piper, q_hold, kp, kd, feed_forward(pc.joint_positions(piper)))
            rate.sleep()
        if choice == "park":
            duration = max(3.0, float(np.max(np.abs(q_hold))) / 0.4)
            print(f"returning to rest pose over {duration:.1f} s ...")
            t0 = time.perf_counter()
            while True:
                t = time.perf_counter() - t0
                q = pc.joint_positions(piper)
                enter_mit(piper)
                send_mit(piper, pc.min_jerk(q_hold, np.zeros(6), duration, t), kp, kd, feed_forward(q))
                if t > duration and (np.max(np.abs(q)) < np.radians(3.0) or t > duration + 3.0):
                    break
                rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
    print(f"hold gains used: --kp-hold {' '.join(f'{v:.1f}' for v in kp)}"
          f" --kd-hold {' '.join(f'{v:.2f}' for v in kd)}")
    print("disabling motors" + (" (done)" if pc.disable(piper) else " FAILED"))
    print("Before position control (step 2 / ROS bridge) run: python3 arm_tool.py reset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--check", action="store_true", help="compare model torque with effort feedback")
    ap.add_argument("--target", type=float, nargs=6, metavar="DEG", help="--check: move here first [deg]")
    ap.add_argument("--effort-x4", action="store_true", help="--check: scale J1-J3 effort by 4 (fw <= V1.8-2)")
    ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], choices=range(1, 7),
                    help="joints to make free (others hold)")
    ap.add_argument("--gain", type=float, default=1.0, help="global model torque multiplier")
    ap.add_argument("--tau-scale", type=float, nargs="+", default=[0.25],
                    help="MIT command per N*m of model torque (1 or 6 values)")
    ap.add_argument("--kd", type=float, nargs="+", default=[0.3], help="free-joint damping (1 or 6 values)")
    ap.add_argument("--kp-hold", type=float, nargs="+", default=[KP_HOLD],
                    help="stiffness of held joints, 1 or 6 values (lower it if holding buzzes)")
    ap.add_argument("--kd-hold", type=float, nargs="+", default=[KD_HOLD],
                    help="damping of held joints, 1 or 6 values")
    ap.add_argument("--gripper", choices=["off", "hold", "keep"], default="off",
                    help="gripper motor while the arm is free: off = disable it (it hums against "
                         "its own target otherwise), hold = keep its current width, keep = untouched")
    ap.add_argument("--k-wall", type=float, default=5.0, help="soft joint-limit spring kp")
    ap.add_argument("--ramp", type=float, default=4.0, help="gain ramp time [s]")
    ap.add_argument("--max-vel", type=float, default=3.0, help="speed watchdog [rad/s]")
    ap.add_argument("--rate", type=float, default=200.0, help="control rate [Hz]")
    ap.add_argument("--no-gripper", action="store_true", help="arm without the gripper")
    ap.add_argument("--tool", choices=sorted(TOOL_URDF), default="piper",
                    help="end-effector actually fitted: piper = standard gripper, pika = Pika gripper")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="extra mass the URDF does not know about, e.g. a camera [kg]")
    ap.add_argument("--payload-x", type=float, default=0.0, help="payload offset along link6 x [m]")
    ap.add_argument("--payload-z", type=float, default=0.15, help="payload offset along link6 z [m]")
    ap.add_argument("--urdf", default=None, help="URDF path; overrides --tool")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    model = build_model(args)
    piper = pc.connect(args.can)
    if args.check:
        run_check(piper, model, args)
    else:
        run_free_drive(piper, model, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
