#!/usr/bin/env python3
"""Cartesian impedance: the tool behaves like a spring-damper anchored at a target pose.

    tau = J(q)^T [ Kp (x_des - x) - Dp v ;  Kr e_rot - Dr w ] + gravity(q)

It goes out through MIT with joint kp = 0, so the stiffness you feel is the Cartesian one
you set. Push the gripper and it springs back; zero one axis and it goes soft in that
direction only while staying stiff in the others. Because the wrench is mapped with J^T
(no inverse), this stays well behaved near singularities.

    python3 impedance.py --tool pika --payload 0.42 --payload-x 0.03 --payload-z 0.07

Keys while running (the terminal must have focus):
    [ / ]     all translational stiffness down / up by 20 N/m
    x, y, z   toggle that axis soft (0) / normal   <- the fun one
    r         toggle rotational stiffness on / off
    t         re-anchor the target at the current pose
    h         hold: drop to zero stiffness, gravity only
    q         stop

Stiffness is limited by CAN latency, not by the maths. A Cartesian stiffness K felt at a
lever arm r loads the joints like K*r^2, and this arm starts to buzz above roughly
20 Nm/rad (the same limit that caps joint kp at about 6). So with the arm tucked in
(r ~ 0.15 m) several hundred N/m is fine, while at full stretch (r ~ 0.55 m) anything over
~70 N/m will hum. Start low, raise with ] until it hums, then back off one step.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import piper_common as pc
from gravity_compensation import (MIT_T_LIMIT, KD_HOLD, KP_HOLD, Rate, announce, enter_mit,
                                  finish, per_joint, send_mit, setup_gripper)
from gravity_model import TOOL_URDF, GravityModel, rotation_error


def clamp_norm(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


def build_model(args):
    urdf = args.urdf or TOOL_URDF[args.tool]
    model = GravityModel(urdf)
    if args.payload > 0:
        model.add_payload(args.payload, (args.payload_x, 0.0, args.payload_z))
    print(f"model: {Path(urdf).name}, tool mass {model.masses[-1]:.3f} kg")
    return model


def run(piper, model, args):
    tcp = model.tool_to_link(args.tcp)  # --tcp is given in the tool frame
    tau_scale = per_joint(args.tau_scale, "--tau-scale")
    joint_kd = per_joint(args.joint_kd, "--joint-kd")
    stiffness = np.full(3, args.kp_lin)
    rot_stiffness = args.kp_rot
    axis_on = np.ones(3, dtype=bool)
    rot_on = True

    print(f"firmware {pc.firmware_version(piper)}   TCP offset {tcp} m (link6 frame)")
    print(f"stiffness {args.kp_lin:.0f} N/m translation, {args.kp_rot:.1f} Nm/rad rotation, "
          f"damping {args.kd_lin:.0f} Ns/m")
    print("keys:  [ ] stiffness   x y z axis soft   r rotation   t re-anchor   h hold   q stop")
    pc.confirm("Keep one hand near the arm: Cartesian impedance is about to start.", args.yes)
    if not pc.enable(piper):
        raise SystemExit("enable failed (after MIT mode run: python3 arm_tool.py reset)")
    setup_gripper(piper, args.gripper)

    q_start = pc.joint_positions(piper)
    target_pos, target_rot = model.tip_pose(q_start, tcp)
    rate = Rate(args.rate)
    half_ramp = max(args.ramp, 1e-3) / 2.0
    kp_hold = per_joint(args.kp_hold, "--kp-hold")
    t0 = time.perf_counter()
    cycle = 0
    fast_cycles = 0
    reason = "user"
    keys = pc.KeyReader()
    keys.start()
    enter_mit(piper)
    try:
        while True:
            key = keys.get()
            if key in ("[", "]"):
                stiffness = np.clip(stiffness + (20.0 if key == "]" else -20.0), 0.0, 2000.0)
                announce(f"translational stiffness -> {stiffness[0]:.0f} N/m")
            elif key in ("x", "y", "z"):
                i = "xyz".index(key)
                axis_on[i] = not axis_on[i]
                announce(f"{key} axis {'stiff' if axis_on[i] else 'SOFT'}")
            elif key == "r":
                rot_on = not rot_on
                announce(f"rotation {'stiff' if rot_on else 'SOFT'}")
            elif key == "t":
                target_pos, target_rot = model.tip_pose(pc.joint_positions(piper), tcp)
                announce(f"target re-anchored at {np.round(target_pos, 3)}")
            elif key == "h":
                axis_on[:] = False
                rot_on = False
                announce("all axes soft: gravity only")
            elif key == "q":
                break

            if pc.feedback_age(piper) > 0.2:
                reason = "feedback lost"
                break
            q = pc.joint_positions(piper)
            dq = pc.joint_velocities(piper)

            # Ramp: hold on joint stiffness first, then fade that out as the wrench fades in.
            elapsed = time.perf_counter() - t0
            gravity_scale = min(elapsed / half_ramp, 1.0)
            hold = 1.0 - min(max(elapsed - half_ramp, 0.0) / half_ramp, 1.0)
            wrench_scale = 1.0 - hold

            position, rotation = model.tip_pose(q, tcp)
            jacobian = model.jacobian(q, tcp)
            twist = jacobian @ dq
            force = np.where(axis_on, stiffness, 0.0) * (target_pos - position) - args.kd_lin * twist[:3]
            moment = (rot_stiffness if rot_on else 0.0) * rotation_error(target_rot, rotation) \
                - args.kd_rot * twist[3:]
            force = clamp_norm(force, args.max_force)
            moment = clamp_norm(moment, args.max_moment)

            tau = jacobian.T @ np.concatenate([force, moment]) * wrench_scale
            tau = tau + model.gravity_torques(q) * args.gain * gravity_scale
            t_ref = np.clip(tau * tau_scale, -MIT_T_LIMIT, MIT_T_LIMIT)
            if cycle % 5 == 0:
                enter_mit(piper)
            send_mit(piper, q_start, kp_hold * hold, np.maximum(joint_kd, KD_HOLD * hold), t_ref)

            fast_cycles = fast_cycles + 1 if np.any(np.abs(dq) > args.max_vel) else 0
            if fast_cycles >= 3:
                reason = f"joint speed above {args.max_vel} rad/s"
                break

            if cycle % max(int(args.rate / 5), 1) == 0:
                soft = "".join(a if not on else "-" for a, on in zip("xyz", axis_on))
                print(f"\r[{'ramp' if hold > 0 else 'IMPEDANCE'}] K {stiffness[0]:4.0f} N/m "
                      f"soft:{soft or '---'}{'' if rot_on else '+rot'} | err {np.linalg.norm(target_pos - position) * 1000:5.1f} mm"
                      f" | F {np.round(force, 1)} N   ", end="", flush=True)
            cycle += 1
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        print(f"\nstiffness used: --kp-lin {stiffness[0]:.0f} --kp-rot {rot_stiffness:.1f}")
    finish(piper, model, args, tau_scale, reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--tool", choices=sorted(TOOL_URDF), default="pika")
    ap.add_argument("--urdf", default=None, help="URDF path; overrides --tool")
    ap.add_argument("--payload", type=float, default=0.42, help="extra tool mass [kg]")
    ap.add_argument("--payload-x", type=float, default=0.03)
    ap.add_argument("--payload-z", type=float, default=0.07)
    ap.add_argument("--tcp", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="tool centre point in the TOOL frame [m]: +X out of the gripper")
    ap.add_argument("--kp-lin", type=float, default=80.0, help="translational stiffness [N/m]")
    ap.add_argument("--kp-rot", type=float, default=3.0, help="rotational stiffness [Nm/rad]")
    ap.add_argument("--kd-lin", type=float, default=25.0, help="translational damping [Ns/m]")
    ap.add_argument("--kd-rot", type=float, default=0.4, help="rotational damping [Nms/rad]")
    ap.add_argument("--max-force", type=float, default=20.0, help="wrench clamp [N]")
    ap.add_argument("--max-moment", type=float, default=4.0, help="wrench clamp [Nm]")
    ap.add_argument("--joint-kd", type=float, nargs="+", default=[0.3],
                    help="extra joint-space damping, 1 or 6 values")
    ap.add_argument("--tau-scale", type=float, nargs="+", default=[0.25],
                    help="MIT command per Nm (the driver executes 4x)")
    ap.add_argument("--gain", type=float, default=1.0, help="gravity model multiplier")
    ap.add_argument("--kp-hold", type=float, nargs="+", default=[KP_HOLD],
                    help="joint stiffness used while ramping and when holding")
    ap.add_argument("--kd-hold", type=float, default=KD_HOLD)
    ap.add_argument("--gripper", choices=["off", "hold", "keep"], default="off")
    ap.add_argument("--ramp", type=float, default=4.0, help="gain ramp time [s]")
    ap.add_argument("--max-vel", type=float, default=3.0, help="speed watchdog [rad/s]")
    ap.add_argument("--rate", type=float, default=100.0, help="control rate [Hz]")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    model = build_model(args)
    run(pc.connect(args.can), model, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
