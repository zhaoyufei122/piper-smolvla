#!/usr/bin/env python3
"""Cartesian impedance: the tool behaves like a spring-damper anchored at a target pose.

    tau = J(q)^T [ Kp (x_des - x) - Dp v ;  Kr e_rot - Dr w ] + gravity(q)

It goes out through MIT with joint kp = 0, so the stiffness you feel is the Cartesian one
you set. Push the gripper and it springs back; zero one axis and it goes soft in that
direction only while staying stiff in the others. Because the wrench is mapped with J^T
(no inverse), this stays well behaved near singularities.

    python3 impedance.py
    python3 analyse_impedance.py          # what the last run did, and why it shook

Keys while running (the terminal must have focus):
    [ / ]     all translational stiffness down / up by 20 N/m
    x, y, z   toggle that axis soft (0) / normal   <- the fun one
    r         toggle rotational stiffness on / off
    t         re-anchor the target at the current pose
    h         hold: drop to zero stiffness, gravity only
    q         stop

What limits it is latency, and it bites the damping before the stiffness. Everything
computed here reaches the motor ~10 ms after the state it was computed from, and a damper
that late pushes with the motion instead of against it: with --kd-lin 25 the arm shook at
25 Hz whenever it was pushed (measured 2026-09-21). So the Cartesian damping is kept small
and most damping is left to the drivers' own kd (--joint-kd), which acts locally with no
delay. Stiffness held up far better: left alone the arm was still at every K up to 900 N/m.
Start at 60 and raise with ].

The J1-J3 drivers multiply MIT torque by 4 and the J4-J6 drivers do not, so --tau-scale
defaults to 0.25 0.25 0.25 1 1 1 (see gravity_compensation.TAU_SCALE). Each joint's commanded
torque is also capped (--max-torque) a little above what position control uses routinely.

Every run is traced cycle by cycle into ~/piper_data/runs/impedance/imp_NNN.jsonl
(--no-trace to skip); python3 analyse_impedance.py reads it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import piper_common as pc
from gravity_compensation import (MIT_T_LIMIT, KD_HOLD, KP_HOLD, TAU_SCALE, Rate, announce,
                                  enter_mit, finish, per_joint, send_mit, setup_gripper)
from gravity_model import TOOL_URDF, GravityModel, rotation_error


def clamp_norm(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


T_STEP = 2 * MIT_T_LIMIT / 255   # one t_ref code: ~0.25 Nm at J1-J3 after the driver's x4
KP_MAX, KP_BITS = 500.0, 12
POS_SPAN, POS_BITS = 25.0, 16     # pos_ref is encoded over [-12.5, 12.5] rad


def mit_torque_code(t_ref):
    """The 8-bit code the SDK actually puts on the bus for t_ref (FloatToUint, truncating)."""
    return ((np.asarray(t_ref) + MIT_T_LIMIT) * 255 / (2 * MIT_T_LIMIT)).astype(int)


def mit_kp_received(kp):
    """kp as the driver decodes it after the SDK's 12-bit truncation."""
    levels = (1 << KP_BITS) - 1
    return np.floor(np.asarray(kp) * levels / KP_MAX) * KP_MAX / levels


def fine_torque(t_ref, q, kp_spring, q_spring, kp_fine):
    """Deliver t_ref at far better than the 8-bit t_ref resolution.

    t_ref alone moves in 0.063 steps (0.25 Nm at J1-J3), so a torque that should change
    smoothly -- a spring being stretched, gravity as the arm turns -- arrives as a string of
    small kicks. Here t_ref is rounded to its nearest code and the remainder goes through the
    driver's own spring instead: kp * (pos_ref - q), with pos_ref 16-bit. The driver applies
    the same gain to both paths (measured: 4x at J1-J3, 1x at J5), so the split is exact
    without knowing that gain. Any hold spring (kp_spring towards q_spring) rides along.

    Returns (t_ref to send, kp to send, pos_ref to send, the remainder carried by the spring).
    """
    code = np.clip(np.round((t_ref + MIT_T_LIMIT) / T_STEP), 0, 255)
    coarse = code * T_STEP - MIT_T_LIMIT
    remainder = t_ref - coarse                      # within half a code
    kp = kp_spring + kp_fine
    kp_real = mit_kp_received(kp)
    pos_ref = q + (kp_spring * (q_spring - q) + remainder) / kp_real
    lsb = POS_SPAN / ((1 << POS_BITS) - 1)
    # The SDK truncates when encoding, so aim at the middle of the intended code / step.
    return coarse + 0.5 * T_STEP, kp, pos_ref + 0.5 * lsb, remainder


def rounded(values, digits=4):
    return [round(float(v), digits) for v in values]


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
    max_torque = per_joint(args.max_torque, "--max-torque")
    capped_before = False

    trace, trace_path = None, None
    if not args.no_trace:
        trace_path = pc.open_trace(args.trace, "imp")
        trace = trace_path.open("w")
        trace.write(json.dumps({"meta": {
            "script": "impedance", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "firmware": str(pc.firmware_version(piper)), "args": vars(args),
            "tcp_link6": rounded(tcp), "q_start": rounded(q_start),
            "tau_scale": rounded(tau_scale), "mit_t_limit": MIT_T_LIMIT,
            "mit_t_bits": 8, "effort_x4_j123": True,
        }}) + "\n")
        print(f"tracing into {trace_path}")

    t0 = time.perf_counter()
    last_start = t0
    cycle = 0
    fast_cycles = 0
    last_problem = None
    reason = "user"
    keys = pc.KeyReader()
    keys.start()
    enter_mit(piper)
    try:
        while True:
            start = time.perf_counter()
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

            tau_wrench = jacobian.T @ np.concatenate([force, moment]) * wrench_scale
            tau_gravity = model.gravity_torques(q) * args.gain * gravity_scale
            tau = tau_wrench + tau_gravity
            capped = np.abs(tau) > max_torque
            tau = np.clip(tau, -max_torque, max_torque)
            if capped.any() and not capped_before:
                announce("torque capped on " + " ".join(f"J{j + 1}" for j in np.nonzero(capped)[0])
                         + f" at {pc.fmt(max_torque, 1)} Nm")
            capped_before = capped.any()
            t_ref = np.clip(tau * tau_scale, -MIT_T_LIMIT, MIT_T_LIMIT)
            kp_sent = kp_hold * hold
            kd_sent = np.maximum(joint_kd, KD_HOLD * hold)
            pos_sent, t_sent, t_fine = q_start, t_ref, np.zeros(6)
            if args.fine_torque:
                t_sent, kp_sent, pos_sent, t_fine = fine_torque(t_ref, q, kp_hold * hold, q_start,
                                                                args.kp_fine)
            if cycle % 5 == 0:
                enter_mit(piper)
            send_mit(piper, pos_sent, kp_sent, kd_sent, t_sent)

            # Driver health, 10x less often than the loop: the low-speed feedback is slow anyway.
            health = pc.health(piper) if cycle % 10 == 0 else None
            problem = pc.health_problem(health) if health else None
            if problem and problem != last_problem:
                announce(f"DRIVER PROBLEM: {problem}")
            last_problem = problem if health else last_problem
            if trace is not None:
                hs = piper.GetArmHighSpdInfoMsgs()
                trace.write(json.dumps({
                    "t": round(start - t0, 5), "dt": round(start - last_start, 5),
                    "fb_age": round(pc.feedback_age(piper), 4),
                    "hs_age": round(time.time() - hs.time_stamp, 4),
                    "key": key,
                    "hold": round(hold, 3), "K": float(stiffness[0]),
                    "axis_on": axis_on.tolist(), "rot_on": rot_on,
                    "q": rounded(q, 5), "qd": rounded(dq, 4),
                    "effort": rounded(pc.joint_efforts(piper, x4_j123=True), 3),
                    "pos": rounded(position, 5), "target": rounded(target_pos, 5),
                    "twist": rounded(twist, 4),
                    "force": rounded(force, 3), "moment": rounded(moment, 3),
                    "capped": capped.tolist(),
                    "tau_wrench": rounded(tau_wrench, 4), "tau_gravity": rounded(tau_gravity, 4),
                    "t_ref": rounded(t_ref, 4), "t_code": mit_torque_code(t_sent).tolist(),
                    "t_fine": rounded(t_fine, 5), "pos_offset": rounded(pos_sent - q, 5),
                    "kp": rounded(kp_sent, 3), "kd": rounded(kd_sent, 3),
                    **({"health": health} if health else {}),
                }) + "\n")
            last_start = start

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
        if trace is not None:
            trace.write(json.dumps({"meta": {
                "stop_reason": reason, "cycles": cycle,
                "joint_feedback_hz": piper.GetArmJointMsgs().Hz,
                "high_speed_feedback_hz": piper.GetArmHighSpdInfoMsgs().Hz,
            }}) + "\n")
            trace.close()
            print(f"trace written to {trace_path}\n"
                  f"  analyse with: python3 analyse_impedance.py {trace_path}")
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
    ap.add_argument("--kd-lin", type=float, default=5.0,
                    help="translational damping [Ns/m]. It arrives ~10 ms late over CAN: 25 made "
                         "the arm shake at 25 Hz when pushed")
    ap.add_argument("--kd-rot", type=float, default=0.1, help="rotational damping [Nms/rad]")
    ap.add_argument("--max-force", type=float, default=20.0, help="wrench clamp [N]")
    ap.add_argument("--max-moment", type=float, default=4.0, help="wrench clamp [Nm]")
    ap.add_argument("--max-torque", type=float, nargs="+", default=[6.0, 12.0, 10.0, 1.0, 2.0, 1.0],
                    help="per-joint cap on the commanded torque [Nm at the joint]. Defaults sit "
                         "above what position control uses routinely (J4 0.65, J5 1.6, J6 0.4 Nm) "
                         "and above gravity + a 20 N push at J1-J3")
    ap.add_argument("--joint-kd", type=float, nargs="+", default=[0.8, 0.8, 0.8, 0.3, 0.3, 0.3],
                    help="extra joint-space damping, 1 or 6 values")
    ap.add_argument("--tau-scale", type=float, nargs="+", default=TAU_SCALE,
                    help="MIT command per Nm, 1 or 6 values. Measured: J1-J3 drivers execute 4x "
                         "(0.25), J5 executes 1x (1.0)")
    ap.add_argument("--fine-torque", action="store_true",
                    help="send the part of the torque below one 8-bit t_ref step through the "
                         "driver's kp/pos_ref path (16-bit) instead of dropping it")
    ap.add_argument("--kp-fine", type=float, default=2.0,
                    help="--fine-torque: driver kp carrying the remainder [MIT units]")
    ap.add_argument("--gain", type=float, default=1.0, help="gravity model multiplier")
    ap.add_argument("--kp-hold", type=float, nargs="+", default=[KP_HOLD],
                    help="joint stiffness used while ramping and when holding")
    ap.add_argument("--kd-hold", type=float, default=KD_HOLD)
    ap.add_argument("--gripper", choices=["off", "hold", "keep"], default="off")
    ap.add_argument("--ramp", type=float, default=4.0, help="gain ramp time [s]")
    ap.add_argument("--max-vel", type=float, default=3.0, help="speed watchdog [rad/s]")
    ap.add_argument("--rate", type=float, default=100.0, help="control rate [Hz]")
    ap.add_argument("--trace", default="~/piper_data/runs/impedance",
                    help="directory (auto-numbered imp_NNN.jsonl) or a .jsonl file")
    ap.add_argument("--no-trace", action="store_true", help="do not record this run")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    model = build_model(args)
    run(pc.connect(args.can), model, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
