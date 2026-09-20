#!/usr/bin/env python3
"""Watch position and torque of all six joints, to catch oscillation.

Read-only: it never sends a command, so it is safe to run in a second terminal while
gravity_compensation.py, joint_position_ctrl.py or the ROS bridge is driving the arm.

  python3 monitor.py                    # live table
  python3 monitor.py --effort-x4        # firmware <= V1.8-2: J1-J3 torque reads 4x too small
  python3 monitor.py --csv shake.csv    # also log every sample for plotting later

A joint that is being moved on purpose shows a large p2p at a low frequency. A joint that
is buzzing shows a small p2p at a high frequency, and its torque p2p is large compared to
how little it actually moves -- that ratio is what gives a self-oscillating loop away.
"""
import argparse
import time
from collections import deque

import numpy as np

import piper_common as pc
from gravity_compensation import Rate


def high_pass(values, samples):
    """Signal minus its own moving average: removes the slow part (you dragging the arm)
    and leaves the fast wiggle (a joint shaking on its own)."""
    if samples < 2 or len(values) < 2 * samples:
        return values - np.mean(values)
    # 'valid' only, no padding: padding the ends fakes a large residual there.
    trend = np.convolve(values, np.ones(samples) / samples, mode="valid")
    start = (samples - 1) // 2
    return values[start:start + len(trend)] - trend


def dominant_frequency(values, duration):
    """Zero-crossing estimate of the dominant frequency of a mean-removed signal [Hz]."""
    if duration <= 0 or len(values) < 4:
        return 0.0
    centred = np.asarray(values) - np.mean(values)
    if np.max(np.abs(centred)) < 1e-6:
        return 0.0
    crossings = int(np.count_nonzero(np.diff(np.signbit(centred))))
    return crossings / (2.0 * duration)


def render(piper, elapsed, times, positions, torques, velocities, shake_deg):
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    period = span / max(len(times) - 1, 1)
    smooth = max(int(0.15 / period), 2) if period > 0 else 2
    lines = [f"monitor  {elapsed:6.1f} s   window {span:4.2f} s   {len(times)} samples   (Ctrl+C to quit)",
             pc.status_line(piper),
             "",
             "joint   pos[deg]  moved[deg]  shake[deg]  f[Hz]   tau[Nm]  tau p2p   shaking"]
    worst = (0.0, 0, 0.0)
    for i in range(6):
        degrees = np.degrees(positions[:, i])
        moved = float(degrees.max() - degrees.min())
        fast = high_pass(degrees, smooth)
        # 1st..99th percentile, not min..max: one fast move (parking, a shove) would
        # otherwise leak through the high-pass and dominate the whole window.
        shake = float(np.percentile(fast, 99) - np.percentile(fast, 1))
        torque_p2p = float(torques[:, i].max() - torques[:, i].min())
        frequency = dominant_frequency(fast, span)
        bar = "#" * min(int(shake / shake_deg * 10), 30)
        lines.append(f"joint{i + 1} {degrees[-1]:9.3f} {moved:11.3f} {shake:11.3f} {frequency:6.1f} "
                     f"{torques[-1, i]:9.3f} {torque_p2p:8.3f}  {bar}")
        if shake > worst[0]:
            worst = (shake, i + 1, frequency)
    shake, joint, frequency = worst
    if shake > shake_deg:
        lines.append(f"\n>>> joint{joint} is shaking: {shake:.2f} deg fast wiggle at {frequency:.0f} Hz"
                     "   (a regular frequency = control loop; irregular = floppy joint or stick-slip)")
    else:
        lines.append("\n    no fast wiggle (moved[deg] is just the arm being moved)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--rate", type=float, default=200.0, help="sampling rate [Hz]")
    ap.add_argument("--refresh", type=float, default=10.0, help="screen refresh rate [Hz]")
    ap.add_argument("--window", type=float, default=1.0, help="analysis window [s]")
    ap.add_argument("--shake-deg", type=float, default=0.3,
                    help="fast-wiggle peak-to-peak above which a joint counts as shaking [deg]. "
                         "Hard dragging leaks about 0.25 deg through the high-pass, hence 0.3")
    ap.add_argument("--effort-x4", action="store_true",
                    help="scale J1-J3 torque by 4 (firmware <= V1.8-2)")
    ap.add_argument("--csv", help="write every sample to this file")
    ap.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl+C)")
    args = ap.parse_args()

    piper = pc.connect(args.can)
    size = max(int(args.window * args.rate), 8)
    times, positions, torques = deque(maxlen=size), deque(maxlen=size), deque(maxlen=size)

    csv = open(args.csv, "w") if args.csv else None
    if csv:
        columns = ["t"] + [f"q{i}" for i in range(1, 7)] + [f"dq{i}" for i in range(1, 7)] \
            + [f"tau{i}" for i in range(1, 7)]
        csv.write(",".join(columns) + "\n")

    rate = Rate(args.rate)
    start = time.perf_counter()
    next_draw = 0.0
    try:
        while True:
            elapsed = time.perf_counter() - start
            q = pc.joint_positions(piper)
            dq = pc.joint_velocities(piper)
            tau = pc.joint_efforts(piper, args.effort_x4)
            times.append(elapsed)
            positions.append(q)
            torques.append(tau)
            if csv:
                csv.write(",".join(f"{v:.5f}" for v in (elapsed, *q, *dq, *tau)) + "\n")
            if elapsed >= next_draw and len(times) > 3:
                next_draw = elapsed + 1.0 / args.refresh
                print("\033[2J\033[H" + render(piper, elapsed, np.array(times), np.array(positions),
                                               np.array(torques), dq, args.shake_deg), flush=True)
            if args.seconds and elapsed >= args.seconds:
                break
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if csv:
            csv.close()
            print(f"\nsamples written to {args.csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
