#!/usr/bin/env python3
"""Work out why Cartesian impedance twitches, from an impedance.py trace.

    python3 impedance.py                       # traces into ~/piper_data/runs/impedance/
    python3 analyse_impedance.py               # newest trace
    python3 analyse_impedance.py ~/piper_data/runs/impedance/imp_003.jsonl

For a useful trace: start it, leave the arm alone for ~10 s, then push it gently along
x, y and z one at a time, then press q.

Five things can make this controller twitch, and each leaves a different fingerprint:

  torque quantisation   the SDK packs MIT t_ref into 8 bits over +-8, and the driver
                        multiplies by 4, so the joint torque moves in ~0.25 Nm steps. The
                        spring is a staircase: the tip drifts through a dead band, the
                        code steps, the joint gets a kick. Twitches follow code changes.
                        Or, when the wanted torque sits on a code boundary, the code
                        flips back and forth every cycle: a 30-50 Hz chatter that the
                        command itself carries.
  loop timing           late cycles or stale feedback: the torque is computed from where
                        the arm was, not where it is. Shows up as long dt / feedback age.
  delayed damping       the Cartesian damping is computed here from measured speed and
                        reaches the motor ~10 ms later over CAN. A damper that late pushes
                        in phase with the motion instead of against it: a 20-30 Hz shake
                        whose torque is almost all the damping term. (Found on imp_000.)
  noisy velocity        the Cartesian damping multiplies measured speed; if that speed is
                        noisy, the damping torque is noise and flips codes at random.
  stiffness too high    K * r^2 over ~20 Nm/rad buzzes: a steady tone above ~8 Hz that the
                        arm makes on its own, with a quiet command.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse_shake import band_energy, peak, spectrum
from gravity_model import TOOL_URDF, GravityModel

AXES = "xyz"


def load(path):
    meta, rows = {}, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        (meta.update(record["meta"]) if "meta" in record else rows.append(record))
    if not rows:
        raise SystemExit(f"{path} has no samples")
    return meta, rows


def newest_trace():
    traces = sorted(Path("~/piper_data/runs/impedance").expanduser().glob("imp_*.jsonl"))
    if not traces:
        raise SystemExit("no traces in ~/piper_data/runs/impedance -- run impedance.py first")
    return traces[-1]


def build_model(meta):
    args = meta.get("args", {})
    model = GravityModel(args.get("urdf") or TOOL_URDF[args.get("tool", "pika")])
    if args.get("payload", 0) > 0:
        model.add_payload(args["payload"], (args.get("payload_x", 0.0), 0.0, args.get("payload_z", 0.0)))
    return model


def column(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def twitches(qd, dt, floor):
    """Cycles where a joint's speed jumps far more than it normally does between samples.

    A twitch is a jolt: a change of speed between two consecutive samples that stands out
    from that joint's usual sample-to-sample change by a wide margin (and by an absolute
    floor, so a joint sitting perfectly still is not flagged for a 0.001 rad/s blip).
    """
    jump = np.abs(np.diff(qd, axis=0))
    median = np.median(jump, axis=0)
    mad = np.median(np.abs(jump - median), axis=0) + 1e-9
    threshold = np.maximum(median + 8 * 1.4826 * mad, floor)
    return jump > threshold, threshold / dt


def reversals(codes):
    """Per joint, code changes that undo the previous one within two cycles: dithering."""
    counts = []
    for j_codes in codes.T:
        delta = np.diff(j_codes)
        steps = np.nonzero(delta)[0]
        signs = np.sign(delta[steps])
        counts.append(int(np.sum((signs[1:] == -signs[:-1]) & (np.diff(steps) <= 2))))
    return np.array(counts)


def preceded_by(events, changes, window):
    """Fraction of events that have a code change in the `window` cycles before them."""
    if not events.any():
        return float("nan")
    hits = [changes[max(i - window, 0):i + 1].any() for i in np.nonzero(events)[0]]
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="imp_NNN.jsonl (default: the newest)")
    ap.add_argument("--jolt", type=float, default=0.05,
                    help="smallest speed jump between samples that counts as a twitch [rad/s]")
    args = ap.parse_args()

    path = Path(args.trace).expanduser() if args.trace else newest_trace()
    meta, rows = load(path)
    model = build_model(meta)
    tcp = np.array(meta.get("tcp_link6", [0.0, 0.0, 0.0]))
    tau_scale = np.array(meta.get("tau_scale", [0.25] * 6))
    t_limit = meta.get("mit_t_limit", 8.0)
    step_cmd = 2 * t_limit / (2 ** meta.get("mit_t_bits", 8) - 1)
    step_nm = step_cmd / tau_scale            # what one code step is at the joint
    period = 1.0 / meta.get("args", {}).get("rate", 100.0)

    t = column(rows, "t")
    dt = column(rows, "dt")[1:]
    q = np.array([r["q"] for r in rows])
    qd = np.array([r["qd"] for r in rows])
    effort = np.array([r["effort"] for r in rows])
    codes = np.array([r["t_code"] for r in rows])
    t_fine = np.array([r.get("t_fine", [0.0] * 6) for r in rows])
    tau_wrench = np.array([r["tau_wrench"] for r in rows])
    twist = np.array([r["twist"] for r in rows])
    force = np.array([r["force"] for r in rows])
    hold = column(rows, "hold")
    live = hold == 0.0
    duration = t[-1] - t[0]

    print(f"{path.name}: {len(rows)} cycles, {duration:.1f} s "
          f"({live.sum() * period:.1f} s in impedance, the rest ramping)")
    a = meta.get("args", {})
    print(f"  K {a.get('kp_lin')} N/m   kd-lin {a.get('kd_lin')} Ns/m   joint-kd {a.get('joint_kd')}"
          f"   rate {a.get('rate')} Hz   stop: {meta.get('stop_reason', '?')}")
    if meta.get("joint_feedback_hz"):
        print(f"  feedback: joints {meta['joint_feedback_hz']:.0f} Hz, "
              f"speed/torque {meta['high_speed_feedback_hz']:.0f} Hz")
    stiffness = column(rows, "K")
    settled = [(t[i], stiffness[i]) for i in range(len(rows))
               if i == 0 or stiffness[i] != stiffness[i - 1]]
    settled = [(ts, k) for n, (ts, k) in enumerate(settled)
               if (settled[n + 1][0] if n + 1 < len(settled) else t[-1]) - ts > 1.0]
    print("  K over time: " + "  ".join(f"{k:.0f}@{ts:.0f}s" for ts, k in settled))
    soft = [r["t"] for r in rows if r.get("key") == "h"]
    if soft:
        print(f"  all axes soft (h) from {soft[0]:.0f} s")

    # 1. Timing: does each cycle act on fresh data, on time?
    print("\n1. loop timing")
    late = dt > 1.5 * period
    fb_age = column(rows, "fb_age")
    stale = fb_age > period            # no joint frame arrived during the last cycle
    print(f"   cycle period  median {np.median(dt) * 1000:.1f} ms   99% {np.percentile(dt, 99) * 1000:.1f} ms"
          f"   worst {dt.max() * 1000:.1f} ms   late (>1.5x) {late.mean():.1%}")
    print(f"   feedback age  median {np.median(fb_age) * 1000:.1f} ms   worst {fb_age.max() * 1000:.1f} ms"
          f"   cycles with no new joint frame {stale.mean():.1%}")
    timing_bad = late.mean() > 0.02 or np.percentile(fb_age, 99) > 0.02

    # 1b. Does each joint get the torque we ask for? At rest with kp = 0, effort = gain * command.
    commanded_mit = codes * step_cmd - t_limit + t_fine       # what was sent, in MIT units
    rest = live & (np.degrees(np.abs(qd[:, :3])).max(axis=1) < 0.5)
    gravity = np.array([r["tau_gravity"] for r in rows])
    print("\n1b. torque gain per joint (at rest: measured effort / MIT command sent)")
    scale_wrong = []
    for j in range(6):
        c, e = commanded_mit[rest, j], effort[rest, j]
        if rest.sum() < 50 or np.abs(gravity[rest, j]).mean() < 0.2:
            print(f"   J{j + 1}: too little load at rest to measure")
            continue
        gain = float(c @ e / (c @ c))
        expected = 1.0 / tau_scale[j]
        ok = abs(gain / expected - 1) < 0.35
        print(f"   J{j + 1}: driver gain {gain:4.2f}   --tau-scale {tau_scale[j]:.2f} assumes "
              f"{expected:.2f}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            scale_wrong.append((j, gain))

    # 2. Quantisation: how coarse is the torque staircase compared with what we ask for?
    print(f"\n2. torque resolution: one t_ref code = {step_cmd:.4f} -> "
          f"{step_nm[0]:.3f} Nm at the joint (8 bits over +-{t_limit:g}, x{1 / tau_scale[0]:.0f} in the driver)")
    changes = np.r_[np.zeros((1, 6), dtype=bool), np.diff(codes, axis=0) != 0]
    live_seconds = max(live.sum() * period, 1e-6)
    dither = reversals(codes[live]) / live_seconds if live.any() else np.zeros(6)
    print(f"   {'':4s}{'wrench torque':>22s}{'  in codes':>11s}{'code changes':>15s}"
          f"{'  back-and-forth':>17s}")
    for j in range(6):
        span = np.ptp(tau_wrench[live, j]) if live.any() else 0.0
        std = tau_wrench[live, j].std() if live.any() else 0.0
        rate_changes = changes[live, j].sum() / live_seconds
        print(f"   J{j + 1}  std {std:6.3f} p2p {span:5.2f} Nm  {std / step_nm[j]:5.2f} std"
              f"   {rate_changes:6.1f} /s   {dither[j]:6.1f} /s")

    # What that staircase means at the tip: how far it moves before any joint's code steps.
    q_mid = np.median(q[live] if live.any() else q, axis=0)
    jacobian = model.jacobian(q_mid, tcp)
    K = float(np.median(column(rows, "K")[live])) if live.any() else a.get("kp_lin", 80.0)
    lever = np.abs(jacobian[:3, :])                 # joint torque per newton, per axis
    print(f"   at the typical pose, with K = {K:.0f} N/m, the tip moves this far before a code steps:")
    dead_band = {}
    for axis in range(3):
        sensitive = lever[axis] > 1e-3
        dead_band[axis] = float(np.min(step_nm[sensitive] / (K * lever[axis, sensitive]))) if sensitive.any() else np.inf
        print(f"     {AXES[axis]}: {dead_band[axis] * 1000:5.1f} mm   (force steps of "
              f"{K * dead_band[axis]:.2f} N)")

    # 3. Twitches, and what came just before them.
    print("\n3. twitches (sudden speed jumps between consecutive samples)")
    jolts, threshold = twitches(qd, period, args.jolt)
    jolts = jolts & live[1:, None]
    window = 3
    explained = []
    for j in range(6):
        n = int(jolts[:, j].sum())
        after_change = preceded_by(jolts[:, j], changes[1:, j], window)
        chance = changes[1:, j][live[1:]].mean() * (window + 1) if live.any() else 0.0
        explained.append((j, n, after_change, min(chance, 1.0)))
        print(f"   J{j + 1}: {n:4d} ({n / max(live.sum() * period, 1e-6):4.1f}/s)"
              + (f"   {after_change:4.0%} came within {window * period * 1000:.0f} ms of a torque code step"
                 f" (chance: {min(chance, 1.0):.0%})" if n else ""))
    total = sum(n for _, n, _, _ in explained)
    lifted = [(j, n, f, c) for j, n, f, c in explained if n >= 5 and f > max(2 * c, c + 0.3)]

    # 4. Damping from measured speed: is the velocity signal clean enough to multiply?
    print("\n4. damping torque from measured speed")
    still = live & (np.linalg.norm(twist[:, :3], axis=1) < 0.01)
    damping_torque = np.array([
        model.jacobian(qi, tcp).T @ np.r_[-a.get("kd_lin", 25.0) * v[:3], -a.get("kd_rot", 0.4) * v[3:]]
        for qi, v in zip(q, twist)])
    if still.sum() > 20:
        noise = damping_torque[still].std(axis=0)
        print(f"   while nearly still ({still.sum() * period:.1f} s): speed noise "
              f"{np.degrees(qd[still].std(axis=0)).round(2)} deg/s")
        print(f"   -> damping torque noise {noise.round(3)} Nm = "
              f"{(noise / step_nm).round(2)} codes")
        damping_noisy = bool(np.any(noise > 0.5 * step_nm))
    else:
        print("   not enough still time to judge (leave the arm alone for a few seconds next run)")
        damping_noisy = False

    # 5. Where the shaking is, and which half of the controller is doing it.
    # tau_wrench = J^T [K e - D v]: split it, so a shake can be pinned on the spring or the damper.
    spring_torque = tau_wrench - damping_torque * (1.0 - hold[:, None])
    window = int(round(2.0 / period))
    print("\n5. shaking, in 2 s windows (fast = above 8 Hz)")
    print(f"   {'t [s]':>7s} {'K':>5s} {'joint':>6s} {'at':>7s} {'measured':>10s} {'sent':>9s}"
          f" {'= damping':>10s} {'+ spring':>9s} {'+ steps':>8s}")
    shaking, calm_changes, still_windows = [], [], 0
    for s0 in range(0, len(rows) - window + 1, window):
        sl = slice(s0, s0 + window)
        if not live[sl].all():
            continue
        measured = np.array([band_energy(*spectrum(effort[sl, j], 1 / period), 8, 0.5 / period) for j in range(6)])
        worst = int(np.argmax(measured))
        moving = np.degrees(np.abs(qd[sl, :3]).mean(axis=0)).max() > 0.5
        if measured[worst] < 0.2:
            if not moving:
                still_windows += 1
                calm_changes.append(changes[sl].sum() / 2.0)
            continue
        from_damping = band_energy(*spectrum(damping_torque[sl, worst], 1 / period), 8, 0.5 / period)
        from_spring = band_energy(*spectrum(spring_torque[sl, worst], 1 / period), 8, 0.5 / period)
        # What actually went out, 8-bit steps included, in joint Nm. Whatever it carries beyond
        # the smooth damping + spring terms is the staircase.
        sent = band_energy(*spectrum(commanded_mit[sl, worst] / tau_scale[worst], 1 / period), 8, 0.5 / period)
        steps = max(sent - from_damping - from_spring, 0.0)
        f, amp = spectrum(qd[sl, worst], 1 / period)
        frequency = peak(f, amp, 8.0)[0]
        shaking.append((t[s0], worst, frequency, measured[worst], sent, steps, from_damping, from_spring))
        print(f"   {t[s0]:7.0f} {column(rows, 'K')[s0]:5.0f} {'J' + str(worst + 1):>6s} {frequency:5.1f}Hz"
              f" {measured[worst]:8.2f}Nm {sent:7.2f}Nm {from_damping:8.2f}Nm {from_spring:7.2f}Nm"
              f" {steps:6.2f}Nm")
    live_windows = max(int(live.sum()) // window, 1)
    print(f"   shaking in {len(shaking)} of {live_windows} windows;"
          f" left alone and still ({still_windows} windows) the torque codes changed "
          f"{np.mean(calm_changes) if calm_changes else float('nan'):.1f}/s")
    total_measured = sum(m for _, _, _, m, *_ in shaking)
    total_sent = sum(se for _, _, _, _, se, *_ in shaking)
    total_steps = sum(st for *_, st, d, sp in shaking)
    total_smooth = sum(d + sp for *_, d, sp in shaking)
    damping_share = sum(d for *_, d, sp in shaking) / max(total_smooth, 1e-9) if shaking else 0.0
    shake_hz = float(np.median([f for _, _, f, *_ in shaking])) if shaking else 0.0
    if shaking:
        print(f"   of the shaking torque, the command sent accounts for {total_sent / total_measured:.0%}; "
              f"of what was sent, {total_steps / max(total_sent, 1e-9):.0%} is 8-bit steps")

    # Other things worth knowing.
    clamped = (np.linalg.norm(force, axis=1) >= a.get("max_force", 20.0) - 1e-3) & live
    if clamped.any():
        print(f"\n   force clamp ({a.get('max_force')} N) active {clamped.mean():.1%} of the time")
    err = np.linalg.norm(np.array([r["target"] for r in rows]) - np.array([r["pos"] for r in rows]), axis=1)
    if live.any():
        print(f"   distance from target: median {np.median(err[live]) * 1000:.1f} mm, "
              f"worst {err[live].max() * 1000:.1f} mm")

    print("\nverdict")
    findings = []
    for j, gain in scale_wrong:
        findings.append(f"torque gain: J{j + 1} executes {gain:.1f}x the MIT command, but --tau-scale "
                        f"{tau_scale[j]:.2f} assumes {1 / tau_scale[j]:.0f}x -- it gets "
                        f"{gain * tau_scale[j]:.0%} of the torque asked for (gravity, stiffness and "
                        f"damping alike). Use --tau-scale {1 / gain:.2f} for it.")
    q0 = np.array(meta.get("q_start", q_mid))
    J0 = model.jacobian(q0, tcp)
    joint_damping = np.diag(J0.T @ np.diag([a.get("kd_lin", 25.0)] * 3 + [a.get("kd_rot", 0.4)] * 3) @ J0)
    if shaking and total_sent < 0.25 * total_measured:
        findings.append(
            f"not the command: only {total_sent / total_measured:.0%} of the shaking torque was in what "
            "was sent. The rest is made on the arm side -- the driver's own kd acting on its speed "
            "estimate, gearbox friction letting go in steps, or the tool rattling. Compare runs with "
            "different --joint-kd, and feel whether it follows the hand or the arm.")
        if total_steps > 0.5 * max(total_sent, 1e-9):
            findings.append("most of what the command did carry was 8-bit t_ref steps: try --fine-torque.")
    elif shaking and total_steps > max(total_smooth, 1e-9):
        findings.append(
            f"torque steps: the command's shake is mostly its own 8-bit staircase "
            f"({total_steps / total_sent:.0%} of it), not the controller's terms. Use --fine-torque.")
    elif shaking and damping_share > 0.6:
        findings.append(
            f"delayed damping: the shake runs at ~{shake_hz:.0f} Hz and {damping_share:.0%} of it is the "
            f"Cartesian damping term. That damping reaches the joints as {np.round(joint_damping, 2)} "
            f"Nms/rad, computed from speed that is ~{1000 / (4 * max(shake_hz, 1)):.0f} ms old by the time "
            f"the torque lands (a damper that late oscillates at 1/(4 x delay)). The driver's own kd "
            f"({a.get('joint_kd')}) has no such delay. Move the damping into the driver: lower --kd-lin / "
            f"--kd-rot, raise --joint-kd.")
    elif shaking:
        findings.append(f"stiffness: the shake (~{shake_hz:.0f} Hz) is mostly the spring term "
                        f"({1 - damping_share:.0%}) -- lower K with [.")
    if lifted:
        joints = ", ".join(f"J{j + 1}" for j, _, _, _ in lifted)
        findings.append(f"torque quantisation (secondary): small jolts on {joints} follow t_ref code steps "
                        f"more often than chance; the spring moves in {min(dead_band.values()) * 1000:.0f}+ mm "
                        f"treads at K={K:.0f}.")
    if timing_bad:
        findings.append("loop timing: late cycles or stale feedback are common enough to matter.")
    if damping_noisy:
        findings.append("noisy damping: the Cartesian damping term flips torque codes on speed noise alone.")
    if not total and not shaking:
        findings.append("no twitches detected in this trace -- was it one of the runs that shook?")
    for finding in findings or ["nothing stands out; send the trace over for a closer look."]:
        print(f"  - {finding}")


if __name__ == "__main__":
    main()
