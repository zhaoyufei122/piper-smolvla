#!/usr/bin/env python3
"""Find out what the arm is shaking at, and whether the command told it to.

    python3 run_policy.py ... --trace /tmp/run.jsonl --trace-every 1
    python3 analyse_shake.py /tmp/run.jsonl

Position alone cannot answer this. A stiff joint fighting a noisy target barely moves --
tenths of a millimetre -- while its motor swings several newton-metres doing it, so torque
is where vibration shows up first. And the question that decides what to fix is not "how
much does it shake" but "at which frequencies, and is the command shaking there too":

    the command has energy at that frequency   the policy is asking for it
                                               -> filter the action, collect better data
    only the measurement does                  the servo or the structure is ringing
                                               -> soften the loop: lower speed percent,
                                                  cap joint acceleration, or drive the arm
                                                  in MIT mode with a low kp

Sampling rate is everything here. A trace written every third cycle of a 100 Hz loop sees
33 Hz, which folds anything above 16 Hz back down into the range it can represent and
reports it at the wrong frequency. Run the trace at --trace-every 1.
"""
import argparse
import json
from pathlib import Path

import numpy as np


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


def spectrum(signal, rate):
    """Amplitude per frequency of the mean-removed signal, ignoring DC."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - signal.mean()
    window = np.hanning(len(signal))
    amplitude = np.abs(np.fft.rfft(signal * window)) * 2 / (window.sum() or 1)
    frequency = np.fft.rfftfreq(len(signal), 1.0 / rate)
    return frequency[1:], amplitude[1:]


def peak(frequency, amplitude, above=2.0):
    """The strongest component above `above` Hz: the slow part is the task, not the shake."""
    keep = frequency >= above
    if not keep.any():
        return 0.0, 0.0
    index = int(np.argmax(amplitude[keep]))
    return float(frequency[keep][index]), float(amplitude[keep][index])


def band_energy(frequency, amplitude, low, high):
    keep = (frequency >= low) & (frequency < high)
    return float(np.sqrt((amplitude[keep] ** 2).sum())) if keep.any() else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--from-seconds", type=float, default=0.0)
    ap.add_argument("--to-seconds", type=float)
    ap.add_argument("--above", type=float, default=2.0, help="ignore slower than this [Hz]")
    args = ap.parse_args()

    meta, rows = load(args.trace)
    rows = [r for r in rows if r.get("running")]
    if len(rows) < 64:
        raise SystemExit("not enough running samples to analyse")
    t = np.array([r["t"] for r in rows])
    keep = t >= args.from_seconds
    if args.to_seconds:
        keep &= t <= args.to_seconds
    rows = [r for r, k in zip(rows, keep) if k]
    t = t[keep]
    rate = 1.0 / float(np.median(np.diff(t)))

    print(f"{Path(args.trace).name}: {len(rows)} running samples, {t[-1] - t[0]:.0f} s, "
          f"sampled at {rate:.0f} Hz (resolves up to {rate / 2:.0f} Hz)")
    if rate < 60:
        print("  WARNING: below 60 Hz this cannot see vibration. Re-run with --trace-every 1.")
    print(f"  policy {meta.get('policy_rate')} Hz   smooth {meta.get('smooth')}   "
          f"speed {meta.get('speed_percent')}%")

    measured = np.degrees([r["q"] for r in rows])
    commanded = np.degrees([r["q_cmd"] for r in rows])
    has_tau = rows[0].get("tau") is not None
    torque = np.array([r["tau"] for r in rows]) if has_tau else None
    velocity = np.degrees([r["qd"] for r in rows]) if rows[0].get("qd") is not None else None

    print(f"\n{'':4s} {'commanded':>22s} {'measured':>22s} {'torque':>22s}")
    print(f"{'':4s} {'peak Hz':>10s}{'amp':>12s} {'peak Hz':>10s}{'amp':>12s} "
          f"{'peak Hz':>10s}{'amp':>12s}")
    verdicts = []
    for j in range(6):
        fc, ac = spectrum(commanded[:, j], rate)
        fm, am = spectrum(measured[:, j], rate)
        pc_f, pc_a = peak(fc, ac, args.above)
        pm_f, pm_a = peak(fm, am, args.above)
        line = (f"J{j + 1:<3d} {pc_f:9.1f}{pc_a:11.4f}° {pm_f:9.1f}{pm_a:11.4f}°")
        if has_tau:
            ft, at = spectrum(torque[:, j], rate)
            pt_f, pt_a = peak(ft, at, args.above)
            line += f" {pt_f:9.1f}{pt_a:11.3f}Nm"
        print(line)
        # Energy the measurement has and the command does not: that is the arm's own.
        high = max(args.above, 8.0)
        own = band_energy(fm, am, high, rate / 2) - band_energy(fc, ac, high, rate / 2)
        verdicts.append((j, own, pm_f, pm_a))

    print(f"\nabove 8 Hz, how much the measurement has that the command does not [deg]")
    for j, own, pm_f, pm_a in verdicts:
        tag = ("the arm is ringing on its own" if own > 0.01 else
               "follows the command, nothing added")
        print(f"  J{j + 1}: {own:+.4f}   {tag}")

    if has_tau:
        print(f"\ntorque swing above {args.above:.0f} Hz [Nm peak-to-peak]")
        for j in range(6):
            f, a = spectrum(torque[:, j], rate)
            print(f"  J{j + 1}: {2 * band_energy(f, a, args.above, rate / 2):6.2f}"
                  f"   (slow part {2 * band_energy(f, a, 0, args.above):6.2f})")
        print("  A joint whose fast torque swing rivals its slow one is fighting itself.")

    if velocity is not None:
        print(f"\nmeasured joint speed [deg/s]")
        for j in range(6):
            print(f"  J{j + 1}: rms {np.sqrt((velocity[:, j] ** 2).mean()):6.2f}   "
                  f"max {np.abs(velocity[:, j]).max():6.2f}")


if __name__ == "__main__":
    main()
