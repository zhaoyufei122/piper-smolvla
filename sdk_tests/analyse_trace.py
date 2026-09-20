#!/usr/bin/env python3
"""Work out why the arm did not go where it was pushed, from a --trace file.

    python3 teleop_spacemouse.py --trace /tmp/run.jsonl
    python3 analyse_trace.py /tmp/run.jsonl

The episode files say what the arm did. This says what the controller was thinking, which
is what you need when the answer is "it refused": every time the stick asked for motion and
the tool did not deliver it, the cause is one of a handful of things, and they are all
recorded, so they can be attributed instead of guessed at.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

AXES = "xyz"
# A push that produced less than this fraction of the speed it asked for did not take effect.
DELIVERED = 0.35


def load(path):
    meta, rows = {}, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "meta" in record:
            meta = record["meta"]
        else:
            rows.append(record)
    if not rows:
        raise SystemExit(f"{path} has no samples")
    return meta, rows


def blame(row, meta, axis):
    """Why this axis did not move, in the order the control loop would have stopped it."""
    if not row["jogging"]:
        return "paused (HOLD)"
    if row["auto"]:
        return "flying to a saved pose"
    if row.get("radius_block") and axis in (0, 1):
        return (f"too close to the base ({row.get('radius', 0) * 100:.0f} cm): "
                "the arm folds back on itself in there")
    if row["refused"][axis]:
        low, high = meta.get("box", [[0, 0]] * 3)[axis]
        edge = "lower" if row["target"][axis] <= (low + high) / 2 else "upper"
        return f"workspace box, {edge} {AXES[axis]} limit"
    if row["lag_mm"] > 0.8 * row["leash_mm"]:
        return "out of reach: the arm is already at the leash"
    if row["ori"] < 0.5 * meta.get("ori_weight", 0.4):
        return "orientation being given up (arm at its limit)"
    at_limit = joints_at_limit(row, meta)
    if at_limit:
        return f"joint {' '.join(at_limit)} at its limit"
    return "moved less than asked but nothing refused it (check gear / speed)"


def joints_at_limit(row, meta, margin_deg=3.0):
    limits = np.array(meta.get("joint_limits_deg", []))
    if limits.size == 0:
        return []
    q = np.degrees(row["q"])
    hit = (q <= limits[:, 0] + margin_deg) | (q >= limits[:, 1] - margin_deg)
    return [f"J{i + 1}" for i in np.nonzero(hit)[0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--axis", choices=list(AXES), help="only report on one axis")
    ap.add_argument("--episodes", type=int, default=6, help="how many incidents to list")
    args = ap.parse_args()

    meta, rows = load(args.trace)
    duration = rows[-1]["t"] - rows[0]["t"]
    dt = duration / max(len(rows) - 1, 1)
    speed = meta.get("lin_speed", 0.12)

    print(f"{Path(args.trace).name}: {len(rows)} samples, {duration:.1f} s, "
          f"{1 / dt:.0f} Hz")
    if meta:
        bx, by, bz = meta["box"]
        print(f"box  x{bx}  y{by}  z{bz} m   leash {meta['leash_m'] * 1000:.0f} mm   "
              f"lin-speed {speed * 100:.0f} cm/s")
    jog = sum(r["jogging"] and not r["auto"] for r in rows)
    print(f"jogging {jog * dt:.1f} s of {duration:.1f} s")

    # How much of what was asked for actually came out, axis by axis.
    print("\nasked for vs delivered, per axis")
    incidents = {}
    for axis in range(3):
        if args.axis and AXES[axis] != args.axis:
            continue
        asked = delivered = 0.0
        stuck, causes = [], Counter()
        for previous, row in zip(rows, rows[1:]):
            want = row["cmd_twist"][axis]
            if abs(want) < 0.05:
                continue
            moved = (row["tcp"][axis] - previous["tcp"][axis]) / max(dt, 1e-6)
            asked += abs(want) * speed * dt
            delivered += abs(moved) * dt if np.sign(moved) == np.sign(want) else 0.0
            if abs(moved) < DELIVERED * abs(want) * speed:
                reason = blame(row, meta, axis)
                causes[reason] += 1
                stuck.append((row["t"], reason, row))
        if asked < 1e-4:
            print(f"  {AXES[axis]}: never pushed")
            continue
        print(f"  {AXES[axis]}: asked {asked * 100:6.1f} cm, delivered {delivered * 100:6.1f} cm"
              f"  ({delivered / asked:5.0%})")
        for reason, count in causes.most_common():
            print(f"        {count * dt:5.1f} s  {reason}")
        incidents[AXES[axis]] = stuck

    # The individual moments, so a specific "it would not go down" can be looked at.
    for axis, stuck in incidents.items():
        if not stuck:
            continue
        print(f"\nwhen {axis} would not move ({len(stuck) * dt:.1f} s total, "
              f"first {args.episodes} runs)")
        runs, last_t = [], -9
        for t, reason, row in stuck:
            if t - last_t > 0.3 or not runs or runs[-1][2] != reason:
                runs.append([t, t, reason, row])
            else:
                runs[-1][1] = t
            last_t = t
        for start, end, reason, row in sorted(runs, key=lambda r: r[1] - r[0],
                                              reverse=True)[:args.episodes]:
            index = AXES.index(axis)
            print(f"  t={start:6.1f}-{end:5.1f}s  {reason}")
            print(f"        tcp {np.round(row['tcp'], 3)} m   target {np.round(row['target'], 3)} m"
                  f"   lag {row['lag_mm']:.0f}/{row['leash_mm']:.0f} mm")
            print(f"        joints {np.degrees(row['q']).round(0)} deg"
                  f"   ori {row['ori']:.2f}   gear {row['gear']}   frame {row['frame']}")
            if meta.get("box") and row["refused"][index]:
                low, high = meta["box"][index]
                print(f"        the {axis} box is [{low}, {high}] m and the target sits at "
                      f"{row['target'][index]:+.3f} -- raise it with "
                      f"--box-{axis} {min(low, row['target'][index] - 0.1):.2f} "
                      f"{max(high, row['target'][index] + 0.1):.2f}")

    # Joints that spent time parked on a limit are the usual root cause behind everything.
    limits = np.array(meta.get("joint_limits_deg", []))
    if limits.size:
        q = np.degrees([r["q"] for r in rows])
        print("\ntime spent within 3 deg of a joint limit")
        any_hit = False
        for j in range(6):
            hit = (q[:, j] <= limits[j, 0] + 3) | (q[:, j] >= limits[j, 1] - 3)
            if hit.any():
                any_hit = True
                print(f"  J{j + 1} ({limits[j, 0]:+.0f} to {limits[j, 1]:+.0f} deg): "
                      f"{hit.sum() * dt:5.1f} s   range used "
                      f"{q[:, j].min():+.0f} to {q[:, j].max():+.0f}")
        if not any_hit:
            print("  none -- no joint ever got within 3 deg of a limit")

    # The configuration trap: joint1 and joint5 both running out at a small radius.
    radii = np.hypot([r["tcp"][0] for r in rows], [r["tcp"][1] for r in rows])
    q_all = np.degrees([r["q"] for r in rows])
    close = radii < meta.get("min_radius", 0.25)
    if close.any():
        print(f"\ntime spent within {meta.get('min_radius', 0.25) * 100:.0f} cm of the base: "
              f"{close.sum() * dt:.1f} s")
        print(f"  joint1 there: {q_all[close, 0].min():+.0f} to {q_all[close, 0].max():+.0f} deg"
              f"   |joint5| up to {np.abs(q_all[close, 4]).max():.0f} deg")
        print(f"  joint1 elsewhere: {q_all[~close, 0].min():+.0f} to "
              f"{q_all[~close, 0].max():+.0f} deg" if (~close).any() else "")

    # Torque residual: how close the collision detector came to firing, and on which joint.
    residuals = np.array([r["residual"] for r in rows if r.get("residual")])
    if residuals.size:
        peak = np.abs(residuals).max(axis=0)
        print("\npeak torque residual per joint [Nm] (the collision check fires above "
              f"{meta.get('collision_torque', 5.0)}):")
        print("  " + "  ".join(f"J{i + 1} {v:4.1f}" for i, v in enumerate(peak)))
        worst = int(np.argmax(peak))
        print(f"  highest on J{worst + 1}; if that is J2/J3 it is usually the gripper "
              "leaning on the table, not a collision")

    lows = [r["tcp"][2] for r in rows]
    print(f"\nlowest the tool reached: z = {min(lows):.3f} m"
          + (f"   (box floor is {meta['box'][2][0]} m)" if meta.get("box") else ""))


if __name__ == "__main__":
    main()
