#!/usr/bin/env python3
"""Judge whether a recorded episode is worth training on.

    python3 check_episode.py ~/piper_data/cubes              # every episode in there
    python3 check_episode.py ~/piper_data/cubes/episode_000  # just one

Behaviour cloning copies whatever is in the data, including the parts where the operator
was thinking. So the things worth measuring before training are: did the demonstration
actually happen (grasp, transport, release), how much of it is the arm standing still, and
did any joint spend the episode pinned against a limit.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import piper_common as pc
from gravity_model import TOOL_URDF, GravityModel

IDLE_DEG_S = 1.0        # every joint slower than this counts as standing still
IDLE_BUDGET = 0.30      # more idle than this and the episode drags
LIMIT_MARGIN_DEG = 3.0
# The patch between the fingers at the moment they close: whatever is in there is what was
# picked up. Fractions of the frame, measured off this gripper's own camera.
GRIP_PATCH = (0.68, 0.82, 0.44, 0.56)
# Measured on this gripper's camera: the red cube leads the other channels by 20 to 72,
# the white cube trails them by 12 to 44. Anything in between separates them, so sit in the
# middle rather than near either edge -- warm desk light pushes the red cube's lead down.
RED_MARGIN = 8


def grasped_colour(video, frame_index):
    """'red', 'white' or None -- what was between the fingers when they closed.

    A task label that does not match what was actually picked up is worse than a missing
    episode: it teaches the policy to fetch the wrong object when it hears the instruction.
    """
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    height, width = frame.shape[:2]
    y0, y1, x0, x1 = GRIP_PATCH
    patch = frame[int(height * y0):int(height * y1), int(width * x0):int(width * x1)]
    blue, green, red = patch.reshape(-1, 3).mean(axis=0)
    return "red" if red - max(green, blue) > RED_MARGIN else "white"


def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    rows = [json.loads(line) for line in (directory / "data.jsonl").read_text().splitlines()
            if line.strip()]
    return meta, rows


def runs(mask, t, minimum=0.0):
    """Contiguous True stretches of mask, as (start, end) times, longest first."""
    edges = np.diff(mask.astype(int))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask) - 1)
    found = [(t[a], t[b]) for a, b in zip(starts, ends) if t[b] - t[a] >= minimum]
    return sorted(found, key=lambda r: r[1] - r[0], reverse=True)


def check(directory, model, tcp, verbose=True):
    meta, rows = load(directory)
    name = Path(directory).name
    t = np.array([r["t"] for r in rows])
    if t[0] != 0:
        t = t - t[0]
    state = np.array([r["observation.state"] for r in rows])
    action = np.array([r["action"] for r in rows])
    q = np.degrees(state[:, :6])
    grip_cmd = action[:, 6] * 1000
    dt = float(np.diff(t).mean()) if len(t) > 1 else 1 / 30

    pos = np.array([model.tip_pose(row, tcp)[0] for row in state[:, :6]])
    speed = np.r_[0, np.abs(np.diff(q, axis=0)).max(axis=1)] / dt
    idle = speed < IDLE_DEG_S

    # The demonstration itself: a close, then a later open, is a pick and a place.
    closed = grip_cmd < (meta.get("gripper_open_mm", 70) + meta.get("gripper_close_mm", 20)) / 2
    grasped = bool(closed.any() and (~closed[np.argmax(closed):]).any())
    i_close = int(np.argmax(closed)) if closed.any() else None
    i_open = (len(closed) - 1 - int(np.argmax(closed[::-1]))) if closed.any() else None

    limits = np.degrees(pc.JOINT_LIMITS)
    pinned = [(j, ((q[:, j] <= limits[j, 0] + LIMIT_MARGIN_DEG)
                   | (q[:, j] >= limits[j, 1] - LIMIT_MARGIN_DEG)).sum() * dt)
              for j in range(6)]
    pinned = [(j, secs) for j, secs in pinned if secs > 0.2]

    # Look a little after the close, so the fingers have actually come together.
    colour = (grasped_colour(Path(directory) / "wrist.mp4", min(i_close + 20, len(rows) - 1))
              if grasped and (Path(directory) / "wrist.mp4").exists() else None)
    labelled = ("red" if "red" in (meta.get("task") or "") else
                "white" if "white" in (meta.get("task") or "") else None)

    tracking = np.degrees(np.abs(action[:, :6] - state[:, :6])).max(axis=1)
    problems = []
    if colour and labelled and colour != labelled:
        problems.append(f"the task says {labelled} but the {colour} cube was picked up "
                        "-- fix the label or drop this one")
    if not grasped:
        problems.append("no complete grasp-and-release in the gripper channel")
    if idle.mean() > IDLE_BUDGET:
        problems.append(f"{idle.mean():.0%} of the episode is the arm standing still")
    for j, secs in pinned:
        problems.append(f"J{j + 1} sat within {LIMIT_MARGIN_DEG:.0f} deg of its limit for {secs:.1f} s")
    if tracking.max() > 10:
        problems.append(f"the arm fell {tracking.max():.0f} deg behind the command at worst")

    if verbose:
        print(f"\n=== {name} ===")
        print(f"task      {meta.get('task')}")
        print(f"length    {t[-1]:.1f} s, {len(rows)} frames at {1 / dt:.0f} Hz")
        print(f"gripper   " + (f"closed at {t[i_close]:.1f} s (z={pos[i_close, 2]:.3f} m), "
                               f"opened at {t[i_open]:.1f} s (z={pos[i_open, 2]:.3f} m)"
                               if grasped else "NO complete grasp-and-release"))
        print(f"tool      z {pos[:, 2].min():.3f}..{pos[:, 2].max():.3f} m,   "
              f"radius {np.hypot(pos[:, 0], pos[:, 1]).min():.3f}.."
              f"{np.hypot(pos[:, 0], pos[:, 1]).max():.3f} m,   "
              f"path {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.2f} m")
        if colour:
            print(f"picked    the {colour} cube"
                  + ("" if colour == labelled else f"   BUT THE TASK SAYS {labelled}"))
        print(f"idle      {idle.mean():.0%} of frames ({idle.sum() * dt:.1f} s)")
        for start, end in runs(idle, t, minimum=1.0)[:3]:
            print(f"            {start:5.1f} - {end:5.1f} s  ({end - start:.1f} s still)")
        print(f"tracking  command vs measured: mean {tracking.mean():.2f} deg, "
              f"worst {tracking.max():.2f} deg")
        print("joints    " + "  ".join(
            f"J{j + 1} {q[:, j].min():+.0f}/{q[:, j].max():+.0f}" for j in range(6)))
        for j, secs in pinned:
            print(f"            J{j + 1} pinned against its limit for {secs:.1f} s "
                  f"(range {limits[j, 0]:+.0f}..{limits[j, 1]:+.0f})")
        mislabelled = bool(colour and labelled and colour != labelled)
        print("verdict   " + ("USABLE" if not problems else
                              "WRONG LABEL" if mislabelled else
                              "USABLE, with caveats" if grasped else "DROP THIS ONE"))
        for problem in problems:
            print(f"            - {problem}")
    return {"name": name, "task": meta.get("task"), "seconds": float(t[-1]),
            "idle": float(idle.mean()), "grasped": grasped, "problems": problems,
            # Where the gripper closed is where the object was, and where it opened is where
            # the box was: the only record of how the table was actually laid out.
            "colour": colour, "labelled": labelled,
            "pick": pos[i_close].tolist() if grasped else None,
            "place": pos[i_open].tolist() if grasped else None,
            "layout": meta.get("layout")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an episode directory, or the directory holding them")
    ap.add_argument("--tool", choices=["pika", "piper"], default="pika")
    ap.add_argument("--tcp", type=float, nargs=3, default=[0.13, 0.0, 0.0])
    ap.add_argument("--quiet", action="store_true", help="only the summary table")
    args = ap.parse_args()

    root = Path(args.path)
    # meta.json is only written when an episode closes, so a directory without one is the
    # episode being recorded right now -- skip it instead of crashing mid-session.
    episodes = ([root] if (root / "data.jsonl").exists()
                else sorted(p for p in root.glob("episode_*")
                            if (p / "data.jsonl").exists() and (p / "meta.json").exists()))
    if not episodes:
        raise SystemExit(f"no finished episodes in {root}")
    recording = [p.name for p in root.glob("episode_*")
                 if (p / "data.jsonl").exists() and not (p / "meta.json").exists()]
    if recording:
        print(f"(skipping {', '.join(recording)}: still being recorded)")

    model = GravityModel(TOOL_URDF[args.tool])
    tcp = model.tool_to_link(args.tcp)
    results = [check(directory, model, tcp, verbose=not args.quiet) for directory in episodes]

    print(f"\n{len(results)} episodes, {sum(r['seconds'] for r in results) / 60:.1f} min total")
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    for task, group in by_task.items():
        print(f"  {len(group):3d}  {task}")
    wrong = [r for r in results if r["colour"] and r["labelled"]
             and r["colour"] != r["labelled"]]
    if wrong:
        print("  WRONG LABEL: " + ", ".join(f"{r['name']} (says {r['labelled']}, "
                                            f"picked {r['colour']})" for r in wrong))
    bad = [r for r in results if not r["grasped"]]
    caveat = [r for r in results if r["grasped"] and r["problems"]]
    print(f"  clean {len(results) - len(bad) - len(caveat)},  with caveats {len(caveat)},"
          f"  drop {len(bad)}")
    if bad:
        print("  drop: " + ", ".join(r["name"] for r in bad))
    picks = np.array([r["pick"] for r in results if r["pick"]])
    places = np.array([r["place"] for r in results if r["place"]])
    if len(picks) >= 2:
        print("\ncoverage -- a policy only generalises over variation it was shown")
        for label, points in (("pick ", picks), ("place", places)):
            if not len(points):
                continue
            span = points.max(axis=0) - points.min(axis=0)
            print(f"  {label}  x {points[:, 0].min():+.3f}..{points[:, 0].max():+.3f}"
                  f"  y {points[:, 1].min():+.3f}..{points[:, 1].max():+.3f}"
                  f"  z {points[:, 2].min():+.3f}..{points[:, 2].max():+.3f} m"
                  f"   (spread {span[0] * 100:.0f} x {span[1] * 100:.0f} cm)")
        spread = (picks.max(axis=0) - picks.min(axis=0))[:2].max()
        if spread < 0.08:
            print("  the pick point barely moves between episodes: the policy will learn one "
                  "trajectory and fail as soon as you move the cube")
        # A coarse map of where on the table the grasps happened.
        print("\n  grasp positions seen so far, looking down on the table:")
        xs, ys = picks[:, 0], picks[:, 1]
        pad = 0.03
        x0, x1 = xs.min() - pad, xs.max() + pad
        y0, y1 = ys.min() - pad, ys.max() + pad
        grid = [[" "] * 21 for _ in range(9)]
        for x, y in zip(xs, ys):
            col = int((x - x0) / max(x1 - x0, 1e-6) * 20)
            row = int((y - y0) / max(y1 - y0, 1e-6) * 8)
            grid[row][col] = "#" if grid[row][col] == " " else "*"
        for row in reversed(grid):
            print("    |" + "".join(row) + "|")
        print(f"    x {x0:+.2f} .. {x1:+.2f} m,  y {y0:+.2f} .. {y1:+.2f} m")

    mean_idle = float(np.mean([r["idle"] for r in results]))
    print(f"  average idle {mean_idle:.0%}"
          + ("  <- worth filtering at conversion time" if mean_idle > IDLE_BUDGET else ""))


if __name__ == "__main__":
    sys.exit(main())
