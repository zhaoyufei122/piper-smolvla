#!/usr/bin/env python3
"""Convert the SpaceMouse episodes into a LeRobot dataset for SmolVLA fine-tuning.

    python3 to_lerobot.py ~/piper_data/cubes --repo-id yufei/piper_cubes \
        --root /root/autodl-tmp/piper/lerobot

Works with both dataset APIs in the wild, because the two differ in where the task
string goes and the version you get depends on the Python on the machine (lerobot >= 0.5
needs Python 3.12, so a 3.11 box gets 0.4.x):
    0.4.x   dataset.add_frame(frame, task=...)     dataset.save_episode()
    0.6.x   dataset.add_frame(frame)               dataset.save_episode()
            ... with "task" as a key inside frame
The signature is inspected at import time rather than guessed at.

About the pauses. Roughly a quarter of every episode is the operator thinking, with the arm
standing still, and behaviour cloning copies that: a policy trained on it learns to stop
mid-reach. Dropping those frames is safe in a way that dropping moving frames is not --
the state does not change while the arm is still, so removing some of it leaves the
state/action sequence continuous and only changes the timing. Leading and trailing stillness
goes entirely; an interior pause is capped at --max-pause seconds so genuine "settle before
grasping" moments survive.
"""
import argparse
import inspect
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ADD_FRAME_TAKES_TASK = "task" in inspect.signature(LeRobotDataset.add_frame).parameters

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
CHANNEL_NAMES = JOINT_NAMES + ["gripper"]
IDLE_DEG_S = 1.0        # every joint slower than this counts as standing still


def keep_mask(idle, max_pause_frames):
    """Which frames to keep: no leading/trailing stillness, interior pauses capped."""
    keep = np.ones(len(idle), dtype=bool)
    first = int(np.argmax(~idle)) if (~idle).any() else 0
    last = len(idle) - 1 - int(np.argmax(~idle[::-1])) if (~idle).any() else len(idle) - 1
    keep[:first] = False
    keep[last + 1:] = False
    run = 0
    for i in range(first, last + 1):
        run = run + 1 if idle[i] else 0
        if run > max_pause_frames:
            keep[i] = False
    return keep


def load_episode(directory):
    meta = json.loads((directory / "meta.json").read_text())
    rows = [json.loads(line) for line in (directory / "data.jsonl").read_text().splitlines()
            if line.strip()]
    state = np.array([r["observation.state"] for r in rows], dtype=np.float32)
    action = np.array([r["action"] for r in rows], dtype=np.float32)
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    t = t - t[0]
    return meta, state, action, t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="directory holding episode_*/")
    ap.add_argument("--repo-id", default="local/piper_cubes")
    ap.add_argument("--root", required=True, help="where to write the dataset")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-pause", type=float, default=0.5,
                    help="longest interior pause to keep [s]; 0 removes every still frame, "
                         "a large value keeps the episodes exactly as recorded")
    ap.add_argument("--camera", default="wrist")
    ap.add_argument("--robot-type", default="piper")
    ap.add_argument("--dry-run", action="store_true", help="report what would be kept, write nothing")
    args = ap.parse_args()

    source = Path(args.source)
    episodes = sorted(p for p in source.glob("episode_*")
                      if (p / "data.jsonl").exists() and (p / "meta.json").exists())
    if not episodes:
        raise SystemExit(f"no finished episodes in {source}")

    video_key = f"observation.images.{args.camera}"
    probe = cv2.VideoCapture(str(episodes[0] / f"{args.camera}.mp4"))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    print(f"{len(episodes)} episodes, video {width}x{height} @ {args.fps} Hz")
    print(f"lerobot add_frame takes an explicit task: {ADD_FRAME_TAKES_TASK}")

    features = {
        video_key: {"dtype": "video", "shape": (height, width, 3),
                    "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": CHANNEL_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": CHANNEL_NAMES},
    }

    dataset = None
    if not args.dry_run:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=args.fps, features=features,
            root=args.root, robot_type=args.robot_type, use_videos=True,
            image_writer_processes=0, image_writer_threads=4,
        )

    max_pause_frames = max(int(round(args.max_pause * args.fps)), 0)
    tasks, total_in, total_out = {}, 0, 0
    for directory in episodes:
        meta, state, action, t = load_episode(directory)
        dt = float(np.diff(t).mean()) if len(t) > 1 else 1.0 / args.fps
        speed = np.r_[0, np.abs(np.diff(np.degrees(state[:, :6]), axis=0)).max(axis=1)] / dt
        keep = keep_mask(speed < IDLE_DEG_S, max_pause_frames)

        capture = cv2.VideoCapture(str(directory / f"{args.camera}.mp4"))
        written = 0
        for index in range(len(state)):
            ok, frame = capture.read()      # sequential: seeking per frame is far slower
            if not ok:
                break
            if not keep[index]:
                continue
            if dataset is not None:
                sample = {
                    video_key: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    "observation.state": state[index],
                    "action": action[index],
                }
                if ADD_FRAME_TAKES_TASK:
                    dataset.add_frame(sample, task=meta["task"])
                else:
                    dataset.add_frame({**sample, "task": meta["task"]})
            written += 1
        capture.release()
        if dataset is not None:
            dataset.save_episode()

        tasks[meta["task"]] = tasks.get(meta["task"], 0) + 1
        total_in += len(state)
        total_out += written
        print(f"  {directory.name}  {len(state):4d} -> {written:4d} frames "
              f"({written / max(len(state), 1):4.0%})  | {meta['task']}")

    print(f"\n{total_in} frames in, {total_out} out "
          f"({1 - total_out / max(total_in, 1):.0%} of the stillness removed)")
    for task, count in tasks.items():
        print(f"  {count:3d}  {task}")
    if dataset is not None:
        print(f"\nwritten to {args.root}")
        print(f"train with:\n"
              f"  lerobot-train --policy.path=lerobot/smolvla_base \\\n"
              f"      --dataset.repo_id={args.repo_id} --dataset.root={args.root} \\\n"
              f"      --batch_size=64 --steps=20000 --policy.device=cuda")


if __name__ == "__main__":
    main()
