#!/usr/bin/env python3
"""Record a demonstration by hand, then replay it to capture clean training data.

Kinesthetic teaching puts your hand in the camera view, and that hand does not exist at
inference time. So a demonstration is collected in two passes:

  record  drag the arm through the task (with gravity_compensation.py running in another
          terminal) while this script only *reads* the joint stream. It sends nothing to
          the arm, so it cannot fight the other process.
  replay  put the scene back, let the arm execute the recorded trajectory on its own, and
          record the cameras together with the (state, action) pairs.

The replay pass is what becomes the dataset: no hand in frame, and the actions are exactly
the commands the arm followed. See SMOLVLA.md for why, and for the limits of this trick.

  # terminal A                        # terminal B
  python3 gravity_compensation.py     python3 teach_replay.py record vla_data/ep_0001 \
                                          --task "pick up the red cube" --gripper-keys

  python3 teach_replay.py replay vla_data/ep_0001 --camera 0
  python3 teach_replay.py info   vla_data/ep_0001

Units in the saved files: joint positions in rad, gripper width in m, time in s.
"""
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import piper_common as pc
from gravity_compensation import Rate
from joint_position_ctrl import move_to


class GripperKeys:
    """Single-key gripper control during teaching (the gripper is not back-drivable)."""

    def __init__(self, piper, effort=1.0, step=0.005):
        self.piper = piper
        self.effort = effort
        self.step = step
        self.target = pc.gripper_state(piper)[0]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import select
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            saved = termios.tcgetattr(fd)
        except termios.error:
            print("no terminal available, gripper keys disabled")
            return
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = sys.stdin.read(1)
                if key == "o":
                    self.target = pc.GRIPPER_MAX_WIDTH
                elif key == "c":
                    self.target = 0.0
                elif key == "+":
                    self.target = min(self.target + self.step, pc.GRIPPER_MAX_WIDTH)
                elif key == "-":
                    self.target = max(self.target - self.step, 0.0)
                else:
                    continue
                pc.send_gripper(self.piper, self.target, self.effort)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=0.5)


class CameraRecorder:
    """Grabs frames in a thread and writes an mp4, so the control loop never blocks on a read."""

    def __init__(self, index, path, fps=30.0, size=(640, 480)):
        import cv2

        self._cv2 = cv2
        self.index = index
        self.path = Path(path)
        self.frames = 0
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise SystemExit(f"cannot open camera {index} (try: ls /dev/video*)")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        actual = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, actual)
        self.size = actual
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok:
                self._writer.write(frame)
                self.frames += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._writer.release()
        self._capture.release()


def cmd_record(args):
    episode = Path(args.episode)
    episode.mkdir(parents=True, exist_ok=True)
    piper = pc.connect(args.can)
    print(f"task: {args.task}")
    print(f"current [deg]: {pc.fmt(np.degrees(pc.joint_positions(piper)))}")
    print("This pass only reads the CAN bus." if not args.gripper_keys else
          "This pass only reads the CAN bus, except for gripper keys: o open, c close, +/- 5 mm.")
    print("Start gravity_compensation.py in another terminal first, then drag the arm.")
    pc.confirm("Ready to record; Ctrl+C stops the recording.", args.yes)

    gripper = GripperKeys(piper) if args.gripper_keys else None
    samples = []
    rate = Rate(args.rate)
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            q = pc.joint_positions(piper)
            width, _ = pc.gripper_state(piper)
            samples.append({
                "t": round(t, 4),
                "q": [round(float(v), 5) for v in q],
                "gripper": round(float(width), 5),
                "gripper_cmd": None if gripper is None else round(float(gripper.target), 5),
            })
            if len(samples) % max(int(args.rate // 2), 1) == 0:
                print(f"\r{t:6.1f} s  {len(samples):5d} samples  q[deg] {pc.fmt(np.degrees(q))}",
                      end="", flush=True)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if gripper is not None:
            gripper.stop()
    print()

    if len(samples) < 2:
        raise SystemExit("nothing recorded")
    path = episode / "teach.json"
    path.write_text(json.dumps({
        "task": args.task,
        "rate": args.rate,
        "can": args.can,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "units": "q in rad, gripper width in m, t in s",
        "samples": samples,
    }, indent=1))
    print(f"saved {len(samples)} samples ({samples[-1]['t']:.1f} s) to {path}")
    print(f"next: put the scene back, then python3 teach_replay.py replay {episode} --camera 0")


def cmd_replay(args):
    episode = Path(args.episode)
    teach = json.loads((episode / "teach.json").read_text())
    samples = teach["samples"]
    times = np.array([s["t"] for s in samples]) * args.time_scale
    q_traj = np.array([s["q"] for s in samples])
    widths = np.array([s["gripper"] if s.get("gripper_cmd") is None else s["gripper_cmd"]
                       for s in samples])
    duration = float(times[-1])

    piper = pc.connect(args.can)
    print(f"task: {teach.get('task')}")
    print(f"{len(samples)} samples, {duration:.1f} s replay")
    print(f"start   [deg]: {pc.fmt(np.degrees(q_traj[0]))}")
    print(f"current [deg]: {pc.fmt(np.degrees(pc.joint_positions(piper)))}")
    print("Put the scene back to how it was at the start of the teach pass.")
    pc.confirm("The arm will enable, move to the start pose and replay the trajectory.", args.yes)
    if not pc.enable(piper):
        raise SystemExit("enable failed (after MIT mode run: python3 arm_tool.py reset)")

    move_to(piper, q_traj[0], np.radians(args.approach_speed), args.speed)
    pc.send_gripper(piper, float(widths[0]))

    cameras = []
    for i, index in enumerate(args.camera):
        name = args.camera_name[i] if i < len(args.camera_name) else f"cam{index}"
        cameras.append((name, CameraRecorder(index, episode / f"{name}.mp4", args.rate)))
    for _, camera in cameras:
        camera.start()

    rate = Rate(args.rate)
    worst_error = 0.0
    steps = 0
    t0 = time.perf_counter()
    try:
        with (episode / "replay.jsonl").open("w") as log:
            while True:
                t = time.perf_counter() - t0
                if t > duration:
                    break
                q_desired = np.array([np.interp(t, times, q_traj[:, j]) for j in range(6)])
                width_desired = float(np.interp(t, times, widths))
                state_q = pc.joint_positions(piper)
                state_width, _ = pc.gripper_state(piper)
                # observation_t is logged before action_t is applied
                log.write(json.dumps({
                    "t": round(t, 4),
                    "observation.state": [round(float(v), 5) for v in state_q] + [round(state_width, 5)],
                    "action": [round(float(v), 5) for v in q_desired] + [round(width_desired, 5)],
                    "frames": {name: camera.frames - 1 for name, camera in cameras},
                }) + "\n")
                pc.send_joint_positions(piper, q_desired, args.speed)
                pc.send_gripper(piper, width_desired)
                worst_error = max(worst_error, float(np.max(np.abs(state_q - q_desired))))
                steps += 1
                if steps % max(int(args.rate // 2), 1) == 0:
                    print(f"\r{t:6.1f} / {duration:.1f} s  tracking error {worst_error:.3f} rad",
                          end="", flush=True)
                rate.sleep()
    except KeyboardInterrupt:
        print("\ninterrupted, arm holds the last setpoint")
    finally:
        for _, camera in cameras:
            camera.stop()
    print()

    meta = {
        "task": teach.get("task"),
        "rate": args.rate,
        "time_scale": args.time_scale,
        "steps": steps,
        "duration": round(duration, 3),
        "max_tracking_error_rad": round(worst_error, 4),
        "replayed_at": datetime.now().isoformat(timespec="seconds"),
        "cameras": {name: {"file": camera.path.name, "frames": camera.frames,
                           "size": list(camera.size)} for name, camera in cameras},
        "state_layout": "joint1..joint6 [rad], gripper width [m]",
        "action_layout": "commanded joint1..joint6 [rad], commanded gripper width [m]",
    }
    (episode / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{steps} steps logged to {episode / 'replay.jsonl'}")
    for name, camera in cameras:
        print(f"  {name}: {camera.frames} frames -> {camera.path}")
    if worst_error > 0.1:
        print(f"WARNING: tracking error reached {worst_error:.3f} rad; the replay may not match "
              "the taught motion. Check the video before keeping this episode.")
    print("Keep the episode only if the replay actually did the task.")


def cmd_info(args):
    episode = Path(args.episode)
    teach = json.loads((episode / "teach.json").read_text())
    samples = teach["samples"]
    q = np.array([s["q"] for s in samples])
    print(f"task:     {teach.get('task')}")
    print(f"recorded: {teach.get('recorded_at')} at {teach.get('rate')} Hz")
    print(f"samples:  {len(samples)} over {samples[-1]['t']:.1f} s")
    print("joint range [deg]:")
    for i, name in enumerate(pc.JOINT_NAMES):
        print(f"  {name}: {np.degrees(q[:, i].min()):8.1f} .. {np.degrees(q[:, i].max()):8.1f}")
    gripper = np.array([s["gripper"] for s in samples])
    print(f"gripper [mm]: {gripper.min() * 1000:.1f} .. {gripper.max() * 1000:.1f}")
    meta_path = episode / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"replay:   {meta['steps']} steps, max tracking error "
              f"{meta['max_tracking_error_rad']} rad, cameras {list(meta['cameras'])}")
    else:
        print("replay:   not run yet")


def main():
    # Shared options sit on the subcommands so they can be given after the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--can", default="can0")
    common.add_argument("--rate", type=float, default=30.0, help="record / replay rate [Hz]")
    common.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", parents=[common],
                            help="log a hand-guided demonstration (reads only)")
    record.add_argument("episode", help="episode directory, e.g. vla_data/ep_0001")
    record.add_argument("--task", required=True, help="language instruction for this episode")
    record.add_argument("--gripper-keys", action="store_true",
                        help="control the gripper while teaching: o open, c close, +/- 5 mm")
    record.set_defaults(func=cmd_record)

    replay = sub.add_parser("replay", parents=[common],
                            help="re-execute the demonstration and record cameras")
    replay.add_argument("episode")
    replay.add_argument("--camera", type=int, action="append", default=[],
                        help="camera index, repeat for several (e.g. --camera 0 --camera 2)")
    replay.add_argument("--camera-name", action="append", default=[],
                        help="name per camera, in the same order (default cam<N>)")
    replay.add_argument("--time-scale", type=float, default=1.0,
                        help="1.0 = taught speed, 2.0 = half speed")
    replay.add_argument("--approach-speed", type=float, default=20.0,
                        help="speed for the move to the start pose [deg/s]")
    replay.add_argument("--speed", type=int, default=100, help="firmware MOVE J speed percent")
    replay.set_defaults(func=cmd_replay)

    info = sub.add_parser("info", parents=[common], help="summarise an episode")
    info.add_argument("episode")
    info.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted")
