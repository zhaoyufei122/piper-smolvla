"""Camera capture and episode writing for teleoperated data collection.

Kept separate from the control loop so it can be tested without hardware, but it is used
from inside teleop_spacemouse.py on purpose: only one process may command the arm, so
recording has to happen in the same loop that drives it.

One episode is a directory:

    episode_000/
        meta.json     task, rates, gripper widths, model, camera info, outcome
        data.jsonl    one line per sample: t, observation.state, action, frame index
        wrist.mp4     H.264, one frame per sample

state  = 6 measured joint angles [rad] + measured gripper width [m]
action = 6 commanded joint angles [rad] + commanded gripper width [m]

The action is what the arm was told to do, and the state is what it did, both straight
from the robot: nothing is retargeted or inferred, which is the whole point of collecting
by teleoperation rather than from a handheld device.
"""
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class RealSenseCamera:
    """D405 colour stream, grabbed on its own thread so the control loop never blocks."""

    def __init__(self, serial=None, width=640, height=480, fps=30, depth=False):
        import pyrealsense2 as rs

        self._rs = rs
        self.size = (width, height)
        self.fps = fps
        self.wants_depth = bool(depth)
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self.wants_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        # Depth pixels must line up with colour pixels, otherwise the two streams cannot
        # be used together later.
        self._align = rs.align(rs.stream.color) if self.wants_depth else None
        self.depth_scale = (profile.get_device().first_depth_sensor().get_depth_scale()
                            if self.wants_depth else None)
        self.frame = None
        self.depth = None
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            if self._align is not None:
                frames = self._align.process(frames)
            colour = frames.get_color_frame()
            if colour:
                self.frame = np.asanyarray(colour.get_data())
                if self.wants_depth:
                    depth = frames.get_depth_frame()
                    if depth:
                        self.depth = np.asanyarray(depth.get_data())
                self.frames += 1

    def wait_ready(self, timeout=5.0):
        deadline = time.time() + timeout
        while self.frame is None and time.time() < deadline:
            time.sleep(0.05)
        return self.frame is not None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._pipeline.stop()


class UvcCamera:
    """Any plain USB camera through OpenCV, grabbed on its own thread."""

    def __init__(self, index, width=640, height=480, fps=30):
        import cv2

        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise SystemExit(f"cannot open camera {index} (in use? try: ls /dev/video*)")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self.size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame = None
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok:
                self.frame = frame
                self.frames += 1

    def wait_ready(self, timeout=5.0):
        deadline = time.time() + timeout
        while self.frame is None and time.time() < deadline:
            time.sleep(0.05)
        return self.frame is not None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._capture.release()


class EpisodeWriter:
    """Writes one episode: an H.264 video per camera plus a jsonl of state/action pairs."""

    def __init__(self, directory, task, sizes, fps, meta=None, crf=23,
                 depth_sizes=None, depth_mode="off", depth_range_m=(0.0, 1.0),
                 depth_scale=0.001):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = fps
        self.sizes = dict(sizes)          # camera name -> (width, height)
        self.samples = 0
        self.started = time.time()          # wall clock, for the meta and the on-screen timer
        self.clock0 = time.perf_counter()   # monotonic, for the per-sample timestamps
        self._meta = dict(meta or {})
        self._log = (self.path / "data.jsonl").open("w")
        self.depth_sizes = dict(depth_sizes or {})
        self.depth_mode = depth_mode
        self.depth_range = depth_range_m
        self.depth_scale = depth_scale
        self._ffmpeg = {}
        for name, size in self.depth_sizes.items():
            if depth_mode == "lossless":       # 16-bit, exact millimetres, ~300x bigger
                codec = ["-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{size[0]}x{size[1]}",
                         "-r", str(fps), "-i", "-", "-c:v", "ffv1"]
            else:                              # 8-bit over a fixed range, same cost as colour
                codec = ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size[0]}x{size[1]}",
                         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
                         "-crf", str(crf), "-pix_fmt", "yuv420p"]
            self._ffmpeg[f"{name}_depth"] = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y"] + codec
                + [str(self.path / f"{name}_depth.{'mkv' if depth_mode == 'lossless' else 'mp4'}")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for name, size in self.sizes.items():
            self._ffmpeg[name] = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                 "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", str(self.path / f"{name}.mp4")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def add(self, elapsed, state, action, frames, depths=None):
        """One sample: measured state, commanded action, one image (and depth) per camera.

        elapsed must be measured against self.clock0: mixing perf_counter with time.time
        writes timestamps offset by the whole Unix epoch.
        """
        self._log.write(json.dumps({
            "t": round(float(elapsed), 4),
            "observation.state": [round(float(v), 5) for v in state],
            "action": [round(float(v), 5) for v in action],
            "frame": self.samples,
        }) + "\n")
        for name, process in self._ffmpeg.items():
            if name.endswith("_depth"):
                raw = (depths or {}).get(name[:-len("_depth")])
                if raw is None or not process.stdin:
                    continue
                if self.depth_mode == "lossless":
                    process.stdin.write(np.ascontiguousarray(raw, dtype="<u2").tobytes())
                else:
                    low, high = self.depth_range
                    metres = raw.astype(np.float32) * self.depth_scale
                    scaled = (metres - low) / max(high - low, 1e-6) * 255.0
                    process.stdin.write(np.clip(scaled, 0, 255).astype(np.uint8).tobytes())
                continue
            frame = (frames or {}).get(name)
            if frame is not None and process.stdin:
                process.stdin.write(frame.tobytes())
        self.samples += 1

    def close(self, outcome="kept"):
        """Finish the episode. outcome 'discarded' deletes the directory."""
        self._log.close()
        for process in self._ffmpeg.values():
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=30)
        if outcome == "discarded":
            shutil.rmtree(self.path, ignore_errors=True)
            return None
        duration = time.time() - self.started
        meta = {
            "task": self.task,
            "outcome": outcome,
            "samples": self.samples,
            "duration_s": round(duration, 2),
            "fps": self.fps,
            "cameras": {n: list(sz) for n, sz in self.sizes.items()},
            "depth_mode": self.depth_mode,
            "depth_range_m": list(self.depth_range) if self.depth_mode == "scaled" else None,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "state_layout": "joint1..joint6 measured [rad], gripper width measured [m]",
            "action_layout": "joint1..joint6 commanded [rad], gripper width commanded [m]",
            **self._meta,
        }
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1))
        size_mb = sum(f.stat().st_size for f in self.path.iterdir()) / 1e6
        return {"samples": self.samples, "duration_s": round(duration, 2),
                "size_mb": round(size_mb, 2), "path": str(self.path)}


def next_episode_dir(root):
    """episode_000, episode_001, ... inside root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    used = [int(p.name.split("_")[-1]) for p in root.glob("episode_*") if p.name.split("_")[-1].isdigit()]
    return root / f"episode_{max(used) + 1 if used else 0:03d}"
