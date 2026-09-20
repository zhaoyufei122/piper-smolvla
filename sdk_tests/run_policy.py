#!/usr/bin/env python3
"""Let a trained SmolVLA policy drive the Piper.

    HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
    ~/miniconda3/envs/lerobot/bin/python run_policy.py \
        ~/piper_data/policies/piper_cubes_020000/pretrained_model \
        --task "put the red cube in the black box"

The environment matters: lerobot 0.6 needs Python 3.12, so it lives in the `lerobot` conda
env rather than `pika`, and PYTHONNOUSERSITE keeps a CPU-only torch in ~/.local from
shadowing the CUDA build.

A trained checkpoint is not just weights. lerobot 0.6 stores the normalisation statistics
and the tokenizer as separate processor pipelines next to the model:

    pre:   rename(wrist -> camera1) -> batch -> newline -> tokenize -> to cuda -> normalise
    post:  un-normalise -> to cpu

Feeding raw values straight to select_action skips all of it: no language tokens at all,
and the returned numbers are normalised units that only look like radians. So both
pipelines are loaded from the same checkpoint and applied around every call.

Two threads. The policy runs in a worker and the control loop keeps its own 100 Hz beat,
so a slow inference can never delay the stop key, the torque check or a deadline:

    worker      30 Hz   observation -> pre -> chunk -> ensemble -> post -> latest action
    control    100 Hz   safety gates -> send, or hold

Temporal ensembling, because a 50-step chunk executed to its end is 1.7 s of open loop and
the first action of the next chunk does not agree with the last of the old one -- measured
on training data, 9.8 deg at the boundary against 1.3 deg inside the demonstration. So a
fresh chunk is predicted every --ensemble-every steps and each timestep is the weighted
average of every prediction that covers it, newest weighted most. The boundary stops being
a boundary: predictions blend instead of switching. Inference is 92 ms, which at 30 Hz
leaves room to re-predict every third step and average about sixteen chunks deep.

Every gate below can only ever cause a hold, never a motion:

    the operator has not pressed space, or pressed it again
    joint feedback older than --max-feedback-age, or the camera stopped producing frames
    the action is not finite, wrong-shaped, or older than --max-action-age
    a joint outside its limits (joint5 is held to +-60: its driver trips near 65)
    the tool inside --min-radius of the base, where the arm folds back on itself
    the tool outside the workspace box
    the command running more than --max-drift ahead of the arm
    torque residual over --collision-torque
    more than --max-seconds since the attempt started

A hold is not "stop sending": MOVE J targets are absolute, so the firmware keeps driving
to the last one. Holding means writing the measured position back as the target.

The policy has no idea when it is finished. Every demonstration simply stops a second or
two after the cube is released, so nothing in the data says "now stand still"; left running,
the policy keeps predicting and the arm oscillates over the box. The gripper channel does
say it, though: closed for a while and then open again is a place. That transition ends the
attempt, flies the arm back to the pose it started from, and holds.
"""
import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import piper_common as pc
from gravity_compensation import Rate, announce
from gravity_model import TOOL_URDF, GravityModel
from recorder import RealSenseCamera


def load_policy(path, device):
    """Weights plus the two processor pipelines that belong to them."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = Path(path)
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(path)
    policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(path))
    return policy, preprocessor, postprocessor, path


class Inference(threading.Thread):
    """Runs the policy on whatever observation is latest, never blocking the control loop."""

    def __init__(self, policy, preprocessor, postprocessor, camera_key, task, rate,
                 every=0, decay=0.1):
        super().__init__(daemon=True)
        self._policy = policy
        self._pre, self._post = preprocessor, postprocessor
        self._camera_key, self._task = camera_key, task
        self._period = 1.0 / rate
        self._every = max(int(every), 0)          # 0 = plain select_action, no ensembling
        self._decay = float(decay)
        self._chunks = deque()                    # (start_step, chunk) newest last
        self._step = 0
        self._lock = threading.Lock()
        self._observation = None        # (frame, state) set by the control loop
        self._action = None             # (action, produced_at) read by the control loop
        self._error = None
        self._enabled = threading.Event()
        self._stopping = threading.Event()   # not _stop: that name is Thread's own method

    def observe(self, frame, state):
        with self._lock:
            self._observation = (frame, state.copy())

    def latest(self):
        with self._lock:
            return self._action

    def start_episode(self):
        """New attempt: drop the queued chunk and any action from the previous one."""
        self._policy.reset()
        self._chunks.clear()
        self._step = 0
        with self._lock:
            self._action, self._error = None, None
        self._enabled.set()

    def _ensemble(self, batch):
        """One action for this step, averaged over every chunk that still covers it."""
        if self._step % self._every == 0 or not self._chunks:
            chunk = self._policy.predict_action_chunk(batch)[0]        # (horizon, dim)
            self._chunks.append((self._step, chunk))
            horizon = chunk.shape[0]
            while self._chunks and self._chunks[0][0] + horizon <= self._step:
                self._chunks.popleft()
        total, weight_sum = None, 0.0
        newest = self._chunks[-1][0]
        for start, chunk in self._chunks:
            offset = self._step - start
            if not 0 <= offset < chunk.shape[0]:
                continue
            # Age in steps, not in list position: a prediction made long ago saw an older
            # scene and should count for less.
            weight = float(np.exp(-self._decay * (newest - start)))
            total = chunk[offset] * weight if total is None else total + chunk[offset] * weight
            weight_sum += weight
        self._step += 1
        return (total / weight_sum).unsqueeze(0)

    def pause(self):
        self._enabled.clear()
        with self._lock:
            self._action = None

    def error(self):
        return self._error

    def stop(self):
        self._stopping.set()
        self._enabled.set()

    def run(self):
        while not self._stopping.is_set():
            if not self._enabled.wait(timeout=0.1) or self._stopping.is_set():
                continue
            with self._lock:
                observation = self._observation
            if observation is None:
                time.sleep(self._period)
                continue
            frame, state = observation
            started_at = time.time()
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
                # The pipeline renames, batches, tokenises and normalises: hand it the raw
                # observation under the name the dataset used.
                batch = self._pre({
                    self._camera_key: image,
                    "observation.state": torch.from_numpy(state),
                    "task": self._task,
                })
                with torch.inference_mode():
                    raw = (self._ensemble(batch) if self._every
                           else self._policy.select_action(batch))
                    action = self._post(raw)
                action = action.squeeze(0).float().cpu().numpy()
                with self._lock:
                    self._action = (action, time.time())
            except Exception as exc:                      # noqa: BLE001 - surfaced to the loop
                self._error = f"{type(exc).__name__}: {exc}"
                self._enabled.clear()
            # Sleep the remainder of the period, not a whole period on top of the work:
            # preprocessing alone is 20 ms and was dragging 30 Hz down to 18.
            time.sleep(max(0.0, started_at + self._period - time.time()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--task", required=True, help="worded exactly as during collection")
    ap.add_argument("--camera-key", default="observation.images.wrist",
                    help="the dataset's name for the camera; the pipeline renames it itself")
    ap.add_argument("--camera-serial", default="315122272699")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rate", type=float, default=100.0, help="control loop rate [Hz]")
    ap.add_argument("--policy-rate", type=float, default=30.0,
                    help="policy query rate [Hz]; must match the rate the data was recorded at")
    ap.add_argument("--speed", type=int, default=100, help="firmware MOVE J speed percent")
    ap.add_argument("--tool", choices=["pika", "piper"], default="pika")
    ap.add_argument("--tcp", type=float, nargs=3, default=[0.13, 0.0, 0.0])
    ap.add_argument("--limit-margin", type=float, nargs="+", default=[2, 2, 2, 2, 10, 2],
                    metavar="DEG", help="stay this far inside each joint limit; joint5 needs 10")
    ap.add_argument("--min-radius", type=float, default=0.25)
    ap.add_argument("--box-x", type=float, nargs=2, default=[-0.65, 0.65])
    ap.add_argument("--box-y", type=float, nargs=2, default=[-0.65, 0.65])
    ap.add_argument("--box-z", type=float, nargs=2, default=[-0.05, 0.75])
    ap.add_argument("--max-joint-vel", type=float, default=1.0, help="joint speed clamp [rad/s]")
    ap.add_argument("--max-drift", type=float, default=0.26,
                    help="hold if the command gets this far ahead of the arm [rad]")
    ap.add_argument("--collision-torque", type=float, default=9.0,
                    help="[Nm], 0 disables. Higher than during teleoperation on purpose: the "
                         "gravity model knows about the gripper but not about the cube in it, "
                         "and lifting adds inertia on top. Measured carrying a 30 mm cube, J2 "
                         "sits around 5 Nm of residual with nothing wrong")
    ap.add_argument("--max-feedback-age", type=float, default=0.2, help="[s]")
    ap.add_argument("--max-action-age", type=float, default=0.5,
                    help="[s]; a chunk computed from a scene that has since changed is stale")
    ap.add_argument("--ensemble-every", type=int, default=0, metavar="STEPS",
                    help="re-predict a chunk this often and average the overlap. 0 falls back "
                         "to consuming one chunk at a time, which is where the boundary jump "
                         "comes from. Inference is 92 ms, so 3 steps at 30 Hz is affordable")
    ap.add_argument("--ensemble-decay", type=float, default=0.4,
                    help="weight falls off as exp(-decay * age in steps). Larger lets the "
                         "newest prediction dominate, which is what keeps the arm moving: at "
                         "0.1 the newest chunk is only 40 percent of the target, at 0.4 it is 86")
    ap.add_argument("--smooth", type=float, default=0.25, metavar="ALPHA",
                    help="low-pass the commanded target: 1 = raw policy output, smaller is "
                         "smoother. The policy predicts 50-step chunks and the first action "
                         "of a new chunk disagrees with the last of the old one -- measured "
                         "on training data, 9.8 deg at the boundary against 1.3 deg in the "
                         "demonstration itself, which is what makes the arm shake")
    ap.add_argument("--gripper-idle-unpowered", action="store_true",
                    help="cut power to the gripper once it is open and settled. Its servo "
                         "self-oscillates at 25 Hz whenever energised -- torque swings 0.23 Nm "
                         "while the width reading does not move at all -- and the drive holds "
                         "its position unpowered. Off by default: a loose flange turns that "
                         "hum into a real shake, and tightening the flange is the actual fix")
    ap.add_argument("--gripper-effort", type=float, default=1.0, metavar="NM",
                    help="how hard the gripper servo works to hold its width -- the gripper's "
                         "own stiffness. At 1.0, the value the demonstrations used, it chases "
                         "every tenth of a millimetre the policy asks for and buzzes even when "
                         "it is just sitting open. Lower it until the buzz stops and the cube "
                         "is still held")
    ap.add_argument("--gripper-open", type=float, default=70.0, metavar="MM",
                    help="the open width the demonstrations used")
    ap.add_argument("--gripper-close", type=float, default=20.0, metavar="MM",
                    help="the closed width the demonstrations used")
    ap.add_argument("--gripper-continuous", action="store_true",
                    help="send the policy's raw gripper output instead of snapping it to the "
                         "two widths it was taught. The demonstrations only ever contain 20 "
                         "and 70 mm; the regressed output wanders between them by a few "
                         "tenths of a millimetre, and at 100 Hz that is a dither the gripper "
                         "motor has to chase forever")
    ap.add_argument("--grasp-gap", type=float, default=0.8, metavar="MM",
                    help="the gripper must settle this much wider than commanded to count as "
                         "holding something. Measured: 56 successful demonstrations sit 1.6 "
                         "to 7.9 mm wider, and a miss lands within 0.2 mm of the command, so "
                         "the gap is real but only if it is read after the fingers stop")
    ap.add_argument("--ignore-missed-grasp", action="store_true",
                    help="carry on to the box even when the fingers came up empty")
    ap.add_argument("--no-auto-stop", action="store_true",
                    help="keep going after the cube is released instead of returning home")
    ap.add_argument("--settle-seconds", type=float, default=0.6,
                    help="the gripper must stay closed, then open, this long for each to count")
    ap.add_argument("--home-seconds", type=float, default=3.0,
                    help="how long the flight back to the observation pose takes")
    ap.add_argument("--trace", metavar="FILE",
                    help="log to jsonl: policy output, smoothed target, measured joints, "
                         "velocity, torque, tcp, gripper, and which gate held it")
    ap.add_argument("--trace-every", type=int, default=1,
                    help="write one trace sample every N control cycles. 1 = the full control "
                         "rate, which is what it takes to see vibration: sampling at 33 Hz "
                         "hides anything above about 16 Hz entirely")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything but never command the arm; prints what it would send")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    model = GravityModel(TOOL_URDF[args.tool])
    model.add_payload(0.42, (0.03, 0.0, 0.07))
    tcp = model.tool_to_link(args.tcp)
    margin = np.radians(np.broadcast_to(np.asarray(args.limit_margin, dtype=float), (6,)))
    limits = pc.JOINT_LIMITS + np.stack([margin, -margin], axis=1)
    box_low = np.array([args.box_x[0], args.box_y[0], args.box_z[0]])
    box_high = np.array([args.box_x[1], args.box_y[1], args.box_z[1]])
    max_step = args.max_joint_vel / args.rate

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint}")
    policy, preprocessor, postprocessor, path = load_policy(args.checkpoint, device)
    print(f"on {device}   pre: {[type(s).__name__ for s in preprocessor.steps]}")
    print(f'task: "{args.task}"')

    piper = pc.connect(args.can)            # waits for complete, settled feedback
    camera = RealSenseCamera(args.camera_serial, fps=int(args.policy_rate))
    if not camera.wait_ready():
        raise SystemExit("no camera frames (is another program holding the D405?)")

    q_now = pc.joint_positions(piper)
    time.sleep(0.1)
    if np.max(np.abs(pc.joint_positions(piper) - q_now)) > np.radians(2.0):
        raise SystemExit("the arm's reported pose changed between two reads 100 ms apart; "
                         "it is still moving or the feedback is incomplete")
    print(f"start pose: {np.degrees(q_now).round(1)} deg")
    if not args.dry_run:
        pc.confirm("The POLICY is about to drive the arm. Keep the workspace clear and stay "
                   "near the power switch.", args.yes)
        if not pc.enable(piper):
            raise SystemExit("enable failed (arm_tool.py status, then clear --joint N)")

    worker = Inference(policy, preprocessor, postprocessor, args.camera_key,
                       args.task, args.policy_rate,
                       every=args.ensemble_every, decay=args.ensemble_decay)
    worker.start()

    t_zero = time.time()
    keys = pc.KeyReader()
    keys.start()
    rate = Rate(args.rate)
    running, started, cycle = False, time.time(), 0
    home_q = q_now.copy()            # the observation pose: where an attempt starts and ends
    q_cmd = q_now.copy()
    smoothed = q_now.copy()          # the low-passed policy target, before the speed clamp
    closed_since = opened_since = None
    grasped_once = False
    going_home = None
    last_action_at = None            # smoothing steps once per NEW action, not per cycle
    grasp_verdict = None
    open_m, close_m = args.gripper_open / 1000, args.gripper_close / 1000
    gripper_target = open_m          # the binary state, with hysteresis around the middle
    gripper_sent = None
    gripper_powered = True
    powered_until = 0.0              # keep the motor on this long after a width change
    grip_stable_since, grip_last = None, None
    gripper = pc.gripper_state(piper)[0]
    last_frames, last_frame_change = camera.frames, time.time()
    raw_action, held_for = None, None
    def next_trace(where, task):
        """run_NNN_<what it was told to fetch>.jsonl, so runs never overwrite each other."""
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        used = [int(f.name[4:7]) for f in where.glob("run_[0-9][0-9][0-9]_*.jsonl")]
        skip = {"put", "the", "cube", "in", "a", "into", "block", "box", "on"}
        words = [w for w in (task or "").lower().split() if w not in skip]
        return where / f"run_{max(used) + 1 if used else 0:03d}_{(words[0] if words else 'run')[:10]}.jsonl"

    trace = None
    trace_path = args.trace
    if trace_path and not str(trace_path).endswith(".jsonl"):
        trace_path = str(next_trace(trace_path, args.task))
    if trace_path:
        trace = Path(trace_path).open("w")
        trace.write(json.dumps({"meta": {
            "checkpoint": str(path), "task": args.task, "smooth": args.smooth,
            "rate": args.rate, "policy_rate": args.policy_rate,
            "max_joint_vel": args.max_joint_vel, "min_radius": args.min_radius,
            "limit_margin_deg": list(np.degrees(margin).round(1)),
            "trace_rate": args.rate / max(args.trace_every, 1),
            "ensemble_every": args.ensemble_every, "ensemble_decay": args.ensemble_decay,
            "gripper_effort": args.gripper_effort,
            "gripper_binary": not args.gripper_continuous,
            "speed_percent": args.speed,
            "joint_command_limits_deg": np.degrees(limits).round(1).tolist(),
        }}) + "\n")
        print(f"tracing into {trace_path}")

    def hold(reason=None):
        """The only way to stop: overwrite the firmware's target with where the arm is."""
        nonlocal running, q_cmd, smoothed, held_for, gripper_sent
        running = False
        held_for = reason
        worker.pause()
        q_cmd = pc.joint_positions(piper)
        smoothed = q_cmd.copy()
        gripper_sent = None
        if not args.dry_run:
            pc.send_joint_positions(piper, q_cmd, args.speed)
            pc.send_gripper(piper, pc.gripper_state(piper)[0], args.gripper_effort)
        if reason:
            announce(f"HOLD: {reason}")

    announce("ready -- SPACE to let the policy drive, SPACE again to hold, q to quit")
    try:
        while True:
            key = keys.get()
            if key == "q":
                hold("quit")
                break
            if key == " ":
                if running:
                    hold("stopped by operator")
                else:
                    q_cmd = pc.joint_positions(piper)
                    smoothed = q_cmd.copy()
                    held_for = None
                    closed_since = opened_since = None
                    grasped_once, going_home = False, None
                    last_action_at, grasp_verdict = None, None
                    gripper_target, gripper_sent = open_m, None
                    gripper_powered, powered_until = True, 0.0
                    grip_stable_since, grip_last = None, None
                    worker.start_episode()
                    running, started = True, time.time()
                    announce("policy driving -- SPACE to hold")

            q_now = pc.joint_positions(piper)
            measured_grip = pc.gripper_state(piper)[0]
            frame = camera.frame
            if camera.frames != last_frames:
                last_frames, last_frame_change = camera.frames, time.time()
            worker.observe(frame, np.concatenate([q_now, [measured_grip]]).astype(np.float32))

            if running:
                # Everything below is checked BEFORE anything is sent.
                if worker.error():
                    hold(f"policy failed -- {worker.error()}")
                elif pc.feedback_age(piper) > args.max_feedback_age:
                    hold(f"joint feedback is {pc.feedback_age(piper) * 1000:.0f} ms old")
                elif time.time() - last_frame_change > args.max_feedback_age * 5:
                    hold("the camera stopped producing frames")
                elif time.time() - started > args.max_seconds:
                    hold(f"gave up after {args.max_seconds:.0f} s")

            if going_home is not None:
                elapsed = time.time() - going_home["t0"]
                q_cmd = pc.min_jerk(going_home["q0"], home_q, args.home_seconds, elapsed)
                q_cmd = np.clip(q_cmd, limits[:, 0], limits[:, 1])
                if not args.dry_run:
                    pc.send_joint_positions(piper, q_cmd, args.speed)
                    pc.send_gripper(piper, pc.GRIPPER_MAX_WIDTH, args.gripper_effort)
                if elapsed >= args.home_seconds:
                    going_home = None
                    hold("task complete -- back at the observation pose")

            if running and going_home is None:
                latest = worker.latest()
                if latest is None:
                    pass                                   # first chunk not ready yet
                elif time.time() - latest[1] > args.max_action_age:
                    hold(f"the last action is {time.time() - latest[1]:.1f} s old")
                else:
                    action = latest[0]
                    if action.shape != (7,) or not np.all(np.isfinite(action)):
                        hold(f"policy returned {action.shape} {action}")
                    else:
                        raw_action = action
                        # One filter step per NEW action, not per control cycle. The control
                        # loop runs at 100 Hz and the policy at 15-30, so filtering every
                        # cycle lets the filter fully converge between actions and reproduce
                        # the raw sequence exactly -- measured: the commanded path wobbled
                        # 0.78 mm RMS, the same as the unfiltered target.
                        if latest[1] != last_action_at:
                            last_action_at = latest[1]
                            alpha = float(np.clip(args.smooth, 1e-3, 1.0))
                            smoothed = smoothed + alpha * (action[:6] - smoothed)
                        target = np.clip(smoothed, limits[:, 0], limits[:, 1])
                        target = q_cmd + np.clip(target - q_cmd, -max_step, max_step)
                        position = model.tip_pose(target, tcp)[0]
                        radius = float(np.hypot(position[0], position[1]))
                        current_radius = float(np.hypot(*model.tip_pose(q_cmd, tcp)[0][:2]))
                        outside = np.any(position < box_low) or np.any(position > box_high)
                        if radius < args.min_radius and radius < current_radius:
                            announce(f"refused: driving into the base ({radius * 100:.0f} cm)")
                        elif outside:
                            announce(f"refused: outside the workspace box {position.round(3)}")
                        else:
                            q_cmd = target

                        drift = float(np.max(np.abs(q_cmd - q_now)))
                        if drift > args.max_drift:
                            hold(f"command {np.degrees(drift):.0f} deg ahead of the arm")
                        else:
                            if args.gripper_continuous:
                                gripper = float(np.clip(action[6], 0.0, pc.GRIPPER_MAX_WIDTH))
                            else:
                                # Snap to the two taught widths, with a dead band either side
                                # of the middle so a wobbling prediction cannot flap the motor.
                                middle = (open_m + close_m) / 2
                                band = abs(open_m - close_m) * 0.15
                                if action[6] > middle + band:
                                    gripper_target = open_m
                                elif action[6] < middle - band:
                                    gripper_target = close_m
                                gripper = gripper_target
                            if not args.dry_run:
                                pc.send_joint_positions(piper, q_cmd, args.speed)
                                # Only on change: re-sending the same width 100 times a second
                                # gives the gripper servo something to chase that is not there.
                                if gripper_sent is None or abs(gripper - gripper_sent) > 1e-6:
                                    pc.send_gripper(piper, gripper, args.gripper_effort)
                                    gripper_sent, gripper_powered = gripper, True
                                    powered_until = time.time() + 1.0
                                elif (args.gripper_idle_unpowered and gripper_powered
                                      and gripper > (open_m + close_m) / 2
                                      and time.time() > powered_until):
                                    # Open and settled: nothing to hold, so stop the buzzing.
                                    pc.send_gripper(piper, gripper, args.gripper_effort,
                                                    powered=False)
                                    gripper_powered = False

                            # Closed for a while, then open again, is a place. Both halves
                            # need to persist: the commanded width wobbles frame to frame.
                            now = time.time()
                            half = pc.GRIPPER_MAX_WIDTH / 2
                            if gripper < half:
                                opened_since = None
                                closed_since = closed_since or now
                                # The fingers are still travelling when the command
                                # changes; judging then reads a width from halfway through
                                # the close. Wait for the measured width to stop moving.
                                if grip_last is None or abs(measured_grip - grip_last) > 1.5e-4:
                                    grip_stable_since = now
                                grip_last = measured_grip
                                settled = (now - closed_since > args.settle_seconds
                                           and grip_stable_since is not None
                                           and now - grip_stable_since > 0.3)
                                if settled:
                                    grasped_once = True
                                    # Did it actually get the cube? On all 56 demonstrations
                                    # the cube stops the fingers 1.6 to 7.9 mm short of the
                                    # commanded width; on the miss we recorded, they closed
                                    # 1.1 mm PAST it. The policy cannot tell the difference --
                                    # it never saw a failed grasp -- so the gap is checked here.
                                    if grasp_verdict is None:
                                        gap = (measured_grip - gripper) * 1000
                                        grasp_verdict = gap > args.grasp_gap
                                        if not grasp_verdict:
                                            announce(f"grasp missed: fingers closed {-gap:.1f} mm "
                                                     "past the commanded width, nothing in them")
                                            if not args.ignore_missed_grasp:
                                                worker.pause()
                                                going_home = {"q0": pc.joint_positions(piper),
                                                              "t0": time.time()}
                                        else:
                                            announce(f"grasp confirmed: held {gap:.1f} mm open")
                            else:
                                closed_since = None
                                grip_stable_since, grip_last = None, None
                                opened_since = opened_since or now
                                if (grasped_once and not args.no_auto_stop
                                        and now - opened_since > args.settle_seconds):
                                    announce("released -- returning to the observation pose")
                                    worker.pause()
                                    going_home = {"q0": pc.joint_positions(piper),
                                                  "t0": time.time()}

            # Collision: 50 ms apart, as during collection.
            if running and args.collision_torque > 0 and cycle % 5 == 0:
                residual = pc.joint_efforts(piper, x4_j123=True) - model.gravity_torques(q_now)
                on_stop = ((q_now <= pc.JOINT_LIMITS[:, 0] + margin)
                           | (q_now >= pc.JOINT_LIMITS[:, 1] - margin))
                if np.max(np.abs(np.where(on_stop, 0.0, residual))) > args.collision_torque:
                    hold(f"collision? residual {np.round(residual, 1)} Nm")

            if not args.no_show and frame is not None and cycle % 5 == 0:
                view = frame.copy()
                colour = (80, 220, 80) if running else (80, 80, 240)
                cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(view, f"{'RUNNING' if running else 'HOLD'}  {args.task[:44]}",
                            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
                cv2.imshow("policy", view)
                cv2.waitKey(1)

            if trace is not None and cycle % max(args.trace_every, 1) == 0:
                latest = worker.latest()
                trace.write(json.dumps({
                    # Two clocks: `t` restarts with every attempt (useful inside one), `wall`
                    # never does (the only one that works across attempts).
                    "t": round(time.time() - started, 3),
                    "wall": round(time.time() - t_zero, 3),
                    "running": bool(running),
                    "held_for": held_for,
                    "q": [round(float(v), 5) for v in q_now],
                    "qd": [round(float(v), 4) for v in pc.joint_velocities(piper)],
                    "tau": [round(float(v), 3) for v in pc.joint_efforts(piper, x4_j123=True)],
                    "q_cmd": [round(float(v), 5) for v in q_cmd],
                    "smoothed": [round(float(v), 5) for v in smoothed],
                    "action": ([round(float(v), 5) for v in raw_action]
                               if raw_action is not None else None),
                    "tcp": [round(float(v), 4) for v in model.tip_pose(q_now, tcp)[0]],
                    "grip": [round(float(gripper), 4), round(float(measured_grip), 4)],
                    "action_age_ms": (round((time.time() - latest[1]) * 1000, 1)
                                      if latest else None),
                    "grasped_once": bool(grasped_once),
                    "grasp_ok": grasp_verdict,
                    "going_home": going_home is not None,
                }) + "\n")

            if cycle % int(args.rate) == 0:
                position = model.tip_pose(q_now, tcp)[0]
                age = (time.time() - worker.latest()[1]) if worker.latest() else float("nan")
                print(f"\r[{'RUN ' if running else 'HOLD'}] tcp {np.round(position, 3)} m  "
                      f"grip {gripper * 1000:4.1f}/{measured_grip * 1000:4.1f} mm  "
                      f"action {age * 1000:4.0f} ms  t {time.time() - started:5.1f}s   ",
                      end="", flush=True)
            cycle += 1
            rate.sleep()
    except KeyboardInterrupt:
        hold("interrupted")
    finally:
        if trace is not None:
            trace.close()
            print(f"\ntrace written to {trace_path}")
        # Join before the interpreter tears down CUDA under the worker's feet.
        worker.stop()
        worker.join(timeout=3.0)
        keys.stop()
        camera.stop()
        if not args.no_show:
            cv2.destroyAllWindows()
    print("\nstopped; the arm holds this pose "
          "(joint_position_ctrl.py --zero, then arm_tool.py disable to park)")


if __name__ == "__main__":
    main()
