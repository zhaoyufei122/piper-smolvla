> Historical planning notes, retained for context. The implemented collection method is SpaceMouse teleoperation with robot-mounted wrist RGB. See the [current guide](../SMOLVLA.md). Installation/version statements below are historical.

# SmolVLA on Piper — plan and data collection decision

Goal: fine-tune SmolVLA (`lerobot/smolvla_base`, 450M) on a simple Piper manipulation task,
as a first hands-on VLA project. This document is the plan; nothing here has been run yet.

> **Superseded (2026-09-17).** Data collection now goes through the Pika Sense handheld
> device (UMI style) in `~/Desktop/Agilex_UMI`, not through kinesthetic teaching on the
> Piper. The comparison below is kept for the reasoning; option B is no longer the plan,
> and gravity compensation is no longer on the critical path. What the Piper side needs
> instead is at the end of this section.

## The data collection question

You have no teleoperation rig. The two options you raised, plus two more:

| option | hand in frame? | hardware cost | effort | verdict |
|---|---|---|---|---|
| A. Kinesthetic drag only (record video while dragging) | **yes** | 0 | low | reject |
| B. **Kinesthetic drag, then autonomous replay for the video** | no | 0 | low | **start here** |
| C. Phone / gamepad teleop driving Piper | no | 0–50 GBP | medium | upgrade later |
| D. Second Piper as leader (AgileX master-slave) | no | ~arm price | low | best, if funded |
| E. D1T mapped onto Piper | no | 0 | **high** | not worth it now |

### Why not A

At inference time there is no hand in the picture. If every training frame contains your
hand next to the gripper, the policy is being trained on a different distribution than it
will see, and the hand also occludes the object exactly at the grasp moment — the frames
that matter most. It usually still "learns something", which is the worst outcome for
learning purposes: you cannot tell whether a failure comes from the method or the data.

### Why B works

Split the demonstration into two passes:

1. **Teach** — drag the arm through the task with gravity compensation on, and log only the
   joint trajectory. Your hand is in the room but no camera data is kept.
2. **Replay** — put the object back where it was, let the arm run the recorded trajectory on
   its own, and record the cameras during that run.

The replay frames contain no hand, and the action labels are exactly the commands the arm
executed, so observations and actions stay consistent. This is the standard trick when
there is no leader arm.

Costs and limits, honestly:

- Each episode takes about twice as long (teach, reset the scene, replay).
- The scene must be reset to the same state. Mark object positions with tape; a sheet of
  paper with a grid is enough.
- Only works for quasi-static tasks with rigid objects. No reactive behaviour, no moving
  targets, no recovery from a slipped grasp.
- Some replays will fail (the grasp misses). Discard those episodes; do not keep a replay
  whose video does not match what the actions were meant to do.
- The policy only ever sees successful, non-corrective trajectories. Fine for a first
  project, a known limitation of pure behaviour cloning.

### Why not E (D1T as leader)

The D1T has different link lengths, joint limits and gripper from the Piper, so a 1:1 joint
mapping gives a different end-effector pose. Doing it properly means D1T forward kinematics
→ Cartesian retargeting → Piper IK, plus a workspace and calibration story, plus the D1T
needs to be in zero-torque mode and readable at 30 Hz. That is a project in itself, and it
is the arm your dissertation runs on. If you want live teleop, option C gets you there in
an afternoon.

### What the Piper side needs with UMI data

Pika Sense records **end-effector poses**, not joint angles, so the arm side changes shape:

- **Retargeting or IK at execution.** Either train the policy in EE space and convert to
  joint commands at 30 Hz, or convert the recorded trajectories to Piper joint space before
  training. Either way something must do IK — `move_group` planning is far too slow for a
  closed-loop policy, so this belongs in `piper_bridge` (or `moveit_servo`).
- **Gripper width calibration.** The Sense records an opening in mm and the Pika Gripper has
  to reproduce it, so the SDK's 0-70 mm scale must be checked against a ruler.
- **Matching viewpoints.** UMI only works when the handheld camera and the robot-mounted
  camera see the same thing. Pika Sense and Pika Gripper are sold as a matched pair for
  exactly this, so verify rather than assume.

### Option C, when B runs out

LeRobot 0.6.1 ships `phone` (phone pose as a 6-DoF input) and `gamepad` teleoperators. Both
need a `Robot` implementation for the Piper, which LeRobot does not have — see below. This
buys you closed-loop, reactive demonstrations, which B cannot produce.

## Where LeRobot stands (checked 2026-09-16)

- `lerobot` 0.6.1 on PyPI, requires Python >= 3.12 (system Python 3.12.3 is fine).
- Robots included: SO-100/101, Koch, OpenArm, ReBot, OMX, LeKiwi, Unitree G1, Reachy2,
  HopeJR. **No AgileX Piper.** Adding one means implementing `lerobot.robots.Robot`:
  `connect`, `disconnect`, `get_observation`, `send_action`, `observation_features`,
  `action_features`, `is_connected`, `calibrate`/`configure`. Our `piper_bridge/hardware.py`
  already does the hard part (SI-unit read/write over the SDK), so this is a thin wrapper —
  roughly a day including debugging.
- Teleoperators included: keyboard, gamepad, phone, plus the leader arms.
- Dataset format is LeRobotDataset v3 (`docs/source/lerobot-dataset-v3.mdx`).
- Fine-tuning entry point:

  ```bash
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=<user>/<dataset> \
    --batch_size=64 --steps=20000 \
    --policy.device=cuda
  ```

  The docs quote ~4 h for 20k steps on a single A100.

You already have the conventions from `~/Desktop/Franka_VLA` (observation_t before
action_t, per-episode json, LeRobot export). Reuse them; only the robot layer changes.

## Two blockers to solve before training

1. **Disk.** 14 GB free on `/`. CUDA PyTorch plus the base model plus checkpoints will eat
   most of that before any data exists. `~/miniconda3` is 18 GB and `~/Desktop/R03_Work` is
   14 GB — either clear space or put the dataset and checkpoints on an external drive.
2. **GPU.** RTX 5070 Ti Laptop, 12 GB. The documented `--batch_size=64` assumes an 80 GB
   A100. Expect batch 2–8 with gradient accumulation and bf16, and a training run measured
   in a day rather than 4 hours. Consider a university GPU or Colab for the actual
   fine-tune, and keep the laptop for data collection and evaluation.

Cameras: one fixed third-person view plus one wrist view is the usual minimum. The D435 can
be the wrist camera (you already worked out a Link6 mount extrinsic); any USB webcam on a
tripod works for the fixed view. Record at 30 fps, 640x480.

## Pipeline

```
teach (drag)  ->  replay (record)  ->  LeRobotDataset v3  ->  lerobot-train  ->  eval
  teach_replay.py record            (converter, TODO)      smolvla_base      async inference
  teach_replay.py replay
```

Suggested first task: "pick up the red cube and put it in the box", object position varied
over a 20x20 cm area, 30–50 episodes, 10–20 s each. That is 1–2 hours of collection with
the replay pass included.

## Status

Prepared now:

- `sdk_tests/teach_replay.py` — the record and replay passes (untested on hardware).

Deliberately deferred:

- LeRobotDataset v3 export — pin the LeRobot version first, then write the converter
  against it rather than guessing at the API.
- `PiperFollower` for LeRobot — only needed for option C, or to use `lerobot-record`
  directly instead of our own recorder.

Prerequisite: finish `TESTING.md` T1–T4 first. The replay pass needs position control
(T2) and the teach pass needs gravity compensation (T4).
