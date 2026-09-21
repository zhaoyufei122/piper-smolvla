> Earlier hardware bring-up notes. For the current SmolVLA workflow, start with the [project README](../README.md). Hardware-specific values below are bench notes, not portable defaults.

# AgileX Piper test bench

Bring-up of the AgileX Piper 6-DoF arm, in four steps:

1. read joint positions with `piper_sdk`
2. command joint positions directly with the SDK
3. control the arm from ROS 2 Jazzy + MoveIt 2
4. zero-torque drag mode (gravity compensation) with MIT joint control

For bring-up on the real arm, follow [TESTING.md](../TESTING.md): the same four steps as a
checklist, with pass criteria, expected values and the numbers to record. Day-to-day
commands — start up, teleoperate, recover, shut down — are on the bench card
[操作速查.md](../操作速查.md).

A follow-on experiment — collecting demonstrations on this arm to fine-tune SmolVLA — is
planned in [SMOLVLA.md](../SMOLVLA.md), with `sdk_tests/teach_replay.py` as its first tool.

```
Agilex_Piper/
├── scripts/can_activate.sh        bring up the USB-CAN adapter (from piper_ros)
├── sdk_tests/                     steps 1, 2, 4: plain Python on top of piper_sdk
│   ├── piper_common.py            unit conversion, enable/disable, feedback helpers
│   ├── read_joints.py             step 1
│   ├── joint_position_ctrl.py     step 2
│   ├── arm_tool.py                status / enable / disable / stop / reset
│   ├── gravity_model.py           URDF gravity torque model (numpy)
│   └── gravity_compensation.py    step 4
└── ros2_ws/src/                   step 3
    ├── piper_description/         URDF + meshes (from piper_ros)
    ├── piper_bridge/              SDK <-> ROS 2 bridge node (joint_states + trajectory actions)
    └── piper_moveit_config/       MoveIt 2 config, launch file, moveit_goal.py client
```

All scripts use SI units: rad, rad/s, N·m, m. Command-line joint targets are in degrees.

## 0. Setup (once)

```bash
sudo apt install can-utils ethtool
python3 -m pip install --user --break-system-packages piper_sdk   # 0.6.2 + python-can 4.6.1

cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Every session: power the arm, plug in the USB-CAN adapter, then

```bash
bash scripts/can_activate.sh can0 1000000   # needs sudo
candump can0                                # should scroll quickly; Ctrl+C
```

## 1. Read joint positions

```bash
cd sdk_tests
python3 read_joints.py          # live table, 10 Hz
python3 read_joints.py --once
```

This script only reads, so it is safe with the motors off. It prints the firmware version,
controller mode, enable flags, joint angles, velocities, torque feedback and the gripper
width. Two firmware notes:

- MIT control (step 4) needs firmware **>= V1.5-2**.
- On firmware **<= V1.8-2**, joints J1–J3 report torque 4× too small. Add `--effort-x4`.

## 2. Direct SDK position control

```bash
python3 joint_position_ctrl.py --demo                       # joint1 ±15°, joint6 +30°, back
python3 joint_position_ctrl.py --target 0 60 -45 0 30 0     # "ready" pose, degrees
python3 joint_position_ctrl.py --target 0 60 -45 0 30 0 --gripper 40
python3 joint_position_ctrl.py --zero                       # folded rest pose
python3 arm_tool.py status
```

The script enables the motors and streams a min-jerk trajectory from the current pose at
100 Hz (MOVE J, `MotionCtrl_2` + `JointCtrl`), so the arm never jumps. When it finishes,
the **arm stays enabled and holds its pose**.

**Disabling cuts power and the arm drops.** Always move to the rest pose with `--zero`
first, then run `arm_tool.py disable`. At the rest pose, gravity pushes joints 2 and 3
into their hard stops, so the arm stays put when the motors turn off.

Only one program should send commands at a time. Stop the ROS bridge before you run
these scripts.

## 3. ROS 2 + MoveIt 2

```
RViz / moveit_goal.py ──> move_group ──FollowJointTrajectory──> piper_bridge ──CAN──> Piper
                              ^                                     │
                              └────────────── /joint_states <───────┘
```

`piper_bridge` publishes `/joint_states` from the real CAN feedback. It also serves
`arm_controller/follow_joint_trajectory` and `gripper_controller/follow_joint_trajectory`.
It samples each trajectory at 200 Hz and streams the samples to the arm as position
setpoints. MoveIt therefore always sees the real arm state. The upstream `piper_ros` demo
works differently: it runs mock ros2_control hardware and copies the mock state onto the
arm, open-loop.

```bash
cd ros2_ws && source install/setup.bash

# a) simulated arm (no hardware): drag the marker in RViz -> Plan & Execute
ros2 launch piper_moveit_config piper_moveit.launch.py

# b) real arm, visualisation only (motors stay off): check that RViz matches the real arm
ros2 launch piper_bridge display.launch.py

# c) real arm with MoveIt (enables the motors and holds the current pose)
ros2 launch piper_moveit_config piper_moveit.launch.py use_sim:=false

# d) scripted goals, in a second terminal
ros2 run piper_moveit_config moveit_goal.py --named ready
ros2 run piper_moveit_config moveit_goal.py --joints 0 30 -30 0 0 0 --vel 0.2
ros2 run piper_moveit_config moveit_goal.py --gripper 40
ros2 run piper_moveit_config moveit_goal.py --named zero
```

Named poses in `piper.srdf`:

- arm: `zero`, `ready` (0, 60, −45, 0, 30, 0)°
- gripper: `open`, `close`

MoveIt's default velocity and acceleration scaling is 0.1, set in `joint_limits.yaml`.

Bridge parameters (`--ros-args -p name:=value`, or edit the launch file):

| parameter | default | meaning |
|---|---|---|
| `use_sim` | false | kinematic simulator instead of CAN |
| `can_port` | can0 | |
| `auto_enable` | true | enable motors at start-up and hold the current pose |
| `rate` | 200.0 | state publish / setpoint stream rate [Hz] |
| `speed_percent` | 100 | firmware MOVE J speed cap |
| `goal_tolerance` | 0.02 | final joint error to report success [rad] |
| `path_tolerance` | 0.5 | abort if tracking error exceeds this [rad], 0 disables |
| `goal_time_margin` | 2.0 | extra time allowed to settle [s] |
| `gripper_effort` | 1.0 | gripper force setting [N·m] |

Enable or disable the motors while the bridge is running (disabling drops the arm):
`ros2 service call /piper_bridge/enable std_srvs/srv/SetBool "{data: true}"`

## 4. Gravity compensation (zero-torque drag)

For each joint *i*, the script sends a MIT command with `kp = 0`, a small `kd` and a
feed-forward torque:

```
t_ref_i = tau_scale_i · gain · tau_g,i(q)
```

`tau_g(q)` comes from the URDF masses and centres of mass (`gravity_model.py`), with the
gripper lumped into link6. It matches `pinocchio.computeGeneralizedGravity` to 1e-14 N·m.
The J1–J3 drivers multiply MIT torque commands by 4 and the J4–J6 drivers do not
(measured 2026-09-21: effort / command is 4.2 on J2/J3 and 1.0 on J5), so the default
`--tau-scale` is `0.25 0.25 0.25 1 1 1`. With 0.25 on every joint the wrist gets a quarter
of its torque and J5 sags ~30°. The default model is the fitted Pika gripper plus a 0.42 kg
payload (`--tool pika --payload 0.42 --payload-x 0.03 --payload-z 0.07`); the standard-gripper
model left out most of the wrist load. Every run is traced to `~/piper_data/runs/gravity/`,
including each driver's enable flag and fault bits.

Procedure:

```bash
cd sdk_tests

# 1) Validate the model in position mode at a few poses
python3 gravity_compensation.py --check --target 0 60 -45 0 30 0
python3 gravity_compensation.py --check --target 0 30 -80 0 -40 0
#    For J2, J3, J5 the measured/model ratio should be close to 1 with the same sign.
#    If J1-J3 read about 0.25, the firmware is <= V1.8-2: add --effort-x4.

# 2) Go back to the rest pose
python3 joint_position_ctrl.py --zero -y

# 3) Free one joint first, the others hold (keep this short -- see the warning below)
python3 gravity_compensation.py --joints 2
python3 gravity_compensation.py --joints 2 3

# 4) All joints free
python3 gravity_compensation.py
```

**Held joints are held by a MIT position spring (kp 10), and that is the fragile part.**
On 2026-09-21, `--joints 5` at the working pose (200 Hz) buzzed J2 at 1.3–2 N·m for six
seconds; then J2 took a jolt, its driver stopped producing torque, and J2 fell 33° before the
speed watchdog stopped the run (`gc_000.jsonl`). With every joint free the same arm was quiet
(0.15 N·m) and no driver dropped out in 66 s. Prefer step 4, run at 100 Hz, and keep a hand
under the arm whenever joints are being held.

Stopping: press Ctrl+C and the arm holds its pose. Then press Enter to move to the rest
pose and disable, or type `d` + Enter to disable at once (support the arm by hand). **Before
any position control afterwards, run `python3 arm_tool.py reset`.** That applies to step 2
and to the ROS bridge.

Tuning, per joint (`--tau-scale` and `--kd` take 1 or 6 values):

| symptom | fix |
|---|---|
| joint slowly sinks | raise its `--tau-scale` (e.g. 0.25 → 0.3), or `--payload` if holding something |
| joint drifts up | lower its `--tau-scale` |
| buzzing / oscillation | raise `--kd` slightly (0.3 → 0.5) |
| feels sticky / heavy to move | lower `--kd` |

Built-in safety:

- gains ramp in over `--ramp` s (first the feed-forward fades in while holding, then kp fades out)
- soft spring 5° inside each joint limit
- joint speed above `--max-vel` (3 rad/s) switches to position hold without feed-forward
- feedback older than 0.2 s stops the loop

The SDK encodes `t_ref` with 8 bits over ±8 (about 0.06 per step), so compensation on the
light wrist joints is coarse.

## Troubleshooting

- `CAN socket can0 does not exist`: the adapter isn't up. Run `bash scripts/can_activate.sh can0 1000000`.
- Enable fails, or position commands are ignored: check the e-stop. After MIT or teach mode, run `arm_tool.py reset` at the rest pose, then enable again.
- MoveIt says the start state deviates from the trajectory: tune `allowed_start_tolerance` in `piper_moveit_config/config/moveit_controllers.yaml`.
- The bridge aborts with `PATH_TOLERANCE_VIOLATED`: the arm couldn't follow the trajectory. Check for collisions and `speed_percent`, or raise `path_tolerance`.

## Provenance

These files come from [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros)
(humble branch, MIT licence):

- `piper_description`
- `piper.srdf`
- `pilz_cartesian_limits.yaml`
- `moveit.rviz`
- `scripts/can_activate.sh`

Local changes: MuJoCo and teach-pendant files removed, RViz fixed frame set to
`base_link`, `ready` pose added to the SRDF. Everything else in this repo is new.
