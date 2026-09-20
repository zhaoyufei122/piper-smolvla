# Piper hardware test protocol

Run the tests in order: T0 → T4. Each test says what to type, what should happen, and what
to write down. Stop and fix before moving on if a pass criterion fails.

Nothing below has been run on the real arm yet, so treat the "expected" values as
predictions to check, not as facts.

## Before you start

- Clear the table around the arm. Keep the e-stop within reach.
- `sudo apt install can-utils ethtool` (once).
- Terminal A is for the arm scripts. Keep a terminal B free for `arm_tool.py`.
- **Ctrl+C aborts any script.** At a confirmation prompt it just exits; during motion the
  arm stops at the last streamed setpoint and stays enabled.
- **Disabling the motors makes the arm fall.** Only disable at the zero (folded) pose,
  where gravity presses joints 2 and 3 into their hard stops, or while supporting it.
- Only one program may command the arm. Close the ROS bridge before running SDK scripts.

Test log:

| | |
|---|---|
| date | |
| firmware (from T0) | |
| gripper fitted | yes / no |
| tester | |

---

## T0 — CAN bring-up

```bash
cd ~/Desktop/Agilex_Piper
bash scripts/can_activate.sh can0 1000000     # asks for sudo
ip -details link show can0 | head -3
candump can0 | head -20                       # Ctrl+C after a moment
```

Expected: `can0` is UP with `bitrate 1000000`, and `candump` scrolls quickly (the arm
sends roughly 3000 frames/s on its own).

- [ ] Pass: interface up at 1 Mbit/s and frames arriving.

If `can_activate.sh` reports no CAN interface: check the USB-CAN adapter and arm power.
If `candump` shows nothing: the adapter is up but the arm is not talking — check the CAN
cable and the bitrate.

---

## T1 — Read feedback (read-only, motors stay off)

```bash
cd ~/Desktop/Agilex_Piper/sdk_tests
python3 read_joints.py --once
python3 read_joints.py                 # live table, Ctrl+C to quit
```

Expected: firmware string printed, `feedback age` under ~50 ms, six joint angles,
`enabled` all `False` at power-up.

Record the firmware version — it decides two things:

| firmware | consequence |
|---|---|
| < V1.5-2 | MIT mode unavailable, **T4 cannot run** |
| <= V1.8-2 | J1–J3 torque feedback is 4x too small, use `--effort-x4` in T1.3 and T4.1 |

**T1.1** With the arm at its folded rest pose, all six angles should read close to 0°.

- [ ] Pass: all joints within a few degrees of 0 at the rest pose.
- Record any joint that does not: joint ____ reads ____ °

**T1.2 Sign check by hand.** With the live view running and the motors off, gently move
each joint by hand (only if the arm is back-drivable; do not force it) and watch the
value change. Note which physical direction gives a positive number — you will compare
this against RViz in T3.2.

**T1.3 Torque feedback sanity.** In the live view, note the `effort` column at rest. With
motors off it should be near zero.

- [ ] Pass: six joints, six velocities, six efforts update live; gripper width shown.

---

## T2 — SDK position control

```bash
cd ~/Desktop/Agilex_Piper/sdk_tests
python3 arm_tool.py status
```

**T2.1 Small demo move.** Keep a hand near the e-stop.

```bash
python3 joint_position_ctrl.py --demo
```

The script prints the current pose and each waypoint, then waits for Enter before
enabling. It moves joint1 ±15° and joint6 +30°, then returns.

- [ ] Pass: motion is smooth (no jerk at the start), final error printed per waypoint is
      below ~0.5°, no `arm_status` warning.
- Record the largest final error: ____ °

**T2.2 Larger move to the ready pose.**

```bash
python3 joint_position_ctrl.py --target 0 60 -45 0 30 0
```

Expected: the arm lifts to a forward-reaching pose, roughly 0.32 m in front of the base
and 0.22 m above it.

- [ ] Pass: reaches the pose, error below ~0.5°.

**T2.3 Gripper.**

```bash
python3 joint_position_ctrl.py --target 0 60 -45 0 30 0 --gripper 40
python3 joint_position_ctrl.py --target 0 60 -45 0 30 0 --gripper 0
```

- [ ] Pass: gripper opens to about 40 mm and closes. Width is visible in `read_joints.py`.

**T2.4 Return and power down.**

```bash
python3 joint_position_ctrl.py --zero
python3 arm_tool.py disable
```

- [ ] Pass: the arm folds back to the rest pose, and after disabling it stays put instead
      of dropping (joints 2 and 3 are resting on their stops).

If enable fails at any point: check the e-stop, then `python3 arm_tool.py reset` at the
rest pose and try again.

---

## T3 — ROS 2 and MoveIt

```bash
cd ~/Desktop/Agilex_Piper/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

**T3.1 Simulated arm first (hardware not involved).**

```bash
ros2 launch piper_moveit_config piper_moveit.launch.py
```

In RViz: drag the interactive marker, then Plan & Execute. In a second terminal:

```bash
ros2 run piper_moveit_config moveit_goal.py --named ready
ros2 run piper_moveit_config moveit_goal.py --named zero
```

- [ ] Pass: RViz plans and executes, `moveit_goal.py` prints `success`.
      (This already passed on my side, so a failure here means an environment problem.)

**T3.2 Real arm, visualisation only — the URDF sign check.** Motors stay off.

```bash
ros2 launch piper_bridge display.launch.py
```

Move each joint by hand and watch the RViz model.

- [ ] Pass: the RViz model matches the real arm's pose, and each joint moves in the **same
      direction** as the real one.
- If a joint is mirrored, stop here and tell me which one — MoveIt would drive it the
  wrong way.

**T3.3 Real arm with MoveIt.** The bridge enables the motors on start and holds the
current pose.

```bash
ros2 launch piper_moveit_config piper_moveit.launch.py use_sim:=false
```

Start with scripted, small goals from a second terminal, then try RViz dragging:

```bash
ros2 run piper_moveit_config moveit_goal.py --joints 0 20 -20 0 0 0
ros2 run piper_moveit_config moveit_goal.py --named ready
ros2 run piper_moveit_config moveit_goal.py --gripper 40
ros2 run piper_moveit_config moveit_goal.py --named zero
```

- [ ] Pass: each goal prints `success`, the arm follows the planned path, and the bridge
      logs `goal reached`.
- Record any `PATH_TOLERANCE_VIOLATED` abort: at which motion? ____

Tuning if it aborts: the arm was lagging more than 0.5 rad behind the plan. Either slow
MoveIt down (`--vel 0.1 --acc 0.1`) or raise the bridge tolerance
(`-p path_tolerance:=0.8`).

**T3.4 Stop mid-motion.** Start a long motion in RViz and press **Stop** on the
MotionPlanning panel.

- [ ] Pass: the arm stops promptly and holds position; the bridge logs `canceled, holding
      the current pose`.

Shut down with Ctrl+C in the launch terminal. The motors stay enabled and keep holding —
that is intentional. To power down: `python3 sdk_tests/joint_position_ctrl.py --zero`
then `python3 sdk_tests/arm_tool.py disable`.

---

## T4 — Gravity compensation (MIT)

Only run this with firmware >= V1.5-2. Keep one hand near the arm throughout.

### T4.1 Validate the model against the motors (no MIT yet, position mode)

This arm carries a **Pika gripper**, not the standard one, so every command below selects
`--tool pika` plus the payload fitted from measurements (see "Model calibration" at the end
of this file). `--effort-x4` is required by firmware S-V1.7-3.

```bash
cd ~/Desktop/Agilex_Piper/sdk_tests
MODEL="--tool pika --payload 0.42 --payload-x 0.03 --payload-z 0.07 --effort-x4"
python3 gravity_compensation.py --check $MODEL --target 0 60 -45 0 30 0      # P1
python3 gravity_compensation.py --check $MODEL --target 0 30 -80 0 -40 0     # P2
python3 gravity_compensation.py --check $MODEL --target 45 90 -60 0 0 0      # P3
```

**Never calibrate at the zero pose.** Joints 2 and 3 rest on their hard stops there, so the
stops carry most of the load and the motors read far less than the model predicts (measured
0.63 N·m against a 2.7 N·m model — which looked like a neat 0.93 ratio after the x4, and was
pure coincidence).

Each command asks for confirmation, moves there, then prints model vs measured torque at
about 5 Hz. Model predictions (N·m):

| pose | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| zero (rest) | 0.00 | 2.26 | −5.41 | −0.01 | −1.26 | 0.00 |
| P1 `0 60 -45 0 30 0` | 0.00 | −2.06 | −5.10 | −0.01 | −0.89 | 0.00 |
| P2 `0 30 -80 0 -40 0` | 0.00 | 4.11 | −2.07 | −0.01 | 0.00 | 0.00 |
| P3 `45 90 -60 0 0 0` | 0.00 | −5.99 | −5.07 | −0.01 | −1.09 | 0.00 |

J1 and J6 are always 0: J1's axis is vertical and J6 carries almost no offset mass, so
gravity puts no load on them. Judge the model on **J2, J3 and J5** (P2 is the one that
flips J2's sign, so it is the best sign check; P2 says nothing useful about J5).

Fill in the measured/model ratio:

| | J2 | J3 | J5 |
|---|---|---|---|
| P1 ratio | | | |
| P2 ratio | | | |
| P3 ratio | | | |

- [ ] Pass: ratios are roughly 1 (say 0.7–1.4) and the **signs match**.
- Ratios near 0.25 on J1–J3 mean old firmware: rerun with `--effort-x4`.
- Ratios far off on one joint only: that link's URDF mass is off, or something is mounted
  on the arm. Note it and tell me.
- Opposite sign: stop. Do not run T4.2 — tell me first.

Friction adds a roughly constant offset, so do not expect an exact match.

### T4.2 One joint free

```bash
python3 joint_position_ctrl.py --zero -y      # start from the rest pose
python3 gravity_compensation.py --joints 2
```

Gains ramp in over 4 s: first the feed-forward torque fades in while the arm holds, then
the stiffness fades out and joint 2 goes free.

Expected at the moment it goes free: **joint 2 drifts up by about 5°**. That is the soft
limit spring — at the rest pose joint 2 sits exactly on its lower stop, and the script
keeps a 5° margin. It is not a fault.

Then lift the forearm by hand:

- [ ] joint 2 feels roughly weightless and stays where you leave it
- [ ] it sinks slowly → raise its scale: `--tau-scale 0.25 0.30 0.25 0.25 0.25 0.25`
- [ ] it rises on its own → lower the second value instead
- [ ] it buzzes or oscillates → add damping: `--kd 0.5`
- Record the value that worked for J2: ____

Exit with Ctrl+C, then Enter (returns to the rest pose and disables).

### T4.3 Add joints

Repeat T4.2 with `--joints 2 3`, then `--joints 2 3 5`, carrying over the scales you
found. Record each joint's final scale:

| | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| final `--tau-scale` | 0.25 | | | 0.25 | | 0.25 |

J1, J4 and J6 carry no gravity load, so they are free with damping only — nothing to tune.

### T4.4 All joints free

```bash
python3 gravity_compensation.py --tau-scale <your six values>
```

- [ ] Pass: you can drag the arm around the workspace, it holds position when released,
      and it does not collapse or climb.
- Watch the printed `tau_model` line as you move it — it should change smoothly.

Safety nets that may trigger (expected behaviour, not faults): moving a joint faster than
3 rad/s drops into position hold, and a joint pushed within 5° of a limit springs back.

### T4.5 Recover to position control — do not skip

```
Ctrl+C  →  Enter        # holds, then returns to the rest pose and disables
```

```bash
python3 arm_tool.py reset
python3 joint_position_ctrl.py --demo
```

- [ ] Pass: after `reset`, position control works again. Without the reset the arm stays
      in MIT mode and ignores position commands — this also applies to the ROS bridge.

---

## Model calibration (this arm, 2026-09-17)

The arm has a **Pika gripper**. Three things came out of the first measurements:

- `--tool pika` loads `piper_pika_gripper.urdf` (copied from `~/Desktop/Agilex_UMI`, meshes
  included). Its end-effector is 0.264 kg, not the 0.507 kg of the standard gripper, and it
  also carries AgileX's newer link2/link3 centres of mass.
- Even so the arm measured **heavier** than that model. Fitting the P3 measurement
  (J2 −6.25, J3 −4.77, J5 −1.14 N·m) gives an extra **0.42 kg at (x=0.03, z=0.07) m** in the
  link6 frame — most likely the D405 and its mount, which the URDF does not include. With it
  the three joints fit to 0.23 N·m RMS. This is a one-pose fit: confirm it at P1 and P2.
- **Only `--check` validates the model.** It holds the pose in position mode, where the
  torque feedback is independent. In MIT mode the feedback tracks the torque we command, so
  model and "measurement" always agree — a static MIT sample showed J2 3.14 against a 3.14
  model, which proves nothing.

## Recovery reference

| symptom | what to do |
|---|---|
| arm must stop now | e-stop, or `python3 arm_tool.py stop` |
| after an e-stop or MIT/teach mode | `python3 arm_tool.py reset` at the rest pose, then enable |
| enable fails | check e-stop, then `reset`, then retry |
| position commands ignored | still in MIT mode → `reset` |
| arm drifts or climbs in T4 | Ctrl+C (it holds), then Enter to park and disable |
| `CAN socket can0 does not exist` | `bash scripts/can_activate.sh can0 1000000` |
| ROS and SDK fighting each other | close one; only one program may command the arm |

## What to send me afterwards

1. The firmware string from T0.
2. The T4.1 ratio table (and whether you needed `--effort-x4`).
3. The final `--tau-scale` values from T4.3.
4. Any error text you did not expect, plus which test it came from.
