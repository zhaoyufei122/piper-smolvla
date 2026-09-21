"""Shared helpers for the Piper SDK test scripts.

Unit conventions used everywhere in this folder: joint angles in rad, velocities in
rad/s, torques in N*m, gripper width in m. The SDK itself uses 0.001 deg, 0.001 rad/s,
0.001 N*m and 0.001 mm, so all conversions live here.
"""
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from piper_sdk import C_PiperInterface_V2

# These scripts are often run through a pipe (an agent shell, tee, nohup), where stdout is
# block-buffered and progress would only appear at exit.
sys.stdout.reconfigure(line_buffering=True)

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]

# Joint limits from the SDK JointCtrl docstring [rad].
JOINT_LIMITS = np.array([
    [-2.6179, 2.6179],
    [0.0, 3.14],
    [-2.967, 0.0],
    [-1.745, 1.745],
    [-1.22, 1.22],
    [-2.09439, 2.09439],
])
GRIPPER_MAX_WIDTH = 0.07  # m, full opening of the standard gripper

MDEG_PER_RAD = 180000.0 / math.pi  # SDK joint unit is 0.001 deg


def connect(can_name="can0", timeout=3.0, settle=True):
    """Open the CAN port and wait for COMPLETE, settled joint feedback.

    The six joint angles arrive in three separate CAN frames (J1-2, J3-4, J5-6) and the SDK
    updates its message object in place, so a read taken the instant the first frame lands
    mixes fresh values with the zeros left from initialisation. Returning then is dangerous:
    a caller that anchors on that pose is anchoring on a pose the arm is not in. It has
    happened -- teleop read (-90, 59, -0.6, 0, 0.6, 0) instead of (-90, 60, -45, 0, 30, 0),
    took the tool to be 13 cm lower than it was, and drove the arm into the table.

    So wait for several frames to have landed AND for two consecutive reads to agree. Pass
    settle=False only where a stale first sample cannot hurt.
    """
    try:
        piper = C_PiperInterface_V2(can_name)
    except ConnectionError as exc:
        raise SystemExit(
            f"Cannot open CAN interface '{can_name}': {exc}\n"
            f"Plug in the USB-CAN adapter and run: bash scripts/can_activate.sh {can_name} 1000000"
        ) from exc
    piper.ConnectPort()
    t0 = time.time()
    stamps, previous, agreed = set(), None, 0
    while time.time() - t0 < timeout:
        stamp = piper.GetArmJointMsgs().time_stamp
        if stamp > 0:
            stamps.add(stamp)
            if not settle:
                return piper
            current = joint_positions(piper)
            # Every frame seen at least once, the gripper reporting too, and the pose the
            # same twice running: anything less can still be half a message.
            agreed = agreed + 1 if (previous is not None
                                    and np.max(np.abs(current - previous)) < 1e-4) else 0
            previous = current
            if len(stamps) >= 4 and agreed >= 2 and piper.GetArmGripperMsgs().time_stamp > 0:
                return piper
        time.sleep(0.01)
    if not stamps:
        raise SystemExit(
            f"'{can_name}' is up but no joint feedback arrived within {timeout:.1f} s.\n"
            "Check that the arm is powered on and the CAN bitrate is 1000000 (candump can0)."
        )
    raise SystemExit(
        f"joint feedback on '{can_name}' never settled within {timeout:.1f} s "
        f"(last read {np.degrees(previous).round(1)} deg).\n"
        "The arm is probably still moving, or frames are being dropped. Wait for it to stop "
        "and try again; check the bus with: candump can0"
    )


def feedback_age(piper):
    """Seconds since the last joint feedback frame."""
    return time.time() - piper.GetArmJointMsgs().time_stamp


def joint_positions(piper):
    js = piper.GetArmJointMsgs().joint_state
    raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return np.array(raw, dtype=float) / MDEG_PER_RAD


def joint_velocities(piper):
    hs = piper.GetArmHighSpdInfoMsgs()
    motors = [hs.motor_1, hs.motor_2, hs.motor_3, hs.motor_4, hs.motor_5, hs.motor_6]
    return np.array([m.motor_speed for m in motors], dtype=float) / 1000.0


def joint_efforts(piper, x4_j123=False):
    """Motor torque feedback [N*m].

    Firmware <= V1.8-2 reports J1-J3 effort 4x too small (see piper_sdk Q&A), so pass
    x4_j123=True on those versions.
    """
    hs = piper.GetArmHighSpdInfoMsgs()
    motors = [hs.motor_1, hs.motor_2, hs.motor_3, hs.motor_4, hs.motor_5, hs.motor_6]
    effort = np.array([m.effort for m in motors], dtype=float) / 1000.0
    if x4_j123:
        effort[:3] *= 4.0
    return effort


def gripper_state(piper):
    """Returns (width [m], effort [N*m])."""
    gs = piper.GetArmGripperMsgs().gripper_state
    return gs.grippers_angle / 1e6, gs.grippers_effort / 1000.0


def arm_status(piper):
    return piper.GetArmStatus().arm_status


def status_line(piper):
    s = arm_status(piper)
    return (f"ctrl_mode={s.ctrl_mode}  arm_status={s.arm_status}  move_mode={s.mode_feed}  "
            f"teach={s.teach_status}  enabled={piper.GetArmEnableStatus()}")


def firmware_version(piper, timeout=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.SearchPiperFirmwareVersion()
        time.sleep(0.05)
        version = piper.GetPiperFirmwareVersion()
        if isinstance(version, str):
            return version
    return "unknown"


# Driver status bits reported per joint (0x261..0x266, see the SDK message docstring).
FOC_FLAGS = ("voltage_too_low", "motor_overheating", "driver_overcurrent",
             "driver_overheating", "collision_status", "driver_error_status",
             "stall_status")


def driver_info(piper):
    """Per-joint driver state: enable flag, bus voltage, temperatures, raised fault bits.

    A single joint driver can trip its own protection and disable itself while the
    arm-level status stays NORMAL, so this is the only place that failure is visible: the
    joint looks alive and communicates, it just produces no torque.
    """
    low = piper.GetArmLowSpdInfoMsgs()
    info = []
    for i in range(1, 7):
        motor = getattr(low, f"motor_{i}")
        status = motor.foc_status
        info.append({
            "joint": i,
            "enabled": bool(status.driver_enable_status),
            "voltage": motor.vol / 10.0,
            "driver_temp": motor.foc_temp,
            "motor_temp": motor.motor_temp,
            "faults": [name for name in FOC_FLAGS if getattr(status, name)],
        })
    return info


def health(piper):
    """Arm status code, per-driver enable flag and raised fault bits, compact enough to trace.

    A driver that trips drops its own torque while the arm keeps talking: on 2026-09-21 J2
    went limp mid-run and fell 33 deg, and nothing recorded why. This is what would have.
    """
    info = driver_info(piper)
    return {"arm": int(arm_status(piper).arm_status),
            "on": [d["enabled"] for d in info],
            "faults": {str(d["joint"]): d["faults"] for d in info if d["faults"]}}


def health_problem(record):
    """Human-readable problem in a health() record, or None."""
    off = [i + 1 for i, on in enumerate(record["on"]) if not on]
    parts = []
    if record["arm"]:
        parts.append(f"arm status 0x{record['arm']:02X} ({FAULT_STATUSES.get(record['arm'], 'unknown')})")
    if off:
        parts.append("driver disabled on " + " ".join(f"J{j}" for j in off))
    parts += [f"J{j}: {', '.join(names)}" for j, names in record["faults"].items()]
    return "; ".join(parts) or None


def protection_levels(piper, timeout=1.0):
    """Collision protection level per joint: 0 = off, 1 = trips easiest, 8 = least sensitive."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.ArmParamEnquiryAndConfig(param_enquiry=0x02)
        time.sleep(0.1)
        feedback = piper.GetCrashProtectionLevelFeedback().crash_protection_level_feedback
        levels = [getattr(feedback, f"joint_{i}_protection_level") for i in range(1, 7)]
        if any(levels):
            return levels
    return None


# ArmStatus values that mean the arm will not move until the cause is cleared.
FAULT_STATUSES = {
    0x01: "emergency stop",
    0x05: "joint communication error",
    0x06: "joint brake not released",
    0x07: "collision detected",
    0x09: "joint status error",
    0x0A: "other error",
    0x0E: "controller over-temperature",
    0x0F: "brake resistor over-temperature",
}


def fault(piper):
    """Description of the current fault, or None when the arm is healthy."""
    return FAULT_STATUSES.get(int(arm_status(piper).arm_status))


def enable(piper, timeout=5.0, settle=0.5):
    """Enable all joint motors and confirm they stay enabled.

    The firmware keeps a joint target of its own, and after a power cycle that target is
    all zeros: the moment the motors are energised the arm drives to it, which from a
    working pose is a fast unexpected swing. So the current position is written as the
    target first, and kept up during the handshake, making it mean "stay where you are".

    Right after a reset the drivers also need a moment: they can report enabled and then
    drop out again, so the state is re-checked after a short settle.
    """
    hold = joint_positions(piper)
    width = gripper_state(piper)[0]

    def keep_target():
        send_joint_positions(piper, hold, speed_percent=20)
        send_gripper(piper, width)

    keep_target()
    time.sleep(0.05)
    t0 = time.time()
    while time.time() - t0 < timeout:
        keep_target()
        if piper.EnablePiper():
            time.sleep(settle)
            keep_target()
            if all(piper.GetArmEnableStatus()):
                return True
        time.sleep(0.05)
    return False


def disable(piper, timeout=5.0):
    """Disable all joint motors. The arm drops unless it rests on its stops!"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not piper.DisablePiper():
            return True
        time.sleep(0.05)
    return False


def clamp_joints(q):
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def send_joint_positions(piper, q, speed_percent=50):
    """Joint position setpoint in MOVE J mode (firmware does the low-level tracking)."""
    mdeg = [int(round(v * MDEG_PER_RAD)) for v in clamp_joints(q)]
    piper.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)
    piper.JointCtrl(*mdeg)


def send_gripper(piper, width, effort=1.0, powered=True):
    """Gripper width setpoint. powered=False de-energises the motor instead.

    The gripper servo self-oscillates at about 25 Hz whenever it is energised: the width
    reading does not move at all, but the torque swings 0.23 Nm peak to peak and the whole
    gripper buzzes. It does this with no command traffic at all, and the effort setting is a
    torque ceiling rather than a gain, so lowering it changes nothing. Cutting power is the
    only thing that stops it, and the drive holds its position when unpowered -- so the
    motor is only worth energising while it actually has to hold something.
    """
    width_um = int(round(np.clip(width, 0.0, GRIPPER_MAX_WIDTH) * 1e6))
    piper.GripperCtrl(width_um, int(round(effort * 1000)), 0x01 if powered else 0x00, 0)


def confirm(message, assume_yes=False):
    """Ask before the arm moves. Refuses instead of hanging when there is no terminal."""
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(f"{message}\n"
                         "No terminal available for the confirmation prompt. Run this in a "
                         "terminal, or pass -y if the workspace is clear and you are watching "
                         "the arm.")
    try:
        input(f"{message}\nPress Enter to continue (Ctrl+C to abort)...")
    except EOFError:
        raise SystemExit("aborted: no input available")


class KeyReader:
    """Non-blocking single-key input, for tuning gains while the arm is running.

    Does nothing when stdin is not a terminal. Ctrl+C still raises KeyboardInterrupt.
    """

    def __init__(self):
        self._fd = None
        self._saved = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self._fd = sys.stdin.fileno()
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error:
            self._fd = None

    def get(self):
        """The next pending key, or None."""
        if self._fd is None or not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.read(1)

    def stop(self):
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._fd = None


def open_trace(where, prefix):
    """<prefix>_NNN.jsonl in a directory, so runs never overwrite each other; or a .jsonl path."""
    where = Path(where).expanduser()
    if where.suffix == ".jsonl":
        where.parent.mkdir(parents=True, exist_ok=True)
        return where
    where.mkdir(parents=True, exist_ok=True)
    width = len(prefix) + 1
    used = [int(f.stem[width:width + 3]) for f in where.glob(f"{prefix}_[0-9][0-9][0-9].jsonl")]
    return where / f"{prefix}_{max(used) + 1 if used else 0:03d}.jsonl"


def min_jerk(q0, q1, duration, t):
    s = np.clip(t / duration, 0.0, 1.0) if duration > 0 else 1.0
    return q0 + (q1 - q0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def fmt(values, digits=1, width=7):
    return " ".join(f"{v:{width}.{digits}f}" for v in values)
