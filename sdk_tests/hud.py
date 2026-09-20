"""Cockpit overlay for the teleoperation camera window.

Teleoperating while reading a scrolling terminal does not work: your eyes are on the
gripper. Everything you need while flying the arm is drawn on the camera image instead --
how much room each joint has left, what the SpaceMouse is actually asking for, whether the
recorder is running, and any warning that would otherwise scroll past unseen.

draw() is a pure function of a frame and a telemetry dict, so the whole overlay can be
rendered and checked without a robot or a camera attached.
"""
import cv2
import numpy as np

# BGR, because OpenCV.
INK = (236, 240, 241)
DIM = (150, 160, 165)
GOOD = (120, 220, 120)
WARN = (60, 200, 250)
BAD = (80, 80, 245)
REC = (80, 80, 245)
ACCENT = (240, 200, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(frame, x0, y0, x1, y1, alpha=0.45):
    """Darken a rectangle so text stays readable over any scene."""
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def _text(frame, text, org, scale=0.4, colour=INK, weight=1):
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, colour, weight, cv2.LINE_AA)


def _headroom_colour(fraction):
    """fraction = how much of the joint's travel is left on the side it is heading for."""
    if fraction > 0.25:
        return GOOD
    return WARN if fraction > 0.1 else BAD


def _joint_bars(frame, joints, limits, x, y, width=112, row=15):
    """One track per joint with the current angle marked, and the mid-point ticked.

    A joint running out of range is the single most common reason the arm stops following,
    and it is invisible until it happens -- so it gets the most prominent part of the HUD.
    """
    _panel(frame, x - 8, y - 16, x + width + 42, y + row * len(joints) + 6)
    _text(frame, "JOINTS", (x - 4, y - 4), 0.36, DIM)
    for i, (q, (low, high)) in enumerate(zip(joints, limits)):
        top = y + i * row
        span = max(high - low, 1e-6)
        position = float(np.clip((q - low) / span, 0.0, 1.0))
        headroom = min(position, 1.0 - position) * 2.0
        colour = _headroom_colour(headroom)
        cv2.rectangle(frame, (x, top), (x + width, top + 7), (60, 60, 60), -1)
        # Filled from the middle outwards: the eye reads "how far from home" directly.
        middle = x + width // 2
        marker = int(x + position * width)
        cv2.rectangle(frame, (min(middle, marker), top), (max(middle, marker), top + 7), colour, -1)
        cv2.line(frame, (middle, top - 2), (middle, top + 9), DIM, 1)
        cv2.line(frame, (marker, top - 3), (marker, top + 10), INK, 2)
        _text(frame, f"J{i + 1}", (x - 26, top + 8), 0.36, colour)
        _text(frame, f"{np.degrees(q):+5.0f}", (x + width + 4, top + 8), 0.34, DIM)


def _gripper(frame, commanded, measured, maximum, x, y, width=92):
    _panel(frame, x - 8, y - 16, x + width + 10, y + 26)
    _text(frame, "GRIPPER", (x - 4, y - 4), 0.36, DIM)
    cv2.rectangle(frame, (x, y), (x + width, y + 9), (60, 60, 60), -1)
    filled = int(width * np.clip(measured / max(maximum, 1e-6), 0, 1))
    cv2.rectangle(frame, (x, y), (x + filled, y + 9), ACCENT, -1)
    target = int(x + width * np.clip(commanded / max(maximum, 1e-6), 0, 1))
    cv2.line(frame, (target, y - 3), (target, y + 12), INK, 2)
    _text(frame, f"{measured * 1000:.0f} -> {commanded * 1000:.0f} mm", (x, y + 24), 0.36)


def _stick(frame, linear, angular, cx, cy, radius=34):
    """What the SpaceMouse is asking for, in the frame you are controlling in.

    Up on the dial is "away from you", which is the direction a forward push means in
    whichever control frame is active -- so a push that comes out sideways here is telling
    you the frame is not the one you thought.
    """
    _panel(frame, cx - radius - 10, cy - radius - 22, cx + radius + 10, cy + radius + 26, 0.35)
    _text(frame, "INPUT", (cx - radius - 6, cy - radius - 10), 0.36, DIM)
    cv2.circle(frame, (cx, cy), radius, (90, 90, 90), 1)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (70, 70, 70), 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (70, 70, 70), 1)
    forward, left, up = float(linear[0]), float(linear[1]), float(linear[2])
    tip = (int(cx - left * radius), int(cy - forward * radius))
    if abs(forward) > 0.01 or abs(left) > 0.01:
        cv2.arrowedLine(frame, (cx, cy), tip, GOOD, 2, cv2.LINE_AA, tipLength=0.3)
    # Vertical is its own bar: a 2D dial cannot show three axes honestly.
    bar_x = cx + radius + 4
    cv2.line(frame, (bar_x, cy - radius), (bar_x, cy + radius), (70, 70, 70), 1)
    if abs(up) > 0.01:
        cv2.line(frame, (bar_x, cy), (bar_x, int(cy - up * radius)), GOOD, 3)
    if np.any(np.abs(angular) > 0.01):
        spin = "  ".join(n for n, v in zip("RPY", angular) if abs(v) > 0.01)
        _text(frame, spin, (cx - radius, cy + radius + 18), 0.38, ACCENT)


def _chip(frame, text, x, y, colour, filled=False):
    (w, h), _ = cv2.getTextSize(text, FONT, 0.38, 1)
    if filled:
        cv2.rectangle(frame, (x, y - h - 5), (x + w + 10, y + 5), colour, -1)
        _text(frame, text, (x + 5, y), 0.38, (20, 20, 20))
    else:
        cv2.rectangle(frame, (x, y - h - 5), (x + w + 10, y + 5), colour, 1)
        _text(frame, text, (x + 5, y), 0.38, colour)
    return x + w + 16


def draw(frame, t):
    """Return a copy of frame with the overlay drawn on it.

    t carries: joints, limits, tcp, error_m, gripper (cmd, measured, max), jogging,
    frame_mode, allow_lin, allow_rot, strict, speed_name, speed_scale, ori_frac, linear,
    angular, recording (None or {samples, seconds}), task, message.
    """
    frame = frame.copy()
    height, width = frame.shape[:2]

    # Top strip: what mode am I in, and is it recording.
    _panel(frame, 0, 0, width, 26, 0.5)
    x = _chip(frame, "FPS" if t["frame_mode"] == "tool" else "BASE", 8, 18, ACCENT)
    if t["strict"]:
        x = _chip(frame, "1-AXIS", x, 18, ACCENT)
    x = _chip(frame, "TRANS", x, 18, GOOD if t["allow_lin"] else DIM)
    x = _chip(frame, "ROT", x, 18, GOOD if t["allow_rot"] else DIM)
    x = _chip(frame, t["speed_name"], x, 18, WARN if t["speed_scale"] != 1.0 else DIM)
    if not t["jogging"]:
        _chip(frame, "HOLD - press SPACE", x, 18, BAD, filled=True)

    recording = t.get("recording")
    if recording:
        label = f"REC {int(recording['seconds']) // 60:d}:{int(recording['seconds']) % 60:02d}"
        (w, _), _ = cv2.getTextSize(label, FONT, 0.42, 2)
        cv2.circle(frame, (width - w - 26, 13), 5, REC, -1)
        _text(frame, label, (width - w - 14, 18), 0.42, REC, 2)
    else:
        _text(frame, "READY  R to record" if t.get("can_record") else "no recording",
              (width - 150, 18), 0.38, DIM)

    _joint_bars(frame, t["joints"], t["limits"], 40, 52)
    _gripper(frame, *t["gripper"], 40, height - 108)
    _stick(frame, t["linear"], t["angular"], width - 60, 88)

    # Bottom strip: the task being demonstrated, where the tool is, how far behind it runs.
    _panel(frame, 0, height - 42, width, height, 0.5)
    if t.get("task"):
        _text(frame, t["task"][:70], (8, height - 26), 0.38, DIM)
    tcp = t["tcp"]
    _text(frame, f"TCP  x{tcp[0]:+.3f}  y{tcp[1]:+.3f}  z{tcp[2]:+.3f} m", (8, height - 8), 0.4)
    lag_mm = t["error_m"] * 1000
    _text(frame, f"lag {lag_mm:3.0f} mm", (width - 190, height - 8), 0.4,
          INK if lag_mm < 15 else WARN if lag_mm < 25 else BAD)
    if t["ori_frac"] < 0.95:
        _text(frame, f"tilting {1 - t['ori_frac']:.0%}", (width - 100, height - 8), 0.4, WARN)

    # One line for whatever just happened, big and in the middle where it cannot be missed.
    if t.get("message"):
        (w, h), _ = cv2.getTextSize(t["message"][:52], FONT, 0.5, 2)
        _panel(frame, width // 2 - w // 2 - 12, height // 2 - h - 10,
               width // 2 + w // 2 + 12, height // 2 + 12, 0.6)
        _text(frame, t["message"][:52], (width // 2 - w // 2, height // 2), 0.5, WARN, 2)
    return frame
