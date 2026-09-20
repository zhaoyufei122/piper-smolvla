#!/usr/bin/env python3
"""3Dconnexion SpaceMouse driver (hidapi with an evdev fallback).

Vendored from ~/Desktop/SpaceMouse/spacemouse.py on 2026-09-17 so this repo is
self-contained. Edit this copy; the Desktop original is no longer the source of truth.

The device is exclusive: only one process can hold it. If open() fails while
hid.enumerate() still lists the device, something else has it open (check with
`fuser -v /dev/bus/usb/*/*`).
"""
"""
3Dconnexion SpaceMouse 驱动与读取库
支持 hidapi (USB 原生 HID) 与 evdev (Linux Input Event) 双后端
提供线程安全的 6-DoF (X, Y, Z, Roll, Pitch, Yaw) 及按键状态获取
"""

import sys
import time
import struct
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# 尝试导入底层支持库
HAS_HIDAPI = False
HAS_EVDEV = False

try:
    import hid
    HAS_HIDAPI = True
except ImportError:
    pass

try:
    import evdev
    from evdev import ecodes
    HAS_EVDEV = True
except ImportError:
    pass


@dataclass
class SpaceMouseState:
    # 平移轴 [-1.0, 1.0]
    x: float = 0.0      # 左右 (Left - / Right +)
    y: float = 0.0      # 前后 (Forward + / Backward -)
    z: float = 0.0      # 上下 (Up + / Down -)
    
    # 旋转轴 [-1.0, 1.0]
    roll: float = 0.0   # 侧倾翻滚 (Tilt Left / Right)
    pitch: float = 0.0  # 俯仰 (Tilt Forward / Backward)
    yaw: float = 0.0    # 偏航扭转 (Twist Clockwise / Counter-Clockwise)

    # 原始数值 (典型范围约 [-350, 350] 或 [-500, 500])
    raw_x: int = 0
    raw_y: int = 0
    raw_z: int = 0
    raw_roll: int = 0
    raw_pitch: int = 0
    raw_yaw: int = 0

    # 按键状态: {0: True/False, 1: True/False, ...}
    buttons: Dict[int, bool] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.buttons is None:
            self.buttons = {0: False, 1: False}


class SpaceMouse:
    VENDOR_ID = 0x256F  # 3Dconnexion
    # 常见型号 PID 映射
    KNOWN_PIDS = {
        0xC635: "SpaceMouse Compact",
        0xC62E: "SpaceMouse Wireless (Cable)",
        0xC652: "SpaceMouse Wireless (Receiver)",
        0xC62B: "SpaceMouse Pro",
        0xC631: "SpaceMouse Pro Wireless (Cable)",
        0xC632: "SpaceMouse Pro Wireless (Receiver)",
        0xC626: "SpaceNavigator",
        0xC628: "SpaceNavigator for Notebooks",
        0xC627: "SpaceExplorer",
        0xC633: "SpacePilot Pro",
    }
    DEFAULT_SCALE = 350.0  # 归一化缩放系数

    def __init__(self, scale: float = 350.0, deadzone: float = 0.02):
        self.scale = scale
        self.deadzone = deadzone
        self.state = SpaceMouseState()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.device_name = "Unknown SpaceMouse"
        self.backend = None
        self._dev = None

    def _normalize(self, val: int) -> float:
        norm = val / self.scale
        norm = max(-1.0, min(1.0, norm))
        if abs(norm) < self.deadzone:
            return 0.0
        return norm

    def open(self) -> bool:
        """尝试连接 SpaceMouse，优先 hidapi，备选 evdev"""
        # 1. 尝试使用 hidapi
        if HAS_HIDAPI:
            try:
                for dev_info in hid.enumerate(self.VENDOR_ID):
                    pid = dev_info['product_id']
                    self.device_name = self.KNOWN_PIDS.get(pid, f"SpaceMouse (0x{pid:04X})")
                    self._dev = hid.device()
                    try:
                        self._dev.open_path(dev_info['path'])
                    except Exception:
                        self._dev.open(self.VENDOR_ID, pid)
                    self._dev.set_nonblocking(True)
                    self.backend = "hidapi"
                    return True
            except Exception:
                self._dev = None

        # 2. 尝试使用 evdev
        if HAS_EVDEV:
            try:
                for path in evdev.list_devices():
                    dev = evdev.InputDevice(path)
                    if dev.info.vendor == self.VENDOR_ID or "spacemouse" in dev.name.lower():
                        self.device_name = dev.name
                        self._dev = dev
                        self.backend = "evdev"
                        return True
            except Exception:
                self._dev = None

        return False

    def start(self):
        """在后台线程中启动数据读取"""
        if not self.open():
            raise RuntimeError(
                "无法打开 SpaceMouse 设备！\n"
                "可能原因：\n"
                "1. 设备未插入 USB\n"
                "2. 权限不足：普通用户无权读取 USB HID / evdev。\n"
                "解决方法：\n"
                "  - 执行: sudo bash setup_udev.sh 配置免密权限\n"
                "  - 或执行: sudo python3 test_spacemouse.py"
            )

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止读取并关闭设备"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    def get_state(self) -> SpaceMouseState:
        """获取当前最新的 6-DoF 和按键状态 (线程安全)"""
        with self._lock:
            return SpaceMouseState(
                x=self.state.x,
                y=self.state.y,
                z=self.state.z,
                roll=self.state.roll,
                pitch=self.state.pitch,
                yaw=self.state.yaw,
                raw_x=self.state.raw_x,
                raw_y=self.state.raw_y,
                raw_z=self.state.raw_z,
                raw_roll=self.state.raw_roll,
                raw_pitch=self.state.raw_pitch,
                raw_yaw=self.state.raw_yaw,
                buttons=dict(self.state.buttons),
                timestamp=self.state.timestamp
            )

    def _read_loop(self):
        if self.backend == "hidapi":
            self._read_loop_hidapi()
        elif self.backend == "evdev":
            self._read_loop_evdev()

    def _read_loop_hidapi(self):
        while self._running:
            try:
                # 读取报文
                data = self._dev.read(64, timeout_ms=10)
                if not data:
                    time.sleep(0.002)
                    continue

                report_id = data[0]
                now = time.time()

                with self._lock:
                    self.state.timestamp = now
                    # 报文 1: 平移 (X, Y, Z)
                    if report_id == 1 and len(data) >= 7:
                        x, y, z = struct.unpack('<hhh', bytes(data[1:7]))
                        self.state.raw_x = x
                        self.state.raw_y = -y  # 常见坐标系对齐
                        self.state.raw_z = -z
                        self.state.x = self._normalize(self.state.raw_x)
                        self.state.y = self._normalize(self.state.raw_y)
                        self.state.z = self._normalize(self.state.raw_z)

                    # 报文 2: 旋转 (Pitch, Roll, Yaw)
                    elif report_id == 2 and len(data) >= 7:
                        pitch, roll, yaw = struct.unpack('<hhh', bytes(data[1:7]))
                        self.state.raw_pitch = -pitch
                        self.state.raw_roll = -roll
                        self.state.raw_yaw = yaw
                        self.state.pitch = self._normalize(self.state.raw_pitch)
                        self.state.roll = self._normalize(self.state.raw_roll)
                        self.state.yaw = self._normalize(self.state.raw_yaw)

                    # 报文 3: 按键 (Buttons)
                    elif report_id == 3 and len(data) >= 2:
                        btn_byte = data[1]
                        self.state.buttons[0] = bool(btn_byte & 0x01)
                        self.state.buttons[1] = bool(btn_byte & 0x02)
                        # 支持更多按键（如 Pro 型号）
                        for b in range(2, 16):
                            byte_idx = 1 + (b // 8)
                            bit_idx = b % 8
                            if byte_idx < len(data):
                                self.state.buttons[b] = bool(data[byte_idx] & (1 << bit_idx))

            except Exception as e:
                if not self._running:
                    break
                time.sleep(0.01)

    def _read_loop_evdev(self):
        while self._running:
            try:
                event = self._dev.read_one()
                if event is None:
                    time.sleep(0.002)
                    continue

                with self._lock:
                    self.state.timestamp = event.timestamp()
                    # 轴事件
                    if event.type == ecodes.EV_ABS:
                        if event.code == ecodes.ABS_X:
                            self.state.raw_x = event.value
                            self.state.x = self._normalize(event.value)
                        elif event.code == ecodes.ABS_Y:
                            self.state.raw_y = -event.value
                            self.state.y = self._normalize(-event.value)
                        elif event.code == ecodes.ABS_Z:
                            self.state.raw_z = -event.value
                            self.state.z = self._normalize(-event.value)
                        elif event.code == ecodes.ABS_RX:
                            self.state.raw_pitch = -event.value
                            self.state.pitch = self._normalize(-event.value)
                        elif event.code == ecodes.ABS_RY:
                            self.state.raw_roll = -event.value
                            self.state.roll = self._normalize(-event.value)
                        elif event.code == ecodes.ABS_RZ:
                            self.state.raw_yaw = event.value
                            self.state.yaw = self._normalize(event.value)

                    # 按键事件
                    elif event.type == ecodes.EV_KEY:
                        # BTN_0 / BTN_1 或其他按键映射
                        btn_idx = 0 if event.code in (ecodes.BTN_0, 256) else (1 if event.code in (ecodes.BTN_1, 257) else event.code)
                        self.state.buttons[btn_idx] = bool(event.value)

            except Exception:
                if not self._running:
                    break
                time.sleep(0.01)
