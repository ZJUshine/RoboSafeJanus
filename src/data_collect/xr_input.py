"""遥操作用的运动手柄输入。

``XRoboToolkitInput`` 通过 ctypes 调用 XRoboToolkit PC 服务(``/opt/apps/roboticsservice``)
的 C SDK 读取 PICO 手柄,因此不需要编译 Python 绑定。``ScriptedControllerInput`` 播放
一段固定的手柄动作,便于在没有头显时测试"遥操作 -> 录制 -> 重放"整条链路。

两者的 ``read()`` 都返回 ``{"left": ControllerState | None, "right": ControllerState | None}``。
位姿为任务(世界)坐标系下的 ``[x, y, z, qw, qx, qy, qz]``;只有其变化量有意义,
遥操作会相对离合按下时的位姿来映射。
"""

import ctypes
from dataclasses import dataclass
import json
import logging
import threading
import time

import numpy as np
import transforms3d as t3d

log = logging.getLogger(__name__)

XRT_SDK_LIB = "/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so"


@dataclass
class ControllerState:
    pose: np.ndarray  # (7,) 世界坐标系
    trigger: float  # 0..1,控制夹爪
    squeeze: float  # 0..1,grip 键 = 离合
    button_0: float  # X(左手)/ A(右手)
    button_1: float  # Y(左手)/ B(右手)


# 以下常量来自 PXREARobotSDK.h
_SERVER_CONNECT, _SERVER_DISCONNECT, _DEVICE_STATE_JSON = 1 << 2, 1 << 3, 1 << 25
_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


class _DevStateJson(ctypes.Structure):
    _fields_ = [("devID", ctypes.c_char * 32), ("stateJson", ctypes.c_char * 16352)]


def _headset_to_world(yaw_deg):
    """PICO 追踪坐标系(x 向右、y 向上、z 指向用户)-> 操作者面向 +y 的任务坐标系的旋转矩阵。"""
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg)) @ base


def parse_tracking(state_json):
    """Tracking 消息的 ``stateJson`` -> ``{"left"/"right": 原始手柄字典}``(非追踪消息返回空字典)。"""
    msg = json.loads(state_json)
    if msg.get("functionName") != "Tracking":
        return {}
    value = msg.get("value", {})
    if isinstance(value, str):
        value = json.loads(value)
    ctrl = value.get("Controller") or {}
    return {hand: ctrl[hand] for hand in ("left", "right") if isinstance(ctrl.get(hand), dict)}


def _pose_xyzw(raw):
    pose = raw.get("pose")
    if isinstance(pose, str):
        pose = [float(v) for v in pose.split(",")]
    pose = np.asarray(pose, dtype=float)
    return pose if pose.shape == (7,) else None


class XRoboToolkitInput:
    """从 XRoboToolkit PC 服务读取最新的 PICO 手柄状态。

    SDK 在自己的线程里对每条追踪消息回调;``read()`` 只转换最新的一条。某只手的
    数据早于 ``stale_s`` 秒(头显休眠、手柄丢失)时返回 ``None``,遥操作随之松开该臂。
    """

    def __init__(self, operator_yaw=0.0, lib_path=XRT_SDK_LIB, stale_s=0.3):
        self._rot = _headset_to_world(operator_yaw)
        self._stale_s = stale_s
        self._lock = threading.Lock()
        self._latest = {}  # 手 -> (monotonic 时间, 原始字典)
        self._first_json = None
        self.connected = False
        self._lib = ctypes.CDLL(lib_path)
        self._lib.PXREAInit.argtypes = [ctypes.c_void_p, _CALLBACK, ctypes.c_uint]
        self._callback = _CALLBACK(self._on_message)  # 必须保留引用:SDK 持有的是裸指针
        mask = _SERVER_CONNECT | _SERVER_DISCONNECT | _DEVICE_STATE_JSON
        if self._lib.PXREAInit(None, self._callback, mask) != 0:
            raise RuntimeError("PXREAInit failed: is the XRoboToolkit PC service (RoboticsServiceProcess) running?")

    def _on_message(self, _ctx, kind, _status, data):
        try:
            if kind == _SERVER_CONNECT:
                self.connected = True
            elif kind == _SERVER_DISCONNECT:
                self.connected = False
            elif kind == _DEVICE_STATE_JSON:
                text = ctypes.cast(data, ctypes.POINTER(_DevStateJson)).contents.stateJson.decode(errors="replace")
                now = time.monotonic()
                hands = parse_tracking(text)
                with self._lock:
                    if self._first_json is None and hands:
                        self._first_json = text
                    for hand, raw in hands.items():
                        self._latest[hand] = (now, raw)
        except Exception:  # 绝不能让异常传进 C 线程
            log.exception("XRoboToolkit message dropped")

    def first_message(self):
        """第一条追踪消息的原始 JSON(用于核对服务端的数据格式)。"""
        with self._lock:
            return self._first_json

    def _state(self, raw):
        pose = _pose_xyzw(raw)
        if pose is None or not np.any(pose[3:]):
            return None
        # SDK 的位姿是 [x, y, z, qx, qy, qz, qw],transforms3d 需要 wxyz 顺序
        rot = t3d.quaternions.quat2mat([pose[6], pose[3], pose[4], pose[5]])
        world = np.concatenate([self._rot @ pose[:3], t3d.quaternions.mat2quat(self._rot @ rot @ self._rot.T)])
        return ControllerState(
            pose=world,
            trigger=float(raw.get("trigger", 0.0)),
            squeeze=float(raw.get("grip", 0.0)),
            button_0=float(bool(raw.get("primaryButton", False))),
            button_1=float(bool(raw.get("secondaryButton", False))),
        )

    def read(self):
        now = time.monotonic()
        with self._lock:
            latest = dict(self._latest)
        return {
            hand: self._state(latest[hand][1]) if hand in latest and now - latest[hand][0] < self._stale_s else None
            for hand in ("left", "right")
        }

    def close(self):
        self._lib.PXREADeinit()


class ScriptedControllerInput:
    """头显的确定性替身:右手按下离合、移动、闭合再张开夹爪,最后按 B 保存。"""

    def __init__(self, ticks=250):
        self.ticks = ticks
        self._tick = 0
        self._origin = np.array([0.3, -0.6, 1.0])

    def read(self):
        t, self._tick = self._tick, self._tick + 1
        n = self.ticks
        # 第 10 tick 起按住 grip,结束前 40 tick 松开;中间 40%~80% 段扣下 trigger
        engaged = 10 <= t < n - 40
        s = np.clip((t - 10) / (n - 60), 0.0, 1.0)
        offset = np.array([-0.08 * np.sin(np.pi * s), 0.06 * s, -0.10 * np.sin(np.pi * s)])
        quat = t3d.euler.euler2quat(0.0, 0.0, 0.4 * np.sin(np.pi * s))
        trigger = 1.0 if 0.4 < s < 0.8 else 0.0
        save = 1.0 if t == n - 10 else 0.0
        if t >= n - 1:
            self._tick = 0
        right = ControllerState(np.concatenate([self._origin + offset, quat]), trigger, float(engaged), 0.0, save)
        left = ControllerState(np.array([-0.3, -0.6, 1.0, 1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0, 0.0)
        return {"left": left, "right": right}

    def close(self):
        pass
