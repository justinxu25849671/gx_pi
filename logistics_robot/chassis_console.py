"""与地面站相同的底盘位置命令。

地面站位置页签发出 ``motion fwdmm|bwdmm|leftmm|rightmm <rpm> <mm> <acc>``，
行尾是 ``\\r\\n``。这些命令只驱动底盘，不包含机械臂。
"""

from __future__ import annotations

import re
import time
from typing import Callable, Protocol


POSITION_KINDS = ("fwdmm", "bwdmm", "leftmm", "rightmm")
DONE_RESULTS = ("done", "done_yaw_drift")

_POSITION_ACK = re.compile(
    r"\[motion\] (fwdmm|bwdmm|leftmm|rightmm) rpm=(-?\d+) acc=(-?\d+) ret=(-?\d+)")
_STOP_ACK = re.compile(r"\[motion\] stop ret=(-?\d+)")
_STATUS = re.compile(r"\[motion\] type=(-?\d+) rpm=(-?\d+) elapsed=\d+ms")
_RESULT = re.compile(
    r"\[motion\] result seq=\d+ type=-?\d+ code=-?\d+\(([a-z_?]+)\) "
    r"target=\d+mm actual=-?\d+mm err=-?\d+mm")


def position_command(kind: str, rpm: int, distance_mm: int, acceleration: int) -> str:
    """生成与地面站 ``positionCommand`` 相同的一行，不含换行。"""
    if kind not in POSITION_KINDS:
        raise ValueError("未知位置动作")
    _require_int(rpm, "转速")
    _require_int(distance_mm, "距离")
    _require_int(acceleration, "位置加速度档")
    if not 1 <= rpm <= 5000:
        raise ValueError("转速须在 1..5000")
    if distance_mm < 1:
        raise ValueError("距离至少 1 mm")
    if not 0 <= acceleration <= 255:
        raise ValueError("位置加速度档须在 0..255")
    if rpm >= 100 and not 1 <= acceleration <= 64:
        raise ValueError("转速达到 100 rpm 时，位置加速度档须在 1..64")
    return f"motion {kind} {rpm} {distance_mm} {acceleration}"


def parse_position_ret(text: str, kind: str) -> int | None:
    """返回指定方向最近一条位置命令的 ``ret``；还没出现则返回 ``None``。"""
    found = [item for item in _POSITION_ACK.finditer(text) if item.group(1) == kind]
    if not found:
        return None
    return int(found[-1].group(4))


def parse_stop_ret(text: str) -> int | None:
    found = list(_STOP_ACK.finditer(text))
    if not found:
        return None
    return int(found[-1].group(1))


def parse_status_type(text: str) -> int | None:
    """``type=0`` 表示底盘空闲。"""
    found = list(_STATUS.finditer(text))
    if not found:
        return None
    return int(found[-1].group(1))


def parse_result_name(text: str) -> str | None:
    found = list(_RESULT.finditer(text))
    if not found:
        return None
    return found[-1].group(1)


def _require_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}必须是整数")


class ConsoleTransport(Protocol):
    def write_line(self, line: str) -> None: ...

    def read_available(self) -> str: ...


class SerialConsoleTransport:
    """连接 USART1 控制台。不要接到二进制路径口。"""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("未安装 pyserial，请先执行 pip install -r requirements.txt") from exc
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, timeout=0.05, write_timeout=0.2)

    def write_line(self, line: str) -> None:
        payload = (line + "\r\n").encode("ascii")
        written = self._serial.write(payload)
        if written != len(payload):
            raise ConnectionError(f"控制台短写：{written}/{len(payload)}")

    def read_available(self) -> str:
        waiting = getattr(self._serial, "in_waiting", 0) or 0
        chunk = self._serial.read(waiting or 1)
        if not chunk:
            return ""
        return chunk.decode("utf-8", errors="replace")

    def close(self) -> None:
        self._serial.close()


class ConsoleChassisLink:
    """按地面站位置命令移动底盘，并查询状态直到空闲。"""

    def __init__(self, transport: ConsoleTransport, rpm: int, acceleration: int,
                 response_timeout_s: float = 1.0, sleep: Callable[[float], None] | None = None) -> None:
        self._transport = transport
        self._rpm = rpm
        self._acceleration = acceleration
        self._timeout = response_timeout_s
        self._sleep = sleep or time.sleep
        self._rx = ""
        position_command("fwdmm", rpm, 1, acceleration)

    def move(self, kind: str, distance_mm: int) -> None:
        line = position_command(kind, self._rpm, distance_mm, self._acceleration)
        ret = self._request(line, lambda text: parse_position_ret(text, kind))
        if ret != 0:
            raise RuntimeError(f"{line} 被拒绝，ret={ret}")

    def poll(self):
        from .alignment import ActionResult

        motion_type = self._request("motion status", parse_status_type)
        if motion_type != 0:
            return ActionResult.BUSY
        name = self._request("motion result", parse_result_name)
        if name in DONE_RESULTS:
            return ActionResult.SUCCEEDED
        if name == "none":
            return ActionResult.BUSY
        return ActionResult.FAILED

    def stop(self) -> None:
        ret = self._request("motion stop", parse_stop_ret)
        if ret != 0:
            raise RuntimeError(f"motion stop 被拒绝，ret={ret}")

    def _request(self, line: str, parse):
        mark = len(self._rx)
        self._transport.write_line(line)
        deadline = time.monotonic() + self._timeout
        while True:
            chunk = self._transport.read_available()
            if chunk:
                self._rx += chunk
            fresh = self._rx[mark:]
            if "[motion] rejected:" in fresh:
                raise RuntimeError(fresh.strip())
            parsed = parse(fresh)
            if parsed is not None:
                if len(self._rx) > 4096:
                    self._rx = self._rx[-2048:]
                return parsed
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待回复超时：{line}")
            self._sleep(0.01)
