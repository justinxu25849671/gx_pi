"""STM32 九点绝对航点二进制协议的模型、编解码与增量解帧。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Iterable, Union


MAGIC = b"\xAA\x55"
VERSION = 0x01
MAX_PAYLOAD = 110


class PathCommand(IntEnum):
    CLEAR = 0x10
    UPLOAD = 0x11
    START = 0x12
    STOP = 0x13
    STATUS_REQ = 0x14
    RESET_ORIGIN = 0x15
    ARM_JOG = 0x20
    ARM_STOP = 0x21
    ARM_STATUS_REQ = 0x22


class PathEventCode(IntEnum):
    ACK = 0x90
    NACK = 0x91
    POINT_DONE = 0x92
    DONE = 0x93
    ERROR = 0x94
    STATUS = 0x95
    ARM_STATUS = 0xA0
    ARM_DONE = 0xA1


class PathState(IntEnum):
    IDLE = 0
    LOADED = 1
    RUN_X = 2
    RUN_Y = 3
    STOPPED = 4
    ERROR = 5


class ArmState(IntEnum):
    IDLE = 0
    MOVING = 1
    STOPPED = 2
    ERROR = 3


class PathSegment(IntEnum):
    X = 0
    Y = 1
    MANAGEMENT = 2


class PathResult(IntEnum):
    OK = 0x00
    BAD_FRAME = 0x01
    BAD_COMMAND = 0x02
    BAD_PARAM = 0x03
    NO_PATH = 0x04
    PATH_ID_MISMATCH = 0x05
    BUSY = 0x06
    MOTION_COMMAND_FAILED = 0x07
    MOTION_TIMEOUT = 0x08
    ENCODER_FAILED = 0x09
    STOPPED = 0x0A


@dataclass(frozen=True)
class PathPoint:
    x_mm: int
    y_mm: int
    rpm: int = 20
    acceleration: int = 10
    flags: int = 0

    def validate(self) -> None:
        if not (-10000 <= self.x_mm <= 10000 and -10000 <= self.y_mm <= 10000):
            raise ValueError("航点坐标必须在 -10000..10000 mm")
        if not 1 <= self.rpm <= 5000:
            raise ValueError("航点 rpm 必须在 1..5000")
        if not 0 <= self.acceleration <= 255:
            raise ValueError("航点 acceleration 必须在 0..255")
        if self.flags != 0:
            raise ValueError("当前协议要求 flags=0")

    def encode(self) -> bytes:
        self.validate()
        return struct.pack("<iiHBB", self.x_mm, self.y_mm, self.rpm,
                           self.acceleration, self.flags)


@dataclass(frozen=True)
class Frame:
    command: int
    sequence: int
    payload: bytes


@dataclass(frozen=True)
class AckEvent:
    request_sequence: int
    path_id: int
    result: PathResult
    accepted: bool


@dataclass(frozen=True)
class PointDoneEvent:
    path_id: int
    point_index: int
    target_x_mm: int
    target_y_mm: int
    estimated_x_mm: int
    estimated_y_mm: int


@dataclass(frozen=True)
class DoneEvent:
    path_id: int
    count: int
    estimated_x_mm: int
    estimated_y_mm: int


@dataclass(frozen=True)
class ErrorEvent:
    path_id: int
    point_index: int
    segment: PathSegment
    error: PathResult
    estimated_x_mm: int
    estimated_y_mm: int


@dataclass(frozen=True)
class StatusEvent:
    path_id: int
    state: PathState
    point_index: int
    segment: PathSegment
    count: int
    estimated_x_mm: int
    estimated_y_mm: int
    active_target_x_mm: int
    active_target_y_mm: int


@dataclass(frozen=True)
class ArmStatusEvent:
    request_sequence: int
    motor_id: int
    state: ArmState
    result: PathResult
    encoder_count: int
    reverse: bool


@dataclass(frozen=True)
class ArmDoneEvent:
    request_sequence: int
    motor_id: int
    result: PathResult
    encoder_count: int
    reverse: bool


PathEvent = Union[AckEvent, PointDoneEvent, DoneEvent, ErrorEvent, StatusEvent,
                  ArmStatusEvent, ArmDoneEvent]


def encode_arm_jog(motor_id: int, direction: int, pulses: int) -> bytes:
    """原始方向符号尚未标定；每次最多 32 个位置脉冲。"""
    if motor_id not in (5, 6, 7) or direction not in (-1, 1) or not 1 <= pulses <= 32:
        raise ValueError("点动只允许 5/6/7 号、方向 +/-、1..32 脉冲")
    return struct.pack("<BBH", motor_id, 0 if direction > 0 else 1, pulses)


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if crc & 1 else crc >> 1
    return crc


def encode_frame(command: int | IntEnum, sequence: int, payload: bytes = b"") -> bytes:
    if not 1 <= sequence <= 255:
        raise ValueError("sequence 必须在 1..255，0 保留不用")
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload 不能超过 {MAX_PAYLOAD} 字节")
    body = struct.pack("<BBBH", VERSION, int(command), sequence, len(payload)) + payload
    return MAGIC + body + struct.pack("<H", crc16_modbus(body))


def encode_upload(path_id: int, points: Iterable[PathPoint]) -> bytes:
    points_tuple = tuple(points)
    if not 1 <= path_id <= 255:
        raise ValueError("path_id 必须在 1..255")
    if not 1 <= len(points_tuple) <= 9:
        raise ValueError("路径必须包含 1..9 个航点")
    return bytes((path_id, len(points_tuple))) + b"".join(point.encode() for point in points_tuple)


class FrameDecoder:
    """保留半帧并从噪声、坏 CRC 或坏版本后恢复的增量解帧器。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.invalid_frames = 0

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[Frame]:
        self._buffer.extend(data)
        frames: list[Frame] = []
        while True:
            marker = self._buffer.find(MAGIC)
            if marker < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer.endswith(MAGIC[:1]) else b""
                break
            if marker:
                del self._buffer[:marker]
            if len(self._buffer) < 7:
                break
            payload_length = int.from_bytes(self._buffer[5:7], "little")
            if payload_length > MAX_PAYLOAD:
                self.invalid_frames += 1
                del self._buffer[0]
                continue
            frame_length = 9 + payload_length
            if len(self._buffer) < frame_length:
                # 截断帧后若已经出现一条 CRC 完整的新帧，优先恢复新帧；
                # 只有 AA55 恰好位于合法 payload 且新帧也 CRC 正确时才可能歧义。
                next_marker = self._buffer.find(MAGIC, 2)
                while next_marker >= 0 and len(self._buffer) - next_marker >= 7:
                    nested_length = int.from_bytes(
                        self._buffer[next_marker + 5:next_marker + 7], "little")
                    nested_total = 9 + nested_length
                    if nested_length <= MAX_PAYLOAD and len(self._buffer) - next_marker >= nested_total:
                        nested = bytes(self._buffer[next_marker:next_marker + nested_total])
                        if nested[2] == VERSION and int.from_bytes(nested[-2:], "little") == crc16_modbus(nested[2:-2]):
                            self.invalid_frames += 1
                            del self._buffer[:next_marker]
                            break
                    next_marker = self._buffer.find(MAGIC, next_marker + 2)
                else:
                    break
                continue
            candidate = bytes(self._buffer[:frame_length])
            body = candidate[2:-2]
            received_crc = int.from_bytes(candidate[-2:], "little")
            if candidate[2] != VERSION or received_crc != crc16_modbus(body):
                self.invalid_frames += 1
                del self._buffer[0]
                continue
            frames.append(Frame(candidate[3], candidate[4], candidate[7:-2]))
            del self._buffer[:frame_length]
        return frames


def parse_event(frame: Frame) -> PathEvent:
    try:
        code = PathEventCode(frame.command)
    except ValueError as exc:
        raise ValueError(f"未知路径事件 0x{frame.command:02X}") from exc
    payload = frame.payload
    expected = {
        PathEventCode.ACK: 3, PathEventCode.NACK: 3,
        PathEventCode.POINT_DONE: 18, PathEventCode.DONE: 10,
        PathEventCode.ERROR: 12, PathEventCode.STATUS: 21,
        PathEventCode.ARM_STATUS: 9, PathEventCode.ARM_DONE: 8,
    }[code]
    if len(payload) != expected:
        raise ValueError(f"事件 {code.name} payload 应为 {expected} 字节，实际 {len(payload)}")
    try:
        if code in (PathEventCode.ACK, PathEventCode.NACK):
            request_sequence, path_id, result = struct.unpack("<BBB", payload)
            return AckEvent(request_sequence, path_id, PathResult(result), code == PathEventCode.ACK)
        if code == PathEventCode.POINT_DONE:
            return PointDoneEvent(*struct.unpack("<BBiiii", payload))
        if code == PathEventCode.DONE:
            return DoneEvent(*struct.unpack("<BBii", payload))
        if code == PathEventCode.ERROR:
            path_id, point_index, segment, error, x_mm, y_mm = struct.unpack("<BBBBii", payload)
            return ErrorEvent(path_id, point_index, PathSegment(segment), PathResult(error), x_mm, y_mm)
        if code == PathEventCode.ARM_STATUS:
            sequence, motor_id, state, result, count, reverse = struct.unpack("<BBBBIB", payload)
            return ArmStatusEvent(sequence, motor_id, ArmState(state),
                                  PathResult(result), count, bool(reverse))
        if code == PathEventCode.ARM_DONE:
            sequence, motor_id, result, count, reverse = struct.unpack("<BBBIB", payload)
            return ArmDoneEvent(sequence, motor_id, PathResult(result), count, bool(reverse))
        path_id, state, point_index, segment, count, x_mm, y_mm, tx_mm, ty_mm = struct.unpack(
            "<BBBBBiiii", payload)
        return StatusEvent(path_id, PathState(state), point_index, PathSegment(segment), count,
                           x_mm, y_mm, tx_mm, ty_mm)
    except ValueError as exc:
        raise ValueError(f"事件 {code.name} 含未知枚举值") from exc
