"""复用现有串口写入的九点路径请求关联与任务状态机。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable

from .path_protocol import (
    AckEvent, ArmDoneEvent, ArmState, ArmStatusEvent, DoneEvent, ErrorEvent,
    FrameDecoder, PathCommand, PathEvent,
    PathPoint, PathResult, PathState, PointDoneEvent, StatusEvent,
    encode_arm_jog, encode_arm_move, encode_frame, encode_servo_angle,
    encode_upload, parse_event,
)


LOG = logging.getLogger(__name__)


class PathProtocolError(RuntimeError):
    pass


class PathRequestTimeout(TimeoutError):
    pass


class PathNackError(PathProtocolError):
    def __init__(self, command: PathCommand, event: AckEvent) -> None:
        super().__init__(f"{command.name} 被 STM32 拒绝：{event.result.name}")
        self.command = command
        self.event = event


class _Pending:
    def __init__(self, sequence: int = 0) -> None:
        self.sequence = sequence
        self.ready = threading.Event()
        self.event: PathEvent | None = None


class PathClient:
    """线程安全的请求客户端；接收线程只解析和分发，不阻塞等待。"""

    def __init__(self, write: Callable[[bytes], None], request_timeout_s: float = 1.0) -> None:
        self._write = write
        self._request_timeout_s = request_timeout_s
        self._decoder = FrameDecoder()
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._workflow_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._status_pending: _Pending | None = None
        self._arm_status_pending: _Pending | None = None
        self._arm_terminal = threading.Event()
        self.last_arm_event: ArmDoneEvent | None = None
        self._active_arm_sequence: int | None = None
        self.last_path_start_sequence: int | None = None
        self._sequence = 0
        self._listeners: list[Callable[[PathEvent], None]] = []
        self._terminal = threading.Event()
        self._last_terminal_event: DoneEvent | ErrorEvent | None = None
        self.current_status: StatusEvent | None = None
        self.last_event: PathEvent | None = None
        self.task_active = False
        self.connection_available = True
        self._last_invalid_log = 0.0
        self._invalid_suppressed = 0

    def add_listener(self, listener: Callable[[PathEvent], None]) -> None:
        self._listeners.append(listener)

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence = self._sequence % 255 + 1
            return self._sequence

    def feed(self, data: bytes) -> None:
        for frame in self._decoder.feed(data):
            try:
                event = parse_event(frame)
            except ValueError as exc:
                now = time.monotonic()
                if now - self._last_invalid_log >= 1.0:
                    suffix = (f"（此前已限频忽略 {self._invalid_suppressed} 条）"
                              if self._invalid_suppressed else "")
                    LOG.warning("忽略非法路径事件：%s%s", exc, suffix)
                    self._last_invalid_log = now
                    self._invalid_suppressed = 0
                else:
                    self._invalid_suppressed += 1
                continue
            self._dispatch(event)

    def _dispatch(self, event: PathEvent) -> None:
        with self._lock:
            self.last_event = event
            if isinstance(event, AckEvent):
                pending = self._pending.pop(event.request_sequence, None)
                if pending is None and self._arm_status_pending is not None:
                    if self._arm_status_pending.sequence == event.request_sequence:
                        pending = self._arm_status_pending
                        self._arm_status_pending = None
                if pending is None and self._status_pending is not None:
                    if self._status_pending.sequence == event.request_sequence:
                        pending = self._status_pending
                        self._status_pending = None
                if pending is not None:
                    pending.event = event
                    pending.ready.set()
            elif isinstance(event, StatusEvent):
                self.current_status = event
                pending = self._status_pending
                self._status_pending = None
                if pending is not None:
                    pending.event = event
                    pending.ready.set()
                self.task_active = event.state in (PathState.RUN_X, PathState.RUN_Y)
            elif isinstance(event, ArmStatusEvent):
                pending = self._arm_status_pending
                if pending is not None and pending.sequence == event.request_sequence:
                    self._arm_status_pending = None
                    pending.event = event
                    pending.ready.set()
            elif isinstance(event, ArmDoneEvent):
                self.last_arm_event = event
                self._arm_terminal.set()
            elif isinstance(event, (DoneEvent, ErrorEvent)):
                self.task_active = False
                self._last_terminal_event = event
                self._terminal.set()
            elif isinstance(event, PointDoneEvent):
                self.task_active = True
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                LOG.exception("路径事件监听器异常")

    def _wait(self, pending: _Pending, command: PathCommand, timeout_s: float | None) -> PathEvent:
        timeout = self._request_timeout_s if timeout_s is None else timeout_s
        if not pending.ready.wait(timeout):
            raise PathRequestTimeout(f"{command.name} 在 {timeout:.2f}s 内未收到响应")
        if pending.event is None:
            raise ConnectionError(f"等待 {command.name} 时串口连接中断")
        return pending.event

    def request(self, command: PathCommand, payload: bytes = b"",
                timeout_s: float | None = None) -> AckEvent | StatusEvent | ArmStatusEvent:
        if not self.connection_available:
            raise ConnectionError("STM32 串口当前不可用")
        with self._request_lock:
            sequence = self._next_sequence()
            pending = _Pending(sequence)
            with self._lock:
                if command == PathCommand.STATUS_REQ:
                    if self._status_pending is not None:
                        raise PathProtocolError("已有状态查询正在等待")
                    self._status_pending = pending
                elif command == PathCommand.ARM_STATUS_REQ:
                    if self._arm_status_pending is not None:
                        raise PathProtocolError("已有机械臂状态查询正在等待")
                    self._arm_status_pending = pending
                else:
                    self._pending[sequence] = pending
            try:
                # 先登记再写串口，避免极速回包抢在 pending 建立之前。
                self._write(encode_frame(command, sequence, payload))
                event = self._wait(pending, command, timeout_s)
            except Exception:
                with self._lock:
                    self._pending.pop(sequence, None)
                    if self._status_pending is pending:
                        self._status_pending = None
                    if self._arm_status_pending is pending:
                        self._arm_status_pending = None
                raise
            if isinstance(event, AckEvent):
                if not event.accepted or event.result != PathResult.OK:
                    raise PathNackError(command, event)
                return event
            if not isinstance(event, (StatusEvent, ArmStatusEvent)):
                raise PathProtocolError(f"{command.name} 收到意外事件")
            return event

    def arm_jog(self, motor_id: int, direction: int, pulses: int) -> None:
        payload = encode_arm_jog(motor_id, direction, pulses)
        with self._lock:
            self._arm_terminal.clear()
            self.last_arm_event = None
            self._active_arm_sequence = None
        event = self.request(PathCommand.ARM_JOG, payload)
        assert isinstance(event, AckEvent)
        if event.path_id != motor_id:
            raise PathProtocolError("机械臂 ACK 电机编号不匹配")
        self._active_arm_sequence = event.request_sequence

    def arm_move(self, motor_id: int, direction: int, pulses: int,
                 rpm: int, acceleration: int) -> AckEvent:
        payload = encode_arm_move(motor_id, direction, pulses, rpm, acceleration)
        with self._lock:
            self._arm_terminal.clear()
            self.last_arm_event = None
            self._active_arm_sequence = None
        event = self.request(PathCommand.ARM_MOVE, payload)
        assert isinstance(event, AckEvent)
        if event.path_id != motor_id:
            raise PathProtocolError("机械臂 ACK 电机编号不匹配")
        self._active_arm_sequence = event.request_sequence
        return event

    def arm_stop(self, motor_id: int) -> None:
        if motor_id not in (5, 6, 7):
            raise ValueError("只能停止 5/6/7 号电机")
        event = self.request(PathCommand.ARM_STOP, bytes((motor_id,)))
        assert isinstance(event, AckEvent)
        if event.path_id != motor_id:
            raise PathProtocolError("机械臂停止 ACK 电机编号不匹配")

    def arm_status(self, motor_id: int) -> ArmStatusEvent:
        if motor_id not in (5, 6, 7):
            raise ValueError("只能查询 5/6/7 号电机")
        event = self.request(PathCommand.ARM_STATUS_REQ, bytes((motor_id,)))
        assert isinstance(event, ArmStatusEvent)
        if event.motor_id != motor_id:
            raise PathProtocolError("机械臂状态电机编号不匹配")
        return event

    def set_servo_angle(self, servo_id: int, angle_tenths: int) -> AckEvent:
        """设置舵机角度；ACK 只表示 F4 已接受并更新 PWM。"""
        payload = encode_servo_angle(servo_id, angle_tenths)
        event = self.request(PathCommand.SERVO_SET_ANGLE, payload)
        assert isinstance(event, AckEvent)
        if event.path_id != servo_id:
            raise PathProtocolError("舵机 ACK 编号不匹配")
        return event

    def wait_arm_done(self, motor_id: int, timeout_s: float = 4.0) -> ArmDoneEvent:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PathRequestTimeout("等待机械臂到位事件超时；请查询状态并人工检查")
            if self._arm_terminal.wait(min(0.2, remaining)):
                event = self.last_arm_event
                if (event is None or event.motor_id != motor_id or
                        event.request_sequence != self._active_arm_sequence):
                    raise PathProtocolError("机械臂到位事件编号或请求序号不匹配")
                return event
            # F4 也通过状态查询看到链路活性；查询失败时不推定动作完成。
            self.arm_status(motor_id)

    def query_status(self, timeout_s: float | None = None) -> StatusEvent:
        event = self.request(PathCommand.STATUS_REQ, timeout_s=timeout_s)
        assert isinstance(event, StatusEvent)
        return event

    def clear(self) -> AckEvent:
        event = self.request(PathCommand.CLEAR)
        assert isinstance(event, AckEvent)
        return event

    def reset_origin(self) -> AckEvent:
        event = self.request(PathCommand.RESET_ORIGIN)
        assert isinstance(event, AckEvent)
        return event

    def stop(self) -> AckEvent:
        # F4 最多用 1 秒确认编码器静止，串口传输和排队需留额外余量。
        event = self.request(PathCommand.STOP,
                             timeout_s=max(self._request_timeout_s, 2.0))
        assert isinstance(event, AckEvent)
        return event

    def upload_and_start(self, path_id: int, points: Iterable[PathPoint],
                         reset_origin: bool = True) -> StatusEvent | None:
        points_tuple = tuple(points)
        if len(points_tuple) != 1:
            raise ValueError("STM32 单段接口一次只能接受一个绝对毫米目标")
        upload_payload = encode_upload(path_id, points_tuple)
        with self._workflow_lock:
            with self._lock:
                if self.task_active:
                    raise PathProtocolError("已有路径正在运行，不能并发启动")
                self._last_terminal_event = None
                self._terminal.clear()
                self.last_path_start_sequence = None
            start_sent = False
            try:
                self.request(PathCommand.CLEAR)
                if reset_origin:
                    self.request(PathCommand.RESET_ORIGIN)
                self.request(PathCommand.UPLOAD, upload_payload)
                start_sent = True
                start_ack = self.request(PathCommand.START, bytes((path_id,)))
                assert isinstance(start_ack, AckEvent)
                self.last_path_start_sequence = start_ack.request_sequence
            except PathRequestTimeout as timeout:
                # ACK 丢失不等于命令未执行；只查询真实状态，绝不盲目重发命令。
                try:
                    status = self.query_status()
                except Exception as status_error:
                    raise PathRequestTimeout(f"{timeout}；状态也无法确认，任务状态未知") from status_error
                if (start_sent and status.path_id == path_id and
                        status.state in (PathState.RUN_X, PathState.RUN_Y)):
                    with self._lock:
                        self.task_active = True
                    return status
                raise PathRequestTimeout(f"{timeout}；STM32 状态为 {status.state.name}，流程已停止") from timeout
            with self._lock:
                if not self._terminal.is_set():
                    self.task_active = True
            return self.current_status

    def wait_until_terminal(self, timeout_s: float | None = None) -> DoneEvent | ErrorEvent:
        if not self._terminal.wait(timeout_s):
            raise PathRequestTimeout("等待路径完成事件超时")
        event = self._last_terminal_event
        if not isinstance(event, (DoneEvent, ErrorEvent)):
            raise PathProtocolError("终止信号缺少 DONE/ERROR 事件")
        return event

    def wait_until_terminal_with_keepalive(self, keepalive_s: float = 0.25,
                                           timeout_s: float | None = None) -> DoneEvent | ErrorEvent:
        """等待段完成，同时周期性查询状态以发现串口失联。

        ``PATH_STATUS_REQ`` 是有效协议帧，也使 ACK 丢失或串口重连后的状态可观察。
        它不根据理论运行时间推进路径；只有 DONE 才表示到达。
        """
        if keepalive_s <= 0.0:
            raise ValueError("keepalive_s 必须大于 0")
        start = time.monotonic()
        while True:
            remaining = None if timeout_s is None else timeout_s - (time.monotonic() - start)
            if remaining is not None and remaining <= 0.0:
                raise PathRequestTimeout("等待路径完成事件超时")
            interval = keepalive_s if remaining is None else min(keepalive_s, remaining)
            if self._terminal.wait(interval):
                event = self._last_terminal_event
                if isinstance(event, (DoneEvent, ErrorEvent)):
                    return event
                raise PathProtocolError("终止信号缺少 DONE/ERROR 事件")
            # 丢失 DONE/ERROR 时，终态状态也必须让等待失败，避免永久等待。
            status = self.query_status(timeout_s=self._request_timeout_s)
            if self._terminal.is_set():
                continue
            if status.state not in (PathState.RUN_X, PathState.RUN_Y):
                raise PathProtocolError(
                    f"未收到 DONE/ERROR，STM32 路径状态已是 {status.state.name}；禁止推进下一段")

    def connection_lost(self) -> None:
        with self._lock:
            self.connection_available = False
            self._decoder.reset()
            pending = list(self._pending.values())
            self._pending.clear()
            if self._status_pending is not None:
                pending.append(self._status_pending)
                self._status_pending = None
            if self._arm_status_pending is not None:
                pending.append(self._arm_status_pending)
                self._arm_status_pending = None
        # 唤醒等待者，由其按“状态未知”处理。
        for item in pending:
            item.ready.set()

    def connection_restored(self) -> None:
        with self._lock:
            self.connection_available = True
            self._decoder.reset()

    def recover_status_async(self) -> None:
        def recover() -> None:
            try:
                self.query_status()
            except Exception as exc:
                LOG.warning("串口重连后 PATH_STATUS 恢复失败：%s", exc)

        threading.Thread(target=recover, name="path-status-recover", daemon=True).start()
