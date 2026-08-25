"""复用现有串口写入的九点路径请求关联与任务状态机。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable

from .path_protocol import (
    AckEvent, DoneEvent, ErrorEvent, FrameDecoder, PathCommand, PathEvent,
    PathPoint, PathResult, PathState, PointDoneEvent, StatusEvent,
    encode_frame, encode_upload, parse_event,
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
    def __init__(self) -> None:
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
        self._sequence = 0
        self._listeners: list[Callable[[PathEvent], None]] = []
        self._terminal = threading.Event()
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
            elif isinstance(event, (DoneEvent, ErrorEvent)):
                self.task_active = False
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
                timeout_s: float | None = None) -> AckEvent | StatusEvent:
        if not self.connection_available:
            raise ConnectionError("STM32 串口当前不可用")
        with self._request_lock:
            sequence = self._next_sequence()
            pending = _Pending()
            with self._lock:
                if command == PathCommand.STATUS_REQ:
                    if self._status_pending is not None:
                        raise PathProtocolError("已有状态查询正在等待")
                    self._status_pending = pending
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
                raise
            if isinstance(event, AckEvent):
                if not event.accepted or event.result != PathResult.OK:
                    raise PathNackError(command, event)
                return event
            if not isinstance(event, StatusEvent):
                raise PathProtocolError(f"{command.name} 收到意外事件")
            return event

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
        event = self.request(PathCommand.STOP)
        assert isinstance(event, AckEvent)
        return event

    def upload_and_start(self, path_id: int, points: Iterable[PathPoint],
                         reset_origin: bool = True) -> StatusEvent | None:
        points_tuple = tuple(points)
        upload_payload = encode_upload(path_id, points_tuple)
        with self._workflow_lock:
            with self._lock:
                if self.task_active:
                    raise PathProtocolError("已有路径正在运行，不能并发启动")
                self._terminal.clear()
            start_sent = False
            try:
                self.request(PathCommand.CLEAR)
                if reset_origin:
                    self.request(PathCommand.RESET_ORIGIN)
                self.request(PathCommand.UPLOAD, upload_payload)
                start_sent = True
                self.request(PathCommand.START, bytes((path_id,)))
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
        event = self.last_event
        if not isinstance(event, (DoneEvent, ErrorEvent)):
            raise PathProtocolError("终止信号缺少 DONE/ERROR 事件")
        return event

    def connection_lost(self) -> None:
        with self._lock:
            self.connection_available = False
            self._decoder.reset()
            pending = list(self._pending.values())
            self._pending.clear()
            if self._status_pending is not None:
                pending.append(self._status_pending)
                self._status_pending = None
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
