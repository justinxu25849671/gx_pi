"""树莓派与 STM32 的九点路径协议双向链路。

@File    : logistics_robot/serial_link.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 提供二进制路径收发。
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from .path_client import PathClient
from .path_protocol import PathPoint, StatusEvent


LOG = logging.getLogger(__name__)


class PathLink(Protocol):
    """真实串口路径链路的最小接口。"""
    def start_path(self, path_id: int, points: tuple[PathPoint, ...],
                   reset_origin: bool = True) -> StatusEvent | None: ...
    def stop_path(self) -> None: ...
    def query_path_status(self) -> StatusEvent: ...
    def raw(self, payload: bytes) -> None: ...
    def close(self) -> None: ...
    def close_after_estop(self) -> None: ...


class SerialPathLink:
    """单一 pyserial 连接承载路径二进制协议的收发。"""

    def __init__(self, port: str, baudrate: int, request_timeout_s: float = 1.0,
                 reconnect_period_s: float = 1.0) -> None:
        """打开 STM32 串口；打开失败时不允许程序进入真实运行。"""
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("未安装 pyserial，请先执行 pip install -r requirements.txt") from exc
        self._serial_module = serial
        self._port = port
        self._baudrate = baudrate
        self._reconnect_period_s = reconnect_period_s
        self._serial = self._open_serial()
        self._serial_lock = threading.Lock()
        self._stop_reader = threading.Event()
        self.path_client = PathClient(self.raw, request_timeout_s)
        self._reader = threading.Thread(target=self._reader_loop, name="stm32-serial-rx", daemon=True)
        self._reader.start()

    def _open_serial(self):
        return self._serial_module.Serial(
            port=self._port, baudrate=self._baudrate, timeout=0.02, write_timeout=0.1)

    def _disconnect(self) -> None:
        with self._serial_lock:
            connection = self._serial
            self._serial = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self.path_client.connection_lost()

    def _reader_loop(self) -> None:
        """后台读取二进制事件；断线后复用同一连接对象自动恢复。"""
        while not self._stop_reader.is_set():
            connection = self._serial
            if connection is None:
                if self._stop_reader.wait(self._reconnect_period_s):
                    break
                try:
                    connection = self._open_serial()
                except Exception as exc:
                    LOG.warning("STM32 串口重连失败：%s", exc)
                    continue
                with self._serial_lock:
                    self._serial = connection
                LOG.info("STM32 串口已重连，正在恢复路径状态")
                self.path_client.connection_restored()
                self.path_client.recover_status_async()
            try:
                data = connection.read(256)
            except (self._serial_module.SerialException, OSError, TypeError) as exc:
                # pyserial 在另一个线程关闭端口时可能以 fd=None 抛出 TypeError。
                if self._stop_reader.is_set():
                    break
                LOG.warning("STM32 串口断开：%s", exc)
                self._disconnect()
                continue
            if data:
                self.path_client.feed(data)

    def raw(self, payload: bytes) -> None:
        """写入已由 ``protocol`` 编码的一帧命令。"""
        write_error = None
        with self._serial_lock:
            connection = self._serial
            if connection is None:
                raise ConnectionError("STM32 串口当前未连接")
            try:
                written = connection.write(payload)
            except (self._serial_module.SerialException, OSError) as exc:
                written = 0
                write_error = exc
        if write_error is not None:
            self._disconnect()
            raise ConnectionError("STM32 串口写入失败") from write_error
        if written != len(payload):
            raise ConnectionError(f"STM32 串口短写：{written}/{len(payload)}")

    def start_path(self, path_id: int, points: tuple[PathPoint, ...],
                   reset_origin: bool = True) -> StatusEvent | None:
        return self.path_client.upload_and_start(path_id, points, reset_origin)

    def stop_path(self) -> None:
        self.path_client.stop()

    def query_path_status(self) -> StatusEvent:
        return self.path_client.query_status()

    def close(self) -> None:
        """关闭路径串口连接。"""
        self._stop_reader.set()
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=0.2)
        self._disconnect()
