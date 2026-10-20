"""相机原画只编码一次，并按可设置的帧率和峰值速度发送。"""

from __future__ import annotations

import threading
import time


def normalize_stream_limits(fps, peak_kib_per_sec, quality=70) -> tuple[float, float, int]:
    """帧率、峰值传输速度和单次 JPEG 质量。"""
    fps_value = float(fps)
    peak_value = float(peak_kib_per_sec)
    quality_value = int(quality)
    if not 1 <= fps_value <= 30:
        raise ValueError("帧率必须在 1..30")
    if not 32 <= peak_value <= 8192:
        raise ValueError("峰值传输必须在 32..8192 KiB/s")
    if not 30 <= quality_value <= 95:
        raise ValueError("JPEG 质量必须在 30..95")
    return fps_value, peak_value, quality_value


def stream_limits_from(config: dict) -> tuple[float, float, int]:
    raw = dict(config.get("vision_stream") or {})
    return normalize_stream_limits(
        raw.get("fps", raw.get("target_fps", 12)),
        raw.get("peak_kib_per_sec", 512),
        raw.get("jpeg_quality", 70),
    )


class FrameSlot:
    """只保留最新一帧。发布不等待识别或传输。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame = None
        self._seq = 0
        self.closed = False

    def publish(self, frame) -> int:
        with self._condition:
            self._frame = frame
            self._seq += 1
            seq = self._seq
            self._condition.notify_all()
            return seq

    def wait_next(self, seen: int, timeout: float):
        """返回比 ``seen`` 更新的帧副本。关闭且没有新帧时返回 ``None``。"""
        with self._condition:
            if not self.closed and self._seq <= seen:
                self._condition.wait(timeout)
            if self._frame is None or self._seq <= seen:
                if self.closed:
                    return None
                return seen, None
            frame = self._frame
            seq = self._seq
        return seq, frame.copy()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()


class StreamBudget:
    """按帧率节拍发送，并用令牌桶限制峰值字节数。超限就丢掉这一帧，不二次编码。"""

    def __init__(self, fps: float, peak_kib_per_sec: float, quality: int = 70) -> None:
        self.fps, self.peak_kib, self.quality = normalize_stream_limits(
            fps, peak_kib_per_sec, quality)
        self._next = time.perf_counter()
        self._tokens = self.peak_kib * 1024.0
        self._token_time = time.perf_counter()
        self._encode_quality = self.quality
        self._sent = 0
        self._frames = 0
        self._window = time.perf_counter()
        self.measured_fps = 0.0
        self.measured_kib = 0.0
        self.dropped = 0

    def configure(self, fps: float, peak_kib_per_sec: float, quality: int) -> None:
        self.fps, self.peak_kib, self.quality = normalize_stream_limits(
            fps, peak_kib_per_sec, quality)
        self._encode_quality = min(self._encode_quality, self.quality)

    @property
    def peak_bytes(self) -> float:
        return self.peak_kib * 1024.0

    def wait_turn(self, stop: threading.Event) -> bool:
        interval = 1.0 / self.fps
        delay = self._next - time.perf_counter()
        if delay > 0 and stop.wait(delay):
            return False
        now = time.perf_counter()
        if self._next < now - interval:
            self._next = now
        self._next += interval
        return not stop.is_set()

    def encode_quality(self) -> int:
        return int(self._encode_quality)

    def accept(self, nbytes: int) -> bool:
        now = time.perf_counter()
        elapsed = max(0.0, now - self._token_time)
        self._token_time = now
        self._tokens = min(self.peak_bytes, self._tokens + elapsed * self.peak_bytes)
        if nbytes > self._tokens:
            self.dropped += 1
            self._encode_quality = max(30, self._encode_quality - 8)
            self._measure(now)
            return False
        self._tokens -= nbytes
        self._sent += nbytes
        self._frames += 1
        if self._encode_quality < self.quality and self._tokens > self.peak_bytes * 0.5:
            self._encode_quality = min(self.quality, self._encode_quality + 4)
        self._measure(now)
        return True

    def _measure(self, now: float) -> None:
        span = now - self._window
        if span < 0.5:
            return
        self.measured_fps = self._frames / span
        self.measured_kib = (self._sent / 1024.0) / span
        self._frames = 0
        self._sent = 0
        self._window = now
