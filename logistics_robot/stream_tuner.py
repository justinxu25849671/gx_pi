"""按实际帧率自动升降检测分辨率、预览宽度和 JPEG 画质。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamLevel:
    detect_width: int
    preview_width: int
    jpeg_quality: int


LEVELS = (
    StreamLevel(960, 1280, 85),
    StreamLevel(640, 960, 72),
    StreamLevel(480, 640, 58),
    StreamLevel(320, 480, 42),
)


class StreamTuner:
    """帧率低于目标就降画质，高于目标一段时间再升回去。"""

    def __init__(self, target_fps: float = 12.0) -> None:
        self.target_fps = float(target_fps)
        self._index = 1
        self._slow = 0
        self._fast = 0
        self.level = LEVELS[self._index]

    def observe(self, fps: float) -> StreamLevel:
        if fps <= 0:
            return self.level
        if fps < self.target_fps * 0.85:
            self._slow += 1
            self._fast = 0
            if self._slow >= 6 and self._index < len(LEVELS) - 1:
                self._index += 1
                self._slow = 0
        elif fps > self.target_fps * 1.35:
            self._fast += 1
            self._slow = 0
            if self._fast >= 10 and self._index > 0:
                self._index -= 1
                self._fast = 0
        else:
            self._slow = 0
            self._fast = 0
        self.level = LEVELS[self._index]
        return self.level
